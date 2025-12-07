#!/usr/bin/env python3
import argparse
import json
import os
import re
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml
from vllm import LLM, SamplingParams

# Repo layout assumptions:
#   model-workbench/
#     config/models.yaml
#     benchmarks/run_bench.py
#
# Models expected at:
#   ~/models/<org>/<repo_id>/...

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "models.yaml"
MODELS_ROOT = Path.home() / "models"

PROMPTS = {
    "short": "Explain speculative decoding in 2 sentences.",
    "medium": "Summarize key tradeoffs between tensor parallelism and pipeline parallelism.",
    "long": "Write a concise technical overview of KV cache and why it matters for long context.",
}

def sanitize(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)

def get_gpu_info():
    """
    Returns a dict with driver_version and a list of GPUs.
    Uses nvidia-smi. Safe to fail quietly.
    """
    try:
        # Query driver version separately for clarity
        drv = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True
        ).strip().splitlines()
        driver_version = drv[0].strip() if drv else None

        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,pci.bus_id,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True
        ).strip()

        gpus = []
        if out:
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    idx, name, uuid, bus, mem = parts[:5]
                    gpus.append({
                        "index": int(idx) if idx.isdigit() else idx,
                        "name": name,
                        #"uuid": uuid,
                        #"pci_bus_id": bus,
                        "memory_total_mib": int(mem) if mem.isdigit() else mem,
                    })

        return {
            "driver_version": driver_version,
            "gpus": gpus,
        }
    except Exception as e:
        return {
            "error": str(e),
            "driver_version": None,
            "gpus": [],
        }

def load_config_models():
    """
    Expects config/models.yaml like:

    models:
      - source: hf
        repo_id: Org/Model
      - source: hf
        repo_id: Org/OtherModel
    """
    if not CONFIG_PATH.exists():
        return []

    data = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    items = data.get("models") or []
    if not isinstance(items, list):
        raise SystemExit("config/models.yaml: 'models' must be a list")

    out = []
    for item in items:
        if isinstance(item, dict) and item.get("source") == "hf" and item.get("repo_id"):
            out.append(item["repo_id"])
    return out

def resolve_model_ref(model_arg: str) -> str:
    """
    Prefer local mirror for most models, BUT avoid rewriting openai/gpt-oss*
    into a filesystem path due to observed vLLM path-handling issues.
    """
    if model_arg.startswith("openai/gpt-oss"):
        return model_arg

    if "/" in model_arg:
        local = MODELS_ROOT / model_arg
        if local.exists():
            return str(local)

    return model_arg

def infer_tag(cli_tag: str | None) -> str:
    if cli_tag:
        return cli_tag

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd:
        return "unknown-gpu"

    devs = [d.strip() for d in cvd.split(",") if d.strip() != ""]
    if len(devs) <= 1:
        return "single-gpu"
    if len(devs) == 2:
        return "dual-gpu"
    return f"{len(devs)}-gpu"

def try_get_token_counts(outputs):
    """
    Best-effort extraction of token counts from vLLM outputs.
    API can vary across versions.
    """
    prompt_tokens = None
    gen_tokens = None
    try:
        ro = outputs[0]
        if hasattr(ro, "prompt_token_ids") and ro.prompt_token_ids is not None:
            prompt_tokens = len(ro.prompt_token_ids)

        comp = ro.outputs[0]
        if hasattr(comp, "token_ids") and comp.token_ids is not None:
            gen_tokens = len(comp.token_ids)
    except Exception:
        pass

    return prompt_tokens, gen_tokens

def bench_once(llm: LLM, prompt: str, max_tokens: int, temperature: float, seed: int):
    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
    )

    t0 = time.perf_counter()
    outputs = llm.generate([prompt], params)
    t1 = time.perf_counter()

    prompt_tokens, gen_tokens = try_get_token_counts(outputs)

    text = ""
    try:
        text = outputs[0].outputs[0].text
    except Exception:
        pass

    wall = t1 - t0
    tok_per_s = None
    if gen_tokens and wall > 0:
        tok_per_s = gen_tokens / wall

    return {
        "wall_s": wall,
        "max_tokens": max_tokens,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": gen_tokens,
        "tok_per_s": tok_per_s,
        "temperature": temperature,
        "seed": seed,
        "output_text": text,
    }

def run_model(model_id: str, args):
    model_ref = resolve_model_ref(model_id)

    llm_kwargs = dict(
        model=model_ref,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,  # engine-level seed
    )
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len

    print(f"\n== Loading model ==")
    print(f"model_id:  {model_id}")
    print(f"model_ref: {model_ref}")
    llm = LLM(**llm_kwargs)
    gpu_info = get_gpu_info()
    prompt = PROMPTS[args.prompt_set]

    # Warmup (not recorded)
    _ = bench_once(
        llm,
        prompt,
        max_tokens=min(64, args.max_tokens),
        temperature=args.temperature,
        seed=args.seed,
    )

    # Timed iterations
    results = []
    for i in range(args.iterations):
        # If you want per-iteration randomness while still reproducible:
        # we can offset the seed slightly.
        iter_seed = args.seed if not args.vary_seed else (args.seed + i)
        results.append(
            bench_once(
                llm,
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=iter_seed,
            )
        )

    def med(key):
        vals = [r.get(key) for r in results if r.get(key) is not None]
        return statistics.median(vals) if vals else None

    summary = {
        "iterations": args.iterations,
        "median_wall_s": med("wall_s"),
        "median_generated_tokens": med("generated_tokens"),
        "median_tok_per_s": med("tok_per_s"),
    }

    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        "model_id": model_id,
        "model_ref": model_ref,
        "gpu_info": gpu_info,
        "tag": args.tag,
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "temperature": args.temperature,
            "seed": args.seed,
            "vary_seed": args.vary_seed,
        },
        "bench": {
            "prompt_set": args.prompt_set,
            "prompt": prompt,
            "iterations": results,
            "summary": summary,
        },
    }

    out_base = ROOT / "benchmarks" / "results" / args.tag
    out_base.mkdir(parents=True, exist_ok=True)

    fname = f"{datetime.now().strftime('%Y-%m-%d')}_{sanitize(model_id)}.json"
    out_path = out_base / fname

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote: {out_path}")

def main():
    ap = argparse.ArgumentParser(description="vLLM benchmark runner")

    ap.add_argument("--model", help="HF repo_id or local path")
    #ap.add_argument("--all", action="store_true", help="Benchmark all models in config/models.yaml")

    # Optional; inferred from CUDA_VISIBLE_DEVICES if omitted
    ap.add_argument("--tag", default=None, help="Optional output tag; inferred if omitted")

    ap.add_argument("--iterations", type=int, default=3,
                    help="Number of timed iterations (median reported)")

    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vary-seed", action="store_true",
                    help="If set, increments seed each iteration (seed+i)")

    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--prompt-set", default="short", choices=tuple(PROMPTS.keys()))

    args = ap.parse_args()

    if not args.model:
        ap.error("Provide --model <id>")

    args.tag = infer_tag(args.tag)

    #if args.all:
    #    models = load_config_models()
    #    if not models:
    #        raise SystemExit("No models found in config/models.yaml")
    #    for m in models:
    #        run_model(m, args)
    #else:
    #    run_model(args.model, args)
    run_model(args.model, args)

if __name__ == "__main__":
    main()
