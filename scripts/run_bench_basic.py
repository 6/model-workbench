#!/usr/bin/env python3
"""
Model Workbench Basic Benchmark Runner

Uses native HuggingFace transformers inference with device_map="auto" for
multi-GPU support via accelerate. Simpler than run_bench.py - no vLLM or
llama-server dependency.

Use this for models that:
  - Require transformers 5.x (e.g., GLM-4.6V-FP8)
  - Don't yet have vLLM/llama.cpp support
  - Need simple multi-GPU inference via accelerate

Examples:

  # Basic usage
  uv run python scripts/run_bench_basic.py --model zai-org/GLM-4.6V-FP8

  # With custom dtype and trust-remote-code
  uv run python scripts/run_bench_basic.py \
    --model zai-org/GLM-4.6V-FP8 \
    --dtype auto \
    --trust-remote-code

  # From local path
  uv run python scripts/run_bench_basic.py \
    --model ~/models/zai-org/GLM-4.6V-FP8
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = Path.home() / "models"
RESULTS_ROOT = ROOT / "benchmarks"

PROMPTS = {
    "short": "Explain speculative decoding in 2 sentences.",
    "medium": "Summarize key tradeoffs between tensor parallelism and pipeline parallelism.",
    "long": "Write a concise technical overview of KV cache and why it matters for long context.",
}

# ----------------------------
# Utility
# ----------------------------

def sanitize(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)

# ----------------------------
# GPU info + tag inference
# ----------------------------

def get_gpu_info():
    """
    Returns a dict with driver_version and a list of GPUs.
    Uses nvidia-smi. Safe to fail quietly.
    """
    try:
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
                        "index": int(idx) if str(idx).isdigit() else idx,
                        "name": name,
                        "memory_total_mib": int(mem) if str(mem).isdigit() else mem,
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

def infer_tag(cli_tag: str | None) -> str:
    if cli_tag:
        return cli_tag

    info = get_gpu_info()
    gpus = info.get("gpus") or []
    n = len(gpus)

    if n == 0:
        return "unknown-gpu"
    if n == 1:
        return "single-gpu"
    if n == 2:
        return "dual-gpu"
    return f"{n}-gpu"

# ----------------------------
# Model resolution
# ----------------------------

def resolve_model_path(model_arg: str) -> str:
    """
    Resolve model path, preferring local mirror under ~/models if it exists.
    """
    if "/" in model_arg:
        local = MODELS_ROOT / model_arg
        if local.exists():
            return str(local)

    p = Path(model_arg).expanduser()
    if p.exists():
        return str(p)

    return model_arg

# ----------------------------
# Benchmarking
# ----------------------------

def bench_once(model, tokenizer, prompt: str, max_tokens: int, temperature: float, seed: int):
    """
    Run a single inference and measure throughput.
    """
    torch.manual_seed(seed)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    prompt_tokens = input_ids.shape[1]

    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            **gen_kwargs
        )
    t1 = time.perf_counter()

    gen_tokens = outputs.shape[1] - prompt_tokens
    text = tokenizer.decode(outputs[0][prompt_tokens:], skip_special_tokens=True)

    wall = t1 - t0
    tok_per_s = gen_tokens / wall if wall > 0 else None

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

def med(results, key):
    vals = [r.get(key) for r in results if r.get(key) is not None]
    return statistics.median(vals) if vals else None

# ----------------------------
# Output
# ----------------------------

def write_payload(payload, tag: str, model_id: str):
    out_base = RESULTS_ROOT / tag
    out_base.mkdir(parents=True, exist_ok=True)

    label = sanitize(model_id)
    fname = f"{datetime.now().strftime('%Y-%m-%d')}_{label}.json"

    out_path = out_base / fname
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote: {out_path}")

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Basic benchmark runner (native transformers)")

    ap.add_argument("--model", required=True, help="HF repo_id or local path")
    ap.add_argument("--tag", default=None, help="Optional output tag; inferred if omitted")
    ap.add_argument("--iterations", type=int, default=3,
                    help="Number of timed iterations (median reported)")

    # Model loading options
    ap.add_argument("--dtype", default="auto",
                    help="Model dtype: auto, float16, bfloat16, float32")
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="Trust remote code for custom model architectures")
    ap.add_argument("--attn-implementation", default=None,
                    help="Attention implementation: eager, sdpa, flash_attention_2")

    # Generation options
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vary-seed", action="store_true",
                    help="If set, increments seed each iteration (seed+i)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--prompt-set", default="short", choices=tuple(PROMPTS.keys()))

    args = ap.parse_args()

    # Tag inference from GPU count
    args.tag = infer_tag(args.tag)

    model_id = args.model
    model_path = resolve_model_path(model_id)

    print(f"\n== Loading model ==")
    print(f"model_id:  {model_id}")
    print(f"model_path: {model_path}")
    print(f"engine:    transformers")
    print(f"dtype:     {args.dtype}")
    print(f"device_map: auto")

    # Resolve dtype
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.dtype, args.dtype)

    # Load model with device_map="auto" for multi-GPU
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs,
    )

    prompt = PROMPTS[args.prompt_set]

    # Warmup
    print("\n== Warmup ==")
    _ = bench_once(
        model, tokenizer, prompt,
        max_tokens=min(64, args.max_tokens),
        temperature=args.temperature,
        seed=args.seed,
    )

    # Benchmark iterations
    print(f"\n== Running {args.iterations} iterations ==")
    results = []
    for i in range(args.iterations):
        iter_seed = args.seed if not args.vary_seed else (args.seed + i)
        result = bench_once(
            model, tokenizer, prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=iter_seed,
        )
        print(f"  iter {i+1}: {result['generated_tokens']} tokens in {result['wall_s']:.2f}s "
              f"({result['tok_per_s']:.1f} tok/s)")
        results.append(result)

    summary = {
        "iterations": args.iterations,
        "median_wall_s": med(results, "wall_s"),
        "median_generated_tokens": med(results, "generated_tokens"),
        "median_tok_per_s": med(results, "tok_per_s"),
    }

    print(f"\n== Summary ==")
    print(f"  median: {summary['median_generated_tokens']} tokens in {summary['median_wall_s']:.2f}s "
          f"({summary['median_tok_per_s']:.1f} tok/s)")

    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        "model_id": model_id,
        "model_ref": model_path,
        "engine": "transformers",
        "gpu_info": get_gpu_info(),
        "tag": args.tag,
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "dtype": args.dtype,
            "torch_dtype": str(torch_dtype),
            "device_map": "auto",
            "trust_remote_code": args.trust_remote_code,
            "attn_implementation": args.attn_implementation,
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

    write_payload(payload, args.tag, model_id)

if __name__ == "__main__":
    main()
