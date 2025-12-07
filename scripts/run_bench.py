#!/usr/bin/env python3
"""
Model Workbench Benchmark Runner

Engines:
  - vLLM for standard HF models
  - llama.cpp llama-server for GGUF (including split shards)

Auto routing:
  --engine auto (default)
    * If --model resolves to a GGUF file or a directory containing GGUF,
      or if local mirror under ~/models contains GGUF variants,
      route to llama-server.
    * Else route to vLLM.

Tag inference:
  * DOES NOT use CUDA_VISIBLE_DEVICES.
  * Always inferred from get_gpu_info().

Result filenames:
  * For llama-server GGUF runs:
      - If GGUF is in a quant folder under ~/models/<org>/<repo>/<quant>/...,
        use "<org>/<repo>/<quant>" as label.
      - If repo stores multiple quant files in the repo root,
        use "<org>/<gguf-stem>" as label to avoid collisions and keep org visible.
      - Otherwise fallback to gguf stem.

Output folders:
  * Simplified to:
      benchmarks/<tag>/
    instead of:
      benchmarks/results/<tag>/

Assumptions:
  * Local mirror layout: ~/models/<org>/<repo>/...
  * Repo structure:
      model-workbench/
        config/models.yaml
        scripts/run_bench.py
        benchmarks/<tag>/...

Examples:

  # vLLM path
  uv run python scripts/run_bench.py --model Qwen/Qwen3-30B-A3B-Instruct-2507

  # llama-server auto path (if GGUF exists locally)
  uv run python scripts/run_bench.py --model unsloth/GLM-4.5-Air-GGUF

  # Explicit quant folder (recommended when multiple quants exist)
  uv run python scripts/run_bench.py \
    --engine llama-server \
    --model unsloth/GLM-4.5-Air-GGUF/UD-Q4_K_XL

  # Root-level quant file in repo (no folders)
  uv run python scripts/run_bench.py \
    --engine llama-server \
    --model unsloth/Ministral-3-14B-Instruct-2512-GGUF/Ministral-3-14B-Instruct-2512-UD-Q4_K_XL.gguf

  # Explicit shard file
  uv run python scripts/run_bench.py \
    --engine llama-server \
    --model ~/models/unsloth/GLM-4.5-Air-GGUF/UD-Q4_K_XL/GLM-4.5-Air-UD-Q4_K_XL-00001-of-00002.gguf

  # Use an already running server
  uv run python scripts/run_bench.py \
    --engine llama-server \
    --llama-no-autostart \
    --model unsloth/GLM-4.5-Air-GGUF/UD-Q4_K_XL
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import time
import socket
import signal
from datetime import datetime
from pathlib import Path

import yaml

# requests is only needed for llama-server path
try:
    import requests
except Exception:  # pragma: no cover
    requests = None

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "models.yaml"
MODELS_ROOT = Path.home() / "models"

# ✅ change: remove "results" nesting
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

def run(cmd):
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)

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
# Config loader (optional)
# ----------------------------

def load_config_models():
    """
    Expects config/models.yaml like:

    models:
      - source: hf
        repo_id: Org/Model
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

# ----------------------------
# GGUF / llama-server resolution
# ----------------------------

_SHARD_RE = re.compile(r".*-\d{5}-of-\d{5}\.gguf$")

def is_gguf_file(p: Path) -> bool:
    return p.is_file() and p.suffix == ".gguf"

def find_shard_entrypoints(dir_path: Path):
    """
    Return all *-00001-of-*.gguf under dir_path (sorted).
    """
    return sorted(dir_path.rglob("*-00001-of-*.gguf"))

def list_all_ggufs(dir_path: Path):
    return sorted(dir_path.rglob("*.gguf"))

def pick_gguf_from_dir(dir_path: Path) -> Path | None:
    """
    Pick a GGUF entrypoint from a directory ONLY if unambiguous.

      1) If exactly one shard entrypoint (*-00001-of-*.gguf) -> return it
      2) Else if exactly one non-sharded gguf anywhere -> return it
      3) Else -> None (caller decides whether to raise ambiguity)
    """
    entrypoints = find_shard_entrypoints(dir_path)
    if len(entrypoints) == 1:
        return entrypoints[0]
    if len(entrypoints) > 1:
        return None

    ggufs = list_all_ggufs(dir_path)
    if len(ggufs) == 1:
        return ggufs[0]

    non_shards = [p for p in ggufs if not _SHARD_RE.match(p.name)]
    if len(non_shards) == 1:
        return non_shards[0]

    return None

def raise_if_multiple_variants(dir_path: Path, model_arg: str):
    """
    Raise helpful errors if GGUF variants exist but are ambiguous:
      - multiple shard entrypoints
      - multiple root-level quant files
    """
    ggufs = list_all_ggufs(dir_path)
    if len(ggufs) <= 1:
        return

    entrypoints = find_shard_entrypoints(dir_path)
    if len(entrypoints) > 1:
        raise SystemExit(
            f"Multiple split GGUF variants found under:\n  {dir_path}\n"
            f"Pass a more specific --model like:\n"
            f"  {model_arg}/UD-Q4_K_XL\n"
            f"  or an exact .gguf file path."
        )

    non_shards = [p for p in ggufs if not _SHARD_RE.match(p.name)]
    if len(non_shards) > 1:
        raise SystemExit(
            f"Multiple GGUF files found under:\n  {dir_path}\n"
            f"This repo appears to store multiple quant files in the root.\n"
            f"Please pass an exact quant file path, e.g.:\n"
            f"  {model_arg}/<model>-UD-Q4_K_XL.gguf"
        )

def resolve_local_gguf(model_arg: str) -> Path | None:
    """
    Resolve a GGUF path from:
      1) explicit filesystem path
      2) path relative to ~/models (e.g. org/repo/UD-Q4_K_XL)
      3) HF-style repo id with local mirror under ~/models/org/repo

    Returns:
      - a GGUF file path (prefer shard entrypoint)
      - None if not GGUF

    Ambiguity:
      - If multiple variants exist and model_arg is too broad,
        raise a helpful error telling you to pass a quant folder or exact file.
    """
    # 1) filesystem-ish path
    p = Path(model_arg).expanduser()
    if is_gguf_file(p):
        return p
    if p.is_dir():
        chosen = pick_gguf_from_dir(p)
        if chosen:
            return chosen
        raise_if_multiple_variants(p, model_arg)
        return None

    # 2) treat as path under ~/models
    candidate = MODELS_ROOT / model_arg
    if candidate.exists():
        if is_gguf_file(candidate):
            return candidate
        if candidate.is_dir():
            chosen = pick_gguf_from_dir(candidate)
            if chosen:
                return chosen
            raise_if_multiple_variants(candidate, model_arg)
            return None

    # 3) plain HF-style repo id
    if "/" in model_arg:
        local_repo = MODELS_ROOT / model_arg
        if local_repo.exists() and local_repo.is_dir():
            chosen = pick_gguf_from_dir(local_repo)
            if chosen:
                return chosen
            raise_if_multiple_variants(local_repo, model_arg)

    return None

def resolve_model_ref_vllm(model_arg: str) -> str:
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

# ----------------------------
# llama-server control + HTTP bench
# ----------------------------

def port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False

def start_llama_server(args, gguf_path: Path):
    """
    Starts llama-server unless:
      - --llama-no-autostart is set, OR
      - port is already open (assume user-managed server)
    """
    host = args.llama_host
    port = args.llama_port

    if port_open(host, port):
        return None  # already running

    if args.llama_no_autostart:
        raise SystemExit(
            f"llama-server not detected on {host}:{port} and --llama-no-autostart was set."
        )

    if not gguf_path.exists():
        raise SystemExit(f"GGUF not found: {gguf_path}")

    cmd = [args.llama_server_bin, "-m", str(gguf_path), "--port", str(port)]

    # GPU offload
    if args.llama_n_gpu_layers is not None:
        cmd += ["-ngl", str(args.llama_n_gpu_layers)]

    # Context length
    if args.llama_ctx:
        cmd += ["-c", str(args.llama_ctx)]

    # Parallel sequences
    if args.llama_parallel and args.llama_parallel > 1:
        cmd += ["-np", str(args.llama_parallel)]

    # Extra raw args
    if args.llama_extra_args:
        cmd += args.llama_extra_args

    print("+", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # readiness poll
    for _ in range(200):
        if port_open(host, port):
            break
        time.sleep(0.05)

    if not port_open(host, port):
        try:
            proc.terminate()
        except Exception:
            pass
        raise SystemExit(
            f"llama-server failed to start on {host}:{port}. "
            f"Check your binary path and that it was built with CUDA."
        )

    return proc

def stop_llama_server(proc):
    if not proc:
        return
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass

def bench_once_llama_http(prompt: str, args):
    if requests is None:
        raise SystemExit("Missing dependency: requests. Install with `pip install requests`.")

    base = f"http://{args.llama_host}:{args.llama_port}"
    url = base.rstrip("/") + "/v1/completions"

    payload = {
        "model": "local",
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    t0 = time.perf_counter()
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    t1 = time.perf_counter()

    data = r.json()

    text = ""
    try:
        text = data["choices"][0].get("text", "")
    except Exception:
        pass

    usage = data.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    gen_tokens = usage.get("completion_tokens")

    wall = t1 - t0
    tok_per_s = (gen_tokens / wall) if isinstance(gen_tokens, int) and wall > 0 else None

    return {
        "wall_s": wall,
        "max_tokens": args.max_tokens,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": gen_tokens,
        "tok_per_s": tok_per_s,
        "temperature": args.temperature,
        "seed": args.seed,
        "output_text": text,
    }

# ----------------------------
# vLLM bench
# ----------------------------

def try_get_token_counts(outputs):
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

def bench_once_vllm(llm, prompt: str, max_tokens: int, temperature: float, seed: int):
    from vllm import SamplingParams

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

# ----------------------------
# Unified helpers
# ----------------------------

def med(results, key):
    vals = [r.get(key) for r in results if r.get(key) is not None]
    return statistics.median(vals) if vals else None

def derive_label_for_payload(payload, model_id: str) -> str:
    """
    For vLLM: use model_id.

    For llama-server:
      - If GGUF is under ~/models, derive label from its relative path.
      - If GGUF is in a quant folder, label = "<org>/<repo>/<quant>".
      - If GGUF is a root-level quant file, label = "<org>/<gguf-stem>".
      - Else fallback to gguf stem.

    This ensures filenames include the org for root-level quant repos
    like unsloth/Ministral-...-GGUF.
    """
    engine = payload.get("engine")
    if engine != "llama-server":
        return model_id

    gguf = payload.get("model_ref")
    if not gguf:
        return model_id

    p = Path(gguf)

    # Prefer path-derived label if under MODELS_ROOT
    try:
        rel = p.relative_to(MODELS_ROOT)
        parts = rel.parts  # e.g. ("unsloth", "Repo-GGUF", "file.gguf") or ("unsloth","Repo","UD-Q4_K_XL","file")
        org = parts[0] if len(parts) >= 1 else None

        parent_rel = Path(*parts[:-1])  # relative parent dir
        parent_str = str(parent_rel)

        # parent like "unsloth/Repo-GGUF"
        if parent_str.count("/") <= 1:
            if org:
                return f"{org}/{p.stem}"
            return p.stem

        # parent like "unsloth/Repo-GGUF/UD-Q4_K_XL"
        return parent_str

    except Exception:
        # If not under MODELS_ROOT, attempt best-effort org extraction from model_id
        # e.g. model_id could be "unsloth/Repo/file.gguf"
        if model_id.endswith(".gguf") and "/" in model_id:
            org = model_id.split("/", 1)[0]
            return f"{org}/{p.stem}"

        return p.stem

def write_payload(payload, tag: str, model_id: str):
    # ✅ change: new folder layout
    out_base = RESULTS_ROOT / tag
    out_base.mkdir(parents=True, exist_ok=True)

    label = derive_label_for_payload(payload, model_id)
    fname = f"{datetime.now().strftime('%Y-%m-%d')}_{sanitize(label)}.json"

    out_path = out_base / fname
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote: {out_path}")

# ----------------------------
# Engine runners
# ----------------------------

def run_model_llama(model_id: str, args, gguf_path: Path):
    prompt = PROMPTS[args.prompt_set]

    proc = start_llama_server(args, gguf_path)
    try:
        _ = bench_once_llama_http(prompt, args)  # warmup

        results = []
        for _ in range(args.iterations):
            results.append(bench_once_llama_http(prompt, args))

        summary = {
            "iterations": args.iterations,
            "median_wall_s": med(results, "wall_s"),
            "median_generated_tokens": med(results, "generated_tokens"),
            "median_tok_per_s": med(results, "tok_per_s"),
        }

        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            "model_id": model_id,
            "model_ref": str(gguf_path),
            "engine": "llama-server",
            "gpu_info": get_gpu_info(),
            "tag": args.tag,
            "env": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "engine": args.engine,
                "llama_server_bin": args.llama_server_bin,
                "llama_host": args.llama_host,
                "llama_port": args.llama_port,
                "llama_ctx": args.llama_ctx,
                "llama_n_gpu_layers": args.llama_n_gpu_layers,
                "llama_parallel": args.llama_parallel,
                "temperature": args.temperature,
                "seed": args.seed,
            },
            "bench": {
                "prompt_set": args.prompt_set,
                "prompt": prompt,
                "iterations": results,
                "summary": summary,
            },
        }

        write_payload(payload, args.tag, model_id)

    finally:
        stop_llama_server(proc)

def run_model_vllm(model_id: str, args):
    from vllm import LLM

    model_ref = resolve_model_ref_vllm(model_id)

    llm_kwargs = dict(
        model=model_ref,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len

    print(f"\n== Loading model ==")
    print(f"model_id:  {model_id}")
    print(f"model_ref: {model_ref}")
    print(f"engine:    vllm")

    llm = LLM(**llm_kwargs)
    prompt = PROMPTS[args.prompt_set]

    _ = bench_once_vllm(
        llm,
        prompt,
        max_tokens=min(64, args.max_tokens),
        temperature=args.temperature,
        seed=args.seed,
    )  # warmup

    results = []
    for i in range(args.iterations):
        iter_seed = args.seed if not args.vary_seed else (args.seed + i)
        results.append(
            bench_once_vllm(
                llm,
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=iter_seed,
            )
        )

    summary = {
        "iterations": args.iterations,
        "median_wall_s": med(results, "wall_s"),
        "median_generated_tokens": med(results, "generated_tokens"),
        "median_tok_per_s": med(results, "tok_per_s"),
    }

    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        "model_id": model_id,
        "model_ref": model_ref,
        "engine": "vllm",
        "gpu_info": get_gpu_info(),
        "tag": args.tag,
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "engine": args.engine,
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

    write_payload(payload, args.tag, model_id)

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Benchmark runner (vLLM + llama-server)")

    ap.add_argument("--model", required=True, help="HF repo_id or local path")

    ap.add_argument("--engine", default="auto",
                    choices=("auto", "vllm", "llama-server"),
                    help="Inference engine")

    ap.add_argument("--tag", default=None,
                    help="Optional output tag; inferred if omitted")

    ap.add_argument("--iterations", type=int, default=3,
                    help="Number of timed iterations (median reported)")

    # vLLM options
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)

    # Shared sampling-ish knobs
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vary-seed", action="store_true",
                    help="If set, increments seed each iteration (seed+i)")

    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--prompt-set", default="short", choices=tuple(PROMPTS.keys()))

    # llama-server options
    ap.add_argument("--llama-server-bin", default="llama-server",
                    help="Path to llama-server binary")
    ap.add_argument("--llama-host", default="127.0.0.1",
                    help="Host for llama-server")
    ap.add_argument("--llama-port", type=int, default=8080,
                    help="Port for llama-server")

    ap.add_argument("--llama-ctx", type=int, default=None,
                    help="Context length for llama-server (-c)")

    ap.add_argument("--llama-n-gpu-layers", type=int, default=999,
                    help="GPU layers to offload for llama-server (-ngl)")

    ap.add_argument("--llama-parallel", type=int, default=1,
                    help="Parallel sequences for llama-server (-np)")

    ap.add_argument("--llama-no-autostart", action="store_true",
                    help="Do not auto-start llama-server; require it already running")

    ap.add_argument("--llama-extra-args", nargs=argparse.REMAINDER,
                    help="Extra raw args appended to llama-server command")

    args = ap.parse_args()

    # Tag inference is purely from get_gpu_info
    args.tag = infer_tag(args.tag)

    model_id = args.model

    # Auto-detect GGUF (may raise ambiguity errors)
    gguf_path = resolve_local_gguf(model_id)

    if args.engine == "auto":
        args.engine = "llama-server" if gguf_path else "vllm"

    if args.engine == "llama-server":
        # Allow explicit .gguf path under MODELS_ROOT via --model org/repo/file.gguf
        if not gguf_path:
            p = Path(model_id).expanduser()
            if is_gguf_file(p):
                gguf_path = p
            else:
                cand = MODELS_ROOT / model_id
                if is_gguf_file(cand):
                    gguf_path = cand

        if not gguf_path:
            raise SystemExit(
                "Engine set to llama-server but no GGUF path could be resolved.\n"
                "Pass --model with:\n"
                "  * a quant subfolder: <repo>/UD-Q4_K_XL\n"
                "  * or an explicit .gguf file path\n"
                "  * or <org>/<repo>/<file>.gguf relative to ~/models\n"
            )

        print(f"\n== Loading model ==")
        print(f"model_id:  {model_id}")
        print(f"model_ref: {gguf_path}")
        print(f"engine:    llama-server")

        run_model_llama(model_id, args, gguf_path)
        return

    if args.engine == "vllm":
        run_model_vllm(model_id, args)
        return

    raise SystemExit(f"Unknown engine: {args.engine}")

if __name__ == "__main__":
    main()
