#!/usr/bin/env python3
"""
Model Workbench Benchmark Runner - llama.cpp / llama-server

Benchmarks GGUF models via llama.cpp's llama-server using OpenAI-compatible API.

--model accepts:
  * Path to .gguf file
  * Directory containing GGUF files (auto-picks if unambiguous)

Output:
  benchmarks/<date>_<label>.json

Examples:

  # Quant folder
  uv run python scripts/run_bench_llama_server.py \\
    --model ~/models/unsloth/GLM-4.5-Air-GGUF/UD-Q4_K_XL

  # Explicit .gguf file
  uv run python scripts/run_bench_llama_server.py \\
    --model ~/models/unsloth/Repo-GGUF/Model-UD-Q4_K_XL.gguf

  # Use an already running server
  uv run python scripts/run_bench_llama_server.py \\
    --no-autostart \\
    --model ~/models/unsloth/GLM-4.5-Air-GGUF/UD-Q4_K_XL
"""

import argparse
import json
import os
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

from bench_utils import (
    MODELS_ROOT,
    RESULTS_ROOT,
    sanitize,
    compact_path,
    extract_repo_id,
    get_gpu_info,
    log,
)
from server_manager import (
    ServerManager,
    wait_for_llama_ready,
    build_llama_cmd,
)

PROMPTS = {
    "short": "Explain speculative decoding in 2 sentences.",
    "medium": "Summarize key tradeoffs between tensor parallelism and pipeline parallelism.",
    "long": "Write a concise technical overview of KV cache and why it matters for long context.",
}

# ----------------------------
# GGUF resolution
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
    Resolve a GGUF path from explicit filesystem path.

    Args:
        model_arg: Explicit path to .gguf file or directory containing GGUF files

    Returns:
        Path to GGUF file (prefer shard entrypoint), or None if not found

    Raises:
        SystemExit if directory contains multiple ambiguous GGUF variants
    """
    p = Path(model_arg).expanduser()
    if is_gguf_file(p):
        return p
    if p.is_dir():
        chosen = pick_gguf_from_dir(p)
        if chosen:
            return chosen
        raise_if_multiple_variants(p, model_arg)
    return None

def bench_once(prompt: str, args):
    """Run a single benchmark using the native /completion endpoint for detailed timings."""
    if requests is None:
        raise SystemExit("Missing dependency: requests. Install with `pip install requests`.")

    base = f"http://{args.host}:{args.port}"
    url = base.rstrip("/") + "/completion"  # Native endpoint with timings

    payload = {
        "prompt": prompt,
        "n_predict": args.max_tokens,
        "temperature": args.temperature,
    }

    t0 = time.perf_counter()
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    t1 = time.perf_counter()

    data = r.json()
    wall = t1 - t0

    # Extract text
    text = data.get("content", "")

    # Extract detailed timings from native endpoint
    timings = data.get("timings", {})
    prompt_tokens = data.get("tokens_evaluated")
    prompt_ms = timings.get("prompt_ms")
    gen_tokens = timings.get("predicted_n")
    gen_ms = timings.get("predicted_ms")
    gen_tok_per_s = timings.get("predicted_per_second")

    # tok_per_s: total throughput including TTFT
    # generation_tok_per_s: pure generation speed excluding prompt processing (llama.cpp native)
    tok_per_s = gen_tokens / wall if wall > 0 and gen_tokens else None

    return {
        "wall_s": wall,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": gen_tokens,
        "ttft_ms": prompt_ms,  # time to first token = prompt processing time
        "tok_per_s": tok_per_s,  # overall throughput (includes TTFT)
        "generation_tok_per_s": gen_tok_per_s,  # pure generation speed (excludes prompt)
        "generation_ms": gen_ms,
        "output_text": text,
    }

# ----------------------------
# Result helpers
# ----------------------------

def med(results, key):
    vals = [r.get(key) for r in results if r.get(key) is not None]
    return statistics.median(vals) if vals else None

def derive_label(model_id: str, gguf_path: Path) -> str:
    """
    Derive a label for result filenames:
      - If GGUF is in a quant folder under ~/models, label = "<org>/<repo>/<quant>"
      - If GGUF is a root-level quant file, label = "<org>/<gguf-stem>"
      - Else fallback to gguf stem
    """
    try:
        rel = gguf_path.relative_to(MODELS_ROOT)
        parts = rel.parts  # e.g. ("unsloth", "Repo-GGUF", "file.gguf") or ("unsloth","Repo","UD-Q4_K_XL","file")
        org = parts[0] if len(parts) >= 1 else None

        parent_rel = Path(*parts[:-1])  # relative parent dir
        parent_str = str(parent_rel)

        # parent like "unsloth/Repo-GGUF"
        if parent_str.count("/") <= 1:
            if org:
                return f"{org}/{gguf_path.stem}"
            return gguf_path.stem

        # parent like "unsloth/Repo-GGUF/UD-Q4_K_XL"
        return parent_str

    except Exception:
        # If not under MODELS_ROOT, attempt best-effort org extraction from model_id
        if model_id.endswith(".gguf") and "/" in model_id:
            org = model_id.split("/", 1)[0]
            return f"{org}/{gguf_path.stem}"

        return gguf_path.stem

def write_payload(payload, label: str):
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    fname = f"{datetime.now().strftime('%Y-%m-%d')}_{sanitize(label)}.json"

    out_path = RESULTS_ROOT / fname
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    log(f"Wrote: {out_path}")

# ----------------------------
# Main runner
# ----------------------------

def run_benchmark(model_id: str, args, gguf_path: Path):
    prompt = PROMPTS[args.prompt_set]

    # Server management
    server = ServerManager(
        host=args.host,
        port=args.port,
        timeout=args.server_timeout,
    )

    # Check if we need to start the server
    if not server.is_running():
        if args.no_autostart:
            raise SystemExit(
                f"llama-server not detected on {args.host}:{args.port} and --no-autostart was set."
            )

    with server:
        # Start server if not already running
        if not server.is_running():
            cmd = build_llama_cmd(
                gguf_path=gguf_path,
                llama_server_bin=args.llama_server_bin,
                port=args.port,
                n_gpu_layers=args.n_gpu_layers,
                ctx=args.ctx,
                parallel=args.parallel,
                extra_args=args.extra_args,
            )
            server.start(
                cmd,
                lambda: wait_for_llama_ready(args.host, args.port),
                stream_stderr=True,  # llama-server outputs to stderr
                label="llama-server",
                sigint_wait=2,  # llama-server shuts down quickly
                term_wait=2,
            )

        # Capture GPU info with memory usage after model loads
        gpu_info = get_gpu_info(include_memory=True)
        log(f"GPU memory: {gpu_info.get('memory_used_mib', '?')} / {gpu_info.get('memory_total_mib', '?')} MiB")

        log("Warmup request...")
        _ = bench_once(prompt, args)

        results = []
        for i in range(args.iterations):
            log(f"Benchmark {i + 1} of {args.iterations}...")
            results.append(bench_once(prompt, args))

        summary = {
            "iterations": args.iterations,
            "median_tok_per_s": med(results, "tok_per_s"),  # unified field (includes TTFT) for cross-backend comparison
            "median_wall_s": med(results, "wall_s"),
            "median_ttft_ms": med(results, "ttft_ms"),
            "median_generation_tok_per_s": med(results, "generation_tok_per_s"),  # pure generation speed
        }

        log(f"Median: {summary['median_generation_tok_per_s']:.1f} tok/s, TTFT: {summary['median_ttft_ms']:.1f} ms")

        label = derive_label(model_id, gguf_path)

        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            "repo_id": extract_repo_id(model_id),
            "model_ref": compact_path(str(gguf_path)),
            "engine": "llama-server",
            "gpu_info": gpu_info,
            "env": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "llama_server_bin": compact_path(args.llama_server_bin),
                "host": args.host,
                "port": args.port,
                "ctx": args.ctx,
                "n_gpu_layers": args.n_gpu_layers,
                "parallel": args.parallel,
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

        write_payload(payload, label)

# ----------------------------
# Main
# ----------------------------

def main():
    default_bin = str(Path.home() / "llama.cpp/build/bin/llama-server")

    ap = argparse.ArgumentParser(description="Benchmark runner for llama.cpp / llama-server")

    ap.add_argument("--model", required=True, help="GGUF path or path under ~/models")


    ap.add_argument("--iterations", type=int, default=3,
                    help="Number of timed iterations (median reported)")

    # Sampling
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--prompt-set", default="short", choices=tuple(PROMPTS.keys()))

    # llama-server options
    ap.add_argument("--llama-server-bin", default=default_bin,
                    help=f"Path to llama-server binary (default: {default_bin})")
    ap.add_argument("--host", default="127.0.0.1",
                    help="Host for llama-server")
    ap.add_argument("--port", type=int, default=8080,
                    help="Port for llama-server")

    ap.add_argument("--ctx", type=int, default=None,
                    help="Context length for llama-server (-c)")

    ap.add_argument("--n-gpu-layers", type=int, default=999,
                    help="GPU layers to offload (-ngl)")

    ap.add_argument("--parallel", type=int, default=1,
                    help="Parallel sequences (-np)")

    ap.add_argument("--no-autostart", action="store_true",
                    help="Do not auto-start llama-server; require it already running")

    ap.add_argument("--server-timeout", type=int, default=180,
                    help="Max seconds to wait for server to be ready (default: 180)")

    ap.add_argument("--extra-args", nargs=argparse.REMAINDER,
                    help="Extra raw args appended to llama-server command")

    args = ap.parse_args()

    model_id = args.model

    # Resolve GGUF path
    gguf_path = resolve_local_gguf(model_id)

    if not gguf_path:
        raise SystemExit(
            f"No GGUF found at: {Path(model_id).expanduser()}\n"
            "Pass --model with an explicit path, e.g.:\n"
            "  --model ~/models/org/repo/quant-folder\n"
            "  --model ~/models/org/repo/model.gguf"
        )

    print(f"\n== Loading model ==")
    print(f"model_id:  {model_id}")
    print(f"model_ref: {gguf_path}")
    print(f"engine:    llama-server")
    print(f"binary:    {args.llama_server_bin}")

    run_benchmark(model_id, args, gguf_path)

if __name__ == "__main__":
    main()
