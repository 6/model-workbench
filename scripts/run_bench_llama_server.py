#!/usr/bin/env python3
"""
Model Workbench Benchmark Runner - llama.cpp / llama-server

Benchmarks GGUF models via llama.cpp's llama-server using OpenAI-compatible API.

Auto resolution:
  --model can be:
    * Full path to .gguf file
    * Directory containing GGUF files (auto-picks if unambiguous)
    * Path relative to ~/models (e.g., org/repo/quant-folder)

Tag inference:
  * Inferred from GPU count via nvidia-smi (single-gpu, dual-gpu, etc.)

Result filenames:
  * For GGUF in quant folder: "<org>/<repo>/<quant>" as label
  * For root-level quant files: "<org>/<gguf-stem>" as label
  * Otherwise: gguf stem

Output:
  benchmarks/<tag>/<date>_<label>.json

Examples:

  # Quant folder (recommended when multiple quants exist)
  uv run python scripts/run_bench_llama_server.py \\
    --model unsloth/GLM-4.5-Air-GGUF/UD-Q4_K_XL

  # Root-level quant file in repo
  uv run python scripts/run_bench_llama_server.py \\
    --model unsloth/Ministral-3-14B-Instruct-2512-GGUF/Ministral-3-14B-Instruct-2512-UD-Q4_K_XL.gguf

  # Explicit shard file
  uv run python scripts/run_bench_llama_server.py \\
    --model ~/models/unsloth/GLM-4.5-Air-GGUF/UD-Q4_K_XL/GLM-4.5-Air-UD-Q4_K_XL-00001-of-00002.gguf

  # Use an already running server
  uv run python scripts/run_bench_llama_server.py \\
    --no-autostart \\
    --model unsloth/GLM-4.5-Air-GGUF/UD-Q4_K_XL
"""

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

from bench_utils import (
    ROOT,
    MODELS_ROOT,
    RESULTS_ROOT,
    sanitize,
    get_gpu_info,
    infer_tag,
    port_open,
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

# ----------------------------
# llama-server control + HTTP bench
# ----------------------------

def start_llama_server(args, gguf_path: Path):
    """
    Starts llama-server unless:
      - --no-autostart is set, OR
      - port is already open (assume user-managed server)

    Waits up to --server-timeout seconds for the server to be ready,
    verifying readiness via API call (not just port open).
    """
    host = args.host
    port = args.port
    timeout = args.server_timeout

    if port_open(host, port):
        return None  # already running

    if args.no_autostart:
        raise SystemExit(
            f"llama-server not detected on {host}:{port} and --no-autostart was set."
        )

    if not gguf_path.exists():
        raise SystemExit(f"GGUF not found: {gguf_path}")

    cmd = [args.llama_server_bin, "-m", str(gguf_path), "--port", str(port)]

    # GPU offload
    if args.n_gpu_layers is not None:
        cmd += ["-ngl", str(args.n_gpu_layers)]

    # Context length
    if args.ctx:
        cmd += ["-c", str(args.ctx)]

    # Parallel sequences
    if args.parallel and args.parallel > 1:
        cmd += ["-np", str(args.parallel)]

    # Extra raw args
    if args.extra_args:
        cmd += args.extra_args

    print("+", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for server to be ready (with API verification)
    print(f"Waiting for server to be ready (timeout: {timeout}s)...")
    start_time = time.time()
    base_url = f"http://{host}:{port}"

    while time.time() - start_time < timeout:
        # Check if process crashed
        if proc.poll() is not None:
            stderr_output = ""
            try:
                stderr_output = proc.stderr.read() if proc.stderr else ""
            except Exception:
                pass
            raise SystemExit(
                f"llama-server exited unexpectedly (exit code: {proc.returncode}).\n"
                f"Last output:\n{stderr_output[-2000:] if stderr_output else '(no output)'}"
            )

        # Check if server is ready via API
        if port_open(host, port):
            try:
                r = requests.get(f"{base_url}/v1/models", timeout=5)
                if r.status_code == 200:
                    elapsed = time.time() - start_time
                    print(f"Server ready in {elapsed:.1f}s")
                    return proc
            except Exception:
                pass  # Not ready yet

        time.sleep(1)

    # Timeout reached
    try:
        proc.terminate()
    except Exception:
        pass
    raise SystemExit(
        f"llama-server failed to become ready within {timeout}s on {host}:{port}.\n"
        f"Check your binary path, that it was built with CUDA, and that the model fits in GPU memory.\n"
        f"Try increasing --server-timeout if the model is very large."
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

def bench_once(prompt: str, args):
    if requests is None:
        raise SystemExit("Missing dependency: requests. Install with `pip install requests`.")

    base = f"http://{args.host}:{args.port}"
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

def write_payload(payload, tag: str, label: str):
    out_base = RESULTS_ROOT / tag
    out_base.mkdir(parents=True, exist_ok=True)

    fname = f"{datetime.now().strftime('%Y-%m-%d')}_{sanitize(label)}.json"

    out_path = out_base / fname
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote: {out_path}")

# ----------------------------
# Main runner
# ----------------------------

def run_benchmark(model_id: str, args, gguf_path: Path):
    prompt = PROMPTS[args.prompt_set]

    proc = start_llama_server(args, gguf_path)
    try:
        _ = bench_once(prompt, args)  # warmup

        results = []
        for _ in range(args.iterations):
            results.append(bench_once(prompt, args))

        summary = {
            "iterations": args.iterations,
            "median_wall_s": med(results, "wall_s"),
            "median_generated_tokens": med(results, "generated_tokens"),
            "median_tok_per_s": med(results, "tok_per_s"),
        }

        label = derive_label(model_id, gguf_path)

        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            "model_id": model_id,
            "model_ref": str(gguf_path),
            "engine": "llama-server",
            "gpu_info": get_gpu_info(),
            "tag": args.tag,
            "env": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "llama_server_bin": args.llama_server_bin,
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

        write_payload(payload, args.tag, label)

    finally:
        stop_llama_server(proc)

# ----------------------------
# Main
# ----------------------------

def main():
    default_bin = str(Path.home() / "llama.cpp/build/bin/llama-server")

    ap = argparse.ArgumentParser(description="Benchmark runner for llama.cpp / llama-server")

    ap.add_argument("--model", required=True, help="GGUF path or path under ~/models")

    ap.add_argument("--tag", default=None,
                    help="Optional output tag; inferred from GPU count if omitted")

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

    # Tag inference from GPU count
    args.tag = infer_tag(args.tag)

    model_id = args.model

    # Resolve GGUF path
    gguf_path = resolve_local_gguf(model_id)

    if not gguf_path:
        # Try explicit .gguf path under MODELS_ROOT
        p = Path(model_id).expanduser()
        if is_gguf_file(p):
            gguf_path = p
        else:
            cand = MODELS_ROOT / model_id
            if is_gguf_file(cand):
                gguf_path = cand

    if not gguf_path:
        raise SystemExit(
            "No GGUF path could be resolved.\n"
            "Pass --model with:\n"
            "  * a quant subfolder: <repo>/UD-Q4_K_XL\n"
            "  * or an explicit .gguf file path\n"
            "  * or <org>/<repo>/<file>.gguf relative to ~/models\n"
        )

    print(f"\n== Loading model ==")
    print(f"model_id:  {model_id}")
    print(f"model_ref: {gguf_path}")
    print(f"engine:    llama-server")
    print(f"binary:    {args.llama_server_bin}")

    run_benchmark(model_id, args, gguf_path)

if __name__ == "__main__":
    main()
