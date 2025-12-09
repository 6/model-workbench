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

  # Vision benchmark with image
  uv run python scripts/run_bench_llama_server.py \\
    --model ~/models/unsloth/Qwen3-VL-235B-A22B-Instruct-GGUF/UD-Q4_K_XL \\
    --image config/example.jpg

  # Use an already running server
  uv run python scripts/run_bench_llama_server.py \\
    --no-autostart \\
    --model ~/models/unsloth/GLM-4.5-Air-GGUF/UD-Q4_K_XL
"""

import argparse
import base64
import os
import statistics
import time
from pathlib import Path

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

from bench_utils import (
    ROOT,
    MODELS_ROOT,
    RESULTS_ROOT,
    TEXT_PROMPTS,
    VISION_PROMPTS,
    compact_path,
    extract_repo_id,
    find_mmproj,
    get_gpu_info,
    log,
    resolve_local_gguf,
    write_benchmark_result,
)
from server_manager import ServerManager

# Built-in test images
BUILTIN_IMAGES = {
    "example": ROOT / "config" / "example.jpg",
}

ALL_PROMPTS = {**TEXT_PROMPTS, **VISION_PROMPTS}


# ----------------------------
# Image helpers
# ----------------------------

def resolve_image_source(image_arg: str | None) -> tuple[str | None, str]:
    """Resolve image argument to path and label.

    Args:
        image_arg: Image path, builtin name, or 'none'

    Returns:
        (image_path, label) tuple
    """
    if image_arg is None or image_arg.lower() == "none":
        return None, "none"

    if image_arg in BUILTIN_IMAGES:
        src = BUILTIN_IMAGES[image_arg]
        return str(src), f"builtin:{image_arg}"

    p = Path(image_arg).expanduser()
    if p.exists():
        return str(p), str(p)

    return image_arg, image_arg


def encode_image_base64(image_path: str) -> str:
    """Encode local image to base64 data URL."""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    # Determine MIME type
    suffix = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime = mime_types.get(suffix, "image/jpeg")
    return f"data:{mime};base64,{data}"


# ----------------------------
# Benchmark functions
# ----------------------------

def bench_once(prompt: str, args, image_path: str | None = None):
    """Run a single benchmark.

    For text-only: uses native /completion endpoint for detailed timings.
    For vision: uses /v1/chat/completions with multimodal messages.
    """
    if requests is None:
        raise SystemExit("Missing dependency: requests. Install with `pip install requests`.")

    base = f"http://{args.host}:{args.port}"

    if image_path is None:
        # Text-only: use native /completion endpoint for detailed timings
        url = base.rstrip("/") + "/completion"

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
        # generation_tok_per_s: pure generation speed excluding prompt processing
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
    else:
        # Vision: use /v1/chat/completions with multimodal messages
        url = base.rstrip("/") + "/v1/chat/completions"

        # Build multimodal message
        if image_path.startswith("http"):
            image_content = {"type": "image_url", "image_url": {"url": image_path}}
        else:
            data_url = encode_image_base64(image_path)
            image_content = {"type": "image_url", "image_url": {"url": data_url}}

        messages = [{
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": prompt},
            ]
        }]

        payload = {
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }

        t0 = time.perf_counter()
        r = requests.post(url, json=payload, timeout=300)
        r.raise_for_status()
        t1 = time.perf_counter()

        data = r.json()
        wall = t1 - t0

        # Extract text from chat response
        text = ""
        if "choices" in data and data["choices"]:
            choice = data["choices"][0]
            if "message" in choice:
                text = choice["message"].get("content", "")

        # Extract token counts from usage
        prompt_tokens = None
        gen_tokens = None
        if "usage" in data:
            prompt_tokens = data["usage"].get("prompt_tokens")
            gen_tokens = data["usage"].get("completion_tokens")

        # llama.cpp chat endpoint also returns timings
        timings = data.get("timings", {})
        prompt_ms = timings.get("prompt_ms")
        gen_ms = timings.get("predicted_ms")
        gen_tok_per_s = timings.get("predicted_per_second")

        tok_per_s = gen_tokens / wall if wall > 0 and gen_tokens else None

        return {
            "wall_s": wall,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": gen_tokens,
            "ttft_ms": prompt_ms,
            "tok_per_s": tok_per_s,
            "generation_tok_per_s": gen_tok_per_s,
            "generation_ms": gen_ms,
            "output_text": text,
        }

# ----------------------------
# Result helpers
# ----------------------------

def med(results, key):
    vals = [r.get(key) for r in results if r.get(key) is not None]
    return statistics.median(vals) if vals else None

# ----------------------------
# Main runner
# ----------------------------

def run_benchmark(
    model_id: str,
    args,
    gguf_path: Path,
    image_path: str | None,
    image_label: str,
    mmproj_path: Path | None,
):
    """Run benchmarks with optional vision support."""
    is_vision = image_path is not None
    mode = "vision" if is_vision else "text-only"

    # Select prompt based on mode
    if is_vision:
        prompt = VISION_PROMPTS.get(args.prompt_set) or VISION_PROMPTS["describe"]
    else:
        prompt = TEXT_PROMPTS[args.prompt_set]

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
            gguf_path = server.start_llama(
                model_path=model_id,
                llama_server_bin=args.llama_server_bin,
                n_gpu_layers=args.n_gpu_layers,
                ctx=args.ctx,
                parallel=args.parallel,
                mmproj_path=mmproj_path,
                extra_args=args.extra_args,
            )

        # Capture GPU info with memory usage after model loads
        gpu_info = get_gpu_info(include_memory=True)
        log(f"GPU memory: {gpu_info.get('memory_used_mib', '?')} / {gpu_info.get('memory_total_mib', '?')} MiB")

        log("Warmup request...")
        _ = bench_once(prompt, args, image_path)

        results = []
        for i in range(args.iterations):
            log(f"Benchmark {i + 1} of {args.iterations}...")
            results.append(bench_once(prompt, args, image_path))

        summary = {
            "median_wall_s": med(results, "wall_s"),
            "median_tok_per_s": med(results, "tok_per_s"),
            "median_ttft_ms": med(results, "ttft_ms"),
            "median_generation_tok_per_s": med(results, "generation_tok_per_s"),
        }

        gen_tok_s = summary.get("median_generation_tok_per_s")
        ttft = summary.get("median_ttft_ms")
        log(f"Median: {gen_tok_s:.1f if gen_tok_s else '?'} tok/s, TTFT: {ttft:.1f if ttft else '?'} ms")

        # Build config
        config = {
            "prompt_set": args.prompt_set,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "llama_server_bin": compact_path(args.llama_server_bin),
            "ctx": args.ctx,
            "n_gpu_layers": args.n_gpu_layers,
            "parallel": args.parallel,
            "seed": args.seed,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }
        if is_vision:
            config["image"] = image_label
            if mmproj_path:
                config["mmproj"] = compact_path(str(mmproj_path))

        # Save results using unified writer
        write_benchmark_result(
            results_dir=RESULTS_ROOT,
            repo_id=extract_repo_id(model_id),
            model_ref=compact_path(str(gguf_path)),
            engine="llama-server",
            mode=mode,
            gpu_info=gpu_info,
            config=config,
            iterations=results,
            summary=summary,
        )

# ----------------------------
# Main
# ----------------------------

def main():
    default_bin = str(Path.home() / "llama.cpp/build/bin/llama-server")

    ap = argparse.ArgumentParser(description="Benchmark runner for llama.cpp / llama-server")

    ap.add_argument("--model", required=True, help="GGUF path or path under ~/models")

    # Vision options
    ap.add_argument("--image", default=None,
                    help="Image for vision benchmark: path, URL, builtin name ('example'), or 'none'")
    ap.add_argument("--mmproj", default=None,
                    help="Path to mmproj file (auto-detected if not specified)")

    ap.add_argument("--iterations", type=int, default=3,
                    help="Number of timed iterations (median reported)")

    # Sampling
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--prompt-set", default="short", choices=tuple(ALL_PROMPTS.keys()))

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

    # Resolve image
    image_path, image_label = resolve_image_source(args.image)
    is_vision = image_path is not None

    # Resolve mmproj for vision
    mmproj_path = None
    if is_vision:
        if args.mmproj:
            mmproj_path = Path(args.mmproj).expanduser()
            if not mmproj_path.exists():
                raise SystemExit(f"mmproj not found: {mmproj_path}")
        else:
            # Auto-detect mmproj
            mmproj_path = find_mmproj(gguf_path)
            if not mmproj_path:
                raise SystemExit(
                    f"Vision mode requires mmproj file but none found.\n"
                    f"Searched from: {gguf_path.parent}\n"
                    "Either:\n"
                    "  1. Download mmproj-*.gguf to the model directory\n"
                    "  2. Pass --mmproj /path/to/mmproj-*.gguf explicitly"
                )

    # Determine mode
    mode = "vision" if is_vision else "text-only"

    print(f"\n== Loading model ==")
    print(f"model_id:  {model_id}")
    print(f"model_ref: {gguf_path}")
    print(f"engine:    llama-server")
    print(f"binary:    {args.llama_server_bin}")
    print(f"mode:      {mode}")
    if is_vision:
        print(f"image:     {image_label}")
        print(f"mmproj:    {mmproj_path}")

    run_benchmark(model_id, args, gguf_path, image_path, image_label, mmproj_path)

if __name__ == "__main__":
    main()
