#!/usr/bin/env python3
"""
Model Workbench vLLM Server Benchmark Runner

Uses vLLM server mode with tensor parallelism. Auto-detects GPU count.
Follows the official vLLM recipe: https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM-V.html

This script:
  1. Starts a vLLM server with tensor parallelism (auto-detects GPUs)
  2. Benchmarks via OpenAI-compatible API
  3. Shuts down the server after completion

Examples:

  # Auto-detect GPUs, use all available
  uv run python scripts/run_bench_vllm_server.py --model ~/models/zai-org/GLM-4.6V-FP8

  # Force single GPU
  uv run python scripts/run_bench_vllm_server.py --model ~/models/zai-org/GLM-4.6V-FP8 --tensor-parallel 1

  # Vision benchmark with local image
  uv run python scripts/run_bench_vllm_server.py \\
    --model ~/models/zai-org/GLM-4.6V-FP8 \\
    --image config/example.jpg

  # Use an already running vLLM server
  uv run python scripts/run_bench_vllm_server.py \\
    --model ~/models/zai-org/GLM-4.6V-FP8 \\
    --no-autostart \\
    --port 8000
"""

import argparse
import base64
import re
import statistics
import time
from pathlib import Path

import requests
from openai import OpenAI

from bench_utils import (
    ROOT,
    RESULTS_ROOT,
    TEXT_PROMPTS,
    VISION_PROMPTS,
    compact_path,
    extract_repo_id,
    get_gpu_info,
    get_gpu_count,
    resolve_model_path,
    model_needs_nightly,
    log,
    write_benchmark_result,
)
from server_manager import ServerManager

# Built-in test images
BUILTIN_IMAGES = {
    "example": ROOT / "config" / "example.jpg",
    "grayscale": "https://upload.wikimedia.org/wikipedia/commons/f/fa/Grayscale_8bits_palette_sample_image.png",
}

ALL_PROMPTS = {**TEXT_PROMPTS, **VISION_PROMPTS}

# Prometheus metrics we care about (vLLM exposes these as histograms)
VLLM_METRICS = [
    "vllm:time_to_first_token_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
]


def scrape_vllm_metrics(host: str, port: int) -> dict[str, float] | None:
    """Scrape vLLM Prometheus /metrics endpoint and parse histogram sums/counts.

    Returns dict with keys like:
        "vllm:time_to_first_token_seconds_sum": 1.234
        "vllm:time_to_first_token_seconds_count": 5
    Or None if scraping fails.
    """
    try:
        resp = requests.get(f"http://{host}:{port}/metrics", timeout=5)
        resp.raise_for_status()
    except Exception:
        return None

    metrics = {}
    # Parse Prometheus text format: metric_name{labels} value
    # We only care about _sum and _count lines for our histograms
    pattern = re.compile(r'^(vllm:[a-z_]+(?:_sum|_count))(?:\{[^}]*\})?\s+([\d.eE+-]+)', re.MULTILINE)

    for match in pattern.finditer(resp.text):
        name, value = match.groups()
        try:
            metrics[name] = float(value)
        except ValueError:
            pass

    return metrics if metrics else None


def compute_metrics_delta(before: dict[str, float] | None, after: dict[str, float] | None) -> dict[str, float]:
    """Compute delta between two metrics snapshots.

    Returns timing values in milliseconds for the single request that ran between snapshots.
    """
    result = {}
    if not before or not after:
        return result

    for metric_base in VLLM_METRICS:
        sum_key = f"{metric_base}_sum"
        count_key = f"{metric_base}_count"

        if sum_key in before and sum_key in after and count_key in before and count_key in after:
            delta_sum = after[sum_key] - before[sum_key]
            delta_count = after[count_key] - before[count_key]

            # For single-request benchmark, delta_count should be 1
            # delta_sum is the time in seconds for that request
            if delta_count > 0:
                # Convert to ms and store with a friendly name
                time_ms = (delta_sum / delta_count) * 1000

                # Map to friendly names
                if "time_to_first_token" in metric_base:
                    result["ttft_ms"] = time_ms
                elif "prefill_time" in metric_base:
                    result["prefill_ms"] = time_ms
                elif "decode_time" in metric_base:
                    result["generation_ms"] = time_ms

    return result


def resolve_image_source(image_arg: str | None) -> tuple[str | None, str]:
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


def bench_once(client: OpenAI, model: str, prompt: str, image_path: str | None,
               max_tokens: int, temperature: float, host: str, port: int) -> dict:
    """Run single inference via OpenAI-compatible API with Prometheus metrics scraping."""

    # Build message content
    if image_path is None:
        # Text-only
        messages = [{"role": "user", "content": prompt}]
    else:
        # Vision - multimodal message
        if image_path.startswith("http"):
            image_content = {"type": "image_url", "image_url": {"url": image_path}}
        else:
            # Local file - encode as base64
            data_url = encode_image_base64(image_path)
            image_content = {"type": "image_url", "image_url": {"url": data_url}}

        messages = [{
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": prompt},
            ]
        }]

    # Scrape metrics before request
    metrics_before = scrape_vllm_metrics(host, port)

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature if temperature > 0 else 0.0,
    )
    t1 = time.perf_counter()

    # Scrape metrics after request
    metrics_after = scrape_vllm_metrics(host, port)

    output_text = response.choices[0].message.content or ""
    wall = t1 - t0

    # Extract token counts from response.usage
    prompt_tokens = None
    generated_tokens = None
    if response.usage:
        prompt_tokens = response.usage.prompt_tokens
        generated_tokens = response.usage.completion_tokens

    # Compute derived metrics
    tok_per_s = None
    generation_tok_per_s = None
    if generated_tokens and wall > 0:
        tok_per_s = generated_tokens / wall  # overall throughput including TTFT

    # Get detailed timing from Prometheus delta
    timing = compute_metrics_delta(metrics_before, metrics_after)

    # Calculate generation tok/s from Prometheus timing if available
    if generated_tokens and timing.get("generation_ms"):
        gen_s = timing["generation_ms"] / 1000
        if gen_s > 0:
            generation_tok_per_s = generated_tokens / gen_s

    result = {
        "wall_s": wall,
        "output_text": output_text,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "tok_per_s": tok_per_s,
        "generation_tok_per_s": generation_tok_per_s,
    }
    # Add Prometheus timing metrics if available
    result.update(timing)

    return result


def med(results: list[dict], key: str) -> float | None:
    """Compute median of a metric across results, ignoring None values."""
    vals = [r.get(key) for r in results if r.get(key) is not None]
    return statistics.median(vals) if vals else None


def main():
    ap = argparse.ArgumentParser(description="vLLM server benchmark runner for GLM-4.5V/GLM-4.6V")
    ap.add_argument("--model", default="zai-org/GLM-4.6V-FP8")
    ap.add_argument("--tensor-parallel", type=int, default=None,
                    help="Tensor parallel size (default: auto-detect GPU count)")
    ap.add_argument("--image", default=None, help="Image path/URL or 'none' for text-only")
    ap.add_argument("--prompt", default=None, help="Custom prompt")
    ap.add_argument("--prompt-set", default="short", choices=list(ALL_PROMPTS.keys()))
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--iterations", type=int, default=3)

    # Server options
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-autostart", action="store_true",
                    help="Don't start server; require it already running")
    ap.add_argument("--server-timeout", type=int, default=180,
                    help="Timeout in seconds waiting for server to start (default: 180)")

    # vLLM server options
    ap.add_argument("--max-model-len", type=int, default=65536,
                    help="Max context length (GLM-4.6V supports 128K)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.95,
                    help="GPU memory fraction")
    ap.add_argument("--max-num-batched-tokens", type=int, default=None,
                    help="Max batched tokens (affects throughput/latency tradeoff)")

    # Environment selection
    env_group = ap.add_mutually_exclusive_group()
    env_group.add_argument("--force-stable", action="store_true",
                           help="Force use of stable venv (ignore model config)")
    env_group.add_argument("--force-nightly", action="store_true",
                           help="Force use of nightly venv (ignore model config)")

    args = ap.parse_args()

    # Auto-detect GPU count for tensor parallelism
    if args.tensor_parallel is None:
        args.tensor_parallel = get_gpu_count()

    model_path = resolve_model_path(args.model)
    image_source, image_label = resolve_image_source(args.image)
    is_vision = image_source is not None

    # Determine which environment to use
    if args.force_nightly:
        use_nightly = True
    elif args.force_stable:
        use_nightly = False
    else:
        use_nightly = model_needs_nightly(args.model)

    env_label = "nightly" if use_nightly else "stable"

    # Select prompt
    if args.prompt:
        prompt_text = args.prompt
    elif is_vision:
        prompt_text = VISION_PROMPTS.get(args.prompt_set, VISION_PROMPTS["describe"])
    else:
        prompt_text = TEXT_PROMPTS.get(args.prompt_set, TEXT_PROMPTS["short"])

    mode = "vision" if is_vision else "text-only"
    print(f"\n== vLLM Server Benchmark ==")
    print(f"model:           {model_path}")
    print(f"mode:            {mode}")
    print(f"environment:     {env_label}")
    print(f"tensor_parallel: {args.tensor_parallel}")
    print(f"image:           {image_label}")
    print(f"prompt:          {prompt_text[:50]}...")
    print(f"max_model_len:   {args.max_model_len}")

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
                f"vLLM server not detected on {args.host}:{args.port} and --no-autostart was set."
            )

    with server:
        # Start server if not already running
        if not server.is_running():
            server.start_vllm(
                model_path=model_path,
                tensor_parallel=args.tensor_parallel,
                use_nightly=use_nightly,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_num_batched_tokens=args.max_num_batched_tokens,
            )

        # Capture GPU info with memory usage after model loads
        gpu_info = get_gpu_info(include_memory=True)
        log(f"GPU memory: {gpu_info.get('memory_used_mib', '?')} / {gpu_info.get('memory_total_mib', '?')} MiB")

        # Create OpenAI client
        client = OpenAI(
            base_url=f"http://{args.host}:{args.port}/v1",
            api_key="dummy",  # vLLM doesn't require auth by default
        )

        # Use model_path for API calls (vLLM registers model under this name)
        api_model = model_path

        # Warmup
        log("Warmup request...")
        bench_once(client, api_model, prompt_text, image_source,
                   min(64, args.max_tokens), args.temperature, args.host, args.port)

        # Benchmark
        results = []
        for i in range(args.iterations):
            log(f"Benchmark {i + 1} of {args.iterations}...")
            r = bench_once(client, api_model, prompt_text, image_source,
                          args.max_tokens, args.temperature, args.host, args.port)
            # Log with detailed metrics if available
            ttft = r.get("ttft_ms")
            gen_tok_s = r.get("generation_tok_per_s")
            if ttft is not None and gen_tok_s is not None:
                log(f"  {r['wall_s']:.2f}s | TTFT: {ttft:.1f}ms | gen: {gen_tok_s:.1f} tok/s")
            else:
                log(f"  {r['wall_s']:.2f}s")
            results.append(r)

        # Build summary with detailed metrics
        summary = {
            "median_wall_s": med(results, "wall_s"),
            "median_tok_per_s": med(results, "tok_per_s"),
            "median_ttft_ms": med(results, "ttft_ms"),
            "median_generation_tok_per_s": med(results, "generation_tok_per_s"),
        }

        # Log summary
        if summary["median_ttft_ms"] is not None and summary["median_generation_tok_per_s"] is not None:
            log(f"Median: {summary['median_generation_tok_per_s']:.1f} tok/s, TTFT: {summary['median_ttft_ms']:.1f} ms")
        else:
            log(f"Median: {summary['median_wall_s']:.2f}s")

        # Save results
        write_benchmark_result(
            results_dir=RESULTS_ROOT,
            repo_id=extract_repo_id(args.model),
            model_ref=compact_path(model_path),
            engine="vllm-server",
            gpu_info=gpu_info,
            config={
                "prompt_set": args.prompt_set,
                "prompt": prompt_text,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "tensor_parallel_size": args.tensor_parallel,
                "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "image": image_label,
            },
            iterations=results,
            summary=summary,
            extra={"mode": mode, "environment": env_label},
        )


if __name__ == "__main__":
    main()
