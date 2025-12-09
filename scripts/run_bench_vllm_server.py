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
  uv run python scripts/run_bench_vllm_server.py --model zai-org/GLM-4.6V-FP8

  # Force single GPU
  uv run python scripts/run_bench_vllm_server.py --model zai-org/GLM-4.6V-FP8 --tensor-parallel 1

  # Vision benchmark with local image
  uv run python scripts/run_bench_vllm_server.py \
    --model zai-org/GLM-4.6V-FP8 \
    --image config/example.jpg

  # Use an already running vLLM server
  uv run python scripts/run_bench_vllm_server.py \
    --model zai-org/GLM-4.6V-FP8 \
    --no-autostart \
    --port 8000
"""

import argparse
import base64
import json
import signal
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from bench_utils import (
    ROOT,
    MODELS_ROOT,
    RESULTS_ROOT,
    sanitize,
    get_gpu_info,
    get_gpu_count,
    infer_tag,
    port_open,
    resolve_model_path,
)

# Built-in test images
BUILTIN_IMAGES = {
    "example": ROOT / "config" / "example.jpg",
    "grayscale": "https://upload.wikimedia.org/wikipedia/commons/f/fa/Grayscale_8bits_palette_sample_image.png",
}

TEXT_PROMPTS = {
    "short": "Explain speculative decoding in 2 sentences.",
    "medium": "Summarize key tradeoffs between tensor parallelism and pipeline parallelism.",
    "long": "Write a concise technical overview of KV cache and why it matters for long context.",
}

VISION_PROMPTS = {
    "describe": "Describe this image in detail.",
    "analyze": "Analyze this image and explain what you see.",
    "caption": "Provide a brief caption for this image.",
}

ALL_PROMPTS = {**TEXT_PROMPTS, **VISION_PROMPTS}


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


def start_vllm_server(args, model_path: str) -> subprocess.Popen | None:
    """Start vLLM server with GLM-specific flags."""
    host = args.host
    port = args.port

    if port_open(host, port):
        print(f"Server already running on {host}:{port}")
        return None

    if args.no_autostart:
        raise SystemExit(
            f"vLLM server not detected on {host}:{port} and --no-autostart was set."
        )

    # Build vLLM serve command following official recipe
    cmd = [
        "vllm", "serve", model_path,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(args.tensor_parallel),
        "--enable-expert-parallel",
        "--allowed-local-media-path", "/",
        "--mm-encoder-tp-mode", "data",
        "--mm_processor_cache_type", "shm",
    ]

    if args.max_model_len:
        cmd += ["--max-model-len", str(args.max_model_len)]

    if args.gpu_memory_utilization:
        cmd += ["--gpu-memory-utilization", str(args.gpu_memory_utilization)]

    if args.max_num_batched_tokens:
        cmd += ["--max-num-batched-tokens", str(args.max_num_batched_tokens)]

    print(f"\n== Starting vLLM server ==")
    print(f"+ {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Stream server output in real-time
    server_output_lines = []

    def read_output():
        """Non-blocking read of server output, print in real-time."""
        if proc.stdout:
            import select
            while select.select([proc.stdout], [], [], 0)[0]:
                line = proc.stdout.readline()
                if line:
                    server_output_lines.append(line)
                    print(f"  [vllm] {line.rstrip()}")
                else:
                    break

    # Wait for server to be ready
    print("Waiting for server to be ready (streaming logs)...")
    max_wait = args.server_timeout
    start_time = time.time()

    while time.time() - start_time < max_wait:
        read_output()

        if port_open(host, port):
            # Additional check: try to hit the models endpoint
            try:
                client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key="dummy")
                client.models.list()
                print(f"Server ready after {time.time() - start_time:.1f}s")
                return proc
            except Exception:
                pass
        time.sleep(2)

        # Check if process died
        if proc.poll() is not None:
            read_output()
            output = "".join(server_output_lines[-50:])  # Last 50 lines
            raise SystemExit(f"vLLM server exited unexpectedly. Last output:\n{output}")

    # Timeout - show recent logs
    read_output()
    proc.terminate()
    output = "".join(server_output_lines[-50:])  # Last 50 lines
    raise SystemExit(
        f"vLLM server failed to start within {max_wait}s.\n"
        f"Last server output:\n{output}"
    )


def stop_vllm_server(proc: subprocess.Popen | None):
    if not proc:
        return
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


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
               max_tokens: int, temperature: float) -> dict:
    """Run single inference via OpenAI-compatible API."""

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

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature if temperature > 0 else 0.0,
    )
    t1 = time.perf_counter()

    output_text = response.choices[0].message.content or ""
    usage = response.usage

    prompt_tokens = usage.prompt_tokens if usage else 0
    gen_tokens = usage.completion_tokens if usage else len(output_text.split())

    wall = t1 - t0
    return {
        "wall_s": wall,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": gen_tokens,
        "tok_per_s": gen_tokens / wall if wall > 0 else 0,
        "output_text": output_text,
    }


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
    ap.add_argument("--tag", default=None)

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

    args = ap.parse_args()

    # Auto-detect GPU count for tensor parallelism
    if args.tensor_parallel is None:
        args.tensor_parallel = get_gpu_count()

    tag = infer_tag(args.tag, args.tensor_parallel)
    model_path = resolve_model_path(args.model)
    image_source, image_label = resolve_image_source(args.image)
    is_vision = image_source is not None

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
    print(f"tensor_parallel: {args.tensor_parallel}")
    print(f"image:           {image_label}")
    print(f"prompt:          {prompt_text[:50]}...")
    print(f"max_model_len:   {args.max_model_len}")

    # Start server
    proc = start_vllm_server(args, model_path)

    try:
        # Create OpenAI client
        client = OpenAI(
            base_url=f"http://{args.host}:{args.port}/v1",
            api_key="dummy",  # vLLM doesn't require auth by default
        )

        # Use model_path for API calls (vLLM registers model under this name)
        api_model = model_path

        # Warmup
        print("\n== Warmup ==")
        bench_once(client, api_model, prompt_text, image_source,
                   min(64, args.max_tokens), args.temperature)

        # Benchmark
        print(f"\n== Running {args.iterations} iterations ==")
        results = []
        for i in range(args.iterations):
            r = bench_once(client, api_model, prompt_text, image_source,
                          args.max_tokens, args.temperature)
            print(f"  iter {i+1}: {r['generated_tokens']} tokens in {r['wall_s']:.2f}s ({r['tok_per_s']:.1f} tok/s)")
            results.append(r)

        med_tps = statistics.median([r["tok_per_s"] for r in results])
        print(f"\n== Median: {med_tps:.1f} tok/s ==")

        # Save results
        out_dir = RESULTS_ROOT / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{datetime.now().strftime('%Y-%m-%d')}_{sanitize(args.model)}_vllm-server_{mode}.json"

        payload = {
            "timestamp": datetime.now().isoformat(),
            "model_id": args.model,
            "engine": "vllm-server",
            "mode": mode,
            "gpu_info": get_gpu_info(),
            "tag": tag,
            "config": {
                "tensor_parallel_size": args.tensor_parallel,
                "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "image": image_label,
                "prompt": prompt_text,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
            },
            "results": results,
            "summary": {"median_tok_per_s": med_tps},
        }

        out_path = out_dir / fname
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote: {out_path}")

    finally:
        stop_vllm_server(proc)


if __name__ == "__main__":
    main()
