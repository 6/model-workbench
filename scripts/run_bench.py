#!/usr/bin/env python3
"""
Model Workbench Unified Benchmark Runner

Auto-detects backend from model format (GGUF -> llama, safetensors -> vLLM).
Use --backend to explicitly select a backend.
Runs backends via Docker for reproducible results with version pinning.

Examples:

  # Safetensors model -> vLLM (auto-detected)
  uv run python scripts/run_bench.py --model ~/models/zai-org/GLM-4.6V-FP8

  # Use vLLM with prebuilt official image
  uv run python scripts/run_bench.py --model ~/models/zai-org/GLM-4.6V-FP8 --image-type prebuilt

  # Use TensorRT-LLM backend
  uv run python scripts/run_bench.py --model ~/models/zai-org/GLM-4.6V-FP8 --backend trtllm

  # GGUF model -> llama.cpp (auto-detected)
  uv run python scripts/run_bench.py --model ~/models/unsloth/GLM-4.5-Air-GGUF/UD-Q4_K_XL

  # Explicitly use ik_llama.cpp for GGUF (ikawrakow's optimized fork)
  uv run python scripts/run_bench.py --model ~/models/unsloth/GLM-GGUF/UD-Q4_K_XL --backend ik_llama

  # Override backend version
  uv run python scripts/run_bench.py --model ~/models/org/model --backend-version v0.8.0

  # Vision benchmark
  uv run python scripts/run_bench.py --model ~/models/org/model --image config/example.jpg

  # Force rebuild Docker image
  uv run python scripts/run_bench.py --model ~/models/org/model --rebuild
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
from bench_utils import (
    ALL_PROMPTS,
    RESULTS_ROOT,
    TEXT_PROMPTS,
    VISION_PROMPTS,
    build_chat_messages,
    compact_path,
    extract_repo_id,
    find_mmproj,
    get_gpu_info,
    get_model_backend_version,
    med,
    resolve_image_source,
    resolve_local_gguf,
    resolve_run_config,
    warmup_model,
    write_benchmark_result,
)
from common import BACKEND_REGISTRY, log
from openai import OpenAI
from server_manager import ServerManager

# ----------------------------
# Cleanup helpers (ensure idempotency)
# ----------------------------


def get_containers_on_port(port: int) -> list[str]:
    """Get list of container IDs running on the target port.

    Args:
        port: Port number to check

    Returns:
        List of container IDs (may be empty)
    """
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"publish={port}", "-q"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        container_ids = [cid.strip() for cid in result.stdout.strip().split("\n") if cid.strip()]
        return container_ids
    except Exception:
        return []


def prompt_cleanup_confirmation(port: int, container_ids: list[str]) -> bool:
    """Ask user if they want to stop existing containers.

    Args:
        port: Port number
        container_ids: List of container IDs to stop

    Returns:
        True if user confirms, False otherwise
    """
    count = len(container_ids)
    plural = "container" if count == 1 else "containers"

    print(f"\n{count} {plural} already running on port {port}:")
    for cid in container_ids:
        print(f"  - {cid[:12]}")

    try:
        response = input(f"\nStop {'it' if count == 1 else 'them'} and start new server? [y/n]: ")
        return response.lower() in ["y", "yes"]
    except (KeyboardInterrupt, EOFError):
        print("\nAborted.")
        return False


def cleanup_existing_containers(port: int):
    """Stop any Docker containers using the target port.

    This ensures benchmark is idempotent - can be run multiple times
    without manual cleanup of orphaned containers.

    Args:
        port: Port number to check for containers
    """
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"publish={port}", "-q"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        container_ids = [cid.strip() for cid in result.stdout.strip().split("\n") if cid.strip()]

        if container_ids:
            log(f"Found {len(container_ids)} existing container(s) on port {port}, cleaning up...")
            for cid in container_ids:
                try:
                    subprocess.run(["docker", "stop", cid], timeout=30, capture_output=True)
                    log(f"Stopped container {cid[:12]}")
                except Exception as e:
                    log(f"Warning: Failed to stop container {cid[:12]}: {e}")
    except Exception as e:
        log(f"Warning: Could not check for existing containers: {e}")


# ----------------------------
# Prometheus metrics scraping (shared for vLLM and TensorRT-LLM)
# ----------------------------

# Metric configs: (metric_name_base, output_key)
PROMETHEUS_METRICS = {
    "vllm": {
        "path": "/metrics",
        "pattern": r"^(vllm:[a-z_]+(?:_sum|_count))(?:\{[^}]*\})?\s+([\d.eE+-]+)",
        "mappings": {
            "vllm:time_to_first_token_seconds": "ttft_ms",
            "vllm:request_prefill_time_seconds": "prefill_ms",
            "vllm:request_decode_time_seconds": "generation_ms",
        },
    },
    "trtllm": {
        "path": "/prometheus/metrics",
        "pattern": r"^(trtllm_[a-z_]+(?:_sum|_count))(?:\{[^}]*\})?\s+([\d.eE+-]+)",
        "mappings": {
            "trtllm_time_to_first_token_seconds": "ttft_ms",
            "trtllm_e2e_latency_seconds": "e2e_latency_ms",
            "trtllm_time_per_output_token_seconds": "tpot_ms",
        },
    },
}


def scrape_prometheus_metrics(host: str, port: int, backend: str) -> dict[str, float] | None:
    """Scrape Prometheus metrics from vLLM or TensorRT-LLM backend."""
    cfg = PROMETHEUS_METRICS.get(backend)
    if not cfg:
        return None

    try:
        resp = requests.get(f"http://{host}:{port}{cfg['path']}", timeout=5)
        resp.raise_for_status()
    except Exception:
        return None

    metrics = {}
    pattern = re.compile(cfg["pattern"], re.MULTILINE)

    for match in pattern.finditer(resp.text):
        name, value = match.groups()
        try:
            metrics[name] = float(value)
        except ValueError:
            pass

    return metrics if metrics else None


def compute_metrics_delta(
    before: dict[str, float] | None,
    after: dict[str, float] | None,
    backend: str,
) -> dict[str, float]:
    """Compute delta between two metrics snapshots. Returns timing in milliseconds."""
    result = {}
    if not before or not after:
        return result

    cfg = PROMETHEUS_METRICS.get(backend)
    if not cfg:
        return result

    for metric_base, output_key in cfg["mappings"].items():
        sum_key = f"{metric_base}_sum"
        count_key = f"{metric_base}_count"

        if sum_key in before and sum_key in after and count_key in before and count_key in after:
            delta_sum = after[sum_key] - before[sum_key]
            delta_count = after[count_key] - before[count_key]

            if delta_count > 0:
                result[output_key] = (delta_sum / delta_count) * 1000

    return result


# ----------------------------
# Backend-specific benchmark functions
# ----------------------------


def bench_once_vllm(
    client: OpenAI,
    model: str,
    prompt: str,
    image_path: str | None,
    max_tokens: int,
    temperature: float,
    host: str,
    port: int,
) -> dict:
    """Run single inference via vLLM OpenAI-compatible API."""
    messages = build_chat_messages(prompt, image_path)

    # Scrape metrics before request
    metrics_before = scrape_prometheus_metrics(host, port, "vllm")

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature if temperature > 0 else 0.0,
    )
    t1 = time.perf_counter()

    # Scrape metrics after request
    metrics_after = scrape_prometheus_metrics(host, port, "vllm")

    output_text = response.choices[0].message.content or ""
    wall = t1 - t0

    prompt_tokens = None
    generated_tokens = None
    if response.usage:
        prompt_tokens = response.usage.prompt_tokens
        generated_tokens = response.usage.completion_tokens

    tok_per_s = None
    generation_tok_per_s = None
    if generated_tokens and wall > 0:
        tok_per_s = generated_tokens / wall

    timing = compute_metrics_delta(metrics_before, metrics_after, "vllm")

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
    result.update(timing)

    return result


def bench_once_llama(
    prompt: str,
    image_path: str | None,
    max_tokens: int,
    temperature: float,
    host: str,
    port: int,
) -> dict:
    """Run single inference via llama.cpp server."""
    base = f"http://{host}:{port}"

    if image_path is None:
        # Text-only: use native /completion endpoint for detailed timings
        url = base.rstrip("/") + "/completion"

        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
        }

        t0 = time.perf_counter()
        r = requests.post(url, json=payload, timeout=300)
        r.raise_for_status()
        t1 = time.perf_counter()

        data = r.json()
        wall = t1 - t0

        text = data.get("content", "")
        timings = data.get("timings", {})
        prompt_tokens = data.get("tokens_evaluated")
        prompt_ms = timings.get("prompt_ms")
        gen_tokens = timings.get("predicted_n")
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
    else:
        # Vision: use /v1/chat/completions with multimodal messages
        url = base.rstrip("/") + "/v1/chat/completions"
        messages = build_chat_messages(prompt, image_path)

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        t0 = time.perf_counter()
        r = requests.post(url, json=payload, timeout=300)
        r.raise_for_status()
        t1 = time.perf_counter()

        data = r.json()
        wall = t1 - t0

        text = ""
        if "choices" in data and data["choices"]:
            choice = data["choices"][0]
            if "message" in choice:
                text = choice["message"].get("content", "")

        prompt_tokens = None
        gen_tokens = None
        if "usage" in data:
            prompt_tokens = data["usage"].get("prompt_tokens")
            gen_tokens = data["usage"].get("completion_tokens")

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


def bench_once_trtllm(
    client: OpenAI,
    model: str,
    prompt: str,
    image_path: str | None,
    max_tokens: int,
    temperature: float,
    host: str,
    port: int,
) -> dict:
    """Run single inference via TensorRT-LLM OpenAI-compatible API."""
    messages = build_chat_messages(prompt, image_path)

    # Scrape TensorRT-LLM metrics before request
    metrics_before = scrape_prometheus_metrics(host, port, "trtllm")

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature if temperature > 0 else 0.0,
    )
    t1 = time.perf_counter()

    # Scrape metrics after request
    metrics_after = scrape_prometheus_metrics(host, port, "trtllm")

    output_text = response.choices[0].message.content or ""
    wall = t1 - t0

    prompt_tokens = None
    generated_tokens = None
    if response.usage:
        prompt_tokens = response.usage.prompt_tokens
        generated_tokens = response.usage.completion_tokens

    tok_per_s = None
    generation_tok_per_s = None
    if generated_tokens and wall > 0:
        tok_per_s = generated_tokens / wall

    timing = compute_metrics_delta(metrics_before, metrics_after, "trtllm")

    # Calculate generation_tok_per_s from TPOT if available
    if generated_tokens and timing.get("tpot_ms"):
        tpot_s = timing["tpot_ms"] / 1000
        if tpot_s > 0:
            generation_tok_per_s = 1 / tpot_s

    result = {
        "wall_s": wall,
        "output_text": output_text,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "tok_per_s": tok_per_s,
        "generation_tok_per_s": generation_tok_per_s,
        "ttft_ms": timing.get("ttft_ms"),
        "tpot_ms": timing.get("tpot_ms"),
        "e2e_latency_ms": timing.get("e2e_latency_ms"),
    }

    return result


# ----------------------------
# Unified benchmark runner
# ----------------------------


def run_benchmark_vllm(
    args, model_path: str, image_path: str | None, image_label: str, image_type: str = "build"
):
    """Run benchmarks using vLLM backend (Docker-only)."""
    is_vision = image_path is not None
    mode = "vision" if is_vision else "text-only"

    # Resolve backend version from config or CLI
    backend_version = args.backend_version or get_model_backend_version(args.model, "vllm")
    if not backend_version:
        raise SystemExit(
            "No backend version specified and none found in config.\n"
            "Either:\n"
            "  1. Set defaults.backends.vllm.version in config/models.yaml\n"
            "  2. Pass --backend-version v0.8.0"
        )

    # Select prompt
    if args.prompt:
        prompt_text = args.prompt
    elif is_vision:
        prompt_text = VISION_PROMPTS.get(args.prompt_set, VISION_PROMPTS["describe"])
    else:
        prompt_text = TEXT_PROMPTS.get(args.prompt_set, TEXT_PROMPTS["short"])

    print("\n== vLLM Benchmark ==")
    print(f"model:           {model_path}")
    print(f"mode:            {mode}")
    print(f"backend_version: {backend_version}")
    print(f"image_type:      {image_type}")
    print(f"tensor_parallel: {args.tensor_parallel}")
    print(f"image:           {image_label}")
    print(f"prompt:          {prompt_text[:50]}...")
    print(f"max_model_len:   {args.max_model_len}")

    server = ServerManager(
        host=args.host,
        port=args.port,
        timeout=args.server_timeout,
    )

    if not server.is_running() and args.no_autostart:
        raise SystemExit(
            f"vLLM server not detected on {args.host}:{args.port} and --no-autostart was set."
        )

    with server:
        # Check for existing containers and prompt user
        if not args.no_autostart and not args.force_cleanup:
            existing = get_containers_on_port(args.port)

            if existing:
                # Non-interactive environment check
                if not sys.stdin.isatty():
                    log("Error: Container running on port and stdin is not interactive")
                    log("Use --force-cleanup to automatically stop containers in CI/CD")
                    sys.exit(1)

                # Interactive prompt
                if not prompt_cleanup_confirmation(args.port, existing):
                    log("Use --no-autostart to benchmark against existing server")
                    sys.exit(0)

                cleanup_existing_containers(args.port)
        elif not args.no_autostart and args.force_cleanup:
            # Force cleanup without prompt (for automation)
            cleanup_existing_containers(args.port)

        if not server.is_running():
            server.start_vllm(
                model_path=model_path,
                tensor_parallel=args.tensor_parallel,
                version=backend_version,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_num_batched_tokens=args.max_num_batched_tokens,
                rebuild=args.rebuild,
                image_type=image_type,
                image_override=args.docker_image,
            )

        gpu_info = get_gpu_info(include_memory=True)
        log(
            f"GPU memory: {gpu_info.get('memory_used_mib', '?')} / {gpu_info.get('memory_total_mib', '?')} MiB"
        )

        client = OpenAI(
            base_url=f"http://{args.host}:{args.port}/v1",
            api_key="dummy",
        )

        api_model = model_path

        # Warmup
        log("Warmup request...")
        warmup_model(
            backend="vllm",
            host=args.host,
            port=args.port,
            api_model=api_model,
            max_tokens=min(64, args.max_tokens),
        )

        # Benchmark
        results = []
        for i in range(args.iterations):
            log(f"Benchmark {i + 1} of {args.iterations}...")
            r = bench_once_vllm(
                client,
                api_model,
                prompt_text,
                image_path,
                args.max_tokens,
                args.temperature,
                args.host,
                args.port,
            )
            ttft = r.get("ttft_ms")
            gen_tok_s = r.get("generation_tok_per_s")
            if ttft is not None and gen_tok_s is not None:
                log(f"  {r['wall_s']:.2f}s | TTFT: {ttft:.1f}ms | gen: {gen_tok_s:.1f} tok/s")
            else:
                log(f"  {r['wall_s']:.2f}s")
            results.append(r)

        summary = {
            "median_wall_s": med(results, "wall_s"),
            "median_tok_per_s": med(results, "tok_per_s"),
            "median_ttft_ms": med(results, "ttft_ms"),
            "median_generation_tok_per_s": med(results, "generation_tok_per_s"),
        }

        if (
            summary["median_ttft_ms"] is not None
            and summary["median_generation_tok_per_s"] is not None
        ):
            log(
                f"Median: {summary['median_generation_tok_per_s']:.1f} tok/s, TTFT: {summary['median_ttft_ms']:.1f} ms"
            )
        else:
            log(f"Median: {summary['median_wall_s']:.2f}s")

        write_benchmark_result(
            results_dir=RESULTS_ROOT,
            repo_id=extract_repo_id(args.model),
            model_ref=compact_path(model_path),
            engine="vllm-server",
            mode=mode,
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
                "backend_version": backend_version,
            },
            iterations=results,
            summary=summary,
        )


def run_benchmark_trtllm(args, model_path: str, image_path: str | None, image_label: str):
    """Run benchmarks using TensorRT-LLM backend (NGC prebuilt images)."""
    is_vision = image_path is not None
    mode = "vision" if is_vision else "text-only"

    # Resolve backend version from config or CLI
    backend_version = args.backend_version or get_model_backend_version(args.model, "trtllm")
    if not backend_version:
        raise SystemExit(
            "No backend version specified and none found in config.\n"
            "Either:\n"
            "  1. Set defaults.backends.trtllm.version in config/models.yaml\n"
            "  2. Pass --backend-version 0.18.0"
        )

    # Select prompt
    if args.prompt:
        prompt_text = args.prompt
    elif is_vision:
        prompt_text = VISION_PROMPTS.get(args.prompt_set, VISION_PROMPTS["describe"])
    else:
        prompt_text = TEXT_PROMPTS.get(args.prompt_set, TEXT_PROMPTS["short"])

    print("\n== TensorRT-LLM Benchmark ==")
    print(f"model:           {model_path}")
    print(f"mode:            {mode}")
    print(f"backend_version: {backend_version}")
    print(f"tensor_parallel: {args.tensor_parallel}")
    print(f"image:           {image_label}")
    print(f"prompt:          {prompt_text[:50]}...")

    server = ServerManager(
        host=args.host,
        port=args.port,
        timeout=args.server_timeout,
    )

    if not server.is_running() and args.no_autostart:
        raise SystemExit(
            f"TensorRT-LLM server not detected on {args.host}:{args.port} and --no-autostart was set."
        )

    with server:
        # Check for existing containers and prompt user
        if not args.no_autostart and not args.force_cleanup:
            existing = get_containers_on_port(args.port)

            if existing:
                # Non-interactive environment check
                if not sys.stdin.isatty():
                    log("Error: Container running on port and stdin is not interactive")
                    log("Use --force-cleanup to automatically stop containers in CI/CD")
                    sys.exit(1)

                # Interactive prompt
                if not prompt_cleanup_confirmation(args.port, existing):
                    log("Use --no-autostart to benchmark against existing server")
                    sys.exit(0)

                cleanup_existing_containers(args.port)
        elif not args.no_autostart and args.force_cleanup:
            # Force cleanup without prompt (for automation)
            cleanup_existing_containers(args.port)

        if not server.is_running():
            server.start_trtllm(
                model_path=model_path,
                tensor_parallel=args.tensor_parallel,
                version=backend_version,
                rebuild=args.rebuild,
            )

        gpu_info = get_gpu_info(include_memory=True)
        log(
            f"GPU memory: {gpu_info.get('memory_used_mib', '?')} / {gpu_info.get('memory_total_mib', '?')} MiB"
        )

        client = OpenAI(
            base_url=f"http://{args.host}:{args.port}/v1",
            api_key="dummy",
        )

        api_model = model_path

        # Warmup
        log("Warmup request...")
        warmup_model(
            backend="trtllm",
            host=args.host,
            port=args.port,
            api_model=api_model,
            max_tokens=min(64, args.max_tokens),
        )

        # Benchmark
        results = []
        for i in range(args.iterations):
            log(f"Benchmark {i + 1} of {args.iterations}...")
            r = bench_once_trtllm(
                client,
                api_model,
                prompt_text,
                image_path,
                args.max_tokens,
                args.temperature,
                args.host,
                args.port,
            )
            ttft = r.get("ttft_ms")
            gen_tok_s = r.get("generation_tok_per_s")
            if ttft is not None and gen_tok_s is not None:
                log(f"  {r['wall_s']:.2f}s | TTFT: {ttft:.1f}ms | gen: {gen_tok_s:.1f} tok/s")
            else:
                log(f"  {r['wall_s']:.2f}s | {r.get('tok_per_s', 0):.1f} tok/s (wall)")
            results.append(r)

        summary = {
            "median_wall_s": med(results, "wall_s"),
            "median_tok_per_s": med(results, "tok_per_s"),
            "median_ttft_ms": med(results, "ttft_ms"),
            "median_generation_tok_per_s": med(results, "generation_tok_per_s"),
            "median_tpot_ms": med(results, "tpot_ms"),
        }

        if (
            summary["median_ttft_ms"] is not None
            and summary["median_generation_tok_per_s"] is not None
        ):
            log(
                f"Median: {summary['median_generation_tok_per_s']:.1f} tok/s, TTFT: {summary['median_ttft_ms']:.1f} ms"
            )
        elif summary["median_tok_per_s"] is not None:
            log(f"Median: {summary['median_tok_per_s']:.1f} tok/s (wall clock)")
        else:
            log(f"Median: {summary['median_wall_s']:.2f}s")

        write_benchmark_result(
            results_dir=RESULTS_ROOT,
            repo_id=extract_repo_id(args.model),
            model_ref=compact_path(model_path),
            engine="trtllm-server",
            mode=mode,
            gpu_info=gpu_info,
            config={
                "prompt_set": args.prompt_set,
                "prompt": prompt_text,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "tensor_parallel_size": args.tensor_parallel,
                "image": image_label,
                "backend_version": backend_version,
            },
            iterations=results,
            summary=summary,
        )


def run_benchmark_gguf(
    args, model_path: str, image_path: str | None, image_label: str, backend: str
):
    """Run benchmarks using GGUF backend (llama.cpp or ik_llama.cpp) via Docker."""
    is_vision = image_path is not None
    mode = "vision" if is_vision else "text-only"

    # Resolve backend version from config or CLI
    backend_version = args.backend_version or get_model_backend_version(args.model, backend)
    if not backend_version:
        raise SystemExit(
            f"No backend version specified and none found in config.\n"
            f"Either:\n"
            f"  1. Set defaults.backends.{backend}.version in config/models.yaml\n"
            f"  2. Pass --backend-version b4521"
        )

    # Resolve GGUF path
    gguf_path = resolve_local_gguf(model_path)
    if not gguf_path:
        raise SystemExit(
            f"No GGUF found at: {Path(model_path).expanduser()}\n"
            "Pass --model with an explicit path, e.g.:\n"
            "  --model ~/models/org/repo/quant-folder\n"
            "  --model ~/models/org/repo/model.gguf"
        )

    # Resolve mmproj for vision
    mmproj_path = None
    if is_vision:
        if args.mmproj:
            mmproj_path = Path(args.mmproj).expanduser()
            if not mmproj_path.exists():
                raise SystemExit(f"mmproj not found: {mmproj_path}")
        else:
            mmproj_path = find_mmproj(gguf_path)
            if not mmproj_path:
                raise SystemExit(
                    f"Vision mode requires mmproj file but none found.\n"
                    f"Searched from: {gguf_path.parent}\n"
                    "Either:\n"
                    "  1. Download mmproj-*.gguf to the model directory\n"
                    "  2. Pass --mmproj /path/to/mmproj-*.gguf explicitly"
                )

    # Select prompt
    if is_vision:
        prompt_text = VISION_PROMPTS.get(args.prompt_set) or VISION_PROMPTS["describe"]
    else:
        prompt_text = TEXT_PROMPTS[args.prompt_set]

    backend_label = BACKEND_REGISTRY[backend]["display_name"]

    print(f"\n== {backend_label} Benchmark ==")
    print(f"model_id:        {model_path}")
    print(f"model_ref:       {gguf_path}")
    print(f"backend:         {backend}")
    print(f"backend_version: {backend_version}")
    print(f"mode:            {mode}")
    if is_vision:
        print(f"image:           {image_label}")
        print(f"mmproj:          {mmproj_path}")

    server = ServerManager(
        host=args.host,
        port=args.port,
        timeout=args.server_timeout,
    )

    if not server.is_running() and args.no_autostart:
        raise SystemExit(
            f"{backend_label} server not detected on {args.host}:{args.port} and --no-autostart was set."
        )

    with server:
        # Check for existing containers and prompt user
        if not args.no_autostart and not args.force_cleanup:
            existing = get_containers_on_port(args.port)

            if existing:
                # Non-interactive environment check
                if not sys.stdin.isatty():
                    log("Error: Container running on port and stdin is not interactive")
                    log("Use --force-cleanup to automatically stop containers in CI/CD")
                    sys.exit(1)

                # Interactive prompt
                if not prompt_cleanup_confirmation(args.port, existing):
                    log("Use --no-autostart to benchmark against existing server")
                    sys.exit(0)

                cleanup_existing_containers(args.port)
        elif not args.no_autostart and args.force_cleanup:
            # Force cleanup without prompt (for automation)
            cleanup_existing_containers(args.port)

        if not server.is_running():
            gguf_path = server.start_gguf_backend(
                engine=backend,
                model_path=model_path,
                version=backend_version,
                n_gpu_layers=args.n_gpu_layers,
                ctx=args.ctx,
                parallel=args.parallel,
                mmproj_path=mmproj_path,
                extra_args=args.extra_args,
                rebuild=args.rebuild,
            )

        gpu_info = get_gpu_info(include_memory=True)
        log(
            f"GPU memory: {gpu_info.get('memory_used_mib', '?')} / {gpu_info.get('memory_total_mib', '?')} MiB"
        )

        log("Warmup request...")
        warmup_model(
            backend=backend,
            host=args.host,
            port=args.port,
            api_model="gpt-3.5-turbo",  # llama.cpp uses this default model name
            max_tokens=min(64, args.max_tokens),
        )

        results = []
        for i in range(args.iterations):
            log(f"Benchmark {i + 1} of {args.iterations}...")
            r = bench_once_llama(
                prompt_text, image_path, args.max_tokens, args.temperature, args.host, args.port
            )
            ttft = r.get("ttft_ms")
            gen_tok_s = r.get("generation_tok_per_s")
            if ttft is not None and gen_tok_s is not None:
                log(f"  {r['wall_s']:.2f}s | TTFT: {ttft:.1f}ms | gen: {gen_tok_s:.1f} tok/s")
            else:
                log(f"  {r['wall_s']:.2f}s")
            results.append(r)

        summary = {
            "median_wall_s": med(results, "wall_s"),
            "median_tok_per_s": med(results, "tok_per_s"),
            "median_ttft_ms": med(results, "ttft_ms"),
            "median_generation_tok_per_s": med(results, "generation_tok_per_s"),
        }

        gen_tok_s = summary.get("median_generation_tok_per_s")
        ttft = summary.get("median_ttft_ms")
        gen_str = f"{gen_tok_s:.1f}" if gen_tok_s else "?"
        ttft_str = f"{ttft:.1f}" if ttft else "?"
        log(f"Median: {gen_str} tok/s, TTFT: {ttft_str} ms")

        config = {
            "prompt_set": args.prompt_set,
            "prompt": prompt_text,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "backend_version": backend_version,
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

        # Use backend-specific engine name for results
        engine_name = "ik_llama-server" if backend == "ik_llama" else "llama-server"

        write_benchmark_result(
            results_dir=RESULTS_ROOT,
            repo_id=extract_repo_id(model_path),
            model_ref=compact_path(str(gguf_path)),
            engine=engine_name,
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
    ap = argparse.ArgumentParser(
        description="Unified benchmark runner - auto-detects backend from model format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    ap.add_argument("--model", required=True, help="Model path (auto-detects GGUF vs safetensors)")

    # Backend selection
    ap.add_argument(
        "--backend",
        choices=list(BACKEND_REGISTRY.keys()),
        default=None,
        help="Backend to use (default: auto-detect from model format)",
    )

    # Common benchmark options
    ap.add_argument("--prompt", default=None, help="Custom prompt (overrides --prompt-set)")
    ap.add_argument(
        "--prompt-set",
        default="long",
        choices=list(ALL_PROMPTS.keys()),
        help="Prompt set to use (default: long)",
    )
    ap.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    ap.add_argument("--iterations", type=int, default=5, help="Number of benchmark iterations")
    ap.add_argument(
        "--image", default=None, help="Image for vision benchmark: path, URL, 'example', or 'none'"
    )

    # Server options
    ap.add_argument("--host", default="127.0.0.1", help="Server host")
    ap.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (default: 8000 for vLLM, 8080 for llama.cpp)",
    )
    ap.add_argument(
        "--no-autostart", action="store_true", help="Don't start server; require it already running"
    )
    ap.add_argument(
        "--force-cleanup",
        action="store_true",
        help="Automatically stop existing containers without prompting (for automation)",
    )
    ap.add_argument(
        "--server-timeout",
        type=int,
        default=360,
        help="Timeout waiting for server to start (default: 360s)",
    )

    # Backend version and image options (Docker-based execution)
    ap.add_argument(
        "--backend-version",
        default=None,
        help="Backend version (e.g., v0.8.0 for vLLM, b4521 for llama.cpp, 0.18.0 for trtllm)",
    )
    ap.add_argument(
        "--rebuild", action="store_true", help="Force rebuild/repull Docker image even if cached"
    )
    ap.add_argument(
        "--build-only",
        action="store_true",
        help="Only build/pull Docker image, don't run benchmark",
    )
    ap.add_argument(
        "--image-type",
        choices=["prebuilt", "build"],
        default=None,
        help="Image type: 'prebuilt' uses official images, 'build' compiles from source",
    )
    ap.add_argument(
        "--docker-image",
        default=None,
        dest="docker_image",
        help="Direct Docker image to use (e.g., vllm/vllm-openai:nightly)",
    )

    # vLLM-specific options (defaults from config, CLI overrides)
    vllm_group = ap.add_argument_group("vLLM options (safetensors models)")
    vllm_group.add_argument(
        "--tensor-parallel",
        type=int,
        default=None,
        help="Tensor parallel size (default: auto-detect GPU count)",
    )
    vllm_group.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Max context length (default: from config or 65536)",
    )
    vllm_group.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="GPU memory fraction (default: from config or 0.95)",
    )
    vllm_group.add_argument(
        "--max-num-batched-tokens", type=int, default=None, help="Max batched tokens"
    )

    # llama.cpp-specific options (defaults from config, CLI overrides)
    llama_group = ap.add_argument_group("llama.cpp options (GGUF models)")
    llama_group.add_argument("--ctx", type=int, default=None, help="Context length (-c)")
    llama_group.add_argument(
        "--n-gpu-layers",
        type=int,
        default=None,
        help="GPU layers to offload (-ngl, default: from config or 999)",
    )
    llama_group.add_argument(
        "--parallel", type=int, default=1, help="Parallel sequences (-np, default: 1)"
    )
    llama_group.add_argument(
        "--mmproj", default=None, help="Multimodal projector path (auto-detected if not specified)"
    )
    llama_group.add_argument("--seed", type=int, default=0, help="Sampling seed")
    llama_group.add_argument(
        "--extra-args", nargs=argparse.REMAINDER, help="Extra args passed to llama-server"
    )

    args = ap.parse_args()

    # Resolve backend config and apply defaults
    backend, model_path, backend_cfg = resolve_run_config(args)

    # Resolve image type (from CLI, model config, or backend defaults)
    image_type = args.image_type or backend_cfg.get("image_type", "build")

    # Resolve image (for vision benchmark, not Docker image)
    image_path, image_label = resolve_image_source(args.image)

    # Handle --build-only flag
    if args.build_only:
        from docker_manager import ensure_image

        version = args.backend_version or get_model_backend_version(args.model, backend)
        if not version:
            raise SystemExit(
                f"No backend version specified. Pass --backend-version or set defaults.backends.{backend}.version in config."
            )

        log(f"Preparing {backend} image for version {version}...")
        image_name = ensure_image(
            backend,
            version,
            rebuild=args.rebuild,
            image_type=image_type,
            image_override=args.docker_image,
        )
        log(f"Image ready: {image_name}")
        return

    # Log backend selection and run benchmark
    backend_label = BACKEND_REGISTRY[backend]["display_name"]

    if args.backend:
        log(f"Using explicit backend: {backend_label}")
    else:
        log(f"Auto-detected model format -> {backend_label} backend")

    if backend == "vllm":
        run_benchmark_vllm(args, model_path, image_path, image_label, image_type)
    elif backend == "trtllm":
        run_benchmark_trtllm(args, model_path, image_path, image_label)
    else:
        run_benchmark_gguf(args, model_path, image_path, image_label, backend)


if __name__ == "__main__":
    main()
