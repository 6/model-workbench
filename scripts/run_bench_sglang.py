#!/usr/bin/env python3
"""
Model Workbench SGLang Benchmark Runner

Uses SGLang for high-performance inference with tensor parallelism for multi-GPU support.
Recommended for models like GLM-4.6V-FP8 that require transformers 5.x (which vLLM doesn't support).

Examples:

  # Basic usage with tensor parallelism
  uv run python scripts/run_bench_sglang.py \
    --model zai-org/GLM-4.6V-FP8 \
    --tensor-parallel 2

  # Text-only benchmark
  uv run python scripts/run_bench_sglang.py \
    --model zai-org/GLM-4.6V-FP8 \
    --tensor-parallel 2 \
    --prompt-set short

  # Vision benchmark with image
  uv run python scripts/run_bench_sglang.py \
    --model zai-org/GLM-4.6V-FP8 \
    --tensor-parallel 2 \
    --image config/example.jpg \
    --prompt-set describe
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

import sglang as sgl
from sglang import Engine

ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = Path.home() / "models"
RESULTS_ROOT = ROOT / "benchmarks"

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


def sanitize(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)


def get_gpu_info():
    try:
        drv = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True
        ).strip().splitlines()
        driver_version = drv[0].strip() if drv else None

        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            text=True
        ).strip()

        gpus = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mib": int(parts[2]),
                })
        return {"driver_version": driver_version, "gpus": gpus}
    except Exception as e:
        return {"error": str(e), "driver_version": None, "gpus": []}


def infer_tag(cli_tag, tensor_parallel):
    if cli_tag:
        return cli_tag
    gpus = get_gpu_info().get("gpus", [])
    n = len(gpus)
    if n == 0:
        return "unknown-gpu"
    if tensor_parallel == 1:
        return "single-gpu"
    if tensor_parallel == 2:
        return "dual-gpu"
    return f"{tensor_parallel}-gpu"


def resolve_model_path(model_arg: str) -> str:
    if "/" in model_arg:
        local = MODELS_ROOT / model_arg
        if local.exists():
            return str(local)
    p = Path(model_arg).expanduser()
    if p.exists():
        return str(p)
    return model_arg


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


def bench_once(engine, prompt: str, image_path: str | None, max_tokens: int, temperature: float):
    """Run single inference and measure throughput using SGLang offline engine."""

    # Build input for SGLang
    if image_path is not None:
        # Vision mode - load image
        from PIL import Image
        if image_path.startswith("http"):
            import requests
            from io import BytesIO
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_path)

        # SGLang multimodal format
        input_data = {
            "text": f"<image>\n{prompt}",
            "image_data": image,
        }
    else:
        input_data = prompt

    sampling_params = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
    }

    t0 = time.perf_counter()

    if isinstance(input_data, dict):
        # Multimodal
        outputs = engine.generate(
            input_data["text"],
            sampling_params=sampling_params,
            image_data=input_data["image_data"],
        )
    else:
        # Text only
        outputs = engine.generate(input_data, sampling_params=sampling_params)

    t1 = time.perf_counter()

    # Extract results
    output = outputs[0] if isinstance(outputs, list) else outputs

    # SGLang output structure
    if hasattr(output, 'text'):
        output_text = output.text
        gen_tokens = output.meta_info.get("completion_tokens", len(output_text.split()))
        prompt_tokens = output.meta_info.get("prompt_tokens", 0)
    else:
        # Fallback for different output format
        output_text = str(output)
        gen_tokens = len(output_text.split())
        prompt_tokens = 0

    wall = t1 - t0
    return {
        "wall_s": wall,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": gen_tokens,
        "tok_per_s": gen_tokens / wall if wall > 0 else 0,
        "output_text": output_text,
    }


def main():
    ap = argparse.ArgumentParser(description="SGLang benchmark runner")
    ap.add_argument("--model", default="zai-org/GLM-4.6V-FP8")
    ap.add_argument("--tensor-parallel", type=int, default=2, help="Tensor parallel size (GPUs)")
    ap.add_argument("--image", default=None, help="Image path/URL or 'none' for text-only")
    ap.add_argument("--prompt", default=None, help="Custom prompt")
    ap.add_argument("--prompt-set", default="short", choices=list(ALL_PROMPTS.keys()))
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--mem-fraction", type=float, default=0.9, help="GPU memory fraction")
    ap.add_argument("--attention-backend", default=None,
                    choices=["flashinfer", "triton", "torch_native", "fa3"],
                    help="Attention backend (default: auto)")
    args = ap.parse_args()

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
    print(f"\n== SGLang Benchmark ==")
    print(f"model:           {model_path}")
    print(f"mode:            {mode}")
    print(f"tensor_parallel: {args.tensor_parallel}")
    print(f"image:           {image_label}")
    print(f"prompt:          {prompt_text[:50]}...")

    # Initialize SGLang engine
    print("\n== Loading model with SGLang ==")
    engine_kwargs = {
        "model_path": model_path,
        "tp_size": args.tensor_parallel,
        "mem_fraction_static": args.mem_fraction,
    }
    if args.attention_backend:
        engine_kwargs["attention_backend"] = args.attention_backend
        print(f"attention_backend: {args.attention_backend}")

    engine = Engine(**engine_kwargs)

    # Warmup
    print("\n== Warmup ==")
    bench_once(engine, prompt_text, image_source, min(64, args.max_tokens), args.temperature)

    # Benchmark
    print(f"\n== Running {args.iterations} iterations ==")
    results = []
    for i in range(args.iterations):
        r = bench_once(engine, prompt_text, image_source, args.max_tokens, args.temperature)
        print(f"  iter {i+1}: {r['generated_tokens']} tokens in {r['wall_s']:.2f}s ({r['tok_per_s']:.1f} tok/s)")
        results.append(r)

    med_tps = statistics.median([r["tok_per_s"] for r in results])
    print(f"\n== Median: {med_tps:.1f} tok/s ==")

    # Save
    out_dir = RESULTS_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{datetime.now().strftime('%Y-%m-%d')}_{sanitize(args.model)}_sglang_{mode}.json"

    payload = {
        "timestamp": datetime.now().isoformat(),
        "model_id": args.model,
        "engine": "sglang",
        "mode": mode,
        "gpu_info": get_gpu_info(),
        "tag": tag,
        "config": {
            "tensor_parallel_size": args.tensor_parallel,
            "mem_fraction_static": args.mem_fraction,
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

    # Cleanup
    engine.shutdown()


if __name__ == "__main__":
    main()
