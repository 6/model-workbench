#!/usr/bin/env python3
"""
Model Workbench vLLM Benchmark Runner

Uses vLLM for high-performance inference with tensor parallelism for multi-GPU support.
Recommended for models like GLM-4.6V-FP8 that have native transformers issues.

Examples:

  # Basic usage with tensor parallelism
  uv run python scripts/run_bench_vllm.py \
    --model zai-org/GLM-4.6V-FP8 \
    --tensor-parallel 2

  # Text-only benchmark
  uv run python scripts/run_bench_vllm.py \
    --model zai-org/GLM-4.6V-FP8 \
    --tensor-parallel 2 \
    --prompt-set short

  # Vision benchmark with image
  uv run python scripts/run_bench_vllm.py \
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

from vllm import LLM, SamplingParams

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


def build_prompt(prompt_text: str, image_path: str | None) -> dict:
    """Build vLLM prompt dict for text or multimodal input."""
    if image_path is None:
        return {"prompt": prompt_text}
    else:
        # For vision models, use multi_modal_data
        from PIL import Image
        image = Image.open(image_path) if not image_path.startswith("http") else None

        if image_path.startswith("http"):
            # For URLs, vLLM can fetch directly in some cases
            # But safer to download first
            import requests
            from io import BytesIO
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content))

        return {
            "prompt": f"<image>\n{prompt_text}",
            "multi_modal_data": {"image": image},
        }


def bench_once(llm, sampling_params, prompt_dict: dict, is_vision: bool):
    """Run single inference and measure throughput."""
    t0 = time.perf_counter()

    if is_vision:
        outputs = llm.generate([prompt_dict], sampling_params=sampling_params)
    else:
        outputs = llm.generate([prompt_dict["prompt"]], sampling_params=sampling_params)

    t1 = time.perf_counter()

    output = outputs[0]
    gen_tokens = len(output.outputs[0].token_ids)
    prompt_tokens = len(output.prompt_token_ids)
    output_text = output.outputs[0].text

    wall = t1 - t0
    return {
        "wall_s": wall,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": gen_tokens,
        "tok_per_s": gen_tokens / wall if wall > 0 else 0,
        "output_text": output_text,
    }


def main():
    ap = argparse.ArgumentParser(description="vLLM benchmark runner")
    ap.add_argument("--model", default="zai-org/GLM-4.6V-FP8")
    ap.add_argument("--tensor-parallel", type=int, default=2, help="Tensor parallel size (GPUs)")
    ap.add_argument("--image", default=None, help="Image path/URL or 'none' for text-only")
    ap.add_argument("--prompt", default=None, help="Custom prompt")
    ap.add_argument("--prompt-set", default="short", choices=list(ALL_PROMPTS.keys()))
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--trust-remote-code", action="store_true")
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
    print(f"\n== vLLM Benchmark ==")
    print(f"model:           {model_path}")
    print(f"mode:            {mode}")
    print(f"tensor_parallel: {args.tensor_parallel}")
    print(f"image:           {image_label}")
    print(f"prompt:          {prompt_text[:50]}...")

    # Initialize vLLM
    print("\n== Loading model with vLLM ==")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Build prompt
    prompt_dict = build_prompt(prompt_text, image_source)

    # Warmup
    print("\n== Warmup ==")
    warmup_params = SamplingParams(max_tokens=min(64, args.max_tokens), temperature=args.temperature)
    bench_once(llm, warmup_params, prompt_dict, is_vision)

    # Benchmark
    print(f"\n== Running {args.iterations} iterations ==")
    results = []
    for i in range(args.iterations):
        r = bench_once(llm, sampling_params, prompt_dict, is_vision)
        print(f"  iter {i+1}: {r['generated_tokens']} tokens in {r['wall_s']:.2f}s ({r['tok_per_s']:.1f} tok/s)")
        results.append(r)

    med_tps = statistics.median([r["tok_per_s"] for r in results])
    print(f"\n== Median: {med_tps:.1f} tok/s ==")

    # Save
    out_dir = RESULTS_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{datetime.now().strftime('%Y-%m-%d')}_{sanitize(args.model)}_vllm_{mode}.json"

    payload = {
        "timestamp": datetime.now().isoformat(),
        "model_id": args.model,
        "engine": "vllm",
        "mode": mode,
        "gpu_info": get_gpu_info(),
        "tag": tag,
        "config": {
            "tensor_parallel_size": args.tensor_parallel,
            "gpu_memory_utilization": args.gpu_memory_utilization,
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


if __name__ == "__main__":
    main()
