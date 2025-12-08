#!/usr/bin/env python3
"""
Model Workbench VLM Benchmark Runner

Uses native HuggingFace transformers inference with device_map="auto" for
multi-GPU support via accelerate. For vision-language models like GLM-4.6V.

Supports both text-only and image+text benchmarking for VLMs.

Use this for models that:
  - Require transformers 5.x
  - Are vision-language models (VLMs)
  - Need simple multi-GPU inference via accelerate

Examples:

  # Text-only benchmark for VLMs
  uv run python scripts/run_bench_vlm.py \
    --model zai-org/GLM-4.6V-FP8 \
    --image none \
    --prompt-set short

  # Use default test image (config/example.jpg)
  uv run python scripts/run_bench_vlm.py \
    --model zai-org/GLM-4.6V-FP8 \
    --prompt-set describe

  # Use custom local image
  uv run python scripts/run_bench_vlm.py \
    --model zai-org/GLM-4.6V-FP8 \
    --image /path/to/test.jpg \
    --prompt-set analyze

  # Use image URL
  uv run python scripts/run_bench_vlm.py \
    --model zai-org/GLM-4.6V-FP8 \
    --image https://example.com/image.jpg
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import time
import warnings
from datetime import datetime
from pathlib import Path

# Suppress MRoPE warning from transformers (harmless for VLMs like GLM-4.6V)
warnings.filterwarnings("ignore", message="Unrecognized keys in.*rope_parameters.*mrope_section")

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = Path.home() / "models"
RESULTS_ROOT = ROOT / "benchmarks"

# Built-in test images
BUILTIN_IMAGES = {
    "example": ROOT / "config" / "example.jpg",  # 400x267 local test image
    "grayscale": "https://upload.wikimedia.org/wikipedia/commons/f/fa/Grayscale_8bits_palette_sample_image.png",
    "color": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/800px-Camponotus_flavomarginatus_ant.jpg",
}
DEFAULT_IMAGE = "example"

# Text-only prompts (for VLMs without image input)
TEXT_PROMPTS = {
    "short": "Explain speculative decoding in 2 sentences.",
    "medium": "Summarize key tradeoffs between tensor parallelism and pipeline parallelism.",
    "long": "Write a concise technical overview of KV cache and why it matters for long context.",
}

# Vision prompts (for image+text input)
VISION_PROMPTS = {
    "describe": "Describe this image in detail.",
    "analyze": "Analyze this image and explain what you see, including any text, objects, colors, and their relationships.",
    "caption": "Provide a brief caption for this image.",
}

# Combined for CLI choices
ALL_PROMPTS = {**TEXT_PROMPTS, **VISION_PROMPTS}

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
# Image handling
# ----------------------------

def resolve_image_source(image_arg: str | None) -> tuple[str | None, str]:
    """
    Resolve image source. Returns (image_url_or_path, source_label).

    If image_arg is "none", returns (None, "none") for text-only mode.
    If image_arg is None, uses default built-in image (config/example.jpg).
    If image_arg is a builtin key (example, grayscale, color), uses that.
    Otherwise treats as URL or local path.
    """
    if image_arg is None:
        image_arg = DEFAULT_IMAGE

    # Text-only mode
    if image_arg.lower() == "none":
        return None, "none"

    if image_arg in BUILTIN_IMAGES:
        src = BUILTIN_IMAGES[image_arg]
        return str(src), f"builtin:{image_arg}"

    # Check if it's a local file
    p = Path(image_arg).expanduser()
    if p.exists():
        return str(p), str(p)

    # Assume it's a URL
    return image_arg, image_arg

# ----------------------------
# Benchmarking
# ----------------------------

def bench_once(model, processor, image_source: str | None, prompt: str, max_tokens: int, temperature: float, seed: int):
    """
    Run a single inference and measure throughput.
    Supports both text-only (image_source=None) and vision (image_source=path/url) modes.
    """
    torch.manual_seed(seed)

    # Build chat message - with or without image
    if image_source is None:
        # Text-only mode
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    else:
        # Vision mode - include image
        if image_source.startswith("http"):
            image_content = {"type": "image", "url": image_source}
        else:
            image_content = {"type": "image", "path": image_source}
        messages = [
            {
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    # Process inputs using chat template
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Move to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    # Remove token_type_ids if present (some models don't use it)
    inputs.pop("token_type_ids", None)

    prompt_tokens = inputs["input_ids"].shape[1]

    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    t1 = time.perf_counter()

    gen_tokens = outputs.shape[1] - prompt_tokens
    text = processor.decode(outputs[0][prompt_tokens:], skip_special_tokens=True)

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

def write_payload(payload, tag: str, model_id: str, is_text_only: bool):
    out_base = RESULTS_ROOT / tag
    out_base.mkdir(parents=True, exist_ok=True)

    suffix = "_vlm_text" if is_text_only else "_vlm_vision"
    label = sanitize(model_id) + suffix
    fname = f"{datetime.now().strftime('%Y-%m-%d')}_{label}.json"

    out_path = out_base / fname
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote: {out_path}")

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Vision benchmark runner (native transformers)")

    ap.add_argument("--model", required=True, help="HF repo_id or local path")
    ap.add_argument("--tag", default=None, help="Optional output tag; inferred if omitted")
    ap.add_argument("--iterations", type=int, default=3,
                    help="Number of timed iterations (median reported)")

    # Image input
    ap.add_argument("--image", default=None,
                    help="Image path, URL, builtin name (example, grayscale, color), or 'none' for text-only. Default: example")

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
    ap.add_argument("--prompt-set", default="describe", choices=tuple(ALL_PROMPTS.keys()),
                    help="Prompt type: short/medium/long (text-only) or describe/analyze/caption (vision)")

    args = ap.parse_args()

    # Tag inference from GPU count
    args.tag = infer_tag(args.tag)

    model_id = args.model
    model_path = resolve_model_path(model_id)
    image_source, image_label = resolve_image_source(args.image)

    mode = "text-only" if image_source is None else "vision"
    print(f"\n== Loading model ==")
    print(f"model_id:   {model_id}")
    print(f"model_path: {model_path}")
    print(f"engine:     transformers-vlm ({mode})")
    print(f"dtype:      {args.dtype}")
    print(f"device_map: auto")
    print(f"image:      {image_label}")

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

    # Load processor and model
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
    )

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        **model_kwargs,
    )

    prompt = ALL_PROMPTS[args.prompt_set]
    is_text_only = image_source is None

    # Warmup
    print("\n== Warmup ==")
    _ = bench_once(
        model, processor, image_source, prompt,
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
            model, processor, image_source, prompt,
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
        "engine": "transformers-vlm-text" if is_text_only else "transformers-vlm-vision",
        "gpu_info": get_gpu_info(),
        "tag": args.tag,
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "dtype": args.dtype,
            "torch_dtype": str(torch_dtype),
            "device_map": "auto",
            "trust_remote_code": args.trust_remote_code,
            "attn_implementation": args.attn_implementation,
            "image_source": image_label,
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

    write_payload(payload, args.tag, model_id, is_text_only)

if __name__ == "__main__":
    main()
