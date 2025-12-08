#!/usr/bin/env python3
"""
GLM-4.6V Benchmark Runner

Dedicated benchmark script for GLM-4.6V models following the official HuggingFace example.
Supports both text-only and vision modes.

Examples:

  # Text-only benchmark
  uv run python scripts/run_bench_glm4v.py \
    --model zai-org/GLM-4.6V-FP8 \
    --image none

  # Vision benchmark with local image
  uv run python scripts/run_bench_glm4v.py \
    --model zai-org/GLM-4.6V-FP8 \
    --image config/example.jpg
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

# Suppress MRoPE warning
warnings.filterwarnings("ignore", message="Unrecognized keys in.*rope_parameters.*mrope_section")

import torch
from transformers import AutoTokenizer, Glm46VForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = Path.home() / "models"
RESULTS_ROOT = ROOT / "benchmarks"

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

def infer_tag(cli_tag):
    if cli_tag:
        return cli_tag
    gpus = get_gpu_info().get("gpus", [])
    n = len(gpus)
    if n == 0: return "unknown-gpu"
    if n == 1: return "single-gpu"
    if n == 2: return "dual-gpu"
    return f"{n}-gpu"

def resolve_model_path(model_arg: str) -> str:
    if "/" in model_arg:
        local = MODELS_ROOT / model_arg
        if local.exists():
            return str(local)
    p = Path(model_arg).expanduser()
    if p.exists():
        return str(p)
    return model_arg

def bench_once_text(model, tokenizer, prompt, max_tokens, temperature, seed):
    """Text-only benchmark using tokenizer directly (bypasses buggy processor)."""
    torch.manual_seed(seed)

    # Build chat message and apply template
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    inputs.pop("token_type_ids", None)  # Model doesn't use this
    prompt_tokens = inputs["input_ids"].shape[1]

    gen_kwargs = {"max_new_tokens": max_tokens}
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    t1 = time.perf_counter()

    gen_tokens = outputs.shape[1] - prompt_tokens
    output_text = tokenizer.decode(outputs[0][prompt_tokens:], skip_special_tokens=True)

    wall = t1 - t0
    return {
        "wall_s": wall,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": gen_tokens,
        "tok_per_s": gen_tokens / wall if wall > 0 else 0,
        "output_text": output_text,
    }

def main():
    ap = argparse.ArgumentParser(description="GLM-4.6V benchmark")
    ap.add_argument("--model", default="zai-org/GLM-4.6V-FP8")
    ap.add_argument("--image", default=None, help="Image path/URL or 'none' for text-only")
    ap.add_argument("--prompt", default=None, help="Custom prompt (overrides --prompt-set)")
    ap.add_argument("--prompt-set", default="short",
                    choices=list(TEXT_PROMPTS.keys()) + list(VISION_PROMPTS.keys()))
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = infer_tag(args.tag)
    model_path = resolve_model_path(args.model)

    # Handle image
    image_path = None
    if args.image and args.image.lower() != "none":
        p = Path(args.image).expanduser()
        image_path = str(p) if p.exists() else args.image

    # Select prompt
    if args.prompt:
        prompt = args.prompt
    elif image_path is None:
        prompt = TEXT_PROMPTS.get(args.prompt_set, TEXT_PROMPTS["short"])
    else:
        prompt = VISION_PROMPTS.get(args.prompt_set, VISION_PROMPTS["describe"])

    mode = "text-only" if image_path is None else "vision"
    print(f"\n== GLM-4.6V Benchmark ==")
    print(f"model:  {model_path}")
    print(f"mode:   {mode}")
    print(f"image:  {image_path or 'none'}")
    print(f"prompt: {prompt[:50]}...")

    print("\n== Loading model ==")
    # Text-only mode: use tokenizer directly to bypass buggy processor (transformers 5.0.0rc0 bug)
    # The Glm46VProcessor requires video_processor which triggers the bug
    if image_path is not None:
        raise NotImplementedError("Vision mode not yet supported - processor bug in transformers 5.0.0rc0")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Explicitly set max_memory to use available VRAM
    n_gpus = torch.cuda.device_count()
    max_memory = {i: "90GiB" for i in range(n_gpus)} if n_gpus > 0 else None

    # MoE models require offload_folder during loading for weight format conversion
    # (weights are re-saved during load, but model runs from VRAM after loading)
    # Use explicit bfloat16 for proper FP8 decompression (not "auto")
    model = Glm46VForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        offload_folder="/tmp/glm4v_offload",
    )

    # Warmup
    print("\n== Warmup ==")
    bench_once_text(model, tokenizer, prompt, min(64, args.max_tokens), args.temperature, 0)

    # Benchmark
    print(f"\n== Running {args.iterations} iterations ==")
    results = []
    for i in range(args.iterations):
        r = bench_once_text(model, tokenizer, prompt, args.max_tokens, args.temperature, i)
        print(f"  iter {i+1}: {r['generated_tokens']} tokens in {r['wall_s']:.2f}s ({r['tok_per_s']:.1f} tok/s)")
        results.append(r)

    med_tps = statistics.median([r["tok_per_s"] for r in results])
    print(f"\n== Median: {med_tps:.1f} tok/s ==")

    # Save
    out_dir = RESULTS_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{datetime.now().strftime('%Y-%m-%d')}_{sanitize(args.model)}_glm4v_{mode}.json"

    payload = {
        "timestamp": datetime.now().isoformat(),
        "model_id": args.model,
        "engine": "glm4v",
        "mode": mode,
        "gpu_info": get_gpu_info(),
        "tag": tag,
        "config": {
            "image": image_path,
            "prompt": prompt,
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
