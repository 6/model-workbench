#!/usr/bin/env python3
"""NVFP4 compression script using llm-compressor (Docker-based).

Runs inside Docker using vLLM base image to ensure compressed-tensors version
alignment between quantizer and runtime.

WARNING: NVFP4 quantization may be SLOWER than BF16 on Blackwell workstation GPUs
(RTX PRO 6000, RTX 5090 - SM120). As of Dec 2025, vLLM kernel optimization for
SM120 is incomplete. See: https://github.com/vllm-project/vllm/issues/31085
NVFP4 is designed for data center Blackwell (B200 - SM100).

Usage:
    uv run python scripts/compress_nvfp4.py --model ~/models/google/gemma-3-1b-it
    uv run python scripts/compress_nvfp4.py --model ~/models/google/gemma-3-1b-it --vllm-version v0.13.0
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Constants
DEFAULT_OUTPUT_BASE = Path.home() / "models-quantized"
DEFAULT_CALIBRATION_SAMPLES = 512
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_VLLM_VERSION = "v0.13.0"
CALIBRATION_DATASET = "HuggingFaceH4/ultrachat_200k"
CALIBRATION_SPLIT = "train_sft"
IMAGE_PREFIX = "model-bench-compressor"
DOCKERFILE = Path(__file__).parent.parent / "docker" / "Dockerfile.compressor"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compress model to NVFP4 format using llm-compressor (Docker-based)"
    )
    parser.add_argument("--model", required=True, help="Path to source model")
    parser.add_argument(
        "--output", help="Output path (default: ~/models-quantized/<org>/<repo>-NVFP4)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_CALIBRATION_SAMPLES,
        help=f"Number of calibration samples (default: {DEFAULT_CALIBRATION_SAMPLES})",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help=f"Maximum sequence length for calibration (default: {DEFAULT_MAX_SEQ_LENGTH})",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip generation test after quantization",
    )
    parser.add_argument(
        "--vllm-version",
        default=DEFAULT_VLLM_VERSION,
        help=f"vLLM version for Docker base image (default: {DEFAULT_VLLM_VERSION})",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild Docker image",
    )
    return parser.parse_args()


def resolve_output_path(model_path: Path, output_arg: str | None) -> Path:
    """Derive output path from model path, preserving org/repo structure."""
    if output_arg:
        return Path(output_arg).expanduser().resolve()

    parts = model_path.parts
    if "models" in parts:
        idx = parts.index("models")
        org_repo = "/".join(parts[idx + 1 :])
    else:
        org_repo = model_path.name

    return DEFAULT_OUTPUT_BASE / f"{org_repo}-NVFP4"


def image_exists(image_name: str) -> bool:
    """Check if Docker image exists locally."""
    result = subprocess.run(
        ["docker", "images", "-q", image_name],
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def build_image(version: str, force: bool = False) -> str:
    """Build the compressor Docker image."""
    image_name = f"{IMAGE_PREFIX}:{version}"

    if not force and image_exists(image_name):
        print(f"Using existing image: {image_name}")
        return image_name

    print(f"Building Docker image: {image_name}")
    cmd = [
        "docker",
        "build",
        "-f",
        str(DOCKERFILE),
        "--build-arg",
        f"VERSION={version}",
        "-t",
        image_name,
        str(DOCKERFILE.parent.parent),
    ]
    subprocess.run(cmd, check=True)
    return image_name


def run_in_docker(args, model_path: Path, output_path: Path):
    """Run compression inside Docker container."""
    image_name = build_image(args.vllm_version, force=args.rebuild)

    print("\nStarting NVFP4 compression in Docker...")
    print(f"  Model:  {model_path}")
    print(f"  Output: {output_path}")

    # Ensure output parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--entrypoint",
        "python3",
        "-e",
        "INSIDE_DOCKER=1",
        "-v",
        f"{model_path}:{model_path}:ro",
        "-v",
        f"{output_path.parent}:{output_path.parent}:rw",
        "-v",
        f"{Path(__file__).resolve()}:/workspace/compress_nvfp4.py:ro",
        image_name,
        "/workspace/compress_nvfp4.py",
        "--model",
        str(model_path),
        "--output",
        str(output_path),
        "--samples",
        str(args.samples),
        "--max-seq-len",
        str(args.max_seq_len),
    ]
    if args.skip_test:
        cmd.append("--skip-test")

    subprocess.run(cmd, check=True)


# ============================================================================
# Docker-internal functions (only run inside container)
# ============================================================================


def run_quantization(args, model_path: Path, output_path: Path):
    """Run the actual quantization (inside Docker)."""
    from datasets import load_dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Source model: {model_path}")
    print(f"Output path:  {output_path}")
    print(f"Calibration:  {args.samples} samples, max_seq_len={args.max_seq_len}")

    # Load model and tokenizer
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    # Prepare calibration data
    print(f"\nLoading {args.samples} samples from {CALIBRATION_DATASET}...")
    ds = load_dataset(CALIBRATION_DATASET, split=f"{CALIBRATION_SPLIT}[:{args.samples}]")
    ds = ds.shuffle(seed=42)

    def preprocess(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=args.max_seq_len,
            truncation=True,
            add_special_tokens=False,
        )

    calibration_ds = ds.map(tokenize, remove_columns=ds.column_names)

    # NVFP4 quantization
    recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])

    print("\nApplying NVFP4 quantization...")
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=calibration_ds,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.samples,
    )

    # Test generation
    if not args.skip_test:
        from llmcompressor.utils import dispatch_for_generation

        print("\n" + "=" * 50)
        print("SAMPLE GENERATION TEST")
        print("=" * 50)
        dispatch_for_generation(model)
        input_ids = tokenizer("Hello, my name is", return_tensors="pt").input_ids.to(model.device)
        output = model.generate(input_ids, max_new_tokens=50)
        print(tokenizer.decode(output[0]))
        print("=" * 50 + "\n")

    # Save
    print(f"Saving to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path), save_compressed=True)
    tokenizer.save_pretrained(str(output_path))

    # Fix config for vLLM compatibility (Gemma3 requires tie_word_embeddings=true)
    import json

    config_path = output_path / "config.json"
    config = json.loads(config_path.read_text())
    if config.get("tie_word_embeddings") is False:
        print("Fixing tie_word_embeddings=false -> true for vLLM compatibility")
        config["tie_word_embeddings"] = True
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    print(f"\nDone! Quantized model saved to: {output_path}")


def main():
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    output_path = resolve_output_path(model_path, args.output)

    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        return 1

    # Check if running inside Docker
    if os.environ.get("INSIDE_DOCKER"):
        run_quantization(args, model_path, output_path)
    else:
        run_in_docker(args, model_path, output_path)
        print("\nTo benchmark:")
        print(f"  uv run python scripts/run_bench.py --model {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
