#!/usr/bin/env python3
"""NVFP4 compression script using llm-compressor.

Compresses BF16/FP16 models to NVFP4 format (4-bit weights, FP4 activations)
for efficient vLLM inference.

Usage:
    uv run python scripts/compress_nvfp4.py --model ~/models/google/gemma-3-1b-it
    uv run python scripts/compress_nvfp4.py --model ~/models/google/gemma-3-1b-it --samples 50
    uv run python scripts/compress_nvfp4.py --model ~/models/google/gemma-3-1b-it --output ~/custom/path
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation
from transformers import AutoModelForCausalLM, AutoTokenizer

# Constants
DEFAULT_OUTPUT_BASE = Path.home() / "models-quantized"
DEFAULT_CALIBRATION_SAMPLES = 512
DEFAULT_MAX_SEQ_LENGTH = 2048
CALIBRATION_DATASET = "HuggingFaceH4/ultrachat_200k"
CALIBRATION_SPLIT = "train_sft"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compress model to NVFP4 format using llm-compressor"
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
    return parser.parse_args()


def resolve_output_path(model_path: Path, output_arg: str | None) -> Path:
    """Derive output path from model path, preserving org/repo structure."""
    if output_arg:
        return Path(output_arg).expanduser().resolve()

    # Extract org/repo from path like ~/models/google/gemma-3-1b-it
    parts = model_path.parts
    if "models" in parts:
        idx = parts.index("models")
        org_repo = "/".join(parts[idx + 1 :])
    else:
        org_repo = model_path.name

    return DEFAULT_OUTPUT_BASE / f"{org_repo}-NVFP4"


def prepare_calibration_data(tokenizer, num_samples: int, max_seq_len: int):
    """Load and tokenize calibration dataset."""
    print(f"Loading {num_samples} samples from {CALIBRATION_DATASET}...")
    ds = load_dataset(CALIBRATION_DATASET, split=f"{CALIBRATION_SPLIT}[:{num_samples}]")
    ds = ds.shuffle(seed=42)

    def preprocess(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_seq_len,
            truncation=True,
            add_special_tokens=False,
        )

    return ds.map(tokenize, remove_columns=ds.column_names)


def test_generation(model, tokenizer):
    """Run a simple generation test to verify the quantized model works."""
    print("\n" + "=" * 50)
    print("SAMPLE GENERATION TEST")
    print("=" * 50)

    dispatch_for_generation(model)
    input_ids = tokenizer("Hello, my name is", return_tensors="pt").input_ids.to(model.device)
    output = model.generate(input_ids, max_new_tokens=50)
    print(tokenizer.decode(output[0]))
    print("=" * 50 + "\n")


def main():
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    output_path = resolve_output_path(model_path, args.output)

    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        return 1

    print(f"Source model: {model_path}")
    print(f"Output path:  {output_path}")
    print(f"Calibration:  {args.samples} samples, max_seq_len={args.max_seq_len}")

    # Load model and tokenizer
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    # Prepare calibration data
    print("\nPreparing calibration data...")
    calibration_ds = prepare_calibration_data(tokenizer, args.samples, args.max_seq_len)

    # NVFP4 quantization recipe
    recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])

    # Apply quantization
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
        print("\nTesting quantized model...")
        test_generation(model, tokenizer)

    # Save
    print(f"Saving to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path), save_compressed=True)
    tokenizer.save_pretrained(str(output_path))

    print(f"\nDone! Quantized model saved to: {output_path}")
    print("\nTo benchmark:")
    print(f"  uv run python scripts/run_bench.py --model {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
