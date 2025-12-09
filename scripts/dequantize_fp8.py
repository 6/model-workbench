#!/usr/bin/env python3
"""
Model Workbench - FP8 to BF16 Dequantization Script

Converts FP8-quantized HuggingFace models to standard BF16 format.
This is needed because llama.cpp cannot directly convert FP8 models to GGUF.

The script leverages transformers' auto-dequantization (when config has "dequantize": true)
to load the FP8 weights as BF16, then saves them as standard safetensors.

NOTE: New models often require nightly transformers/tokenizers. Run from nightly/ directory.

Examples:

  # Dequantize with GPU (run from nightly for new models)
  cd nightly
  uv run python ../scripts/dequantize_fp8.py \
    --model ~/models/mistralai/Devstral-Small-2-24B-Instruct-2512 \
    --output ~/models/mistralai/Devstral-Small-2-24B-bf16 \
    --device cuda:0

  # CPU-only (slower, uses ~100GB RAM for 24B model)
  cd nightly
  uv run python ../scripts/dequantize_fp8.py \
    --model ~/models/mistralai/Devstral-Small-2-24B-Instruct-2512 \
    --output ~/models/mistralai/Devstral-Small-2-24B-bf16

  # Then convert to GGUF (from project root)
  uv run python scripts/quantize_gguf.py \
    --model ~/models/mistralai/Devstral-Small-2-24B-bf16
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from bench_utils import (
    MODELS_ROOT,
    compact_path,
    log,
    resolve_model_path,
)


def get_model_size_gb(model_path: Path) -> float:
    """Estimate model size from safetensors files."""
    safetensors = list(model_path.glob("*.safetensors"))
    if not safetensors:
        return 0.0
    total_bytes = sum(f.stat().st_size for f in safetensors)
    return total_bytes / 1e9


def check_is_fp8_model(model_path: Path) -> bool:
    """Check if model is FP8 quantized."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        return False

    with open(config_path) as f:
        config = json.load(f)

    quant_config = config.get("quantization_config", {})
    return quant_config.get("quant_method") == "fp8"


def dequantize_model(
    model_path: Path,
    output_path: Path,
    device: str = "cpu",
    trust_remote_code: bool = True,
) -> None:
    """Load FP8 model (auto-dequantizes) and save as BF16."""

    log(f"Loading model from: {compact_path(str(model_path))}")
    log(f"Device: {device}")

    # Determine device_map
    if device.startswith("cuda"):
        device_map = device
        log(f"Using GPU: {device}")
    else:
        device_map = "cpu"
        log("Using CPU (slower but uses less VRAM)")

    # Load tokenizer first (lightweight)
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )

    # Load model - transformers will auto-dequantize FP8 → BF16
    log("Loading model (FP8 → BF16 dequantization happens here)...")
    log("This may take several minutes...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )

    log(f"Model loaded. dtype: {model.dtype}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model without quantization config
    log(f"Saving dequantized model to: {compact_path(str(output_path))}")
    log("This may take several minutes...")

    # Move to CPU before saving if on GPU (to avoid CUDA tensors in safetensors)
    if device_map != "cpu":
        log("Moving model to CPU for saving...")
        model = model.to("cpu")

    # Save the model - this saves as standard safetensors without FP8
    model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="5GB",
    )

    # Save tokenizer
    tokenizer.save_pretrained(output_path)

    # Fix the config to remove quantization_config
    config_path = output_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    if "quantization_config" in config:
        log("Removing quantization_config from saved config...")
        del config["quantization_config"]
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    # Copy any additional files that might be needed (chat templates, etc.)
    for extra_file in ["tokenizer_config.json", "special_tokens_map.json",
                       "chat_template.json", "generation_config.json"]:
        src = model_path / extra_file
        dst = output_path / extra_file
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    # Report final size
    output_size = get_model_size_gb(output_path)
    log(f"Done! Output size: {output_size:.2f} GB")
    log(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert FP8-quantized models to BF16 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to FP8-quantized HuggingFace model",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for dequantized BF16 model",
    )
    parser.add_argument(
        "--device", "-d",
        default="cpu",
        help="Device to use for loading (cpu, cuda:0, cuda:1). "
             "GPU is faster if you have enough VRAM (~50GB for 24B model)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code (default: True, needed for most models)",
    )

    args = parser.parse_args()

    # Resolve paths
    model_path = Path(resolve_model_path(args.model))
    output_path = Path(args.output).expanduser()

    # Validate input is FP8
    if not check_is_fp8_model(model_path):
        log("WARNING: Model does not appear to be FP8 quantized")
        log("This script is designed for FP8 → BF16 conversion")
        log("Proceeding anyway...")

    # Check if output already exists
    if output_path.exists() and any(output_path.glob("*.safetensors")):
        raise SystemExit(
            f"Output directory already contains safetensors: {output_path}\n"
            f"Delete it first if you want to re-convert."
        )

    # Estimate memory requirements
    input_size = get_model_size_gb(model_path)
    log(f"Input model size: {input_size:.2f} GB (FP8)")
    log(f"Expected output: ~{input_size:.2f} GB (BF16)")  # Similar size after dequant

    if args.device == "cpu":
        log(f"Estimated RAM needed: ~{input_size * 2:.0f} GB")
    else:
        log(f"Estimated VRAM needed: ~{input_size:.0f} GB")

    # Run dequantization
    dequantize_model(
        model_path=model_path,
        output_path=output_path,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )

    log("")
    log("Next step - convert to GGUF:")
    log(f"  uv run python scripts/quantize_gguf.py --model {output_path}")


if __name__ == "__main__":
    main()
