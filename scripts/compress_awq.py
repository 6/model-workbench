#!/usr/bin/env python3
"""AWQ 4-bit compression script using llm-compressor (Docker-based).

Runs inside Docker using vLLM base image to ensure compressed-tensors version
alignment between quantizer and runtime.

AWQ (Activation-aware Weight Quantization) produces W4A16 quantized models
that run efficiently on vLLM with ~50% memory reduction.

MoE Quantization Notes (applies to all MoE models):
- W4A16 (symmetric) required: vLLM doesn't support asymmetric for MoE models
- MoE gates kept full precision: routing accuracy is critical for model quality
- group_size=32: smaller groups = better quality, larger output file

MiniMax-Specific Notes:
- Gate layer pattern: "block_sparse_moe.gate" (not "mlp.gate" like Qwen/Mixtral)
- Expert path: "block_sparse_moe.experts.*" (not "mlp.experts.*")
- Architecture: MiniMaxM2ForCausalLM (not in llm-compressor registry, needs explicit mappings)

Reference: cyankiwi/MiniMax-M2-AWQ-4bit used these settings successfully

Usage:
    uv run python scripts/compress_awq.py --model ~/models/MiniMaxAI/MiniMax-M2.1
    uv run python scripts/compress_awq.py --model ~/models/org/model --samples 512
    uv run python scripts/compress_awq.py --model ~/models/org/model --group-size 64
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Constants
DEFAULT_OUTPUT_BASE = Path.home() / "models-quantized"
DEFAULT_CALIBRATION_SAMPLES = 512  # Higher for better quality
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_GROUP_SIZE = 32  # Smaller = better quality (cyankiwi used 32)
DEFAULT_VLLM_VERSION = "v0.13.0"
CALIBRATION_DATASET = "nvidia/Llama-Nemotron-Post-Training-Dataset"
CALIBRATION_CONFIG = "SFT"
CALIBRATION_SPLIT = "code"
IMAGE_PREFIX = "model-bench-awq"
DOCKERFILE = Path(__file__).parent.parent / "docker" / "Dockerfile.awq"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compress model to AWQ 4-bit format using llm-compressor (Docker-based)"
    )
    parser.add_argument("--model", required=True, help="Path to source model")
    parser.add_argument(
        "--output", help="Output path (default: ~/models-quantized/<org>/<repo>-AWQ)"
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
        "--group-size",
        type=int,
        default=DEFAULT_GROUP_SIZE,
        help=f"Quantization group size - smaller=better quality, larger file (default: {DEFAULT_GROUP_SIZE})",
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

    return DEFAULT_OUTPUT_BASE / f"{org_repo}-AWQ"


def image_exists(image_name: str) -> bool:
    """Check if Docker image exists locally."""
    result = subprocess.run(
        ["docker", "images", "-q", image_name],
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def build_image(version: str, force: bool = False) -> str:
    """Build the AWQ compressor Docker image."""
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

    print("\nStarting AWQ 4-bit compression in Docker...")
    print(f"  Model:      {model_path}")
    print(f"  Output:     {output_path}")
    print(f"  Group size: {args.group_size}")

    # Ensure output parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--ipc",
        "host",
        "--ulimit",
        "memlock=-1",
        "--entrypoint",
        "python3",
        "-e",
        "INSIDE_DOCKER=1",
        "-v",
        f"{model_path}:{model_path}:ro",
        "-v",
        f"{output_path.parent}:{output_path.parent}:rw",
        "-v",
        f"{Path(__file__).resolve()}:/workspace/compress_awq.py:ro",
        "-v",
        f"{Path.home()}/.cache:/root/.cache:rw",
        image_name,
        "/workspace/compress_awq.py",
        "--model",
        str(model_path),
        "--output",
        str(output_path),
        "--samples",
        str(args.samples),
        "--max-seq-len",
        str(args.max_seq_len),
        "--group-size",
        str(args.group_size),
    ]
    if args.skip_test:
        cmd.append("--skip-test")

    subprocess.run(cmd, check=True)


# ============================================================================
# Docker-internal functions (only run inside container)
# ============================================================================


def get_moe_mappings():
    """Get MoE-aware AWQ mappings for MiniMax models.

    These mappings define which layers' activations are used to compute
    scaling factors for weight quantization. MoE models need special handling
    because their expert layers have a different structure than dense models.

    MiniMax-specific: Uses "block_sparse_moe.experts" path instead of the
    "mlp.experts" pattern used by Qwen/Mixtral MoE models.

    General MoE pattern: The mapping structure itself follows the standard
    _moe_default_mappings from llm-compressor, just with MiniMax layer names.
    """
    from llmcompressor.modifiers.awq import AWQMapping

    return [
        # Attention: input_layernorm -> q/k/v projections
        AWQMapping(
            "re:.*input_layernorm$",
            ["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
        ),
        # Attention: v_proj -> o_proj
        AWQMapping("re:.*v_proj$", ["re:.*o_proj$"]),
        # MLP/MoE: post_attention_layernorm -> expert gate/up projections
        # MiniMax-specific: uses "block_sparse_moe.experts" not "mlp.experts"
        AWQMapping(
            "re:.*post_attention_layernorm$",
            [
                "re:.*block_sparse_moe.experts.*.gate_proj$",
                "re:.*block_sparse_moe.experts.*.up_proj$",
            ],
        ),
        # MLP/MoE: up_proj -> down_proj
        AWQMapping(
            "re:.*up_proj$",
            ["re:.*down_proj$"],
        ),
    ]


def run_quantization(args, model_path: Path, output_path: Path):
    """Run the actual AWQ quantization (inside Docker)."""
    from datasets import load_dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier
    from llmcompressor.utils import dispatch_for_generation
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Source model: {model_path}")
    print(f"Output path:  {output_path}")
    print(f"Calibration:  {args.samples} samples, max_seq_len={args.max_seq_len}")
    print(f"Group size:   {args.group_size}")

    # Load model and tokenizer
    print("\nLoading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype="auto",
        trust_remote_code=True,
    )
    print("Model loaded successfully", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )
    print("Tokenizer loaded", flush=True)

    # Detect if model is MoE by checking for block_sparse_moe layers
    # (can be slow on large models - iterates all named_modules)
    print("Detecting model architecture...", flush=True)
    is_moe = any("block_sparse_moe" in name for name, _ in model.named_modules())
    print(f"Model type:   {'MoE (MiniMax)' if is_moe else 'Dense'}", flush=True)

    # Prepare calibration data
    print(f"\nLoading {args.samples} samples from {CALIBRATION_DATASET}...", flush=True)
    ds = load_dataset(
        CALIBRATION_DATASET,
        CALIBRATION_CONFIG,
        split=f"{CALIBRATION_SPLIT}[:{args.samples}]",
    )
    print(f"Dataset loaded: {len(ds)} samples", flush=True)
    ds = ds.shuffle(seed=42)

    def preprocess(example):
        # Build conversation from input messages + output
        messages = []
        for msg in example.get("input", []):
            messages.append({"role": msg["role"], "content": msg["content"]})
        if example.get("output"):
            messages.append({"role": "assistant", "content": example["output"]})
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    print("Preprocessing calibration data...", flush=True)
    ds = ds.map(preprocess)
    print("Preprocessing complete", flush=True)

    # Build AWQ recipe based on model type
    # MoE models require special handling:
    # - W4A16 (symmetric): vLLM doesn't support asymmetric for MoE models
    # - Ignore MoE gates: routing accuracy is critical for model quality
    # - Explicit MoE mappings: MiniMax isn't in llm-compressor's registry
    if is_moe:
        print("\nUsing MoE-optimized AWQ settings:")
        print("  - W4A16 symmetric (required for MoE in vLLM)")
        print("  - Preserving MoE gate layers in full precision")
        print("  - Using MiniMax MoE-aware layer mappings")

        recipe = [
            AWQModifier(
                # Ignore lm_head (standard) + MoE gates (MiniMax-specific pattern)
                # Keeping gates full precision preserves routing quality
                ignore=["lm_head", "re:.*block_sparse_moe.gate$"],
                # Explicit MoE mappings since MiniMax isn't in the registry
                mappings=get_moe_mappings(),
                # Config with symmetric=True (W4A16) - required for MoE in vLLM
                # Note: can't use both scheme= and config_groups=
                config_groups={
                    "group_0": {
                        "targets": ["Linear"],
                        "weights": {
                            "num_bits": 4,
                            "type": "int",
                            "symmetric": True,
                            "strategy": "group",
                            "group_size": args.group_size,
                        },
                    }
                },
            ),
        ]
    else:
        # Dense model - can use asymmetric quantization
        print("\nUsing standard dense model AWQ settings")
        recipe = [
            AWQModifier(
                ignore=["lm_head"],
                duo_scaling="both",
                # Config with symmetric=False (W4A16_ASYM)
                # Note: can't use both scheme= and config_groups=
                config_groups={
                    "group_0": {
                        "targets": ["Linear"],
                        "weights": {
                            "num_bits": 4,
                            "type": "int",
                            "symmetric": False,
                            "strategy": "group",
                            "group_size": args.group_size,
                        },
                    }
                },
            ),
        ]

    print("\nApplying AWQ 4-bit quantization...", flush=True)
    print("(This may take 30-60+ minutes for large models)", flush=True)
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.samples,
        trust_remote_code_model=True,  # Required for MiniMax custom code
    )

    print("Quantization complete!", flush=True)

    # Test generation
    if not args.skip_test:
        print("\n" + "=" * 50, flush=True)
        print("SAMPLE GENERATION TEST", flush=True)
        print("=" * 50, flush=True)
        dispatch_for_generation(model)
        input_ids = tokenizer("Hello, my name is", return_tensors="pt").input_ids.to(model.device)
        output = model.generate(input_ids, max_new_tokens=50)
        print(tokenizer.decode(output[0]))
        print("=" * 50 + "\n")

    # Save
    print(f"\nSaving to {output_path}...", flush=True)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path), save_compressed=True)
    print("Model saved", flush=True)
    tokenizer.save_pretrained(str(output_path))
    print("Tokenizer saved", flush=True)

    print(f"\nDone! AWQ quantized model saved to: {output_path}", flush=True)


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
