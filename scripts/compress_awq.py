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
DEFAULT_VLLM_VERSION = "nightly"
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
    ]
    if force:
        cmd.append("--no-cache")
    cmd.extend(
        [
            "--build-arg",
            f"VERSION={version}",
            "-t",
            image_name,
            str(DOCKERFILE.parent.parent),
        ]
    )
    subprocess.run(cmd, check=True)
    return image_name


def patch_minimax_model(model_path: Path) -> str | None:
    """Remove decorators that break llmcompressor AST tracing.

    MiniMax-M2.1's modeling code uses transformers decorators like
    @check_model_inputs, @auto_docstring, @can_return_tuple that wrap
    the forward() methods. llmcompressor's AST parser can't handle these
    and fails with KeyError: 'forward' in ast_helpers.autowrap_forward().

    This function patches the modeling file to remove these decorators,
    allowing llmcompressor to trace the model correctly.

    Returns the original content so it can be restored after quantization.
    """
    modeling_file = model_path / "modeling_minimax_m2.py"
    if not modeling_file.exists():
        return None

    original = modeling_file.read_text()
    content = original

    # These decorators wrap forward() and break AST parsing
    decorators = ["@check_model_inputs", "@auto_docstring", "@can_return_tuple"]
    for d in decorators:
        content = content.replace(f"    {d}\n", "")

    if content != original:
        modeling_file.write_text(content)
        print(f"Patched {modeling_file.name}: removed AST-breaking decorators")
        return original
    return None


def restore_minimax_model(model_path: Path, original_content: str):
    """Restore original modeling file after patching."""
    modeling_file = model_path / "modeling_minimax_m2.py"
    modeling_file.write_text(original_content)
    print(f"Restored {modeling_file.name}")


def run_in_docker(args, model_path: Path, output_path: Path):
    """Run compression inside Docker container."""
    image_name = build_image(args.vllm_version, force=args.rebuild)

    print("\nStarting AWQ 4-bit compression in Docker...")
    print(f"  Model:      {model_path}")
    print(f"  Output:     {output_path}")
    print(f"  Group size: {args.group_size}")

    # Ensure output parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Patch MiniMax model to remove AST-breaking decorators (host-side)
    # Model is mounted read-only in Docker, so patching must happen before
    original_content = patch_minimax_model(model_path)

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

    try:
        subprocess.run(cmd, check=True)
    finally:
        # Always restore the original file, even if Docker fails
        if original_content:
            restore_minimax_model(model_path, original_content)


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
        # MLP/MoE: post_attention_layernorm -> expert w1/w3 (gate/up equivalents)
        # MiniMax uses w1, w2, w3 instead of gate_proj, up_proj, down_proj
        AWQMapping(
            "re:.*post_attention_layernorm$",
            [
                "re:.*block_sparse_moe.experts.*.w1$",  # gate_proj equivalent
                "re:.*block_sparse_moe.experts.*.w3$",  # up_proj equivalent
            ],
        ),
        # MLP/MoE: w3 (up) -> w2 (down)
        AWQMapping(
            "re:.*w3$",
            ["re:.*w2$"],  # down_proj equivalent
        ),
    ]


def create_clean_model_symlink(model_path: Path) -> tuple[Path, Path | None]:
    """Create a temporary symlink with a clean name for torch.fx compatibility.

    When using trust_remote_code=True, transformers registers custom modules
    with names derived from the path. torch.fx then escapes special characters
    (- → _hyphen_, . → _dot_), which causes issues during graph tracing.

    This creates a temp symlink with a sanitized name so module registration
    uses clean identifiers from the start.

    Returns (load_path, cleanup_path) - cleanup_path is None if no symlink needed.
    """
    import re
    import tempfile

    model_name = model_path.name
    # Check if name has problematic characters for torch.fx
    if re.search(r"[-.]", model_name):
        clean_name = re.sub(r"[-.]", "_", model_name)
        temp_dir = Path(tempfile.mkdtemp(prefix="awq_"))
        symlink_path = temp_dir / clean_name
        symlink_path.symlink_to(model_path)
        print(f"Created temp symlink: {symlink_path} -> {model_path}")
        return symlink_path, temp_dir
    return model_path, None


def cleanup_model_symlink(temp_dir: Path | None):
    """Remove temporary symlink directory created for model loading."""
    if temp_dir and temp_dir.exists():
        import shutil

        shutil.rmtree(temp_dir)
        print(f"Cleaned up temp symlink: {temp_dir}")


def replace_fp8_with_bf16(model):
    """Replace FP8Linear modules with dequantized nn.Linear for AWQ compatibility.

    MiniMax-M2.1 uses fine-grained FP8 quantization with block-wise scale factors.
    AWQ requires FP16/BF16 weights (abs_cuda not implemented for FP8), so we must:
    1. Find all FP8Linear modules
    2. Dequantize weights using scale factors: dequantized = fp8_weight * scale_inv
    3. Replace with standard nn.Linear containing BF16 weights

    Memory optimization: Process modules one at a time and free memory after each
    to avoid OOM when converting large models (FP8→BF16 doubles memory).
    """
    import gc

    import torch
    import torch.nn as nn

    try:
        from transformers.integrations.finegrained_fp8 import FP8Linear
    except ImportError:
        print("FP8Linear not available, skipping FP8 conversion")
        return model, 0

    # First pass: collect all FP8Linear module paths
    fp8_modules = []
    for name, module in model.named_modules():
        if isinstance(module, FP8Linear):
            fp8_modules.append(name)

    if not fp8_modules:
        return model, 0

    print(f"Found {len(fp8_modules)} FP8Linear modules to convert", flush=True)
    replaced_count = 0

    # Second pass: replace one at a time to minimize peak memory
    for name in fp8_modules:
        # Navigate to parent module
        *parent_path, child_name = name.split(".")
        parent = model
        for p in parent_path:
            parent = getattr(parent, p)
        module = getattr(parent, child_name)

        # Dequantize weight using block-wise scale factors
        # weight: (out_features, in_features) in FP8
        # scale_inv: (num_out_blocks, num_in_blocks) in float32
        with torch.no_grad():
            weight = module.weight.data
            scale_inv = module.weight_scale_inv.data
            block_h, block_w = module.block_size
            out_features, in_features = weight.shape

            # Convert FP8 to float32 for math
            weight_float = weight.to(torch.float32)
            del weight  # Free FP8 weight immediately

            # Expand scale_inv to match weight shape
            scale_expanded = scale_inv.repeat_interleave(block_h, dim=0).repeat_interleave(
                block_w, dim=1
            )
            scale_expanded = scale_expanded[:out_features, :in_features]
            del scale_inv  # Free scale factors

            # Apply scale and convert to BF16
            dequantized = (weight_float * scale_expanded).to(torch.bfloat16)
            del weight_float, scale_expanded

        # Create replacement nn.Linear (don't allocate new weights, reuse dequantized)
        new_linear = nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=dequantized.device,
            dtype=torch.bfloat16,
        )
        # Directly assign dequantized tensor (avoid double allocation)
        new_linear.weight = nn.Parameter(dequantized, requires_grad=False)
        if module.bias is not None:
            new_linear.bias = nn.Parameter(module.bias.data.to(torch.bfloat16), requires_grad=False)

        # Replace module and free old one
        setattr(parent, child_name, new_linear)
        del module
        replaced_count += 1

        # Periodic memory cleanup
        if replaced_count % 100 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  Converted {replaced_count}/{len(fp8_modules)} modules", flush=True)

    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model, replaced_count


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

    # Create temp symlink if model path has special chars (torch.fx compatibility)
    load_path, temp_dir = create_clean_model_symlink(model_path)

    try:
        # Load model and tokenizer
        print("\nLoading model...", flush=True)
        import torch

        # Use BF16 instead of "auto" - MiniMax ships with FP8 weights, but AWQ
        # requires FP16/BF16 for smoothing (abs_cuda not implemented for FP8)
        model = AutoModelForCausalLM.from_pretrained(
            str(load_path),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        print("Model loaded successfully", flush=True)

        # Replace FP8Linear modules with dequantized nn.Linear for AWQ compatibility
        # MiniMax-M2.1 uses fine-grained FP8 with block-wise scale factors
        model, fp8_count = replace_fp8_with_bf16(model)
        if fp8_count > 0:
            print(f"Replaced {fp8_count} FP8Linear modules with BF16 nn.Linear", flush=True)

        # Sanitize config for torch.fx compatibility
        # The quantization_config may contain enum values (e.g., QuantizationMethod.FP8)
        # that serialize to invalid Python syntax like <QuantizationMethod.FP8: 'fp8'>
        # We remove quantization_config entirely since AWQ will re-quantize anyway
        if hasattr(model.config, "quantization_config"):
            # Delete the config entirely to prevent torch.fx serialization issues
            delattr(model.config, "quantization_config")
            print("Removed quantization_config from model config (will be replaced by AWQ)")

        tokenizer = AutoTokenizer.from_pretrained(
            str(load_path),
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
            tokenizer=tokenizer,  # Pass explicitly to avoid re-loading from sanitized path
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
            input_ids = tokenizer("Hello, my name is", return_tensors="pt").input_ids.to(
                model.device
            )
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

    finally:
        # Clean up temp symlink if created
        cleanup_model_symlink(temp_dir)


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
