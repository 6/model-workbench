#!/usr/bin/env python3
"""NVFP4 compression using NVIDIA Model Optimizer (Docker-based).

Produces checkpoints compatible with TensorRT-LLM, vLLM, and SGLang.

WARNING: NVFP4 quantization may be SLOWER than BF16 on Blackwell workstation GPUs
(RTX PRO 6000, RTX 5090 - SM120). As of Dec 2025, vLLM kernel optimization for
SM120 is incomplete. See: https://github.com/vllm-project/vllm/issues/31085
TensorRT-LLM may have better SM120 NVFP4 support.

Usage:
    uv run python scripts/compress_modelopt.py --model ~/models/google/gemma-3-1b-it
    uv run python scripts/compress_modelopt.py --model ~/models/google/gemma-3-1b-it --auto-quantize --effective-bits 4.5
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
DEFAULT_TRTLLM_VERSION = "1.2.0rc5"
CALIBRATION_DATASET = "cnn_dailymail"
CALIBRATION_SPLIT = "train"
IMAGE_PREFIX = "model-bench-modelopt"
DOCKERFILE = Path(__file__).parent.parent / "docker" / "Dockerfile.modelopt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compress model to NVFP4 using NVIDIA Model Optimizer (Docker-based)"
    )
    parser.add_argument("--model", required=True, help="Path to source model")
    parser.add_argument(
        "--output", help="Output path (default: ~/models-quantized/<org>/<repo>-NVFP4-modelopt)"
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
        "--auto-quantize",
        action="store_true",
        help="Use AutoQuantize for per-layer optimal mixed NVFP4/FP8 quantization",
    )
    parser.add_argument(
        "--effective-bits",
        type=float,
        default=4.5,
        help="Target effective bits for AutoQuantize (default: 4.5)",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip generation test after quantization",
    )
    parser.add_argument(
        "--trtllm-version",
        default=DEFAULT_TRTLLM_VERSION,
        help=f"TensorRT-LLM version for Docker base image (default: {DEFAULT_TRTLLM_VERSION})",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild Docker image",
    )
    parser.add_argument(
        "--offload-folder",
        default="/tmp/modelopt_offload",
        help="Folder for disk offload of model layers (default: /tmp/modelopt_offload)",
    )
    parser.add_argument(
        "--max-gpu-memory",
        help="Max GPU memory to use (e.g., '80GiB'). Leaves headroom for quantization.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code for custom models like MiniMax",
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

    return DEFAULT_OUTPUT_BASE / f"{org_repo}-NVFP4-modelopt"


def image_exists(image_name: str) -> bool:
    """Check if Docker image exists locally."""
    result = subprocess.run(
        ["docker", "images", "-q", image_name],
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def build_image(version: str, force: bool = False) -> str:
    """Build the Model Optimizer Docker image."""
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
    image_name = build_image(args.trtllm_version, force=args.rebuild)

    print("\nStarting NVFP4 compression with Model Optimizer in Docker...")
    print(f"  Model:  {model_path}")
    print(f"  Output: {output_path}")
    if args.auto_quantize:
        print(f"  Mode:   AutoQuantize (effective_bits={args.effective_bits})")
    else:
        print("  Mode:   Standard NVFP4")

    # Ensure output parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure offload folder exists
    offload_path = Path(args.offload_folder)
    offload_path.mkdir(parents=True, exist_ok=True)

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
        f"{Path(__file__).resolve()}:/workspace/compress_modelopt.py:ro",
        "-v",
        f"{Path.home()}/.cache:/root/.cache:rw",
        "-v",
        f"{offload_path}:{offload_path}:rw",  # Mount offload folder
        image_name,
        "/workspace/compress_modelopt.py",
        "--model",
        str(model_path),
        "--output",
        str(output_path),
        "--samples",
        str(args.samples),
        "--max-seq-len",
        str(args.max_seq_len),
        "--offload-folder",
        str(offload_path),
    ]
    if args.auto_quantize:
        cmd.extend(["--auto-quantize", "--effective-bits", str(args.effective_bits)])
    if args.skip_test:
        cmd.append("--skip-test")
    if args.max_gpu_memory:
        cmd.extend(["--max-gpu-memory", args.max_gpu_memory])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")

    subprocess.run(cmd, check=True)


# ============================================================================
# Docker-internal functions (only run inside container)
# ============================================================================


def run_quantization(args, model_path: Path, output_path: Path):
    """Run the actual quantization (inside Docker)."""
    import modelopt.torch.quantization as mtq
    import torch
    from datasets import load_dataset
    from modelopt.torch.export import export_hf_checkpoint
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Source model: {model_path}")
    print(f"Output path:  {output_path}")
    print(f"Calibration:  {args.samples} samples, max_seq_len={args.max_seq_len}")
    if args.max_gpu_memory:
        print(f"Max GPU mem:  {args.max_gpu_memory}")
    print(f"Offload dir:  {args.offload_folder}")

    # Ensure offload folder exists
    Path(args.offload_folder).mkdir(parents=True, exist_ok=True)

    # Load model with device_map="auto" for large model support
    # Note: FP8 models (like MiniMax-M2.1) must be loaded as BF16 because:
    # 1. Disk offload doesn't support FP8 (numpy can't handle Float8_e4m3fn)
    # 2. modelopt quantization works on FP16/BF16 weights
    print("\nLoading model with auto device mapping...")
    load_kwargs = {
        "torch_dtype": torch.bfloat16,  # Force BF16 (required for FP8 models)
        "device_map": "auto",
        # Note: Don't use offload_folder for FP8 models - numpy can't handle FP8
    }
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    if args.max_gpu_memory:
        # Detect all available GPUs and assign max_memory to each
        num_gpus = torch.cuda.device_count()
        max_memory = {i: args.max_gpu_memory for i in range(num_gpus)}
        max_memory["cpu"] = "500GiB"
        load_kwargs["max_memory"] = max_memory
        print(f"  Max memory: {num_gpus} GPUs @ {args.max_gpu_memory} each, CPU = 500GiB")

    model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
    print(
        f"Model loaded. Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}"
    )

    tokenizer_kwargs = {}
    if args.trust_remote_code:
        tokenizer_kwargs["trust_remote_code"] = True
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), **tokenizer_kwargs)

    # Set pad_token if not present (required for calibration padding)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    # Prepare calibration data
    print(f"\nLoading {args.samples} samples from {CALIBRATION_DATASET}...")
    ds = load_dataset(CALIBRATION_DATASET, "3.0.0", split=f"{CALIBRATION_SPLIT}[:{args.samples}]")
    ds = ds.shuffle(seed=42)

    # Tokenize calibration data
    def tokenize(sample):
        text = sample["article"][:2000]  # Truncate long articles
        return tokenizer(
            text,
            padding="max_length",
            max_length=args.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

    calib_data = [tokenize(sample) for sample in ds]

    # Get input device for device_map="auto" models (use embedding layer's device)
    def get_input_device(m):
        """Get the device for model inputs (first layer's device for device_map models)."""
        if hasattr(m, "device"):
            return m.device
        # For device_map models, get the embedding layer's device
        embed = m.get_input_embeddings()
        if embed is not None:
            return next(embed.parameters()).device
        # Fallback to first parameter's device
        return next(m.parameters()).device

    input_device = get_input_device(model)
    print(f"Input device: {input_device}")

    # Create forward loop for calibration
    def forward_loop(model):
        print("Running calibration...")
        device = get_input_device(model)
        for i, batch in enumerate(calib_data):
            if i % 100 == 0:
                print(f"  Calibration sample {i}/{len(calib_data)}")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)

    # Quantize
    if args.auto_quantize:
        print(f"\nApplying AutoQuantize (effective_bits={args.effective_bits})...")

        def forward_step(model, data):
            device = get_input_device(model)
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            return model(input_ids=input_ids, attention_mask=attention_mask)

        def loss_func(output, data):
            # Simple cross-entropy proxy
            logits = output.logits
            labels = data["input_ids"].to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
            )
            return loss

        model, search_state = mtq.auto_quantize(
            model,
            constraints={"effective_bits": args.effective_bits},
            quantization_formats=[mtq.NVFP4_DEFAULT_CFG, mtq.FP8_DEFAULT_CFG],
            data_loader=calib_data,
            forward_step=forward_step,
            loss_func=loss_func,
        )
        print(f"AutoQuantize complete. Search state: {search_state}")
    else:
        print("\nApplying NVFP4 quantization...")
        model = mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop)

    # Test generation
    if not args.skip_test:
        print("\n" + "=" * 50)
        print("SAMPLE GENERATION TEST")
        print("=" * 50)
        device = get_input_device(model)
        input_ids = tokenizer("Hello, my name is", return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=50)
        print(tokenizer.decode(output[0]))
        print("=" * 50 + "\n")

    # Export - try HF format first, fall back to TensorRT-LLM format
    print(f"Exporting to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if model has offloaded layers (meta device)
    has_offloaded = any("cpu" in str(v) or v == "disk" for v in model.hf_device_map.values()) if hasattr(model, "hf_device_map") else False

    if has_offloaded:
        print("  Warning: Model has offloaded layers. Trying TensorRT-LLM export format...")
        try:
            from modelopt.torch.export import export_tensorrt_llm_checkpoint
            with torch.inference_mode():
                export_tensorrt_llm_checkpoint(
                    model=model,
                    decoder_type="llama",  # MiniMax uses similar architecture
                    export_dir=str(output_path),
                    dtype=torch.bfloat16,
                )
            tokenizer.save_pretrained(str(output_path))
            print("  Exported using TensorRT-LLM format")
        except Exception as e:
            print(f"  TensorRT-LLM export failed: {e}")
            print("  Falling back to HF export (may not preserve quantization)...")
            with torch.inference_mode():
                export_hf_checkpoint(model=model, export_dir=str(output_path))
            tokenizer.save_pretrained(str(output_path))
    else:
        with torch.inference_mode():
            export_hf_checkpoint(model=model, export_dir=str(output_path))
        tokenizer.save_pretrained(str(output_path))

    print(f"\nDone! Quantized model saved to: {output_path}")
    print("\nTo benchmark with TensorRT-LLM:")
    print(f"  uv run python scripts/run_bench.py --model {output_path} --backend trtllm")
    print("\nTo benchmark with vLLM:")
    print(f"  uv run python scripts/run_bench.py --model {output_path} --backend vllm")


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

    return 0


if __name__ == "__main__":
    sys.exit(main())
