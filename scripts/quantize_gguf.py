#!/usr/bin/env python3
"""
Model Workbench - GGUF Quantization Script

Converts HuggingFace safetensors models to quantized GGUF format using llama.cpp.

Requires:
  - llama.cpp built with llama-quantize (default: ~/llama.cpp)
  - Sufficient RAM (~1.5-2x model size for conversion)

Examples:

  # Basic Q4_K_M quantization (default)
  uv run python scripts/quantize_gguf.py \\
    --model ~/models/mistralai/Devstral-Small-2-24B-Instruct-2512

  # Different quant type
  uv run python scripts/quantize_gguf.py \\
    --model ~/models/org/repo --quant Q5_K_M

  # Custom llama.cpp location
  uv run python scripts/quantize_gguf.py \\
    --model ~/models/org/repo --llama-cpp /path/to/llama.cpp

  # Keep intermediate F16 GGUF
  uv run python scripts/quantize_gguf.py \\
    --model ~/models/org/repo --keep-f16

Available quant types:
  Q4_K_M (default), Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q8_0,
  IQ4_XS, IQ4_NL, Q3_K_M, Q3_K_S, Q2_K, F16, F32
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from bench_utils import (
    MODELS_ROOT,
    compact_path,
    log,
    resolve_model_path,
)

# Default llama.cpp location
DEFAULT_LLAMA_CPP = Path.home() / "llama.cpp"

# Common quantization types
QUANT_TYPES = [
    "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0",
    "IQ4_XS", "IQ4_NL", "Q3_K_M", "Q3_K_S", "Q2_K",
    "F16", "F32",
]


def find_llama_cpp(llama_cpp_arg: str | None) -> Path:
    """Find and validate llama.cpp installation."""
    if llama_cpp_arg:
        llama_cpp = Path(llama_cpp_arg).expanduser()
    else:
        llama_cpp = DEFAULT_LLAMA_CPP

    if not llama_cpp.exists():
        raise SystemExit(
            f"llama.cpp not found at: {llama_cpp}\n"
            f"Clone it with: git clone https://github.com/ggerganov/llama.cpp {llama_cpp}"
        )

    return llama_cpp


def find_convert_script(llama_cpp: Path) -> Path:
    """Find the convert_hf_to_gguf.py script."""
    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise SystemExit(
            f"Convert script not found: {convert_script}\n"
            f"Make sure llama.cpp is up to date: cd {llama_cpp} && git pull"
        )
    return convert_script


def find_quantize_binary(llama_cpp: Path) -> Path:
    """Find the llama-quantize binary."""
    # Check common build locations
    candidates = [
        llama_cpp / "build" / "bin" / "llama-quantize",
        llama_cpp / "llama-quantize",
        llama_cpp / "build" / "llama-quantize",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    raise SystemExit(
        f"llama-quantize not found in {llama_cpp}\n"
        f"Build it with:\n"
        f"  cd {llama_cpp}\n"
        f"  cmake -B build -DGGML_CUDA=ON\n"
        f"  cmake --build build --config Release -j $(nproc)"
    )


def get_model_name(model_path: Path) -> str:
    """Extract a clean model name for output files."""
    # Try to get org/repo from path
    try:
        rel = model_path.relative_to(MODELS_ROOT)
        parts = rel.parts
        if len(parts) >= 2:
            # e.g., mistralai/Devstral-Small-2-24B -> devstral-small-24b
            repo_name = parts[1].lower()
            # Simplify common patterns
            repo_name = repo_name.replace("-instruct", "").replace("-2512", "")
            return repo_name
    except ValueError:
        pass
    return model_path.name.lower()


def check_disk_space(model_path: Path, output_dir: Path) -> None:
    """Warn if disk space might be insufficient."""
    try:
        # Estimate model size from safetensors
        safetensors = list(model_path.glob("*.safetensors"))
        model_size = sum(f.stat().st_size for f in safetensors)

        # Check available space (need ~2x for F16 intermediate + final)
        stat = shutil.disk_usage(output_dir)
        available = stat.free

        # F16 is roughly same size as original, quantized is ~25-30% of F16
        needed = model_size * 1.5  # Conservative estimate

        if available < needed:
            log(f"WARNING: Low disk space!")
            log(f"  Model size: {model_size / 1e9:.1f} GB")
            log(f"  Available:  {available / 1e9:.1f} GB")
            log(f"  Estimated need: {needed / 1e9:.1f} GB")
    except Exception:
        pass  # Don't fail on space check errors


def convert_to_f16_gguf(
    model_path: Path,
    output_path: Path,
    convert_script: Path,
) -> None:
    """Convert HuggingFace model to F16 GGUF."""
    log(f"Converting to F16 GGUF: {compact_path(str(output_path))}")

    cmd = [
        sys.executable,
        str(convert_script),
        str(model_path),
        "--outfile", str(output_path),
        "--outtype", "f16",
    ]

    log(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise SystemExit(f"Conversion failed with exit code {result.returncode}")

    if not output_path.exists():
        raise SystemExit(f"Conversion completed but output not found: {output_path}")

    size_gb = output_path.stat().st_size / 1e9
    log(f"F16 GGUF created: {size_gb:.2f} GB")


def quantize_gguf(
    input_path: Path,
    output_path: Path,
    quantize_bin: Path,
    quant_type: str,
) -> None:
    """Quantize GGUF to target precision."""
    log(f"Quantizing to {quant_type}: {compact_path(str(output_path))}")

    cmd = [
        str(quantize_bin),
        str(input_path),
        str(output_path),
        quant_type,
    ]

    log(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise SystemExit(f"Quantization failed with exit code {result.returncode}")

    if not output_path.exists():
        raise SystemExit(f"Quantization completed but output not found: {output_path}")

    size_gb = output_path.stat().st_size / 1e9
    log(f"Quantized GGUF created: {size_gb:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace models to quantized GGUF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to HuggingFace model directory (with safetensors)",
    )
    parser.add_argument(
        "--quant", "-q",
        default="Q4_K_M",
        choices=QUANT_TYPES,
        help="Quantization type (default: Q4_K_M)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory for GGUF files (default: same as model)",
    )
    parser.add_argument(
        "--llama-cpp",
        help=f"Path to llama.cpp directory (default: {DEFAULT_LLAMA_CPP})",
    )
    parser.add_argument(
        "--keep-f16",
        action="store_true",
        help="Keep intermediate F16 GGUF file",
    )
    parser.add_argument(
        "--name",
        help="Custom name for output files (default: derived from model path)",
    )

    args = parser.parse_args()

    # Resolve paths
    model_path = Path(resolve_model_path(args.model))
    output_dir = Path(args.output).expanduser() if args.output else model_path

    # Validate model has safetensors
    safetensors = list(model_path.glob("*.safetensors"))
    if not safetensors:
        raise SystemExit(
            f"No safetensors files found in: {model_path}\n"
            f"This script converts HuggingFace models, not existing GGUFs."
        )

    log(f"Model: {compact_path(str(model_path))}")
    log(f"Found {len(safetensors)} safetensors files")

    # Find llama.cpp tools
    llama_cpp = find_llama_cpp(args.llama_cpp)
    convert_script = find_convert_script(llama_cpp)
    quantize_bin = find_quantize_binary(llama_cpp)

    log(f"llama.cpp: {compact_path(str(llama_cpp))}")
    log(f"Quantize binary: {compact_path(str(quantize_bin))}")

    # Determine output filenames
    model_name = args.name or get_model_name(model_path)
    f16_path = output_dir / f"{model_name}-f16.gguf"
    quant_path = output_dir / f"{model_name}-{args.quant.lower()}.gguf"

    log(f"Output: {compact_path(str(quant_path))}")

    # Check disk space
    check_disk_space(model_path, output_dir)

    # Step 1: Convert to F16 GGUF
    if f16_path.exists():
        log(f"F16 GGUF already exists, skipping conversion")
    else:
        convert_to_f16_gguf(model_path, f16_path, convert_script)

    # Step 2: Quantize
    if quant_path.exists():
        log(f"Quantized GGUF already exists: {quant_path}")
        log(f"Delete it first if you want to re-quantize")
    else:
        quantize_gguf(f16_path, quant_path, quantize_bin, args.quant)

    # Step 3: Cleanup F16 if requested
    if not args.keep_f16 and f16_path.exists() and quant_path.exists():
        log(f"Removing intermediate F16 file...")
        f16_path.unlink()
        log(f"Removed: {compact_path(str(f16_path))}")

    # Done
    log(f"Done! Quantized model at:")
    log(f"  {quant_path}")
    log(f"")
    log(f"Test with:")
    log(f"  ~/llama.cpp/build/bin/llama-server -m {quant_path} -ngl 999 --port 8080")


if __name__ == "__main__":
    main()
