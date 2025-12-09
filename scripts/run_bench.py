#!/usr/bin/env python3
"""
Unified benchmark runner - auto-detects model format.

Auto-detects GGUF (llama.cpp) vs safetensors (vLLM) based on file extension.

Examples:

  # Safetensors model -> vLLM
  uv run python scripts/run_bench.py --model ~/models/zai-org/GLM-4.6V-FP8

  # GGUF model -> llama.cpp
  uv run python scripts/run_bench.py --model ~/models/unsloth/GLM-4.5-Air-GGUF/UD-Q4_K_XL

  # All other arguments are passed through to the backend
  uv run python scripts/run_bench.py --model ~/models/org/model --iterations 5
"""

import sys

from bench_utils import detect_model_format, log


def find_model_arg() -> str | None:
    """Extract --model argument from sys.argv."""
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--model" and i < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--model="):
            return arg.split("=", 1)[1]
    return None


def main():
    model_arg = find_model_arg()

    if not model_arg:
        print("Usage: run_bench.py --model <path> [options]")
        print("Auto-detects GGUF (llama.cpp) vs safetensors (vLLM)")
        sys.exit(1)

    fmt = detect_model_format(model_arg)

    if fmt == "gguf":
        log("Auto-detected GGUF model -> llama.cpp backend")
        from run_bench_llama_server import main as backend_main
    else:
        log("Auto-detected safetensors model -> vLLM backend")
        from run_bench_vllm_server import main as backend_main

    backend_main()


if __name__ == "__main__":
    main()
