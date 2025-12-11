#!/usr/bin/env python3
"""
Standalone Server Runner - Start vLLM, llama.cpp, ik_llama.cpp, or TensorRT-LLM server.

Auto-detects backend from model format (GGUF -> llama, safetensors -> vLLM).
Use --backend to explicitly select a backend.
Server keeps running until Ctrl+C.

Examples:
  # Start vLLM server (safetensors model, auto-detected)
  uv run python scripts/run_server.py --model ~/models/zai-org/GLM-4.6V-FP8

  # Start vLLM with prebuilt official image
  uv run python scripts/run_server.py --model ~/models/zai-org/GLM-4.6V-FP8 --image-type prebuilt

  # Start vLLM with specific image (e.g., nightly)
  uv run python scripts/run_server.py --model ~/models/zai-org/GLM-4.6V-FP8 --image vllm/vllm-openai:nightly

  # Start TensorRT-LLM server
  uv run python scripts/run_server.py --model ~/models/zai-org/GLM-4.6V-FP8 --backend trtllm

  # Start llama.cpp server (GGUF model, auto-detected)
  uv run python scripts/run_server.py --model ~/models/unsloth/GLM-4.5-Air-GGUF/UD-Q4_K_XL

  # Explicitly use ik_llama.cpp for GGUF (ikawrakow's optimized fork)
  uv run python scripts/run_server.py --model ~/models/unsloth/GLM-GGUF/UD-Q4_K_XL --backend ik_llama

  # Start and test the endpoint
  uv run python scripts/run_server.py --model ~/models/org/model --test

  # Just test an already-running server
  uv run python scripts/run_server.py --model ~/models/org/model --test-only
"""

import argparse
import signal
import sys
import time
from pathlib import Path

from bench_utils import (
    BACKENDS,
    get_gpu_count,
    get_image_type,
    get_model_backend_config,
    get_model_backend_version,
    log,
    resolve_backend,
    resolve_model_path,
)
from server_manager import ServerManager


def test_chat_completion(host: str, port: int, model_name: str) -> bool:
    """Send a test chat completion request."""
    from openai import OpenAI

    log(f"Testing endpoint http://{host}:{port}/v1 ...")
    client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key="dummy")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=10,
        )
        text = response.choices[0].message.content or ""
        log(f"Response: {text.strip()}")
        return True
    except Exception as e:
        log(f"Test failed: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(
        description="Start a standalone vLLM, llama.cpp, ik_llama.cpp, or TensorRT-LLM server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    ap.add_argument("--model", required=True, help="Model path (auto-detects GGUF vs safetensors)")

    # Backend selection
    ap.add_argument("--backend", choices=list(BACKENDS.keys()), default=None,
                    help="Backend to use (default: auto-detect from model format)")

    # Server options
    ap.add_argument("--host", default="127.0.0.1", help="Server host")
    ap.add_argument("--port", type=int, default=None,
                    help="Server port (default: 8000 for vLLM/trtllm, 8080 for llama.cpp)")
    ap.add_argument("--server-timeout", type=int, default=360,
                    help="Timeout waiting for server to start (default: 360s)")

    # Test options
    ap.add_argument("--test", action="store_true",
                    help="Send a test chat completion after server starts")
    ap.add_argument("--test-only", action="store_true",
                    help="Only test existing server, don't start a new one")

    # Backend version and image options
    ap.add_argument("--backend-version", default=None,
                    help="Backend version (e.g., v0.8.0 for vLLM, b4521 for llama.cpp, 0.18.0 for trtllm)")
    ap.add_argument("--rebuild", action="store_true",
                    help="Force rebuild/repull Docker image even if cached")
    ap.add_argument("--image-type", choices=["prebuilt", "build"], default=None,
                    help="Image type: 'prebuilt' uses official images, 'build' compiles from source")
    ap.add_argument("--image", default=None,
                    help="Direct Docker image to use (e.g., vllm/vllm-openai:nightly)")

    # vLLM-specific options (defaults from config, CLI overrides)
    vllm_group = ap.add_argument_group("vLLM options (safetensors models)")
    vllm_group.add_argument("--tensor-parallel", type=int, default=None,
                            help="Tensor parallel size (default: auto-detect GPU count)")
    vllm_group.add_argument("--max-model-len", type=int, default=None,
                            help="Max context length (default: from config or 65536)")
    vllm_group.add_argument("--gpu-memory-utilization", type=float, default=None,
                            help="GPU memory fraction (default: from config or 0.95)")
    vllm_group.add_argument("--max-num-batched-tokens", type=int, default=None,
                            help="Max batched tokens")

    # llama.cpp-specific options (defaults from config, CLI overrides)
    llama_group = ap.add_argument_group("llama.cpp options (GGUF models)")
    llama_group.add_argument("--ctx", type=int, default=None, help="Context length (-c)")
    llama_group.add_argument("--n-gpu-layers", type=int, default=None,
                             help="GPU layers to offload (-ngl, default: from config or 999)")
    llama_group.add_argument("--parallel", type=int, default=1,
                             help="Parallel sequences (-np, default: 1)")
    llama_group.add_argument("--mmproj", default=None,
                             help="Multimodal projector path (auto-detected if not specified)")

    args = ap.parse_args()

    # Resolve backend (auto-detect or explicit)
    backend = resolve_backend(args.model, args.backend)
    backend_info = BACKENDS[backend]

    # Get merged config for this model + backend (defaults + model overrides)
    backend_cfg = get_model_backend_config(args.model, backend)
    backend_args = backend_cfg.get("args", {})

    # Set default port based on backend
    if args.port is None:
        args.port = backend_info["default_port"]

    # Auto-detect tensor parallel for vLLM and trtllm
    if backend in ("vllm", "trtllm") and args.tensor_parallel is None:
        args.tensor_parallel = get_gpu_count()

    # Resolve model path
    model_path = resolve_model_path(args.model) if backend in ("vllm", "trtllm") else args.model

    # Resolve backend version
    backend_version = args.backend_version or backend_cfg.get("version")
    if not backend_version and not args.test_only:
        raise SystemExit(
            f"No backend version specified and none found in config.\n"
            f"Either:\n"
            f"  1. Set defaults.backends.{backend}.version in config/models.yaml\n"
            f"  2. Pass --backend-version"
        )

    # Resolve image type (from CLI, model config, or backend defaults)
    image_type = args.image_type or backend_cfg.get("image_type", "build")

    # Resolve backend-specific args from config if not provided via CLI
    # vLLM args
    if args.max_model_len is None:
        args.max_model_len = backend_args.get("max_model_len", 65536)
    if args.gpu_memory_utilization is None:
        args.gpu_memory_utilization = backend_args.get("gpu_memory_utilization", 0.95)

    # llama.cpp / ik_llama args
    if args.n_gpu_layers is None:
        args.n_gpu_layers = backend_args.get("n_gpu_layers", 999)

    # Create server manager
    server = ServerManager(
        host=args.host,
        port=args.port,
        timeout=args.server_timeout,
    )

    # Model name for API (vLLM/trtllm use full path, llama.cpp uses gpt-3.5-turbo)
    api_model = model_path if backend in ("vllm", "trtllm") else "gpt-3.5-turbo"

    # Test-only mode
    if args.test_only:
        if not server.is_running():
            raise SystemExit(f"No server running on {args.host}:{args.port}")
        if test_chat_completion(args.host, args.port, api_model):
            log("Server is working!")
        else:
            raise SystemExit("Server test failed")
        return

    # Check if server already running
    if server.is_running():
        log(f"Server already running on {args.host}:{args.port}")
        if args.test:
            test_chat_completion(args.host, args.port, api_model)
        return

    # Start server
    backend_labels = {"vllm": "vLLM", "llama": "llama.cpp", "ik_llama": "ik_llama.cpp", "trtllm": "TensorRT-LLM"}
    backend_label = backend_labels.get(backend, backend)
    log(f"Starting {backend_label} server...")
    log(f"  Model: {model_path}")
    log(f"  Backend: {backend}")
    log(f"  Backend version: {backend_version}")
    log(f"  Image type: {image_type}")
    if args.image:
        log(f"  Image override: {args.image}")
    log(f"  Endpoint: http://{args.host}:{args.port}/v1")

    if backend == "vllm":
        server.start_vllm(
            model_path=model_path,
            tensor_parallel=args.tensor_parallel,
            version=backend_version,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_batched_tokens=args.max_num_batched_tokens,
            rebuild=args.rebuild,
            image_type=image_type,
            image_override=args.image,
        )
    elif backend == "trtllm":
        server.start_trtllm(
            model_path=model_path,
            tensor_parallel=args.tensor_parallel,
            version=backend_version,
            rebuild=args.rebuild,
        )
    elif backend in ("llama", "ik_llama"):
        mmproj_path = None
        if args.mmproj:
            mmproj_path = Path(args.mmproj).expanduser()

        server.start_gguf_backend(
            engine=backend,
            model_path=model_path,
            version=backend_version,
            n_gpu_layers=args.n_gpu_layers,
            ctx=args.ctx,
            parallel=args.parallel,
            mmproj_path=mmproj_path,
            rebuild=args.rebuild,
        )

    # Run test if requested
    if args.test:
        log("Running endpoint test...")
        test_chat_completion(args.host, args.port, api_model)

    # Keep server running until Ctrl+C
    log(f"\nServer running at http://{args.host}:{args.port}/v1")
    log("Press Ctrl+C to stop...")

    def signal_handler(sig, frame):
        log("\nShutting down...")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Wait forever (server runs in subprocess)
    while True:
        time.sleep(1)
        # Check if server process died
        if server.proc and server.proc.poll() is not None:
            log("Server process exited unexpectedly")
            sys.exit(1)


if __name__ == "__main__":
    main()
