#!/usr/bin/env python3
"""
Standalone Server Runner - Start vLLM, llama.cpp, TensorRT-LLM, SGLang, or ExLlamaV3 server.

Auto-detects backend from model format (GGUF -> llama, safetensors -> vLLM).
Use --backend to explicitly select a backend.
Server keeps running until Ctrl+C.

Examples:
  # Start vLLM server (safetensors model, auto-detected)
  uv run python scripts/run_server.py --model ~/models/zai-org/GLM-4.6V-FP8

  # Start vLLM with prebuilt official image
  uv run python scripts/run_server.py --model ~/models/zai-org/GLM-4.6V-FP8 --image-type prebuilt

  # Start vLLM with specific image (e.g., nightly)
  uv run python scripts/run_server.py --model ~/models/zai-org/GLM-4.6V-FP8 --docker-image vllm/vllm-openai:nightly

  # Start TensorRT-LLM server
  uv run python scripts/run_server.py --model ~/models/zai-org/GLM-4.6V-FP8 --backend trtllm

  # Start SGLang server (e.g., for MiMo models)
  uv run python scripts/run_server.py --model ~/models/cyankiwi/MiMo-V2-Flash-AWQ-4bit --backend sglang

  # Start llama.cpp server (GGUF model, auto-detected)
  uv run python scripts/run_server.py --model ~/models/unsloth/GLM-4.5-Air-GGUF/UD-Q4_K_XL

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

from bench_utils import resolve_run_config, warmup_model
from common import BACKEND_REGISTRY, log
from server_manager import ServerManager, start_open_webui, stop_container


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
        description="Start a standalone vLLM, llama.cpp, TensorRT-LLM, SGLang, or ExLlamaV3 server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    ap.add_argument("--model", required=True, help="Model path (auto-detects GGUF vs safetensors)")

    # Backend selection
    ap.add_argument(
        "--backend",
        choices=list(BACKEND_REGISTRY.keys()),
        default=None,
        help="Backend to use (default: auto-detect from model format)",
    )

    # Server options
    ap.add_argument("--host", default="127.0.0.1", help="Server host")
    ap.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (default: 8000 for vLLM/trtllm, 8080 for llama.cpp)",
    )
    ap.add_argument(
        "--server-timeout",
        type=int,
        default=360,
        help="Timeout waiting for server to start (default: 360s)",
    )

    # Test options
    ap.add_argument(
        "--test", action="store_true", help="Send a test chat completion after server starts"
    )
    ap.add_argument(
        "--test-only", action="store_true", help="Only test existing server, don't start a new one"
    )

    # Backend version and image options
    ap.add_argument(
        "--backend-version",
        default=None,
        help="Backend version (e.g., v0.8.0 for vLLM, b4521 for llama.cpp, 0.18.0 for trtllm)",
    )
    ap.add_argument(
        "--rebuild", action="store_true", help="Force rebuild/repull Docker image even if cached"
    )
    ap.add_argument(
        "--image-type",
        choices=["prebuilt", "build"],
        default=None,
        help="Image type: 'prebuilt' uses official images, 'build' compiles from source",
    )
    ap.add_argument(
        "--docker-image",
        default=None,
        dest="docker_image",
        help="Direct Docker image to use (e.g., vllm/vllm-openai:nightly)",
    )

    # vLLM-specific options (defaults from config, CLI overrides)
    vllm_group = ap.add_argument_group("vLLM options (safetensors models)")
    vllm_group.add_argument(
        "--tensor-parallel",
        type=int,
        default=None,
        help="Tensor parallel size (default: auto-detect GPU count)",
    )
    vllm_group.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Max context length (default: from config or 65536)",
    )
    vllm_group.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="GPU memory fraction (default: from config or 0.95)",
    )
    vllm_group.add_argument(
        "--max-num-batched-tokens", type=int, default=None, help="Max batched tokens"
    )
    vllm_group.add_argument(
        "--cpu-offload-gb",
        type=float,
        default=None,
        help="CPU offload in GB per GPU (default: none)",
    )
    vllm_group.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Max concurrent sequences (default: from config or vLLM default)",
    )

    # llama.cpp-specific options (defaults from config, CLI overrides)
    llama_group = ap.add_argument_group("llama.cpp options (GGUF models)")
    llama_group.add_argument("--ctx", type=int, default=None, help="Context length (-c)")
    llama_group.add_argument(
        "--n-gpu-layers",
        type=int,
        default=None,
        help="GPU layers to offload (-ngl, default: from config or 999)",
    )
    llama_group.add_argument(
        "--parallel", type=int, default=1, help="Parallel sequences (-np, default: 1)"
    )
    llama_group.add_argument(
        "--mmproj", default=None, help="Multimodal projector path (auto-detected if not specified)"
    )
    # CPU offloading options
    llama_group.add_argument(
        "--jinja",
        action="store_true",
        default=None,
        help="Enable Jinja template engine (default: on)",
    )
    llama_group.add_argument(
        "--no-jinja", action="store_true", dest="no_jinja", help="Disable Jinja template engine"
    )
    llama_group.add_argument(
        "--flash-attn",
        choices=["on", "off"],
        default=None,
        help="Flash attention mode (default: on)",
    )
    llama_group.add_argument(
        "--cache-type-k",
        default=None,
        help="KV cache K quantization (f16, q8_0, q4_0, q4_1, etc.)",
    )
    llama_group.add_argument(
        "--cache-type-v",
        default=None,
        help="KV cache V quantization (requires flash-attn)",
    )
    llama_group.add_argument(
        "--tensor-offload",
        "-ot",
        action="append",
        default=None,
        dest="tensor_offload",
        help="Tensor offload pattern (e.g., '.ffn_.*_exps.=CPU'). Can be repeated.",
    )
    llama_group.add_argument(
        "--fit",
        action="store_true",
        default=None,
        help="Enable auto-fit mode for GPU/CPU balancing (--fit on)",
    )

    # Warmup options
    ap.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip model preloading (warmup request). By default, models are preloaded into GPU memory on startup.",
    )

    # Open WebUI options (enabled by default for OpenAI-compatible backends)
    webui_group = ap.add_argument_group("Open WebUI options")
    webui_group.add_argument(
        "--no-webui",
        action="store_true",
        help="Don't launch Open WebUI (default: WebUI is launched for OpenAI-compatible backends)",
    )
    webui_group.add_argument(
        "--webui-port",
        type=int,
        default=8080,
        help="Port for Open WebUI (default: 8080)",
    )
    webui_group.add_argument(
        "--webui-image",
        default="ghcr.io/open-webui/open-webui:main",
        help="Open WebUI Docker image (default: ghcr.io/open-webui/open-webui:main)",
    )

    args = ap.parse_args()

    # Resolve --jinja / --no-jinja into single value (CLI overrides config)
    if getattr(args, "no_jinja", False):
        args.jinja = False
    elif getattr(args, "jinja", None) is True:
        args.jinja = True
    else:
        args.jinja = None  # Will be resolved from config

    # Resolve backend config and apply defaults
    backend, model_path, backend_cfg = resolve_run_config(args)

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

    # Resolve docker_image (CLI override takes precedence over config)
    docker_image = args.docker_image or backend_cfg.get("docker_image")

    # Create server manager
    server = ServerManager(
        host=args.host,
        port=args.port,
        timeout=args.server_timeout,
    )

    # Model name for API (llama.cpp uses gpt-3.5-turbo, exl uses model dir name, others use full path)
    if backend == "llama":
        api_model = "gpt-3.5-turbo"
    elif backend == "exl":
        api_model = Path(model_path).name  # TabbyAPI uses model_name (directory name)
    else:
        api_model = model_path

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
    backend_label = BACKEND_REGISTRY[backend]["display_name"]
    log(f"Starting {backend_label} server...")
    log(f"  Model: {model_path}")
    log(f"  Backend: {backend}")
    log(f"  Backend version: {backend_version}")
    log(f"  Image type: {image_type}")
    if docker_image:
        log(f"  Image override: {docker_image}")
    log(f"  Endpoint: http://{args.host}:{args.port}/v1")

    if backend == "vllm":
        server.start_vllm(
            model_path=model_path,
            tensor_parallel=args.tensor_parallel,
            version=backend_version,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_batched_tokens=args.max_num_batched_tokens,
            cpu_offload_gb=args.cpu_offload_gb,
            max_num_seqs=args.max_num_seqs,
            env_vars=args.env_vars,
            extra_vllm_args=args.extra_vllm_args,
            rebuild=args.rebuild,
            image_type=image_type,
            image_override=docker_image,
        )
    elif backend == "trtllm":
        server.start_trtllm(
            model_path=model_path,
            tensor_parallel=args.tensor_parallel,
            version=backend_version,
            rebuild=args.rebuild,
        )
    elif backend == "llama":
        mmproj_path = None
        if args.mmproj:
            mmproj_path = Path(args.mmproj).expanduser()

        # Get llama-specific args from config
        backend_args = backend_cfg.get("args", {})
        n_gpu_layers = args.n_gpu_layers or backend_args.get("n_gpu_layers")
        repeat_penalty = backend_args.get("repeat_penalty")
        repeat_last_n = backend_args.get("repeat_last_n")
        ctx = args.ctx or backend_args.get("ctx")

        server.start_gguf_backend(
            engine=backend,
            model_path=model_path,
            version=backend_version,
            n_gpu_layers=n_gpu_layers,
            ctx=ctx,
            parallel=args.parallel,
            mmproj_path=mmproj_path,
            repeat_penalty=repeat_penalty,
            repeat_last_n=repeat_last_n,
            jinja=args.jinja,
            flash_attn=args.flash_attn,
            cache_type_k=args.cache_type_k,
            cache_type_v=args.cache_type_v,
            tensor_offload=args.tensor_offload,
            fit=args.fit,
            rebuild=args.rebuild,
        )
    elif backend == "sglang":
        # Get SGLang-specific args from config
        mem_fraction = backend_cfg.get("args", {}).get("mem_fraction_static")
        max_model_len = args.max_model_len or backend_cfg.get("args", {}).get("max_model_len")

        server.start_sglang(
            model_path=model_path,
            tensor_parallel=args.tensor_parallel,
            version=backend_version,
            mem_fraction_static=mem_fraction,
            max_model_len=max_model_len,
            rebuild=args.rebuild,
            image_override=docker_image,
        )
    elif backend == "exl":
        # Get ExLlamaV3-specific args from config
        backend_args = backend_cfg.get("args", {})
        cache_size = backend_args.get("cache_size")
        max_seq_len = args.max_model_len or backend_args.get("max_seq_len")
        gpu_split_auto = backend_args.get("gpu_split_auto", True)
        gpu_split = backend_args.get("gpu_split")  # e.g., [24, 24] for explicit split

        server.start_exl(
            model_path=model_path,
            version=backend_version,
            cache_size=cache_size,
            max_seq_len=max_seq_len,
            gpu_split_auto=gpu_split_auto,
            gpu_split=gpu_split,
            rebuild=args.rebuild,
        )
    elif backend == "ktransformers":
        # Get KTransformers-specific args from config
        backend_args = backend_cfg.get("args", {})
        cpu_threads = backend_args.get("cpu_threads")
        numa_nodes = backend_args.get("numa_nodes")
        kt_method = backend_args.get("kt_method")
        cache_lens = backend_args.get("cache_lens")

        server.start_ktransformers(
            model_path=model_path,
            tensor_parallel=args.tensor_parallel,
            version=backend_version,
            cpu_threads=cpu_threads,
            numa_nodes=numa_nodes,
            kt_method=kt_method,
            cache_lens=cache_lens,
            rebuild=args.rebuild,
        )

    # Warmup model (preload into GPU memory by default)
    # Note: In the future, --model may be optional for dynamic loading
    should_warmup = args.model and not args.no_warmup

    if should_warmup:
        # Extra verification: ensure server is truly ready
        if not server.is_running():
            log("ERROR: Server not running, skipping warmup")
        else:
            log("Preloading model into GPU memory...")
            success = warmup_model(
                backend=backend,
                host=args.host,
                port=args.port,
                api_model=api_model,
            )
            if success:
                log("Model preloaded successfully")
            else:
                log("WARNING: Model preload failed - first request may be slow")
                log("Check server logs for errors")

    # Run test if requested
    if args.test:
        log("Running endpoint test...")
        test_chat_completion(args.host, args.port, api_model)

    # Start Open WebUI by default (except for llama.cpp which has built-in UI)
    webui_container_id = None
    should_start_webui = not args.no_webui and backend != "llama"
    if should_start_webui:
        webui_container_id = start_open_webui(
            backend_port=args.port,
            webui_port=args.webui_port,
            image=args.webui_image,
        )
        if webui_container_id:
            log(f"Open WebUI available at http://localhost:{args.webui_port}")
        else:
            log("WARNING: Failed to start Open WebUI")

    # Keep server running until Ctrl+C
    log(f"\nServer running at http://{args.host}:{args.port}/v1")
    if webui_container_id:
        log(f"Open WebUI running at http://localhost:{args.webui_port}")
    log("Press Ctrl+C to stop...")

    def signal_handler(sig, frame):
        log("\nShutting down...")
        if webui_container_id:
            stop_container(webui_container_id, label="Open WebUI")
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
            if webui_container_id:
                stop_container(webui_container_id, label="Open WebUI")
            sys.exit(1)


if __name__ == "__main__":
    main()
