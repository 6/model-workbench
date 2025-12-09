#!/usr/bin/env python3
"""
Model Workbench Eval Runner

Runs IFEval and GSM8K benchmarks against local model servers using DeepEval.
Auto-detects GGUF (llama.cpp) vs safetensors (vLLM) based on model format.

Examples:
  # Safetensors model -> vLLM backend
  uv run python scripts/run_eval.py --model ~/models/zai-org/GLM-4.6V-FP8

  # GGUF model -> llama.cpp backend
  uv run python scripts/run_eval.py --model ~/models/unsloth/GLM-4.5-Air-GGUF/UD-Q4_K_XL

  # Run IFEval only (541 instruction-following prompts)
  uv run python scripts/run_eval.py --model ~/models/org/model --benchmark ifeval

  # Run GSM8K only (1319 math reasoning problems)
  uv run python scripts/run_eval.py --model ~/models/org/model --benchmark gsm8k

  # Use an already running server
  uv run python scripts/run_eval.py --model ~/models/org/model --no-autostart
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from deepeval.benchmarks import IFEval, GSM8K

from bench_utils import (
    ROOT,
    sanitize,
    extract_repo_id,
    get_gpu_info,
    get_gpu_count,
    resolve_model_path,
    detect_model_format,
    model_needs_nightly,
    log,
)
from server_manager import ServerManager
from eval_model_wrapper import LocalServerLLM


EVAL_RESULTS_ROOT = ROOT / "evals"


def run_ifeval(model: LocalServerLLM) -> dict:
    """Run IFEval benchmark (541 instruction-following prompts).

    Returns:
        Dict with benchmark name, question count, and overall score.
    """
    log("Running IFEval (541 prompts)...")
    benchmark = IFEval()
    benchmark.evaluate(model=model)

    return {
        "benchmark": "ifeval",
        "questions": 541,
        "overall_score": benchmark.overall_score,
    }


def run_gsm8k(model: LocalServerLLM) -> dict:
    """Run GSM8K benchmark (1319 grade school math problems).

    GSM8K tests multi-step mathematical reasoning with problems requiring 2-8 steps.
    Uses 3-shot prompting with chain-of-thought enabled for best results.

    Returns:
        Dict with benchmark name, question count, and overall score.
    """
    log("Running GSM8K (1319 problems)...")
    benchmark = GSM8K(n_shots=3, enable_cot=True)
    benchmark.evaluate(model=model)

    return {
        "benchmark": "gsm8k",
        "questions": 1319,
        "overall_score": benchmark.overall_score,
    }


def save_results(results: dict, repo_id: str, benchmark_name: str) -> Path:
    """Save results to evals/.

    Args:
        results: The results dict to save.
        repo_id: Model repo ID for filename.
        benchmark_name: Benchmark name(s) for filename.

    Returns:
        Path to saved results file.
    """
    EVAL_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    safe_name = sanitize(repo_id)
    filename = f"{date_str}_{safe_name}_{benchmark_name}.json"
    path = EVAL_RESULTS_ROOT / filename

    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    log(f"Results saved to {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Run evals (IFEval, HumanEval) on local model servers"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to model",
    )
    parser.add_argument(
        "--benchmark",
        nargs="+",
        choices=["ifeval", "gsm8k"],
        default=["ifeval", "gsm8k"],
        help="Benchmark(s) to run (default: both)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens for generation (default: 512)",
    )

    # Server options
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (default: 8000 for vLLM, 8080 for llama.cpp)",
    )
    parser.add_argument(
        "--no-autostart",
        action="store_true",
        help="Don't start server; require it already running",
    )
    parser.add_argument(
        "--server-timeout",
        type=int,
        default=180,
        help="Timeout in seconds waiting for server to start (default: 180)",
    )

    # vLLM server options (used when auto-starting safetensors models)
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=None,
        help="Tensor parallel size (default: auto-detect GPU count)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=65536,
        help="Max context length (default: 65536)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="GPU memory fraction (default: 0.95)",
    )

    # llama.cpp server options (used when auto-starting GGUF models)
    default_llama_bin = str(Path.home() / "llama.cpp/build/bin/llama-server")
    parser.add_argument(
        "--llama-server-bin",
        default=default_llama_bin,
        help=f"Path to llama-server binary (default: {default_llama_bin})",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=999,
        help="GPU layers to offload for llama.cpp (default: 999 = all)",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=None,
        help="Context length for llama.cpp (default: model default)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Parallel sequences for llama.cpp (default: 1)",
    )

    # Environment selection
    env_group = parser.add_mutually_exclusive_group()
    env_group.add_argument(
        "--force-stable",
        action="store_true",
        help="Force use of stable venv (ignore model config)",
    )
    env_group.add_argument(
        "--force-nightly",
        action="store_true",
        help="Force use of nightly venv (ignore model config)",
    )

    args = parser.parse_args()

    # Auto-detect GPU count for tensor parallelism
    if args.tensor_parallel is None:
        args.tensor_parallel = get_gpu_count()

    # Resolve and validate model path
    model_path = resolve_model_path(args.model)
    repo_id = extract_repo_id(model_path)
    model_format = detect_model_format(model_path)

    # Set default port based on backend
    if args.port is None:
        args.port = 8080 if model_format == "gguf" else 8000

    # Determine which environment to use (only applies to vLLM)
    if args.force_nightly:
        use_nightly = True
    elif args.force_stable:
        use_nightly = False
    else:
        use_nightly = model_needs_nightly(args.model)

    env_label = "nightly" if use_nightly else "stable"
    backend_label = "llama-server" if model_format == "gguf" else "vLLM"

    log(f"Model: {repo_id} ({model_format})")
    log(f"Backend: {backend_label}")
    if model_format != "gguf":
        log(f"Environment: {env_label}")

    # Server management
    server = ServerManager(
        host=args.host,
        port=args.port,
        timeout=args.server_timeout,
    )

    # Check if we need to start the server
    if not server.is_running():
        if args.no_autostart:
            raise SystemExit(
                f"Server not running on {args.host}:{args.port} and --no-autostart was set.\n"
                f"Start the server first or remove --no-autostart to auto-start."
            )

    with server:
        # Start server if not already running
        if not server.is_running():
            if model_format == "gguf":
                # llama.cpp backend for GGUF models
                server.start_llama(
                    model_path=model_path,
                    llama_server_bin=args.llama_server_bin,
                    n_gpu_layers=args.n_gpu_layers,
                    ctx=args.ctx,
                    parallel=args.parallel,
                )
            else:
                # vLLM backend for safetensors models
                server.start_vllm(
                    model_path=model_path,
                    tensor_parallel=args.tensor_parallel,
                    use_nightly=use_nightly,
                    max_model_len=args.max_model_len,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                )

        # Model name differs by backend:
        # - vLLM: registers model under full path
        # - llama.cpp: uses "gpt-3.5-turbo" as default model name
        model_name = "gpt-3.5-turbo" if model_format == "gguf" else model_path

        base_url = f"http://{args.host}:{args.port}/v1"
        model = LocalServerLLM(
            base_url=base_url,
            model_name=model_name,
            max_tokens=args.max_tokens,
        )

        log(f"Connected to server at {base_url}")

        # Build results structure
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "repo_id": repo_id,
            "model_path": str(model_path),
            "model_format": model_format,
            "environment": env_label,
            "gpu_info": get_gpu_info(include_memory=True),
            "benchmarks": {},
        }

        # Run requested benchmarks
        for benchmark in args.benchmark:
            if benchmark == "ifeval":
                result = run_ifeval(model)
            elif benchmark == "gsm8k":
                result = run_gsm8k(model)

            all_results["benchmarks"][benchmark] = result
            log(f"{benchmark.upper()} score: {result['overall_score']:.3f}")

        # Save combined results
        benchmarks_str = "-".join(args.benchmark)
        save_results(all_results, repo_id, benchmarks_str)


if __name__ == "__main__":
    main()
