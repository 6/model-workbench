#!/usr/bin/env python3
"""
Model Workbench Eval Runner

Runs IFEval and GSM8K benchmarks against local model servers using DeepEval.
Auto-detects GGUF (llama.cpp) vs safetensors (vLLM) based on model format.
Use --backend to explicitly select a backend.

Examples:
  # Safetensors model -> vLLM backend (auto-detected)
  uv run python scripts/run_eval.py --model ~/models/zai-org/GLM-4.6V-FP8

  # Use TensorRT-LLM backend
  uv run python scripts/run_eval.py --model ~/models/zai-org/GLM-4.6V-FP8 --backend trtllm

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

from deepeval.benchmarks import GSM8K, IFEval

from bench_utils import extract_repo_id, get_gpu_info, resolve_run_config, sanitize
from common import BACKEND_REGISTRY, EVAL_RESULTS_ROOT, log
from eval_model_wrapper import LocalServerLLM
from server_manager import ServerManager


def run_ifeval(model: LocalServerLLM) -> dict:
    """Run IFEval benchmark (541 instruction-following prompts).

    Returns:
        Dict with benchmark name, question count, correct count, overall score,
        and per-instruction-type breakdown (~25 instruction types).
    """
    log("Running IFEval (541 prompts)...")
    benchmark = IFEval()
    benchmark.evaluate(model=model)

    # Count correct vs total from predictions
    correct = benchmark.predictions["All_Instructions_Correct"].sum()
    total = len(benchmark.predictions)

    return {
        "benchmark": "ifeval",
        "questions": total,
        "correct": int(correct),
        "overall_score": benchmark.overall_score,
        "instruction_breakdown": benchmark.instruction_breakdown,
    }


def run_gsm8k(model: LocalServerLLM) -> dict:
    """Run GSM8K benchmark (1319 grade school math problems).

    GSM8K tests multi-step mathematical reasoning with problems requiring 2-8 steps.
    Uses 3-shot prompting with chain-of-thought enabled for best results.

    Returns:
        Dict with benchmark name, question count, correct count, and overall score.
    """
    log("Running GSM8K (1319 problems)...")
    benchmark = GSM8K(n_shots=3, enable_cot=True)
    benchmark.evaluate(model=model)

    # Count correct vs total from predictions
    correct = benchmark.predictions["Correct"].sum()
    total = len(benchmark.predictions)

    return {
        "benchmark": "gsm8k",
        "questions": total,
        "correct": int(correct),
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
        description="Run evals (IFEval, GSM8K) on local model servers"
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

    # Backend selection
    parser.add_argument(
        "--backend",
        choices=list(BACKEND_REGISTRY.keys()),
        default=None,
        help="Backend to use (default: auto-detect from model format)",
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
        help="Server port (default: 8000 for vLLM/trtllm, 8080 for llama.cpp)",
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

    # vLLM/trtllm server options (defaults from config, CLI overrides)
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=None,
        help="Tensor parallel size (default: auto-detect GPU count)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Max context length (default: from config or 65536)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="GPU memory fraction (default: from config or 0.95)",
    )

    # llama.cpp server options (defaults from config, CLI overrides)
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=None,
        help="GPU layers to offload for llama.cpp (default: from config or 999)",
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

    # Backend version (Docker-based execution)
    parser.add_argument(
        "--backend-version",
        default=None,
        help="Backend version (e.g., v0.8.0 for vLLM, b4521 for llama.cpp, 0.18.0 for trtllm)",
    )

    args = parser.parse_args()

    # Resolve backend config and apply defaults
    backend, model_path, backend_cfg = resolve_run_config(args)
    repo_id = extract_repo_id(model_path)

    # Resolve backend version
    backend_version = args.backend_version or backend_cfg.get("version")
    if not backend_version:
        raise SystemExit(
            f"No backend version specified and none found in config.\n"
            f"Either:\n"
            f"  1. Set defaults.backends.{backend}.version in config/models.yaml\n"
            f"  2. Pass --backend-version"
        )

    backend_label = BACKEND_REGISTRY[backend]["display_name"]

    log(f"Model: {repo_id}")
    log(f"Backend: {backend_label}")
    log(f"Backend version: {backend_version}")

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
            if backend in ("llama", "ik_llama"):
                # llama.cpp or ik_llama.cpp backend for GGUF models
                server.start_gguf_backend(
                    engine=backend,
                    model_path=model_path,
                    version=backend_version,
                    n_gpu_layers=args.n_gpu_layers,
                    ctx=args.ctx,
                    parallel=args.parallel,
                )
            elif backend == "trtllm":
                # TensorRT-LLM backend
                server.start_trtllm(
                    model_path=model_path,
                    tensor_parallel=args.tensor_parallel,
                    version=backend_version,
                )
            else:
                # vLLM backend for safetensors models
                server.start_vllm(
                    model_path=model_path,
                    tensor_parallel=args.tensor_parallel,
                    version=backend_version,
                    max_model_len=args.max_model_len,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                )

        # Model name differs by backend:
        # - vLLM/trtllm: registers model under full path
        # - llama.cpp: uses "gpt-3.5-turbo" as default model name
        model_name = "gpt-3.5-turbo" if backend in ("llama", "ik_llama") else model_path

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
            "backend": backend,
            "backend_version": backend_version,
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
