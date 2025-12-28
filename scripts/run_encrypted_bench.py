#!/usr/bin/env python3
"""Run all encrypted prompts against a model and store encrypted results.

Usage:
    uv run python scripts/run_encrypted_bench.py --model ~/models/unsloth/Qwen3-30B-A3B
    uv run python scripts/run_encrypted_bench.py --model ~/models/... --backend sglang

Results are stored in perf/encrypted_results.yaml with partial encryption:
- Visible: prompt_key, model, timestamp, stats
- Encrypted: reasoning, answer
"""

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bench_utils import (
    build_chat_messages,
    extract_repo_id,
    get_gpu_count,
    get_gpu_info,
    get_model_backend_config,
    get_model_backend_version,
    log,
    resolve_model_path,
)
from server_manager import ServerManager

from prompts.secret import SecretPromptError, get_secret_prompt, list_secret_prompts

RESULTS_FILE = ROOT / "perf" / "encrypted_results.yaml"


def load_results() -> dict:
    """Load existing results (decrypted via SOPS)."""
    if not RESULTS_FILE.exists():
        return {"results": []}
    result = subprocess.run(
        ["sops", "--decrypt", str(RESULTS_FILE)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # File exists but can't decrypt - might be first run or corrupted
        return {"results": []}
    return yaml.safe_load(result.stdout) or {"results": []}


def save_results(data: dict) -> None:
    """Save results with partial encryption (only reasoning/answer fields)."""
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    # Write plaintext first
    with open(RESULTS_FILE, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    # Encrypt in-place (uses .sops.yaml rules for partial encryption)
    result = subprocess.run(
        ["sops", "--encrypt", "--in-place", str(RESULTS_FILE)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Warning: SOPS encryption failed: {result.stderr}")
        print("Results saved unencrypted. Run 'sops --encrypt --in-place' manually.")


def run_single_prompt(
    client: OpenAI,
    model: str,
    prompt_text: str,
    max_tokens: int,
    temperature: float,
    frequency_penalty: float,
    host: str,
    port: int,
) -> dict:
    """Run a single prompt and return result with reasoning/answer separated."""
    messages = build_chat_messages(prompt_text, None)  # No image for encrypted prompts

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature if temperature > 0 else 0.0,
        frequency_penalty=frequency_penalty,
        seed=0,
    )
    t1 = time.perf_counter()
    wall = t1 - t0

    # Extract reasoning and answer SEPARATELY
    msg = response.choices[0].message
    reasoning_content = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    answer_content = msg.content or ""

    # If no explicit reasoning field, try to parse thinking from content
    if not reasoning_content:
        # Format 1: <think>reasoning</think>answer (MiniMax, Qwen, etc.)
        # Use regex to handle optional leading whitespace
        match = re.match(r"\s*<think>(.*?)</think>\s*(.*)", answer_content, re.DOTALL)
        if match:
            reasoning_content = match.group(1).strip()
            answer_content = match.group(2).strip()
        # Format 2: reasoning</think>answer (GLM-4.7 - no opening tag)
        elif "</think>" in answer_content:
            parts = answer_content.split("</think>", 1)
            if len(parts) == 2:
                reasoning_content = parts[0].strip()
                answer_content = parts[1].strip()

    # Extract token counts
    prompt_tokens = None
    generated_tokens = None
    if response.usage:
        prompt_tokens = response.usage.prompt_tokens
        generated_tokens = response.usage.completion_tokens

    # Calculate tok_per_s from wall time
    tok_per_s = None
    if generated_tokens and wall > 0:
        tok_per_s = generated_tokens / wall

    return {
        "wall_s": wall,
        "input_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "tok_per_s": tok_per_s,
        "reasoning": reasoning_content,
        "answer": answer_content,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Run all encrypted prompts against a model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python scripts/run_encrypted_bench.py --model ~/models/unsloth/Qwen3-30B-A3B
    uv run python scripts/run_encrypted_bench.py --model ~/models/... --backend sglang --max-tokens 8192
        """,
    )
    ap.add_argument("--model", required=True, help="Path to model directory")
    ap.add_argument(
        "--backend",
        default="vllm",
        choices=["vllm", "sglang", "trtllm"],
        help="Backend to use (default: vllm)",
    )
    ap.add_argument("--max-tokens", type=int, default=50000, help="Max tokens (default: 50000)")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    ap.add_argument(
        "--frequency-penalty", type=float, default=0.2, help="Frequency penalty (default: 0.2)"
    )
    ap.add_argument("--host", default="127.0.0.1", help="Server host")
    ap.add_argument("--port", type=int, default=8000, help="Server port")
    ap.add_argument("--server-timeout", type=int, default=600, help="Server startup timeout")
    ap.add_argument("--tensor-parallel", type=int, default=None, help="Tensor parallel size")
    ap.add_argument("--max-model-len", type=int, default=None, help="Max model length")
    ap.add_argument(
        "--gpu-memory-utilization", type=float, default=0.95, help="GPU memory utilization"
    )
    ap.add_argument("--backend-version", default=None, help="Backend version override")
    ap.add_argument("--no-autostart", action="store_true", help="Don't auto-start server")
    ap.add_argument("--rebuild", action="store_true", help="Force rebuild Docker image")

    args = ap.parse_args()

    # Resolve model path
    model_path = resolve_model_path(args.model)
    if model_path is None:
        print(f"Model not found: {args.model}")
        sys.exit(1)

    repo_id = extract_repo_id(model_path)

    # Get encrypted prompts
    try:
        prompt_keys = list_secret_prompts()
    except SecretPromptError as e:
        print(f"Error loading encrypted prompts: {e}")
        sys.exit(1)

    if not prompt_keys:
        print("No encrypted prompts found. Create prompts/secret.yaml first.")
        print("Run: sops prompts/secret.yaml")
        sys.exit(1)

    print("\n== Encrypted Benchmark ==")
    print(f"Model: {repo_id}")
    print(f"Backend: {args.backend}")
    print(f"Prompts: {len(prompt_keys)}")
    print(f"Max tokens: {args.max_tokens}")

    # Resolve backend version and image type
    backend_version = args.backend_version or get_model_backend_version(args.model, args.backend)
    if not backend_version:
        print(f"No backend version found for {args.backend}. Using default.")
        backend_version = None

    backend_cfg = get_model_backend_config(args.model, args.backend) or {}
    image_type = backend_cfg.get("image_type", "prebuilt")

    # Auto-detect tensor parallel from GPU count if not specified
    tensor_parallel = args.tensor_parallel or get_gpu_count()

    # Ensure max_model_len is sufficient for max_tokens + input headroom
    if args.max_model_len is None or args.max_model_len < args.max_tokens:
        args.max_model_len = max(args.max_tokens + 5000, 65536)

    # Setup server
    server = ServerManager(
        host=args.host,
        port=args.port,
        timeout=args.server_timeout,
    )

    if not server.is_running() and args.no_autostart:
        print(f"Server not running on {args.host}:{args.port} and --no-autostart was set.")
        sys.exit(1)

    # Load existing results
    results = load_results()
    run_timestamp = datetime.now().isoformat()

    with server:
        # Start server if needed
        if not server.is_running():
            log(f"Starting {args.backend} server...")
            if args.backend == "vllm":
                server.start_vllm(
                    model_path=model_path,
                    tensor_parallel=tensor_parallel,
                    version=backend_version,
                    max_model_len=args.max_model_len,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    rebuild=args.rebuild,
                    image_type=image_type,
                )
            elif args.backend == "sglang":
                server.start_sglang(
                    model_path=model_path,
                    tensor_parallel=tensor_parallel,
                    version=backend_version,
                    rebuild=args.rebuild,
                    image_type=image_type,
                )
            elif args.backend == "trtllm":
                server.start_trtllm(
                    model_path=model_path,
                    tensor_parallel=tensor_parallel,
                    version=backend_version,
                    rebuild=args.rebuild,
                    image_type=image_type,
                )

        gpu_info = get_gpu_info(include_memory=True)
        log(
            f"GPU memory: {gpu_info.get('memory_used_mib', '?')} / "
            f"{gpu_info.get('memory_total_mib', '?')} MiB"
        )

        client = OpenAI(
            base_url=f"http://{args.host}:{args.port}/v1",
            api_key="dummy",
        )

        # Warmup
        log("Warmup request...")
        try:
            client.chat.completions.create(
                model=model_path,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
            )
        except Exception as e:
            log(f"Warmup failed: {e}")

        # Run all prompts
        print()
        for i, key in enumerate(prompt_keys, 1):
            print(f"[{i}/{len(prompt_keys)}] Running {key}...")

            try:
                prompt_text = get_secret_prompt(key)
            except SecretPromptError as e:
                print(f"  Error getting prompt: {e}")
                continue

            try:
                result = run_single_prompt(
                    client=client,
                    model=model_path,
                    prompt_text=prompt_text,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    frequency_penalty=args.frequency_penalty,
                    host=args.host,
                    port=args.port,
                )
            except Exception as e:
                print(f"  Error running prompt: {e}")
                continue

            # Report stats
            print(f"  wall_s: {result['wall_s']:.2f}")
            print(f"  tokens: {result['input_tokens']} -> {result['generated_tokens']}")
            if result["tok_per_s"]:
                print(f"  tok/s: {result['tok_per_s']:.1f}")
            if result["reasoning"]:
                print(f"  reasoning: {len(result['reasoning'])} chars")
            print(f"  answer: {len(result['answer'])} chars")

            # Append result
            results["results"].append(
                {
                    "prompt_key": key,
                    "model": repo_id,
                    "timestamp": run_timestamp,
                    "stats": {
                        "wall_s": result["wall_s"],
                        "input_tokens": result["input_tokens"],
                        "generated_tokens": result["generated_tokens"],
                        "tok_per_s": result["tok_per_s"],
                    },
                    "reasoning": result["reasoning"],
                    "answer": result["answer"],
                }
            )

    # Save results
    print()
    save_results(results)
    print(f"Results saved to {RESULTS_FILE}")
    print(f"View with: sops {RESULTS_FILE}")
    print(f"Decrypt to stdout: sops -d {RESULTS_FILE}")


if __name__ == "__main__":
    main()
