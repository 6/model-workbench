model-workbench: personal local LLM workbench to download, serve, and benchmark models

Quick start:
    ./scripts/bootstrap.sh

If you update the model list in `config/models.yaml`, re-run:
    uv run python scripts/fetch_models.py

Benchmarking:

vLLM (safetensors, multi-GPU with tensor parallelism):
    # Auto-detects GPU count, uses all available GPUs
    uv run python scripts/run_bench_vllm_server.py --model zai-org/GLM-4.6V-FP8

    # Force single GPU
    uv run python scripts/run_bench_vllm_server.py --model zai-org/GLM-4.6V-FP8 --tensor-parallel 1

    # Vision benchmark
    uv run python scripts/run_bench_vllm_server.py --model zai-org/GLM-4.6V-FP8 --image config/example.jpg

llama-server (GGUF):
    # Auto-starts server, runs benchmark (auto-splits across GPUs if needed)
    uv run python scripts/run_bench_llama_server.py --model unsloth/GLM-4.5-Air-GGUF/UD-Q4_K_XL

    # Non-sharded GGUF
    uv run python scripts/run_bench_llama_server.py --model unsloth/Ministral-3-14B-Instruct-2512-GGUF/Ministral-3-14B-Instruct-2512-UD-Q4_K_XL.gguf

Results are saved to benchmarks/<tag>/ where tag is inferred from GPU count (single-gpu, dual-gpu, etc.)
