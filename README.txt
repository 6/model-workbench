model-workbench: personal local LLM workbench to download, serve, and benchmark models

Quick start:
    ./scripts/bootstrap.sh
    uv run python scripts/run_bench.py --model ~/models/zai-org/GLM-4.6V-FP8

Benchmarks (auto-detects backend from model format):
    uv run python scripts/run_bench.py --model ~/models/zai-org/GLM-4.6V-FP8                        # vLLM
    uv run python scripts/run_bench.py --model ~/models/zai-org/GLM-4.6V-FP8 --backend trtllm       # TensorRT-LLM
    uv run python scripts/run_bench.py --model ~/models/unsloth/GLM-4.5-Air-GGUF/UD-Q4_K_XL         # llama.cpp
    uv run python scripts/run_bench.py --model ~/models/unsloth/GLM-4.5-Air-GGUF/UD-Q4_K_XL --backend ik_llama

Standalone server:
    uv run python scripts/run_server.py --model ~/models/zai-org/GLM-4.6V-FP8

Evals (IFEval + GSM8K):
    uv run python scripts/run_eval.py --model ~/models/zai-org/GLM-4.6V-FP8

Results saved to perf/ and evals/
