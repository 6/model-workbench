model-workbench: personal local LLM workbench to download, serve, and benchmark models

Quick start:
    ./scripts/bootstrap.sh
    uv run python scripts/run_bench.py --model ~/models/mistralai/Devstral-Small-2-24B-Instruct-2512

Backends (all run in Docker, pinned to specific releases/commits per model):
    Safetensors:  vllm (default), sglang, trtllm (TensorRT-LLM)
    GGUF:         llama (default)
    EXL3:         exl (ExLlamaV3)

Benchmarks (auto-detects backend from model format):
    uv run python scripts/run_bench.py --model ~/models/openai/gpt-oss-120b                # vLLM
    uv run python scripts/run_bench.py --model ~/models/unsloth/GLM-4.7-GGUF/UD-Q3_K_XL    # llama.cpp

Standalone server:
    uv run python scripts/run_server.py --model ~/models/mistralai/Devstral-Small-2-24B-Instruct-2512

Evals (IFEval + GSM8K):
    uv run python scripts/run_eval.py --model ~/models/mistralai/Devstral-Small-2-24B-Instruct-2512

Results saved to perf/ and evals/
