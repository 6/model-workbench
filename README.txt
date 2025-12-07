model-workbench: personal local LLM workbench to download, serve, and benchmark models

Quick start:
    ./scripts/bootstrap.sh

If you update the model list in `config/models.yaml`, re-run:
    uv run python scripts/fetch_models.py

Benchmarking:

Single GPU (vLLM):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_bench.py \
      --model Qwen/Qwen3-VL-8B-Instruct

Dual GPU (vLLM):
    CUDA_VISIBLE_DEVICES=0,1 uv run python scripts/run_bench.py \
      --model openai/gpt-oss-120b

GGUF (llama-server):
    # start server on port 8080
    ./scripts/llama-server.sh unsloth/GLM-4.5-Air-GGUF/UD-Q4_K_XL

    # run benchmark
    uv run python scripts/run_bench.py --model unsloth/GLM-4.5-Air-GGUF
