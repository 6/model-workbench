model-workbench: personal local LLM workbench to download, serve, and benchmark models

Quick start:
    ./scripts/bootstrap.sh

If you update the model list in `config/models.yaml`, re-run:
    uv run python scripts/fetch_models.py

Benchmarking:

Single GPU:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_bench.py \
      --model Qwen/Qwen3-VL-8B-Instruct

Dual GPU:
    CUDA_VISIBLE_DEVICES=0,1 uv run python scripts/run_bench.py \
      --model openai/gpt-oss-120b
