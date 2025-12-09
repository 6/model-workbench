# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Model Workbench is a local LLM benchmarking suite for downloading, serving, and benchmarking models. It supports:
- Multi-GPU setups with tensor parallelism
- Both safetensors (vLLM) and GGUF (llama.cpp) formats
- Vision and text-only benchmarks
- Dual environments: stable releases and nightly/git-based dependencies

## Common Commands

```bash
# Initial setup (validates uv/nvidia, creates envs, downloads models)
./scripts/bootstrap.sh

# Sync dependencies
uv sync --all-extras                    # stable environment
cd nightly && uv sync --all-extras      # nightly environment

# Download/update models from config
uv run python scripts/fetch_models.py

# Benchmark with vLLM (safetensors models)
uv run python scripts/run_bench_vllm_server.py --model ~/models/zai-org/GLM-4.6V-FP8
uv run python scripts/run_bench_vllm_server.py --model ~/models/org/model --tensor-parallel 1
uv run python scripts/run_bench_vllm_server.py --model ~/models/org/model --image config/example.jpg

# Benchmark with llama.cpp (GGUF models)
uv run python scripts/run_bench_llama_server.py --model ~/models/unsloth/GLM-4.5-Air-GGUF/UD-Q4_K_XL
```

## Architecture

### Dual Environment System
- **Stable** (`.venv/`): Production models with released dependencies
- **Nightly** (`nightly/.venv/`): Models requiring git-master transformers/tokenizers
- Models marked `nightly: true` in config auto-select nightly env; override with `--force-nightly` or `--force-stable`

### Key Directories
- `scripts/` - Benchmark runners and utilities
- `config/models.yaml` - Model definitions with quantization variants, sources, special flags
- `perf/` - JSON benchmark results with GPU info, timing stats, outputs
- `nightly/` - Separate pyproject.toml for bleeding-edge deps

### Benchmark Engines
Both engines expose OpenAI-compatible APIs for unified benchmarking:
- **vLLM** (`run_bench_vllm_server.py`): High-performance serving with tensor parallelism, FP8 support, vision models
- **llama.cpp** (`run_bench_llama_server.py`): GGUF quantized models with GPU sharding

### Shared Utilities (`bench_utils.py`)
- GPU detection via nvidia-smi
- Model path resolution and config loading
- Environment selection (stable vs nightly)
- Result file naming and path utilities
