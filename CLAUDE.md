# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Model Workbench is a local LLM benchmarking suite for downloading, serving, and benchmarking models. It supports:
- Multi-GPU setups with tensor parallelism
- Both safetensors (vLLM) and GGUF (llama.cpp) formats
- Vision and text-only benchmarks
- Docker-based execution for reproducible benchmarks with version pinning

## Common Commands

```bash
# Initial setup (validates uv/nvidia/docker, syncs deps, downloads models)
./scripts/bootstrap.sh

# Sync dependencies
uv sync --all-extras

# Download/update models from config
uv run python scripts/fetch_models.py

# Run benchmarks (uses version from config/models.yaml)
uv run python scripts/run_bench.py --model ~/models/zai-org/GLM-4.6V-FP8
uv run python scripts/run_bench.py --model ~/models/unsloth/GLM-4.5-Air-GGUF/UD-Q4_K_XL

# Override backend version for a specific run
uv run python scripts/run_bench.py --model ~/models/... --backend-version v0.8.0

# Force rebuild Docker image
uv run python scripts/run_bench.py --model ~/models/... --rebuild

# Build image only (no benchmark)
uv run python scripts/run_bench.py --model ~/models/... --build-only

# Run evals (IFEval + GSM8K via DeepEval)
uv run python scripts/run_eval.py --model ~/models/zai-org/GLM-4.6V-FP8
uv run python scripts/run_eval.py --model ~/models/org/model --benchmark ifeval
uv run python scripts/run_eval.py --model ~/models/org/model --benchmark gsm8k
```

## Architecture

### Docker-Based Execution
All benchmarks run via Docker for reproducibility with version pinning.

- **Version pinning**: Specify commit SHA or release tag (e.g., `v0.8.0`, `b4521`)
- **On-demand builds**: Docker images built locally and cached
- **Config-driven**: `config/models.yaml` specifies default versions and per-model overrides

### Version Resolution
1. CLI override via `--backend-version`
2. Model's `backend_version` in config (if specified)
3. Global `defaults.vllm_version` or `defaults.llama_version`

### Key Directories
- `scripts/` - Benchmark runners and utilities
- `docker/` - Dockerfiles for vLLM and llama.cpp
- `config/models.yaml` - Model definitions with backend versions
- `perf/` - JSON benchmark results (unified schema across backends)
- `evals/` - DeepEval results (IFEval, GSM8K)

### Benchmark Engine
- **Unified** (`run_bench.py`): Single entry point for all benchmarks
  - Auto-detects model format (GGUF -> llama.cpp, safetensors -> vLLM)
  - Runs backends via Docker for reproducible results
- **vLLM backend**: OpenAI-compatible API, tensor parallelism, FP8, vision models
  - Auto-uses all GPUs (`--tensor-parallel 1` for single)
  - Scrapes Prometheus `/metrics` for detailed timing
- **llama.cpp backend**: Native `/completion` endpoint, GGUF models, GPU sharding
- **Shared metrics**: `wall_s`, `ttft_ms`, `generation_tok_per_s`, `prompt_tokens`, `generated_tokens`

### Key Modules
- **`bench_utils.py`**: GPU detection, model resolution, config loading, prompts, version resolution
- **`server_manager.py`**: Server lifecycle, `start_vllm()`/`start_llama()` for Docker
- **`docker_manager.py`**: Image building, GPU Docker validation, command builders
