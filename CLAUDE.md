# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Model Workbench is a local LLM benchmarking suite for downloading, serving, and benchmarking models. It supports:
- Multi-GPU setups with tensor parallelism
- Both safetensors (vLLM, TensorRT-LLM) and GGUF (llama.cpp, ik_llama.cpp) formats
- Vision and text-only benchmarks
- Docker-based execution for reproducible benchmarks with version pinning
- Multiple backends: `--backend vllm`, `--backend sglang`, `--backend llama`, etc. (see `config/models.yaml` for full list and version pinning)
- Pre-built Docker images for vLLM and TensorRT-LLM; build-from-source for llama.cpp

## Common Commands

```bash
# Initial setup (validates uv/nvidia/docker, syncs deps, downloads models)
./scripts/bootstrap.sh

# Sync dependencies
uv sync --all-extras

# Lint and format (run after making Python changes)
uv run ruff check --fix .
uv run ruff format .

# Download/update models from config
uv run python scripts/fetch_models.py

# Run benchmarks (uses version from config/models.yaml)
uv run python scripts/run_bench.py --model ~/models/zai-org/GLM-4.6V-FP8
uv run python scripts/run_bench.py --model ~/models/unsloth/GLM-4.5-Air-GGUF/UD-Q4_K_XL

# Use ik_llama.cpp backend for GGUF (ikawrakow's optimized fork)
uv run python scripts/run_bench.py --model ~/models/unsloth/GLM-GGUF/UD-Q4_K_XL --backend ik_llama

# Use TensorRT-LLM backend for safetensors models
uv run python scripts/run_bench.py --model ~/models/zai-org/GLM-4.6V-FP8 --backend trtllm

# Override backend version for a specific run
uv run python scripts/run_bench.py --model ~/models/... --backend-version v0.8.0

# Use specific Docker image (e.g., nightly vLLM)
uv run python scripts/run_bench.py --model ~/models/... --docker-image vllm/vllm-openai:nightly

# Force build from source instead of prebuilt
uv run python scripts/run_bench.py --model ~/models/... --image-type build

# Force rebuild/repull Docker image
uv run python scripts/run_bench.py --model ~/models/... --rebuild

# Build image only (no benchmark)
uv run python scripts/run_bench.py --model ~/models/... --build-only

# Run evals (IFEval + GSM8K via DeepEval)
uv run python scripts/run_eval.py --model ~/models/zai-org/GLM-4.6V-FP8
uv run python scripts/run_eval.py --model ~/models/org/model --benchmark ifeval
uv run python scripts/run_eval.py --model ~/models/org/model --benchmark gsm8k

# Start standalone server for inference (keeps running until Ctrl+C)
uv run python scripts/run_server.py --model ~/models/zai-org/GLM-4.6V-FP8
uv run python scripts/run_server.py --model ~/models/org/model --test  # starts + verifies endpoint
uv run python scripts/run_server.py --model ~/models/unsloth/GLM-GGUF/UD-Q4_K_XL --backend ik_llama
uv run python scripts/run_server.py --model ~/models/GLM-FP8 --backend trtllm  # TensorRT-LLM
uv run python scripts/run_server.py --model ~/models/GLM-FP8 --docker-image vllm/vllm-openai:nightly  # nightly vLLM
```

## Architecture

### Docker-Based Execution
All benchmarks run via Docker for reproducibility with version pinning.

- **Version pinning**: Specify commit SHA or release tag (e.g., `v0.8.0`, `b4521`, `0.18.0`)
- **Pre-built images**: vLLM uses `vllm/vllm-openai`, TensorRT-LLM uses NGC images
- **Build from source**: llama.cpp and ik_llama.cpp build from Dockerfiles
- **Config-driven**: `config/models.yaml` specifies default versions and image types

### Configuration Schema

Backend defaults are configured in `config/models.yaml`:

```yaml
defaults:
  backends:
    vllm:
      version: v0.8.0
      image_type: prebuilt    # "prebuilt" (Docker Hub) or "build" (from source)
      args:
        gpu_memory_utilization: 0.95
        max_model_len: 65536
    llama:
      version: b4521
      image_type: build
      args:
        n_gpu_layers: 999
    ik_llama:
      version: b4521          # Explicit version (no fallback to llama)
      image_type: build
      args:
        n_gpu_layers: 999
    trtllm:
      version: "0.18.0"
      image_type: prebuilt

models:
  - repo_id: org/model
    backend: vllm              # Optional: explicit backend selection
    backends:                  # Optional: per-backend overrides
      vllm:
        version: v0.8.1
        args:
          max_model_len: 32768
```

### Resolution Priority (highest to lowest)

**Version/Image Type**:
1. CLI override (`--backend-version`, `--image-type`, `--docker-image`)
2. Model-specific config (`model.backends.{engine}.version`)
3. Global defaults (`defaults.backends.{engine}.version`)

**Backend Args** (gpu_memory_utilization, max_model_len, n_gpu_layers):
1. CLI override (`--gpu-memory-utilization 0.9`)
2. Model-specific config (`model.backends.{engine}.args.gpu_memory_utilization`)
3. Global defaults (`defaults.backends.{engine}.args.gpu_memory_utilization`)
4. Hardcoded fallback in script

### Key Directories
- `scripts/` - Benchmark runners and utilities
- `docker/` - Dockerfiles for vLLM and llama.cpp
- `config/models.yaml` - Model definitions with backend versions
- `perf/` - JSON benchmark results (unified schema across backends)
- `evals/` - DeepEval results (IFEval, GSM8K)

### Benchmark Engine
- **Unified** (`run_bench.py`): Single entry point for all benchmarks
  - Auto-detects model format (GGUF -> llama.cpp, safetensors -> vLLM)
  - Use `--backend` to explicitly select backend (e.g., `--backend ik_llama`, `--backend trtllm`)
  - Runs backends via Docker for reproducible results
- **vLLM backend**: OpenAI-compatible API, tensor parallelism, FP8, vision models
  - Auto-uses all GPUs (`--tensor-parallel 1` for single)
  - Scrapes Prometheus `/metrics` for detailed timing
- **TensorRT-LLM backend**: OpenAI-compatible API via `trtllm-serve`, NGC container images
  - Uses `nvcr.io/nvidia/tensorrt-llm/release:{version}` images
  - Automatic engine compilation and caching
- **llama.cpp backend**: Native `/completion` endpoint, GGUF models, GPU sharding
- **ik_llama.cpp backend**: ikawrakow's optimized fork of llama.cpp (same API, faster inference)
- **Shared metrics**: `wall_s`, `ttft_ms`, `generation_tok_per_s`, `prompt_tokens`, `generated_tokens`

### Key Modules
- **`bench_utils.py`**: GPU detection, model resolution, config loading, prompts, version resolution, backend registry
- **`server_manager.py`**: Server lifecycle, `start_vllm()`/`start_llama()`/`start_trtllm()` for Docker
- **`docker_manager.py`**: Image building/pulling, GPU Docker validation, command builders
- **`run_server.py`**: Standalone server runner for general inference (not tied to benchmarking)

## Troubleshooting

### Benchmark Connection Issues

If benchmark fails with "Server disconnected" or "Connection error":

**Quick checks:**
```bash
docker ps                                    # Check if server still running
docker logs <container-id> --tail 50         # Check for errors
curl http://localhost:8000/v1/models         # Test server health
```

**Common fixes:**

1. **Server still initializing** (large models take 60-120s):
   ```bash
   # Increase timeout (default: 360s)
   uv run python scripts/run_bench.py --model ~/models/... --server-timeout 600

   # Or connect to already-running server
   uv run python scripts/run_bench.py --model ~/models/... --no-autostart
   ```

2. **Check Docker logs for errors:**
   ```bash
   docker logs <container-id> 2>&1 | grep -i "error\|oom\|cuda"
   ```

3. **Stop stuck containers:**
   ```bash
   docker stop <container-id>
   # Or stop all: docker ps | grep vllm | awk '{print $1}' | xargs docker stop
   ```

**Platform note:** `nvidia-smi` and GPU commands only work on Linux, not macOS.
