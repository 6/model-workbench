# Model Workbench Refactor Roadmap

Future enhancements beyond the current architecture.

## 1. Multi-Backend Support ✅ DONE

**Goal**: Run the same model on different backends for comparison.

### Implemented
- Added `--backend` CLI flag: `--backend vllm|llama|ik_llama`
- Auto-detection remains default (GGUF -> llama.cpp, safetensors -> vLLM)
- ik_llama.cpp support (ikawrakow's optimized fork)
- Backend registry in `bench_utils.py` for extensibility
- `docker/Dockerfile.ik_llama` for building ik_llama.cpp images

### Usage
```bash
# Auto-detect (default)
run_bench.py --model ~/models/unsloth/GLM-GGUF/UD-Q4_K_XL

# Explicit backend
run_bench.py --model ~/models/unsloth/GLM-GGUF/UD-Q4_K_XL --backend ik_llama
run_server.py --model ~/models/unsloth/GLM-GGUF/UD-Q4_K_XL --backend ik_llama
```

---

## 2. Pre-Built Image Support ✅ DONE

**Goal**: Use official Docker images alongside build-from-source.

### Implemented
- Support for pre-built images like `vllm/vllm-openai:v0.8.0`
- `--image-type` CLI flag: `prebuilt` (official images) vs `build` (from source)
- `--image` CLI flag for direct image specification (e.g., `--image vllm/vllm-openai:nightly`)
- Config defaults: `vllm_image_type: prebuilt` in `config/models.yaml`
- `docker_manager.py`: `pull_image()`, `get_prebuilt_image_name()`, updated `ensure_image()`

### Usage
```bash
# Use prebuilt (default if configured)
run_server.py --model ~/models/GLM-FP8

# Force build from source
run_server.py --model ~/models/GLM-FP8 --image-type build

# Direct image specification (highest priority)
run_server.py --model ~/models/GLM-FP8 --image vllm/vllm-openai:nightly
run_bench.py --model ~/models/GLM-FP8 --image vllm/vllm-openai:nightly
```

---

## 3. TensorRT-LLM Backend Integration ✅ DONE

**Goal**: Add TensorRT-LLM as a third backend option.

### Implemented
- `--backend trtllm` flag for all scripts (run_server.py, run_bench.py, run_eval.py)
- Uses NGC prebuilt images: `nvcr.io/nvidia/tensorrt-llm/release:{version}`
- `trtllm-serve` command with OpenAI-compatible API
- `start_trtllm()` method in `ServerManager`
- `build_trtllm_docker_cmd()` in `docker_manager.py`
- `run_benchmark_trtllm()` in `run_bench.py`
- Engine caching via `~/.cache` mount to container
- Config default: `trtllm_version: "0.18.0"`

### Usage
```bash
# Run benchmark with TensorRT-LLM
run_bench.py --model ~/models/GLM-FP8 --backend trtllm

# Start server
run_server.py --model ~/models/GLM-FP8 --backend trtllm

# Run evals
run_eval.py --model ~/models/GLM-FP8 --backend trtllm

# Specific version
run_bench.py --model ~/models/GLM-FP8 --backend trtllm --backend-version 0.18.0
```

### References
- https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html
- NGC Container: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release

---

## 4. LiteLLM Integration

**Goal**: Normalize response types across all backends.

### Current Pain Points
- vLLM and llama.cpp both expose OpenAI-compatible APIs, but with subtle differences
- Model name handling differs (vLLM uses path, llama.cpp uses "gpt-3.5-turbo")
- Response metadata varies
- Streaming behavior differs

### Proposed Solution
- Option to run LiteLLM proxy in front of backends
- Single consistent API regardless of backend
- Automatic model aliasing

### Implementation
1. Add optional `--use-litellm` flag to `run_server.py`
2. When enabled, start LiteLLM proxy container alongside backend
3. Configure LiteLLM to route to actual backend
4. Client connects to LiteLLM instead of backend directly
5. Add LiteLLM config generation based on backend type

### Config
```yaml
litellm:
  enabled: false
  port: 4000
  model_alias: "local-model"
```

### References
- https://docs.litellm.ai/

---

## 5. Configuration Refactor

**Goal**: More explicit configuration with sensible defaults.

### Current Config Schema
```yaml
defaults:
  vllm_version: v0.8.0
  llama_version: b4521

models:
  - repo_id: org/model
    backend_version: v0.8.1  # Override default
```

### Proposed Extended Schema
```yaml
defaults:
  backends:
    vllm:
      version: v0.8.0
      image_type: prebuilt  # 'prebuilt' uses official images, 'build' compiles from source
      default_args:
        gpu_memory_utilization: 0.95
        max_model_len: 65536
    llama:
      version: b4521
      image_type: build
      default_args:
        n_gpu_layers: 999
    trtllm:
      version: v0.16.0
      engine_cache: ~/.cache/trtllm-engines

models:
  - repo_id: org/model
    backend: vllm  # Explicit backend selection
    backend_version: v0.8.1
    backend_args:
      max_model_len: 32768
```

### Migration
- Old config format continues to work (backward compatible)
- New fields are optional additions

---

## Priority Order

1. **Multi-backend support** - Low effort, high value for benchmarking comparisons
2. **Pre-built image support** - Faster startup, useful for testing new vLLM releases
3. **LiteLLM** - Normalizes API differences, useful for general inference
4. **Configuration refactor** - Clean up before it gets more complex
5. **TensorRT-LLM** - Significant effort, but useful for production inference

---

## Implementation Checklist

### Multi-Backend Support ✅
- [x] Add `--backend` flag to `run_bench.py`
- [x] Add `--backend` flag to `run_server.py`
- [x] Add `--backend` flag to `run_eval.py`
- [x] Validate backend vs model format compatibility
- [x] Update help text and examples
- [x] Add ik_llama.cpp backend support

### Pre-Built Images ✅
- [x] Add `--image` flag for direct image specification
- [x] Update `docker_manager.ensure_image()` to skip build for prebuilt
- [x] Add `docker pull` for prebuilt images
- [x] Add `image_type` to config schema
- [x] Add `get_image_type()` to bench_utils.py

### TensorRT-LLM ✅
- [x] Use NGC prebuilt images (no Dockerfile needed)
- [x] Add `start_trtllm()` to `ServerManager`
- [x] Add `build_trtllm_docker_cmd()` to `docker_manager.py`
- [x] Implement engine caching (via ~/.cache mount)
- [x] Add `run_benchmark_trtllm()` to `run_bench.py`
- [x] Add trtllm support to `run_eval.py`

### LiteLLM
- [ ] Add `--use-litellm` flag to `run_server.py`
- [ ] Create LiteLLM config generation
- [ ] Handle LiteLLM container lifecycle
- [ ] Test response normalization
- [ ] Document model aliasing

### Configuration
- [ ] Define new config schema
- [ ] Update `bench_utils.py` parsing
- [ ] Migrate existing config
- [ ] Update CLAUDE.md documentation
