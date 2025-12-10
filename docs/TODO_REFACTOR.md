# Model Workbench Refactor Roadmap

Future enhancements beyond the current single-backend architecture.

## 1. Multi-Backend Support

**Goal**: Run the same model on different backends for comparison.

### Current State
- Backend is auto-detected from model format (GGUF -> llama.cpp, safetensors -> vLLM)
- No way to explicitly choose backend

### Proposed Changes
- Add `--backend` CLI flag: `--backend vllm|llama|trtllm`
- For models available in multiple formats, allow explicit backend selection
- Example: `run_bench.py --model ~/models/qwen --backend llama`

### Implementation
1. Add `--backend` argument to `run_bench.py`, `run_server.py`, `run_eval.py`
2. When `--backend` is specified, skip auto-detection
3. Validate backend is compatible with model format

---

## 2. Pre-Built Image Support

**Goal**: Use official Docker images alongside build-from-source.

### Current State
- All images built from source via `docker/Dockerfile.vllm` and `docker/Dockerfile.llama`
- Version pinning via git checkout

### Proposed Changes
- Support pre-built images like `vllm/vllm-openai:nightly`, `vllm/vllm-openai:v0.8.0`
- Add `image_type` config: `build` (current) vs `prebuilt`

### Config Schema Extension
```yaml
defaults:
  vllm_version: v0.8.0
  vllm_image_type: prebuilt  # or 'build'
  # When prebuilt: uses vllm/vllm-openai:v0.8.0
  # When build: uses docker/Dockerfile.vllm with VERSION=v0.8.0
```

### CLI Override
```bash
# Use official image directly
run_server.py --model ... --image vllm/vllm-openai:nightly

# Force build from source even if prebuilt available
run_server.py --model ... --build-from-source
```

### Implementation
1. Add `--image` CLI flag for direct image specification
2. Update `docker_manager.py` to handle prebuilt images (skip build, just pull)
3. Add `--build-from-source` to force build behavior
4. Update config schema parsing in `bench_utils.py`

---

## 3. TensorRT-LLM Backend Integration

**Goal**: Add TensorRT-LLM as a third backend option.

### Key Considerations
- TensorRT-LLM requires model compilation step (engine building)
- Different from vLLM/llama.cpp which load models directly
- Need to handle engine caching

### Implementation Steps
1. Create `docker/Dockerfile.trtllm` with TensorRT-LLM
2. Add `start_trtllm()` method to `ServerManager`
3. Add `build_trtllm_docker_cmd()` to `docker_manager.py`
4. Handle engine compilation caching (store in `~/.cache/trtllm-engines/`)
5. Add `--backend trtllm` flag
6. Add readiness check for TensorRT-LLM server

### Config Addition
```yaml
defaults:
  trtllm_version: v0.16.0
```

### References
- https://nvidia.github.io/TensorRT-LLM/
- TensorRT-LLM uses Triton Inference Server for serving

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

### Multi-Backend Support
- [ ] Add `--backend` flag to `run_bench.py`
- [ ] Add `--backend` flag to `run_server.py`
- [ ] Add `--backend` flag to `run_eval.py`
- [ ] Validate backend vs model format compatibility
- [ ] Update help text and examples

### Pre-Built Images
- [ ] Add `--image` flag for direct image specification
- [ ] Update `docker_manager.ensure_image()` to skip build for prebuilt
- [ ] Add `docker pull` for prebuilt images
- [ ] Add `image_type` to config schema
- [ ] Update `get_model_backend_version()` to return image info

### TensorRT-LLM
- [ ] Create `docker/Dockerfile.trtllm`
- [ ] Add `start_trtllm()` to `ServerManager`
- [ ] Add `build_trtllm_docker_cmd()` to `docker_manager.py`
- [ ] Implement engine caching strategy
- [ ] Add `run_benchmark_trtllm()` to `run_bench.py`
- [ ] Test with sample model

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
