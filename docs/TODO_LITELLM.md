# LiteLLM Integration

Add LiteLLM proxy support to expose standardized API endpoints across all backends.

## Goal

Run a LiteLLM proxy in front of any backend (vLLM, llama.cpp, TensorRT-LLM, SGLang) to provide **two standardized API endpoints**:

1. **OpenAI Chat Completions** (`/v1/chat/completions`) - for OpenAI SDK compatibility
2. **Anthropic Messages** (`/v1/messages`) - for Anthropic SDK compatibility

This allows any local model to be used as a drop-in replacement for OpenAI or Anthropic APIs, regardless of which backend is serving the model.

### Why This Matters

- **SDK compatibility**: Use the official OpenAI or Anthropic Python SDKs with local models
- **Tool/agent frameworks**: Many AI frameworks (LangChain, CrewAI, etc.) expect OpenAI or Anthropic APIs
- **Consistent interface**: Same client code works regardless of backend (vLLM, llama.cpp, TensorRT-LLM)
- **Model aliasing**: Use friendly names like `local-model` instead of full paths

### Supported Features

| Feature | OpenAI API | Anthropic API | Notes |
|---------|------------|---------------|-------|
| Chat completions | Yes | Yes | Standard text generation |
| Streaming | Yes | Yes | Server-sent events (SSE) |
| Tool/function calling | Yes | Yes | For agentic workflows |
| Vision/multimodal | Yes | Yes | Image inputs (base64 or URL) |
| System prompts | Yes | Yes | Via system message/parameter |

## Backend Feature Support

LiteLLM handles API translation, but **feature availability depends on the underlying backend and model**. Not all features work with all backends.

### Feature Matrix

| Feature | vLLM | llama.cpp | TensorRT-LLM |
|---------|------|----------------------|--------------|
| **Chat completions** | Yes | Yes | Yes |
| **Streaming** | Yes | Yes | Yes |
| **Tool/function calling** | Yes | Partial | Limited |
| **Vision/multimodal** | Yes | Yes | Limited |

### Feature Details

#### Streaming
All backends support streaming via Server-Sent Events (SSE). No special configuration needed.

#### Tool/Function Calling
Tool calling support varies significantly:

- **vLLM**: Native support for models trained with tool-calling capabilities (Llama 3.1+, Qwen 2.5, Mistral, etc.). Works out of the box with `--enable-auto-tool-choice`.
- **llama.cpp**: Uses grammar-constrained generation to force JSON output. Works but may be less reliable than native support. Requires models fine-tuned for tool use.
- **TensorRT-LLM**: Limited support, depends on model and TRT-LLM version. Not all tool-calling models are supported.

**Recommendation**: Use vLLM for agentic workloads requiring tool calling.

#### Vision/Multimodal
Vision support requires both a vision-capable model AND backend support:

- **vLLM**: Best VLM support. Works with LLaVA, Qwen-VL, Pixtral, InternVL, GLM-4V, and other vision models. Pass images as base64 or URLs in the `content` array.
- **llama.cpp**: Supports LLaVA and other GGUF vision models via `--mmproj` flag for the multimodal projector. Requires separate projector file.
- **TensorRT-LLM**: Limited vision model support. Check TRT-LLM docs for supported VLMs.

**Recommendation**: Use vLLM for vision workloads. For GGUF vision models, use llama.cpp with the appropriate `--mmproj` projector.

### Model Requirements

The features above require appropriate models:

| Feature | Model Requirements |
|---------|-------------------|
| Tool calling | Models trained for tool use: Llama 3.1+, Qwen 2.5, Mistral v0.3+, Hermes 2 Pro |
| Vision | Vision-language models: LLaVA, Qwen-VL, GLM-4V, Pixtral, InternVL |
| Streaming | Any model (universal support) |

### Anthropic API Translation Notes

When using the Anthropic Messages API (`/v1/messages`), LiteLLM translates requests to OpenAI format internally:

- `tool_use` blocks → OpenAI `tool_calls`
- `tool_result` blocks → OpenAI tool response messages
- Image content blocks → OpenAI image_url format

Most translations work transparently, but edge cases may behave differently than the real Anthropic API.

## Current Pain Points

| Backend | Model Name | API Endpoint | OpenAI SDK | Anthropic SDK |
|---------|------------|--------------|------------|---------------|
| vLLM | Full model path | `/v1/chat/completions` | Yes | No |
| llama.cpp | "gpt-3.5-turbo" | `/v1/chat/completions` | Yes | No |
| TensorRT-LLM | Full model path | `/v1/chat/completions` | Yes | No |

**With LiteLLM**, all backends expose both APIs:

```python
# OpenAI SDK
from openai import OpenAI
client = OpenAI(base_url="http://localhost:4000/v1", api_key="dummy")
response = client.chat.completions.create(model="local-model", messages=[...])

# Anthropic SDK
from anthropic import Anthropic
client = Anthropic(base_url="http://localhost:4000", api_key="dummy")
response = client.messages.create(model="local-model", messages=[...], max_tokens=1024)
```

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│  Host                                                      │
│                                                            │
│  ┌──────────────────┐      ┌──────────────────┐           │
│  │ Backend Container│      │ LiteLLM Container│           │
│  │ (vLLM/llama/etc) │◄─────│                  │◄────────  │ Client
│  │                  │      │                  │           │
│  │ port:8000        │      │ port:4000        │           │
│  │ (host network)   │      │ (host network)   │           │
│  └──────────────────┘      └──────────────────┘           │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

Both containers use host networking for simplicity. LiteLLM connects to backend via `localhost:{backend_port}`.

## Usage

### Starting the Server

```bash
# Start vLLM with LiteLLM proxy
uv run python scripts/run_server.py --model ~/models/GLM-FP8 --with-litellm

# Start llama.cpp with LiteLLM proxy
uv run python scripts/run_server.py --model ~/models/GLM-GGUF --with-litellm

# Start TensorRT-LLM with LiteLLM proxy
uv run python scripts/run_server.py --model ~/models/GLM-FP8 --backend trtllm --with-litellm

# Custom LiteLLM port and model alias
uv run python scripts/run_server.py --model ~/models/GLM-FP8 --with-litellm \
    --litellm-port 4000 --litellm-alias my-model
```

### Client Examples

**OpenAI Chat Completions API:**
```bash
curl http://localhost:4000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "local-model", "messages": [{"role": "user", "content": "Hello"}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:4000/v1", api_key="dummy")
response = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Hello"}]
)
```

**Anthropic Messages API:**
```bash
curl http://localhost:4000/v1/messages \
    -H "Content-Type: application/json" \
    -H "x-api-key: dummy" \
    -d '{"model": "local-model", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 1024}'
```

```python
from anthropic import Anthropic
client = Anthropic(base_url="http://localhost:4000", api_key="dummy")
response = client.messages.create(
    model="local-model",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=1024
)
```

## Implementation

### 1. CLI Arguments (`scripts/run_server.py`)

Add to existing argparse after line 132:

```python
# LiteLLM proxy options
litellm_group = parser.add_argument_group("LiteLLM proxy options")
litellm_group.add_argument(
    "--with-litellm",
    action="store_true",
    help="Run LiteLLM proxy in front of backend for normalized API",
)
litellm_group.add_argument(
    "--litellm-port",
    type=int,
    default=4000,
    help="Port for LiteLLM proxy (default: 4000)",
)
litellm_group.add_argument(
    "--litellm-alias",
    type=str,
    default="local-model",
    help="Model alias exposed by LiteLLM (default: local-model)",
)
litellm_group.add_argument(
    "--litellm-version",
    type=str,
    default=None,
    help="LiteLLM Docker image version (default: from config)",
)
```

### 2. Config Schema (`config/models.yaml`)

Add to `defaults.backends`:

```yaml
defaults:
  backends:
    # ... existing backends ...
    litellm:
      version: "1.56.5"  # LiteLLM version
      image: "ghcr.io/berriai/litellm"
      port: 4000
      alias: "local-model"
```

### 3. LiteLLM Config Generation (`scripts/litellm_config.py`)

New file to generate LiteLLM config YAML:

```python
"""Generate LiteLLM configuration for proxying to local backends."""

import tempfile
from pathlib import Path
import yaml


def generate_litellm_config(
    backend: str,
    backend_port: int,
    model_alias: str,
    model_name: str,
) -> Path:
    """
    Generate a LiteLLM config file for proxying to a local backend.

    Args:
        backend: Backend type (vllm, llama, trtllm, sglang)
        backend_port: Port the backend is listening on
        model_alias: Model name to expose via LiteLLM
        model_name: Actual model name/path used by backend

    Returns:
        Path to generated config file
    """
    # Map backend to LiteLLM provider
    if backend in ("vllm", "trtllm"):
        # OpenAI-compatible backends
        litellm_model = f"openai/{model_name}"
        api_base = f"http://localhost:{backend_port}/v1"
    elif backend == "llama":
        # llama.cpp uses custom endpoint, but LiteLLM supports it
        litellm_model = f"openai/{model_name}"
        api_base = f"http://localhost:{backend_port}/v1"
    else:
        raise ValueError(f"Unknown backend: {backend}")

    config = {
        "model_list": [
            {
                "model_name": model_alias,
                "litellm_params": {
                    "model": litellm_model,
                    "api_base": api_base,
                    "api_key": "dummy",
                },
            }
        ],
        "litellm_settings": {
            "drop_params": True,  # Drop unsupported params instead of erroring
            "set_verbose": False,
        },
    }

    # Write to temp file
    config_file = Path(tempfile.gettempdir()) / "litellm_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_file
```

### 4. Docker Command Builder (`scripts/docker_manager.py`)

Add function after `build_trtllm_docker_cmd()` (line ~414):

```python
def build_litellm_docker_cmd(
    config_path: Path,
    port: int = 4000,
    version: str = "1.56.5",
    image: str = "ghcr.io/berriai/litellm",
) -> list[str]:
    """
    Build docker command for LiteLLM proxy.

    Args:
        config_path: Path to LiteLLM config YAML
        port: Port to expose (default: 4000)
        version: LiteLLM version tag
        image: Base image name

    Returns:
        Docker run command as list
    """
    image_name = f"{image}:{version}"

    cmd = [
        "docker", "run", "--rm",
        "--network", "host",  # Use host network to reach backend on localhost
        "-v", f"{config_path}:/app/config.yaml:ro",
        "--name", "litellm-proxy",
        image_name,
        "--config", "/app/config.yaml",
        "--port", str(port),
        "--host", "0.0.0.0",
    ]

    return cmd
```

### 5. Server Manager (`scripts/server_manager.py`)

Add method to `ServerManager` class after `start_trtllm()` (line ~431):

```python
def start_litellm_proxy(
    self,
    config_path: Path,
    port: int = 4000,
    version: str = "1.56.5",
    image: str = "ghcr.io/berriai/litellm",
    timeout: int = 60,
) -> bool:
    """
    Start LiteLLM proxy container.

    Args:
        config_path: Path to LiteLLM config YAML
        port: Port to expose
        version: LiteLLM version
        image: Docker image
        timeout: Startup timeout in seconds

    Returns:
        True if started successfully
    """
    from docker_manager import build_litellm_docker_cmd

    cmd = build_litellm_docker_cmd(
        config_path=config_path,
        port=port,
        version=version,
        image=image,
    )

    def check_ready():
        return wait_for_litellm_ready(port=port, timeout=5)

    return self._start_generic(
        cmd=cmd,
        readiness_check=check_ready,
        stream_stderr=True,
        label="LiteLLM",
        start_timeout=timeout,
        ready_timeout=timeout,
    )
```

Add readiness check function after `wait_for_llama_ready()` (line ~456):

```python
def wait_for_litellm_ready(
    port: int = 4000,
    host: str = "localhost",
    timeout: int = 30,
) -> bool:
    """
    Wait for LiteLLM proxy to be ready.

    Args:
        port: LiteLLM port
        host: Host address
        timeout: Max wait time in seconds

    Returns:
        True if ready, False if timeout
    """
    import httpx

    url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            resp = httpx.get(url, timeout=2)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)

    return False
```

### 6. Main Integration (`scripts/run_server.py`)

Update `main()` function to handle `--with-litellm`:

```python
def main():
    args = parse_args()
    # ... existing setup code ...

    # Track processes for cleanup
    litellm_manager = None

    try:
        # Start backend as usual
        if backend == "vllm":
            manager.start_vllm(...)
        elif backend == "llama":
            manager.start_gguf_backend(...)
        elif backend == "trtllm":
            manager.start_trtllm(...)

        # Start LiteLLM proxy if requested
        if args.with_litellm:
            from litellm_config import generate_litellm_config

            config_path = generate_litellm_config(
                backend=backend,
                backend_port=port,
                model_alias=args.litellm_alias,
                model_name=model_name,
            )

            litellm_version = args.litellm_version or get_litellm_version(config)

            litellm_manager = ServerManager()
            litellm_manager.start_litellm_proxy(
                config_path=config_path,
                port=args.litellm_port,
                version=litellm_version,
            )

            print(f"LiteLLM proxy ready at http://localhost:{args.litellm_port}")
            print(f"Model alias: {args.litellm_alias}")

        # ... rest of existing code ...

    finally:
        if litellm_manager:
            litellm_manager.stop()
        manager.stop()
```

### 7. Helper Function (`scripts/bench_utils.py`)

Add version resolution for LiteLLM:

```python
def get_litellm_version(config: dict) -> str:
    """Get LiteLLM version from config or default."""
    try:
        return config["defaults"]["backends"]["litellm"]["version"]
    except KeyError:
        return "1.56.5"  # Fallback default
```

## Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Add `--with-litellm`, `--litellm-port`, `--litellm-alias`, `--litellm-version` to `run_server.py`
- [ ] Create `scripts/litellm_config.py` with `generate_litellm_config()`
- [ ] Add `build_litellm_docker_cmd()` to `docker_manager.py`
- [ ] Add `start_litellm_proxy()` to `server_manager.py`
- [ ] Add `wait_for_litellm_ready()` to `server_manager.py`
- [ ] Add `get_litellm_version()` to `bench_utils.py`

### Phase 2: Integration
- [ ] Update `run_server.py` main() to orchestrate backend + LiteLLM
- [ ] Handle dual process lifecycle (cleanup both on exit)
- [ ] Add LiteLLM config to `config/models.yaml`

### Phase 3: Testing & Docs
- [ ] Test with vLLM backend
- [ ] Test with llama.cpp backend
- [ ] Test with TensorRT-LLM backend
- [ ] Update CLAUDE.md with LiteLLM usage examples

## Config Schema

```yaml
defaults:
  backends:
    vllm:
      version: v0.12.0
      image_type: prebuilt
      args:
        gpu_memory_utilization: 0.95
        max_model_len: 65536
    llama:
      version: b7349
      image_type: build
      args:
        n_gpu_layers: 999
    trtllm:
      version: "0.18.0"
      image_type: prebuilt
    litellm:
      version: "1.56.5"
      image: "ghcr.io/berriai/litellm"
      port: 4000
      alias: "local-model"
```

## References

- LiteLLM Docs: https://docs.litellm.ai/
- LiteLLM Docker: https://docs.litellm.ai/docs/proxy/docker_deploy
- LiteLLM Config: https://docs.litellm.ai/docs/proxy/configs
- LiteLLM Health Check: https://docs.litellm.ai/docs/proxy/health
