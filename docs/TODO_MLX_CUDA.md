# TODO: MLX CUDA Backend Support

**Status**: Blocked - pip wheels don't properly enable GPU
**Last Updated**: 2025-12-21

## Overview

[MLX](https://github.com/ml-explore/mlx) is Apple's ML framework, primarily designed for Apple Silicon. It now has experimental CUDA support for NVIDIA GPUs on Linux.

## Current System

- 2x NVIDIA RTX PRO 6000 Blackwell (~98GB VRAM each, ~196GB total)
- CUDA 13.0, Driver 580.95.05
- Ubuntu Linux

## Implementation Progress

### Backend Code Added

The following files were created/modified to add MLX backend support:

| File | Status | Description |
|------|--------|-------------|
| `docker/Dockerfile.mlx` | ✅ Created | Docker image for mlx_lm.server |
| `scripts/common.py` | ✅ Modified | Added `mlx` to BACKEND_REGISTRY |
| `scripts/docker_manager.py` | ✅ Modified | Added `build_mlx_docker_cmd()` |
| `scripts/server_manager.py` | ✅ Modified | Added `start_mlx()` method |
| `scripts/run_bench.py` | ✅ Modified | Added MLX benchmark support |
| `scripts/run_server.py` | ✅ Modified | Added MLX standalone server |
| `scripts/bench_utils.py` | ✅ Modified | Backend resolution fix, warmup support |
| `config/models.yaml` | ✅ Modified | Added mlx defaults and test model |

### What Works

- Docker image builds successfully
- mlx_lm.server starts and becomes ready (~5s)
- Backend selection from `config/models.yaml` works

### What Doesn't Work

The first inference request hangs indefinitely (3.5+ min on a 1B model that should take <5s), indicating MLX is **not using the GPU** and is falling back to CPU inference.

## Installation Issues

### CUDA 13 (`mlx[cuda13]`) - BROKEN

```bash
pip install "mlx[cuda13]==0.30.1"
```

**Error**: Deprecated package names
```
ERROR: Failed building wheel for nvidia-cublas-cu13
⚠️ THIS PROJECT 'nvidia-cublas-cu13' IS DEPRECATED.
Please use 'nvidia-cublas' instead.
```

The CUDA 13 extra pulls deprecated NVIDIA packages that refuse to install.

### CUDA 12 (`mlx[cuda12]`) - BROKEN

```bash
pip install "mlx[cuda12]==0.30.1"
```

**Error**: GPU not used, CPU fallback
- Installation succeeds
- Server starts successfully
- First inference hangs (3.5+ min on 1B model)
- Appears to run on CPU instead of GPU

The pip wheels install but don't actually enable GPU acceleration. The `libmlx.so` shared library may not be compiled with CUDA support in the pre-built wheels.

## Current Limitations

### Multi-GPU Not Supported on CUDA

Per [MLX maintainer (July 2025)](https://github.com/ml-explore/mlx-examples/issues/1378):

> "Right now `mx.distributed` doesn't support the CUDA back-end but we plan to update it in the future."

### Quantization Not Supported on CUDA

Per [GitHub issue #2536](https://github.com/ml-explore/mlx/issues/2536), the CUDA backend only supports BF16 and FP32 models. Quantized models (4-bit, 8-bit) are not supported.

Use full-precision models like `mlx-community/gemma-3-1b-it-bf16` instead of quantized variants.

### Pip Wheels Don't Enable GPU

The PyPI wheels for MLX don't appear to have working CUDA GPU support on Linux. Options:
1. Build MLX from source with `-DMLX_BUILD_CUDA=ON`
2. Wait for improved pip wheel support

## Potential Fix: Build from Source

If pip wheels don't work, MLX can be built from source with CUDA:

```dockerfile
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    cmake ninja-build git

# Clone and build MLX with CUDA
RUN git clone --depth 1 --branch v0.30.1 https://github.com/ml-explore/mlx.git /tmp/mlx \
    && cd /tmp/mlx \
    && pip3 install nanobind \
    && mkdir build && cd build \
    && cmake .. -G Ninja -DMLX_BUILD_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
    && ninja \
    && ninja install \
    && pip3 install /tmp/mlx \
    && rm -rf /tmp/mlx

RUN pip3 install mlx-lm
```

**Note**: This approach is untested and may have additional issues.

## Alternatives for Now

For models requiring GPU inference, use existing backends:

| Backend | Multi-GPU | Format | Status |
|---------|-----------|--------|--------|
| vLLM | ✅ TP | safetensors | Production-ready |
| SGLang | ✅ TP | safetensors | Production-ready |
| llama.cpp | ✅ sharding | GGUF | Production-ready |
| TensorRT-LLM | ✅ TP | safetensors | Production-ready |
| MLX | ❌ single | mlx | **Broken on Linux/CUDA** |

## Action Items

- [ ] Monitor MLX CUDA pip wheel fixes: https://github.com/ml-explore/mlx/issues
- [ ] Test building from source with `-DMLX_BUILD_CUDA=ON`
- [ ] Consider removing MLX backend code until pip wheels work

## References

- [MLX Installation Docs](https://ml-explore.github.io/mlx/build/html/install.html)
- [MLX CUDA Backend PR](https://github.com/ml-explore/mlx/pull/2433)
- [GitHub Issue: Quantization not supported](https://github.com/ml-explore/mlx/issues/2536)
- [mlx-community Models](https://huggingface.co/mlx-community)
