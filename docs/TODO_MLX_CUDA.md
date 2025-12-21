# TODO: MLX CUDA Backend Support

**Status**: Blocked on upstream multi-GPU support
**Last Updated**: 2025-12-21

## Overview

[MLX](https://github.com/ml-explore/mlx) is Apple's ML framework, primarily designed for Apple Silicon. It now has experimental CUDA support for NVIDIA GPUs on Linux.

## Current System

- 2x NVIDIA RTX PRO 6000 Blackwell (~98GB VRAM each, ~196GB total)
- CUDA 13.0, Driver 580.95.05
- Ubuntu Linux

## MLX CUDA Requirements

| Requirement | Our System | Status |
|-------------|------------|--------|
| NVIDIA SM 7.5+ | Blackwell | ✅ |
| Driver >= 550 (CUDA 12) or >= 580 (CUDA 13) | 580.95.05 | ✅ |
| CUDA 12.0+ or 13.0+ | 13.0 | ✅ |
| glibc >= 2.35 | Ubuntu 22.04+ | ✅ |
| Python >= 3.10 | 3.11+ | ✅ |

Installation:
```bash
pip install mlx[cuda13]  # or mlx[cuda12] for CUDA 12
pip install mlx-lm       # for LLM inference
```

## Current Limitations

### Multi-GPU Not Supported on CUDA

Per [MLX maintainer (July 2025)](https://github.com/ml-explore/mlx-examples/issues/1378):

> "Right now `mx.distributed` doesn't support the CUDA back-end but we plan to update it in the future."

The [distributed docs](https://ml-explore.github.io/mlx/build/html/usage/distributed.html) mention NCCL support, but this is not yet implemented for the CUDA backend.

### Implications

Models requiring >98GB VRAM cannot run on our system via MLX until multi-GPU support lands.

Example: [mlx-community/MiMo-V2-Flash-4bit](https://huggingface.co/mlx-community/MiMo-V2-Flash-4bit)
- 309B parameters at 4-bit ≈ 155GB VRAM
- Needs both GPUs with tensor parallelism
- **Cannot run on MLX CUDA currently**

## What Works Today

Single-GPU inference with models that fit in ~98GB:

```bash
pip install mlx[cuda13] mlx-lm
python -m mlx_lm.generate --model mlx-community/some-model-that-fits
```

## Alternatives for Multi-GPU

For models requiring multi-GPU tensor parallelism, use existing backends:

| Backend | Multi-GPU | Format | Notes |
|---------|-----------|--------|-------|
| vLLM | ✅ TP | safetensors | Production-ready, FP8 support |
| SGLang | ✅ TP | safetensors | Fast, good for large models |
| llama.cpp | ✅ sharding | GGUF | Layer-wise GPU sharding |
| TensorRT-LLM | ✅ TP | safetensors | NVIDIA-optimized |

## Action Items

- [ ] Monitor MLX CUDA multi-GPU progress: https://github.com/ml-explore/mlx/issues
- [ ] When multi-GPU lands, consider adding `mlx` backend to model-workbench
- [ ] For MiMo-V2-Flash specifically, find safetensors/GGUF version for vLLM/llama.cpp

## References

- [MLX Installation Docs](https://ml-explore.github.io/mlx/build/html/install.html)
- [MLX Distributed Docs](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)
- [GitHub Issue: Exo + mlx-cuda](https://github.com/ml-explore/mlx-examples/issues/1378)
- [mlx-community Models](https://huggingface.co/mlx-community)
