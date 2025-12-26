# TODO: KTransformers Backend

## Status: Implemented but Disabled (AMD CPU Incompatibility)

The ktransformers backend is implemented in the codebase but not documented for users because it requires **Intel CPUs** for the primary use case (CPU-GPU hybrid inference with FP8 weights).

## Hardware Requirements

ktransformers CPU offloading uses different backends depending on the `--kt-method`:

| --kt-method | Intel Haswell+ | Intel Skylake+ | Intel Sapphire+ | AMD Zen+ |
|-------------|----------------|----------------|-----------------|----------|
| LLAMAFILE   | Yes            | Yes            | Yes             | Yes      |
| RAWINT4     | No             | Yes            | Yes             | No       |
| FP8         | No             | Yes            | Yes             | No       |
| AMXINT4/8   | No             | No             | Yes             | No       |

**Key limitations for AMD CPUs:**
- FP8 method (used by MiniMax M2.1 native weights) requires Intel AVX512
- RAWINT4 method also requires Intel AVX512
- AMD only supports LLAMAFILE (requires GGUF weights) or BLIS (int8 only)

## When This Would Work

ktransformers is useful for models that don't fit entirely in GPU VRAM. To use it:

1. **Intel CPU required**: Skylake+ for FP8/RAWINT4, Sapphire Rapids+ for best performance (AMX)
2. **Large system RAM**: Overflow weights stored in system memory
3. **Model support**: Currently optimized for MoE models (MiniMax M2.1, DeepSeek V3)

## Code Already Implemented

The following code exists and is functional on Intel systems:

- `docker/Dockerfile.ktransformers` - Docker image with kt-kernel + SGLang
- `scripts/common.py` - Backend registry entry
- `scripts/docker_manager.py` - `build_ktransformers_docker_cmd()`
- `scripts/server_manager.py` - `start_ktransformers()`
- `scripts/run_bench.py` - Benchmark dispatch
- `scripts/run_server.py` - Server dispatch
- `config/models.yaml` - Default configuration

## To Enable for Users

1. Verify user has Intel CPU (Skylake+ for FP8, Sapphire Rapids+ for best performance)
2. Add ktransformers back to README.txt and CLAUDE.md
3. Test with MiniMax M2.1 or DeepSeek V3 on Intel hardware

## Example Usage (Intel Only)

```bash
# Start ktransformers server for MiniMax M2.1
uv run python scripts/run_server.py \
  --model ~/models/MiniMaxAI/MiniMax-M2.1 \
  --backend ktransformers \
  --test

# Run benchmark
uv run python scripts/run_bench.py \
  --model ~/models/MiniMaxAI/MiniMax-M2.1 \
  --backend ktransformers
```

## References

- [kt-kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md) - Hardware compatibility
- [MiniMax M2.1 Tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/MiniMax-M2.1-Tutorial.md)
- [SGLang + KTransformers Integration](https://lmsys.org/blog/2025-10-22-KTransformers/)
- [Issue #1753 - Cascadelake not supported](https://github.com/kvcache-ai/ktransformers/issues/1753)
