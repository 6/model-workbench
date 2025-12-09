"""Docker image management for vLLM and llama.cpp backends."""

import subprocess
from pathlib import Path

from bench_utils import ROOT, log


# Image naming convention
VLLM_IMAGE_PREFIX = "model-bench-vllm"
LLAMA_IMAGE_PREFIX = "model-bench-llama"

# Dockerfile locations
DOCKERFILE_VLLM = ROOT / "docker" / "Dockerfile.vllm"
DOCKERFILE_LLAMA = ROOT / "docker" / "Dockerfile.llama"


def get_image_name(engine: str, version: str) -> str:
    """Get Docker image name for engine and version.

    Args:
        engine: 'vllm' or 'llama'
        version: Version tag or commit SHA

    Returns:
        Full image name (e.g., 'model-bench-vllm:v0.8.0')
    """
    prefix = VLLM_IMAGE_PREFIX if engine == "vllm" else LLAMA_IMAGE_PREFIX
    return f"{prefix}:{version}"


def image_exists(engine: str, version: str) -> bool:
    """Check if Docker image exists locally.

    Args:
        engine: 'vllm' or 'llama'
        version: Version tag or commit SHA

    Returns:
        True if image exists locally
    """
    image_name = get_image_name(engine, version)
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image_name],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def build_image(engine: str, version: str, force: bool = False) -> str:
    """Build Docker image for engine and version.

    Args:
        engine: 'vllm' or 'llama'
        version: Version tag or commit SHA
        force: If True, rebuild even if image exists

    Returns:
        Image name that was built

    Raises:
        SystemExit if build fails
    """
    image_name = get_image_name(engine, version)

    if not force and image_exists(engine, version):
        log(f"Image {image_name} already exists (use --rebuild to force)")
        return image_name

    dockerfile = DOCKERFILE_VLLM if engine == "vllm" else DOCKERFILE_LLAMA

    if not dockerfile.exists():
        raise SystemExit(f"Dockerfile not found: {dockerfile}")

    log(f"Building {image_name} from {dockerfile}...")
    log(f"This may take several minutes for first build...")

    cmd = [
        "docker", "build",
        "-f", str(dockerfile),
        "--build-arg", f"VERSION={version}",
        "-t", image_name,
        str(ROOT),
    ]

    log(f"+ {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            timeout=3600,  # 1 hour timeout for builds
        )
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"Docker build failed with exit code {e.returncode}")
    except subprocess.TimeoutExpired:
        raise SystemExit("Docker build timed out (>1 hour)")

    log(f"Successfully built {image_name}")
    return image_name


def docker_gpu_available() -> bool:
    """Check if Docker with GPU support is available.

    Returns:
        True if Docker can access GPUs
    """
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all",
             "nvidia/cuda:12.4.0-base-ubuntu22.04", "nvidia-smi"],
            capture_output=True,
            timeout=60,
        )
        return result.returncode == 0
    except Exception:
        return False


def build_vllm_docker_cmd(
    image_name: str,
    model_path: str,
    host: str,
    port: int,
    tensor_parallel: int,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
    max_num_batched_tokens: int | None = None,
    extra_vllm_args: list[str] | None = None,
) -> list[str]:
    """Build Docker run command for vLLM server.

    Args:
        image_name: Docker image to use
        model_path: Path to model directory (will be bind-mounted)
        host: Server host (inside container)
        port: Server port
        tensor_parallel: Tensor parallel size
        max_model_len: Max context length (optional)
        gpu_memory_utilization: GPU memory fraction (optional)
        max_num_batched_tokens: Max batched tokens (optional)
        extra_vllm_args: Additional vLLM arguments (optional)

    Returns:
        Docker run command list
    """
    model_path_resolved = str(Path(model_path).expanduser().resolve())

    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--ipc", "host",
        "-p", f"{port}:{port}",
        "-v", f"{model_path_resolved}:{model_path_resolved}:ro",
        image_name,
        "--model", model_path_resolved,
        "--host", "0.0.0.0",  # Bind to all interfaces inside container
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel),
    ]

    if max_model_len is not None:
        cmd += ["--max-model-len", str(max_model_len)]

    if gpu_memory_utilization is not None:
        cmd += ["--gpu-memory-utilization", str(gpu_memory_utilization)]

    if max_num_batched_tokens is not None:
        cmd += ["--max-num-batched-tokens", str(max_num_batched_tokens)]

    # Model-specific flags (detect from path)
    lower = model_path.lower()

    # GLM vision models
    if "glm" in lower and "v" in lower.split("glm")[-1]:
        cmd += [
            "--enable-expert-parallel",
            "--allowed-local-media-path", "/",
            "--mm-encoder-tp-mode", "data",
            "--mm_processor_cache_type", "shm",
        ]

    # Mistral/Devstral models
    if "mistral" in lower or "devstral" in lower or "ministral" in lower:
        cmd += [
            "--tokenizer_mode", "mistral",
            "--config_format", "mistral",
            "--load_format", "mistral",
        ]

    if extra_vllm_args:
        cmd += extra_vllm_args

    return cmd


def build_llama_docker_cmd(
    image_name: str,
    gguf_path: str,
    port: int,
    n_gpu_layers: int | None = None,
    ctx: int | None = None,
    parallel: int | None = None,
    mmproj_path: str | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build Docker run command for llama-server.

    Args:
        image_name: Docker image to use
        gguf_path: Path to GGUF model file (will be bind-mounted)
        port: Server port
        n_gpu_layers: GPU layers to offload (optional)
        ctx: Context length (optional)
        parallel: Parallel sequences (optional)
        mmproj_path: Path to multimodal projector (optional)
        extra_args: Additional llama-server arguments (optional)

    Returns:
        Docker run command list
    """
    gguf_path_resolved = str(Path(gguf_path).expanduser().resolve())
    model_dir = str(Path(gguf_path_resolved).parent)

    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "-p", f"{port}:{port}",
        "-v", f"{model_dir}:{model_dir}:ro",
    ]

    # Also mount mmproj directory if different
    if mmproj_path:
        mmproj_resolved = str(Path(mmproj_path).expanduser().resolve())
        mmproj_dir = str(Path(mmproj_resolved).parent)
        if mmproj_dir != model_dir:
            cmd += ["-v", f"{mmproj_dir}:{mmproj_dir}:ro"]

    cmd += [
        image_name,
        "-m", gguf_path_resolved,
        "--host", "0.0.0.0",
        "--port", str(port),
    ]

    if n_gpu_layers is not None:
        cmd += ["-ngl", str(n_gpu_layers)]

    if ctx is not None:
        cmd += ["-c", str(ctx)]

    if parallel is not None and parallel > 1:
        cmd += ["-np", str(parallel)]

    if mmproj_path:
        mmproj_resolved = str(Path(mmproj_path).expanduser().resolve())
        cmd += ["--mmproj", mmproj_resolved]

    if extra_args:
        cmd += extra_args

    return cmd


def ensure_image(engine: str, version: str, rebuild: bool = False) -> str:
    """Ensure Docker image exists, building if necessary.

    Args:
        engine: 'vllm' or 'llama'
        version: Version tag or commit SHA
        rebuild: Force rebuild even if exists

    Returns:
        Image name

    Raises:
        SystemExit if GPU Docker not available or build fails
    """
    if not docker_gpu_available():
        raise SystemExit(
            "Docker with GPU support not available.\n"
            "Ensure nvidia-container-toolkit is installed and Docker can access GPUs:\n"
            "  docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi"
        )

    return build_image(engine, version, force=rebuild)
