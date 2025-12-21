"""Docker image management for vLLM, llama.cpp, and TensorRT-LLM backends."""

import os
import re
import subprocess
from pathlib import Path

from common import BACKEND_REGISTRY, ROOT, log


def get_image_name(engine: str, version: str, prebuilt: bool = False) -> str:
    """Get Docker image name for an engine.

    Args:
        engine: Backend name ('vllm', 'llama', 'ik_llama', 'trtllm')
        version: Version tag or commit SHA
        prebuilt: True for prebuilt registry images, False for local builds

    Returns:
        Full image name with tag
    """
    cfg = BACKEND_REGISTRY.get(engine)
    if not cfg:
        raise SystemExit(f"Unknown engine: {engine}")

    if prebuilt:
        if not cfg["prebuilt_image"]:
            available = [k for k, v in BACKEND_REGISTRY.items() if v["prebuilt_image"]]
            raise SystemExit(
                f"No prebuilt images available for '{engine}'.\n"
                f"Engines with prebuilt images: {', '.join(available)}"
            )
        return f"{cfg['prebuilt_image']}:{version}"

    if not cfg["image_prefix"]:
        raise SystemExit(f"Engine '{engine}' only supports prebuilt images")
    return f"{cfg['image_prefix']}:{version}"


def _pull_image(image_name: str) -> bool:
    """Pull a Docker image from registry.

    Args:
        image_name: Full image name with tag

    Returns:
        True if pull succeeded, False otherwise
    """
    log(f"Pulling {image_name}...")
    try:
        subprocess.run(
            ["docker", "pull", image_name],
            check=True,
            timeout=1800,  # 30 min timeout for large images
        )
        log(f"Successfully pulled {image_name}")
        return True
    except subprocess.CalledProcessError as e:
        log(f"Failed to pull {image_name}: exit code {e.returncode}")
        return False
    except subprocess.TimeoutExpired:
        log(f"Timeout pulling {image_name}")
        return False


def _image_exists_local(image_name: str) -> bool:
    """Check if Docker image exists locally.

    Args:
        image_name: Full image name with tag

    Returns:
        True if image exists locally
    """
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image_name],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _build_image(engine: str, version: str, force: bool = False) -> str:
    """Build Docker image for engine and version.

    Args:
        engine: 'vllm', 'llama', or 'ik_llama'
        version: Version tag or commit SHA
        force: If True, rebuild even if image exists

    Returns:
        Image name that was built

    Raises:
        SystemExit if build fails
    """
    cfg = BACKEND_REGISTRY.get(engine)
    if not cfg or not cfg["dockerfile"]:
        raise SystemExit(f"Engine '{engine}' does not support building from source")

    image_name = get_image_name(engine, version)

    if not force and _image_exists_local(image_name):
        log(f"Image {image_name} already exists (use --rebuild to force)")
        return image_name

    dockerfile = cfg["dockerfile"]

    if not dockerfile.exists():
        raise SystemExit(f"Dockerfile not found: {dockerfile}")

    log(f"Building {image_name} from {dockerfile}...")
    log("This may take several minutes for first build...")

    cmd = [
        "docker",
        "build",
        "-f",
        str(dockerfile),
        "--build-arg",
        f"VERSION={version}",
        "-t",
        image_name,
        str(ROOT),
    ]

    log(f"+ {' '.join(cmd)}")

    try:
        subprocess.run(
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


def _docker_gpu_available() -> bool:
    """Check if Docker with GPU support is available.

    Returns:
        True if Docker can access GPUs
    """
    try:
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "nvidia/cuda:12.4.0-base-ubuntu22.04",
                "nvidia-smi",
            ],
            capture_output=True,
            timeout=60,
        )
        return result.returncode == 0
    except Exception:
        return False


def _docker_run_base(
    engine: str,
    image_name: str,
    port: int,
    mounts: list[tuple[str, str, str]],
) -> list[str]:
    """Build common Docker run prefix.

    Args:
        engine: Backend name (uses BACKEND_REGISTRY for base flags)
        image_name: Docker image to use
        port: Port to expose
        mounts: List of (host_path, container_path, mode) tuples

    Returns:
        Docker run command prefix (up to and including image name)
    """
    cfg = BACKEND_REGISTRY[engine]
    cmd = ["docker", "run", "--rm"]
    cmd.extend(cfg["docker_base"])
    cmd.extend(["-p", f"{port}:{port}"])
    for host_path, container_path, mode in mounts:
        cmd.extend(["-v", f"{host_path}:{container_path}:{mode}"])
    cmd.append(image_name)
    return cmd


def _get_model_specific_args(model_path: str) -> list[str]:
    """Get model-specific vLLM args from config patterns.

    Args:
        model_path: Path to model (checked against patterns)

    Returns:
        List of additional CLI args for vLLM
    """
    from bench_utils import get_backend_config

    cfg = get_backend_config("vllm")
    patterns = cfg.get("model_patterns", [])

    extra_args = []
    lower = model_path.lower()
    for p in patterns:
        if re.search(p["pattern"], lower, re.IGNORECASE):
            extra_args.extend(p["args"])
    return extra_args


def _get_trtllm_model_specific_args(model_path: str) -> list[str]:
    """Get model-specific TensorRT-LLM args from config patterns.

    Args:
        model_path: Path to model (checked against patterns)

    Returns:
        List of additional CLI args for trtllm-serve
    """
    from bench_utils import get_backend_config

    cfg = get_backend_config("trtllm")
    patterns = cfg.get("model_patterns", [])

    extra_args = []
    lower = model_path.lower()
    for p in patterns:
        if re.search(p["pattern"], lower, re.IGNORECASE):
            extra_args.extend(p["args"])
    return extra_args


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
    """Build Docker run command for vLLM server."""
    model_path_resolved = str(Path(model_path).expanduser().resolve())

    cmd = _docker_run_base(
        "vllm",
        image_name,
        port,
        [(model_path_resolved, model_path_resolved, "ro")],
    )
    cmd += [
        "--model",
        model_path_resolved,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel),
        "--max-model-len",
        str(max_model_len if max_model_len is not None else 10000),
    ]

    if gpu_memory_utilization is not None:
        cmd += ["--gpu-memory-utilization", str(gpu_memory_utilization)]
    if max_num_batched_tokens is not None:
        cmd += ["--max-num-batched-tokens", str(max_num_batched_tokens)]

    # Model-specific flags from config
    cmd += _get_model_specific_args(model_path)

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
    """Build Docker run command for llama-server."""
    gguf_path_resolved = str(Path(gguf_path).expanduser().resolve())
    model_dir = str(Path(gguf_path_resolved).parent)

    mounts = [(model_dir, model_dir, "ro")]
    if mmproj_path:
        mmproj_resolved = str(Path(mmproj_path).expanduser().resolve())
        mmproj_dir = str(Path(mmproj_resolved).parent)
        if mmproj_dir != model_dir:
            mounts.append((mmproj_dir, mmproj_dir, "ro"))

    cmd = _docker_run_base("llama", image_name, port, mounts)
    cmd += ["-m", gguf_path_resolved, "--host", "0.0.0.0", "--port", str(port)]

    if n_gpu_layers is not None:
        cmd += ["-ngl", str(n_gpu_layers)]
    if ctx is not None:
        cmd += ["-c", str(ctx)]
    if parallel is not None and parallel > 1:
        cmd += ["-np", str(parallel)]
    if mmproj_path:
        cmd += ["--mmproj", str(Path(mmproj_path).expanduser().resolve())]
    if extra_args:
        cmd += extra_args
    return cmd


def build_trtllm_docker_cmd(
    image_name: str,
    model_path: str,
    port: int,
    tensor_parallel: int,
    max_batch_size: int | None = None,
    max_num_tokens: int | None = None,
    max_seq_len: int | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build Docker run command for TensorRT-LLM trtllm-serve."""
    model_path_resolved = str(Path(model_path).expanduser().resolve())
    cache_dir = os.path.expanduser("~/.cache")

    # Build mounts list
    mounts = [
        (model_path_resolved, model_path_resolved, "ro"),
        (cache_dir, "/root/.cache", "rw"),
    ]

    # Mount config directory if it exists (for model-specific YAML configs)
    config_dir = ROOT / "config" / "trtllm"
    if config_dir.exists():
        mounts.append((str(config_dir), "/config", "ro"))

    cmd = _docker_run_base(
        "trtllm",
        image_name,
        port,
        mounts,
    )
    cmd += [
        "trtllm-serve",
        model_path_resolved,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tp_size",
        str(tensor_parallel),
    ]

    if max_batch_size is not None:
        cmd += ["--max_batch_size", str(max_batch_size)]
    if max_num_tokens is not None:
        cmd += ["--max_num_tokens", str(max_num_tokens)]
    if max_seq_len is not None:
        cmd += ["--max_seq_len", str(max_seq_len)]

    # Model-specific flags from config (e.g., --backend _autodeploy for NemotronH)
    cmd += _get_trtllm_model_specific_args(model_path)

    if extra_args:
        cmd += extra_args
    return cmd


def ensure_image(
    engine: str,
    version: str,
    rebuild: bool = False,
    image_type: str = "build",
    image_override: str | None = None,
) -> str:
    """Ensure Docker image exists, building or pulling as needed.

    Args:
        engine: 'vllm', 'llama', 'ik_llama', or 'trtllm'
        version: Version tag or commit SHA
        rebuild: Force rebuild/repull even if exists
        image_type: 'prebuilt' to use official images, 'build' to build from source
        image_override: Direct image name to use (highest priority, skips build/prebuilt logic)

    Returns:
        Image name

    Raises:
        SystemExit if GPU Docker not available or build/pull fails
    """
    if not _docker_gpu_available():
        raise SystemExit(
            "Docker with GPU support not available.\n"
            "Ensure nvidia-container-toolkit is installed and Docker can access GPUs:\n"
            "  docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi"
        )

    # Direct image override (highest priority)
    if image_override:
        if not _image_exists_local(image_override) or rebuild:
            if not _pull_image(image_override):
                raise SystemExit(f"Failed to pull image: {image_override}")
        else:
            log(f"Using existing image: {image_override}")
        return image_override

    # Prebuilt images (vLLM or TensorRT-LLM)
    cfg = BACKEND_REGISTRY.get(engine)
    use_prebuilt = image_type == "prebuilt" or (cfg and not cfg["dockerfile"])

    if use_prebuilt:
        image_name = get_image_name(engine, version, prebuilt=True)
        if not _image_exists_local(image_name) or rebuild:
            if not _pull_image(image_name):
                raise SystemExit(f"Failed to pull prebuilt image: {image_name}")
        else:
            log(f"Using existing prebuilt image: {image_name}")
        return image_name

    # Build from source (default for llama/ik_llama, optional for vllm)
    return _build_image(engine, version, force=rebuild)
