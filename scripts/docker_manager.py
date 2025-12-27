"""Docker image management for vLLM, llama.cpp, and TensorRT-LLM backends."""

import os
import re
import subprocess
from pathlib import Path

from common import BACKEND_REGISTRY, ROOT, log


def get_image_name(engine: str, version: str, prebuilt: bool = False) -> str:
    """Get Docker image name for an engine.

    Args:
        engine: Backend name ('vllm', 'llama', 'trtllm', 'sglang', etc.)
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
        engine: 'vllm', 'llama', etc.
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
    ]
    if force:
        cmd.append("--no-cache")
    cmd.extend(
        [
            "--build-arg",
            f"VERSION={version}",
            "-t",
            image_name,
            str(ROOT),
        ]
    )

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
    env_vars: dict[str, str] | None = None,
) -> list[str]:
    """Build common Docker run prefix.

    Args:
        engine: Backend name (uses BACKEND_REGISTRY for base flags)
        image_name: Docker image to use
        port: Port to expose
        mounts: List of (host_path, container_path, mode) tuples
        env_vars: Environment variables to set in container

    Returns:
        Docker run command prefix (up to and including image name)
    """
    cfg = BACKEND_REGISTRY[engine]
    cmd = ["docker", "run", "--rm"]
    cmd.extend(cfg["docker_base"])
    cmd.extend(["-p", f"{port}:{port}"])
    for host_path, container_path, mode in mounts:
        cmd.extend(["-v", f"{host_path}:{container_path}:{mode}"])
    for key, val in (env_vars or {}).items():
        cmd.extend(["-e", f"{key}={val}"])
    cmd.append(image_name)
    return cmd


def _get_model_specific_args(model_path: str, backend: str = "vllm") -> list[str]:
    """Get model-specific args from config patterns.

    Args:
        model_path: Path to model (checked against patterns)
        backend: Backend name (vllm, sglang, etc.)

    Returns:
        List of additional CLI args
    """
    from bench_utils import get_backend_config

    cfg = get_backend_config(backend)
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
    cpu_offload_gb: float | None = None,
    max_num_seqs: int | None = None,
    env_vars: dict[str, str] | None = None,
    extra_vllm_args: list[str] | None = None,
) -> list[str]:
    """Build Docker run command for vLLM server."""
    model_path_resolved = str(Path(model_path).expanduser().resolve())

    cmd = _docker_run_base(
        "vllm",
        image_name,
        port,
        [(model_path_resolved, model_path_resolved, "ro")],
        env_vars=env_vars,
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
        "--trust-remote-code",
    ]

    if gpu_memory_utilization is not None:
        cmd += ["--gpu-memory-utilization", str(gpu_memory_utilization)]
    if max_num_batched_tokens is not None:
        cmd += ["--max-num-batched-tokens", str(max_num_batched_tokens)]
    if cpu_offload_gb is not None:
        cmd += ["--cpu-offload-gb", str(cpu_offload_gb)]
    if max_num_seqs is not None:
        cmd += ["--max-num-seqs", str(max_num_seqs)]

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
    repeat_penalty: float | None = None,
    repeat_last_n: int | None = None,
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
    cmd += ["-m", gguf_path_resolved, "--host", "0.0.0.0", "--port", str(port), "-v"]

    if n_gpu_layers is not None:
        cmd += ["-ngl", str(n_gpu_layers)]
    if ctx is not None:
        cmd += ["-c", str(ctx)]
    if parallel is not None and parallel > 1:
        cmd += ["-np", str(parallel)]
    if mmproj_path:
        cmd += ["--mmproj", str(Path(mmproj_path).expanduser().resolve())]
    if repeat_penalty is not None:
        cmd += ["--repeat-penalty", str(repeat_penalty)]
    if repeat_last_n is not None:
        cmd += ["--repeat-last-n", str(repeat_last_n)]
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

    cmd = _docker_run_base(
        "trtllm",
        image_name,
        port,
        [(model_path_resolved, model_path_resolved, "ro"), (cache_dir, "/root/.cache", "rw")],
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
    if extra_args:
        cmd += extra_args
    return cmd


def build_sglang_docker_cmd(
    image_name: str,
    model_path: str,
    host: str,
    port: int,
    tensor_parallel: int,
    mem_fraction_static: float | None = None,
    max_model_len: int | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build Docker run command for SGLang server (prebuilt images only)."""
    model_path_resolved = str(Path(model_path).expanduser().resolve())

    cmd = _docker_run_base(
        "sglang",
        image_name,
        port,
        [(model_path_resolved, model_path_resolved, "ro")],
    )

    # Disable DeepGEMM for Blackwell (SM120) - not supported, causes CUDA graph capture to fail
    # https://github.com/sgl-project/sglang/issues/12320
    cmd.insert(-1, "-e")
    cmd.insert(-1, "SGLANG_ENABLE_JIT_DEEPGEMM=0")

    # Prebuilt images need explicit command (no ENTRYPOINT)
    cmd += ["python", "-m", "sglang.launch_server"]

    cmd += [
        "--model-path",
        model_path_resolved,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tp",
        str(tensor_parallel),
        "--trust-remote-code",
        "--random-seed",  # Note: doesn't achieve determinism for MiMo, but no performance cost
        "0",
    ]

    # Model-specific flags from config
    cmd += _get_model_specific_args(model_path, "sglang")

    if mem_fraction_static is not None:
        cmd += ["--mem-fraction-static", str(mem_fraction_static)]
    if max_model_len is not None:
        cmd += ["--context-length", str(max_model_len)]
    if extra_args:
        cmd += extra_args
    return cmd


def build_exl_docker_cmd(
    image_name: str,
    model_path: str,
    host: str,
    port: int,
    cache_size: int | None = None,
    max_seq_len: int | None = None,
    gpu_split_auto: bool = True,
    gpu_split: list[int] | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build Docker run command for ExLlamaV3 via TabbyAPI server.

    Args:
        image_name: Docker image to use
        model_path: Path to EXL3 model directory
        host: Host to bind to
        port: Port to expose
        cache_size: Cache size in tokens (optional)
        max_seq_len: Maximum sequence length (optional)
        gpu_split_auto: Enable automatic GPU splitting (default: True)
        gpu_split: Explicit GPU memory split in GB per GPU (e.g., [24, 24])
        extra_args: Additional TabbyAPI arguments (optional)
    """
    model_path_resolved = str(Path(model_path).expanduser().resolve())
    model_dir = str(Path(model_path_resolved).parent)
    model_name = Path(model_path_resolved).name

    # Build mounts list
    mounts = [(model_dir, model_dir, "ro")]

    cmd = _docker_run_base(
        "exl",
        image_name,
        port,
        mounts,
    )
    cmd += [
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--disable-auth",
        "true",  # Disable token auth for local benchmarking
        "--model-dir",
        model_dir,
        "--model-name",
        model_name,
        "--backend",
        "exllamav3",
    ]

    if cache_size is not None:
        cmd += ["--cache-size", str(cache_size)]
    if max_seq_len is not None:
        cmd += ["--max-seq-len", str(max_seq_len)]

    # GPU split settings via CLI args (config.yml parsing has issues)
    if gpu_split:
        # Explicit split: --gpu-split 24 24
        cmd += ["--gpu-split"] + [str(g) for g in gpu_split]
        cmd += ["--gpu-split-auto", "false"]
    else:
        # Auto split
        cmd += ["--gpu-split-auto", str(gpu_split_auto).lower()]

    if extra_args:
        cmd += extra_args
    return cmd


def build_ktransformers_docker_cmd(
    image_name: str,
    model_path: str,
    host: str,
    port: int,
    tensor_parallel: int | None = None,
    cpu_threads: int | None = None,
    numa_nodes: int | None = None,
    kt_method: str | None = None,
    max_new_tokens: int | None = None,
    cache_lens: int | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build Docker run command for KTransformers server.

    Args:
        image_name: Docker image to use
        model_path: Path to safetensors model directory
        host: Host to bind to
        port: Port to expose
        tensor_parallel: Number of GPUs for tensor parallelism
        cpu_threads: CPU threads for inference (default: auto-detect)
        numa_nodes: Number of NUMA nodes (default: 1)
        kt_method: KTransformers method (e.g., "FP8" for native FP8 weights)
        max_new_tokens: Maximum new tokens to generate
        cache_lens: KV cache length in tokens
        extra_args: Additional ktransformers arguments (optional)
    """
    model_path_resolved = str(Path(model_path).expanduser().resolve())

    cmd = _docker_run_base(
        "ktransformers",
        image_name,
        port,
        [(model_path_resolved, model_path_resolved, "ro")],
    )

    # Use SGLang's launch_server with ktransformers integration
    # (kvcache-ai's SGLang fork has kt-kernel integration built-in)
    cmd += [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path_resolved,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--trust-remote-code",
    ]

    # Only add tensor parallel if explicitly set (SGLang will auto-detect GPUs)
    if tensor_parallel is not None and tensor_parallel > 0:
        cmd += ["--tp", str(tensor_parallel)]

    if cpu_threads is not None:
        cmd += ["--kt-cpuinfer", str(cpu_threads)]
    if numa_nodes is not None:
        cmd += ["--kt-threadpool-count", str(numa_nodes)]
    if kt_method is not None:
        cmd += ["--kt-method", kt_method]
    if cache_lens is not None:
        cmd += ["--max-total-tokens", str(cache_lens)]
    if extra_args:
        cmd += extra_args
    return cmd


def generate_tabby_config(
    gpu_split_auto: bool = True,
    gpu_split: list[float] | None = None,
    autosplit_reserve: list[int] | None = None,
) -> str:
    """Generate a TabbyAPI config.yml file and return its path.

    Args:
        gpu_split_auto: Enable automatic GPU splitting (ignored if gpu_split provided)
        gpu_split: Explicit GPU memory split in GB per GPU (e.g., [24, 24])
        autosplit_reserve: VRAM to reserve per GPU in MB (default: [96])

    Returns:
        Path to the generated config file
    """
    import tempfile

    import yaml

    if autosplit_reserve is None:
        autosplit_reserve = [96]

    config = {
        "autosplit_reserve": autosplit_reserve,
    }

    # Explicit gpu_split takes priority over gpu_split_auto
    if gpu_split:
        config["gpu_split"] = gpu_split
        config["gpu_split_auto"] = False  # Disable auto when explicit split provided
    else:
        config["gpu_split_auto"] = gpu_split_auto

    # Create a persistent temp file (not deleted on close)
    fd, path = tempfile.mkstemp(suffix=".yml", prefix="tabby_config_")
    with os.fdopen(fd, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return path


def ensure_image(
    engine: str,
    version: str,
    rebuild: bool = False,
    image_type: str = "build",
    image_override: str | None = None,
) -> str:
    """Ensure Docker image exists, building or pulling as needed.

    Args:
        engine: 'vllm', 'llama', 'trtllm', 'sglang', 'exl', etc.
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

    # Build from source (default for llama, optional for vllm)
    return _build_image(engine, version, force=rebuild)
