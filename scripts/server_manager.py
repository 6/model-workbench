"""Server lifecycle management for vLLM and llama.cpp servers."""

import select
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from bench_utils import port_open, resolve_local_gguf
from common import log

# ----------------------------
# Server Manager
# ----------------------------


def _get_gpu_memory_usage() -> str:
    """Get GPU memory usage from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            usages = []
            for i, line in enumerate(lines):
                used, total = line.split(",")
                usages.append(f"GPU{i}: {used.strip()}/{total.strip()}MB")
            return " | ".join(usages)
    except Exception:
        pass
    return ""


class ServerManager:
    """Manages local model server lifecycle with context manager support.

    Example:
        server = ServerManager(host="127.0.0.1", port=8000)
        with server:
            server.start(cmd, readiness_check, label="vLLM")
            # ... run benchmarks/evals
        # Server automatically stopped on exit
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        timeout: int = 180,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.proc: subprocess.Popen | None = None
        self._output_lines: list[str] = []
        self._we_started_it = False
        self.container_id: str | None = None  # Track Docker container ID for robust cleanup

    def is_running(self) -> bool:
        """Check if server is already running on port."""
        return port_open(self.host, self.port)

    def _get_container_id(self) -> str | None:
        """Extract Docker container ID for the port we're using.

        Returns container ID if found, None otherwise.
        """
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"publish={self.port}", "-q", "--no-trunc"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            container_ids = [
                cid.strip() for cid in result.stdout.strip().split("\n") if cid.strip()
            ]
            if container_ids:
                return container_ids[0]  # Return first match
        except Exception:
            pass
        return None

    def start(
        self,
        cmd: list[str],
        readiness_check: Callable[[], bool],
        stream_stderr: bool = False,
        label: str = "server",
        sigint_wait: float = 10,
        term_wait: float = 5,
    ) -> bool:
        """Start server and wait for readiness.

        Args:
            cmd: Command to run
            readiness_check: Function that returns True when server is ready
            stream_stderr: If True, stream stderr instead of stdout
            label: Label for log messages
            sigint_wait: Seconds to wait after SIGINT before terminate
            term_wait: Seconds to wait after terminate before kill

        Returns:
            True if we started the server, False if already running

        Raises:
            SystemExit if server fails to start or times out
        """
        if self.is_running():
            log(f"{label} already running on {self.host}:{self.port}")
            return False

        log(f"Starting {label}")
        log(f"+ {' '.join(cmd)}")

        # Configure output streaming
        if stream_stderr:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",  # Handle non-UTF-8 bytes gracefully
            )
            stream = self.proc.stderr
        else:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",  # Handle non-UTF-8 bytes gracefully
            )
            stream = self.proc.stdout

        self._we_started_it = True
        self._sigint_wait = sigint_wait
        self._term_wait = term_wait
        self._output_lines = []

        def read_output():
            """Non-blocking read of server output, print in real-time."""
            if stream:
                while select.select([stream], [], [], 0)[0]:
                    line = stream.readline()
                    if line:
                        self._output_lines.append(line)
                        print(f"  [{label}] {line.rstrip()}")
                    else:
                        break

        # Wait for server to be ready
        log(f"Waiting for {label} to be ready (timeout: {self.timeout}s)...")
        start_time = time.time()
        last_status_time = 0.0

        while time.time() - start_time < self.timeout:
            read_output()

            # Check if server is ready
            if self.is_running():
                try:
                    if readiness_check():
                        log(f"{label} ready in {time.time() - start_time:.1f}s")
                        # Capture container ID for robust cleanup
                        self.container_id = self._get_container_id()
                        return True
                except Exception:
                    pass

            # Show elapsed time and GPU memory every 10s
            elapsed = time.time() - start_time
            if elapsed - last_status_time >= 10:
                gpu_mem = _get_gpu_memory_usage()
                status_msg = f"Waiting... {elapsed:.0f}s elapsed"
                if gpu_mem:
                    status_msg += f" | {gpu_mem}"
                print(f"  [{label}] {status_msg}")
                last_status_time = elapsed

            time.sleep(2)

            # Check if process died
            if self.proc.poll() is not None:
                read_output()
                output = "".join(self._output_lines[-50:])
                raise SystemExit(f"{label} exited unexpectedly. Last output:\n{output}")

        # Timeout - show recent logs and cleanup
        read_output()
        self.stop()
        output = "".join(self._output_lines[-50:])
        raise SystemExit(f"{label} failed to start within {self.timeout}s.\nLast output:\n{output}")

    def stop(self):
        """Graceful shutdown: SIGINT → wait → terminate → wait → kill.

        Also ensures Docker container is stopped via container ID as fallback.
        """
        # First: try to stop via subprocess (original logic)
        if self.proc and self._we_started_it:
            sigint_wait = getattr(self, "_sigint_wait", 10)
            term_wait = getattr(self, "_term_wait", 5)

            try:
                self.proc.send_signal(signal.SIGINT)
                self.proc.wait(timeout=sigint_wait)
            except Exception:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=term_wait)
                except Exception:
                    self.proc.kill()

            self.proc = None
            self._we_started_it = False

        # Fallback: stop Docker container directly if we have container ID
        if self.container_id:
            try:
                log(f"Stopping Docker container {self.container_id[:12]}...")
                subprocess.run(
                    ["docker", "stop", self.container_id],
                    timeout=30,
                    capture_output=True,
                )
            except Exception as e:
                log(f"Warning: Failed to stop container: {e}")
            finally:
                self.container_id = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()

    def start_vllm(
        self,
        model_path: str,
        tensor_parallel: int,
        version: str,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | None = None,
        max_num_batched_tokens: int | None = None,
        cpu_offload_gb: float | None = None,
        max_num_seqs: int | None = None,
        env_vars: dict[str, str] | None = None,
        extra_vllm_args: list[str] | None = None,
        rebuild: bool = False,
        image_type: str = "build",
        image_override: str | None = None,
        pr_number: int | None = None,
        pr_overlay: bool = False,
    ) -> None:
        """Start vLLM server via Docker with version pinning.

        Args:
            model_path: Path to model directory
            tensor_parallel: Tensor parallel size
            version: vLLM version (release tag like 'v0.8.0' or commit SHA)
            max_model_len: Max context length (optional)
            gpu_memory_utilization: GPU memory fraction (optional)
            max_num_batched_tokens: Max batched tokens (optional)
            cpu_offload_gb: CPU offload in GB per GPU (optional)
            max_num_seqs: Max concurrent sequences (optional)
            env_vars: Environment variables for Docker container (optional)
            extra_vllm_args: Extra vLLM CLI arguments (optional)
            rebuild: Force rebuild image even if cached
            image_type: 'prebuilt' to use official images, 'build' to build from source
            image_override: Direct image name to use (highest priority)
            pr_number: PR number for unmerged PRs (optional)
            pr_overlay: If True, use nightly + overlay PR files (fast mode)
        """
        from docker_manager import (
            build_vllm_docker_cmd as build_versioned_vllm_cmd,
        )
        from docker_manager import (
            ensure_image,
        )

        # Ensure image exists (builds or pulls as needed)
        image_name = ensure_image(
            "vllm",
            version,
            rebuild=rebuild,
            image_type=image_type,
            image_override=image_override,
            pr_number=pr_number,
            pr_overlay=pr_overlay,
        )

        cmd = build_versioned_vllm_cmd(
            image_name=image_name,
            model_path=model_path,
            host=self.host,
            port=self.port,
            tensor_parallel=tensor_parallel,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_batched_tokens=max_num_batched_tokens,
            cpu_offload_gb=cpu_offload_gb,
            max_num_seqs=max_num_seqs,
            env_vars=env_vars,
            extra_vllm_args=extra_vllm_args,
        )

        # Label based on image source
        if image_override:
            label = f"vLLM ({image_override})"
        elif image_type == "prebuilt":
            label = f"vLLM (prebuilt {version})"
        else:
            label = f"vLLM (Docker {version})"

        self.start(
            cmd,
            lambda: wait_for_openai_ready(self.host, self.port),
            label=label,
        )

    def start_gguf_backend(
        self,
        engine: str,
        model_path: str,
        version: str,
        n_gpu_layers: int | None = None,
        ctx: int | None = None,
        parallel: int | None = None,
        mmproj_path: Path | None = None,
        repeat_penalty: float | None = None,
        repeat_last_n: int | None = None,
        # CPU offloading options
        jinja: bool | None = None,
        flash_attn: str | None = None,
        cache_type_k: str | None = None,
        cache_type_v: str | None = None,
        tensor_offload: list[str] | None = None,
        fit: bool | None = None,
        extra_args: list[str] | None = None,
        rebuild: bool = False,
    ) -> Path:
        """Start a GGUF backend (llama.cpp) via Docker.

        Args:
            engine: 'llama'
            model_path: Path to .gguf file or directory containing GGUF files
            version: Backend version (release tag or commit SHA)
            n_gpu_layers: GPU layers to offload (optional)
            ctx: Context length (optional)
            parallel: Parallel sequences (optional)
            mmproj_path: Path to multimodal projector (optional)
            repeat_penalty: Repetition penalty (optional, default 1.0)
            repeat_last_n: Tokens to consider for repetition penalty (optional)
            jinja: Enable Jinja template engine (optional)
            flash_attn: Flash attention mode "on" or "off" (optional)
            cache_type_k: KV cache K quantization type (optional)
            cache_type_v: KV cache V quantization type (optional)
            tensor_offload: List of tensor offload patterns (optional)
            fit: Enable auto-fit mode for GPU/CPU balancing (optional)
            extra_args: Extra raw args (optional)
            rebuild: Force rebuild image even if cached

        Returns:
            Path to resolved GGUF file
        """
        from docker_manager import (
            build_llama_docker_cmd,
            ensure_image,
        )

        # Resolve GGUF path
        gguf_path = resolve_local_gguf(model_path)
        if not gguf_path:
            raise SystemExit(
                f"No GGUF found at: {Path(model_path).expanduser()}\n"
                "Pass --model with an explicit path"
            )

        # Ensure image exists (builds if needed)
        image_name = ensure_image(engine, version, rebuild=rebuild)

        cmd = build_llama_docker_cmd(
            image_name=image_name,
            gguf_path=str(gguf_path),
            port=self.port,
            n_gpu_layers=n_gpu_layers,
            ctx=ctx,
            parallel=parallel,
            mmproj_path=str(mmproj_path) if mmproj_path else None,
            repeat_penalty=repeat_penalty,
            repeat_last_n=repeat_last_n,
            jinja=jinja,
            flash_attn=flash_attn,
            cache_type_k=cache_type_k,
            cache_type_v=cache_type_v,
            tensor_offload=tensor_offload,
            fit=fit,
            extra_args=extra_args,
        )

        self.start(
            cmd,
            lambda: wait_for_llama_ready(self.host, self.port),
            label=f"llama.cpp (Docker {version})",
        )

        return gguf_path

    def start_trtllm(
        self,
        model_path: str,
        tensor_parallel: int,
        version: str,
        max_batch_size: int | None = None,
        max_num_tokens: int | None = None,
        max_seq_len: int | None = None,
        rebuild: bool = False,
        extra_args: list[str] | None = None,
    ) -> None:
        """Start TensorRT-LLM server via Docker with NGC prebuilt image.

        Args:
            model_path: Path to model directory
            tensor_parallel: Tensor parallel size (tp_size)
            version: TensorRT-LLM version (NGC release tag like '0.18.0')
            max_batch_size: Maximum batch size (optional)
            max_num_tokens: Maximum number of tokens (optional)
            max_seq_len: Maximum sequence length (optional)
            rebuild: Force re-pull image even if cached
            extra_args: Additional trtllm-serve arguments (optional)
        """
        from docker_manager import (
            build_trtllm_docker_cmd,
            ensure_image,
        )

        # TensorRT-LLM always uses prebuilt NGC images
        image_name = ensure_image(
            "trtllm",
            version,
            rebuild=rebuild,
            image_type="prebuilt",
        )

        cmd = build_trtllm_docker_cmd(
            image_name=image_name,
            model_path=model_path,
            port=self.port,
            tensor_parallel=tensor_parallel,
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens,
            max_seq_len=max_seq_len,
            extra_args=extra_args,
        )

        self.start(
            cmd,
            lambda: wait_for_openai_ready(self.host, self.port),  # Uses OpenAI API like vLLM
            label=f"TensorRT-LLM ({version})",
        )

    def start_sglang(
        self,
        model_path: str,
        tensor_parallel: int,
        version: str,
        mem_fraction_static: float | None = None,
        max_model_len: int | None = None,
        rebuild: bool = False,
        image_override: str | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        """Start SGLang server via Docker (prebuilt images only).

        Args:
            model_path: Path to model directory
            tensor_parallel: Tensor parallel size
            version: SGLang version tag (e.g., 'nightly-dev-20251221-1d90b194')
            mem_fraction_static: Memory fraction for static allocation (optional)
            max_model_len: Max context length (optional)
            rebuild: Force re-pull image even if cached
            image_override: Direct image name to use (highest priority)
            extra_args: Additional SGLang server arguments (optional)
        """
        from docker_manager import (
            build_sglang_docker_cmd,
            ensure_image,
        )

        # Ensure image exists (pulls as needed) - SGLang is prebuilt-only
        image_name = ensure_image(
            "sglang",
            version,
            rebuild=rebuild,
            image_type="prebuilt",
            image_override=image_override,
        )

        cmd = build_sglang_docker_cmd(
            image_name=image_name,
            model_path=model_path,
            host=self.host,
            port=self.port,
            tensor_parallel=tensor_parallel,
            mem_fraction_static=mem_fraction_static,
            max_model_len=max_model_len,
            extra_args=extra_args,
        )

        label = f"SGLang ({image_override or version})"

        self.start(
            cmd,
            lambda: wait_for_openai_ready(self.host, self.port),  # SGLang uses OpenAI API
            label=label,
        )

    def start_exl(
        self,
        model_path: str,
        version: str,
        cache_size: int | None = None,
        max_seq_len: int | None = None,
        gpu_split_auto: bool = True,
        gpu_split: list[int] | None = None,
        rebuild: bool = False,
        extra_args: list[str] | None = None,
    ) -> None:
        """Start ExLlamaV3 server via TabbyAPI in Docker.

        Args:
            model_path: Path to EXL3 model directory
            version: ExLlamaV3 version (release tag like 'v0.0.18')
            cache_size: Cache size in tokens (optional)
            max_seq_len: Maximum sequence length (optional)
            gpu_split_auto: Enable automatic GPU splitting (default: True)
            gpu_split: Explicit GPU memory split in GB per GPU (e.g., [24, 24])
            rebuild: Force rebuild image even if cached
            extra_args: Additional TabbyAPI arguments (optional)
        """
        from docker_manager import (
            build_exl_docker_cmd,
            ensure_image,
        )

        model_path_resolved = str(Path(model_path).expanduser().resolve())

        # Ensure image exists (builds if needed)
        image_name = ensure_image("exl", version, rebuild=rebuild)

        cmd = build_exl_docker_cmd(
            image_name=image_name,
            model_path=model_path_resolved,
            host=self.host,
            port=self.port,
            cache_size=cache_size,
            max_seq_len=max_seq_len,
            gpu_split_auto=gpu_split_auto,
            gpu_split=gpu_split,
            extra_args=extra_args,
        )

        self.start(
            cmd,
            lambda: wait_for_openai_ready(self.host, self.port),  # TabbyAPI uses OpenAI API
            label=f"ExLlamaV3 (Docker {version})",
        )

    def start_ktransformers(
        self,
        model_path: str,
        tensor_parallel: int,
        version: str,
        cpu_threads: int | None = None,
        numa_nodes: int | None = None,
        kt_method: str | None = None,
        max_new_tokens: int | None = None,
        cache_lens: int | None = None,
        rebuild: bool = False,
        extra_args: list[str] | None = None,
    ) -> None:
        """Start KTransformers server via Docker for CPU-GPU hybrid inference.

        Args:
            model_path: Path to model directory (safetensors)
            tensor_parallel: Tensor parallel size (number of GPUs)
            version: KTransformers version (branch or commit SHA)
            cpu_threads: CPU threads for inference (default: auto-detect)
            numa_nodes: Number of NUMA nodes (default: 1)
            kt_method: KTransformers method (e.g., "FP8" for native FP8 weights)
            max_new_tokens: Maximum new tokens to generate
            cache_lens: KV cache length in tokens
            rebuild: Force rebuild image even if cached
            extra_args: Additional ktransformers arguments (optional)
        """
        from docker_manager import (
            build_ktransformers_docker_cmd,
            ensure_image,
        )

        model_path_resolved = str(Path(model_path).expanduser().resolve())

        # Ensure image exists (builds if needed) - KTransformers is build-from-source only
        image_name = ensure_image("ktransformers", version, rebuild=rebuild)

        cmd = build_ktransformers_docker_cmd(
            image_name=image_name,
            model_path=model_path_resolved,
            host=self.host,
            port=self.port,
            tensor_parallel=tensor_parallel,
            cpu_threads=cpu_threads,
            numa_nodes=numa_nodes,
            kt_method=kt_method,
            max_new_tokens=max_new_tokens,
            cache_lens=cache_lens,
            extra_args=extra_args,
        )

        self.start(
            cmd,
            lambda: wait_for_openai_ready(self.host, self.port),  # KTransformers uses OpenAI API
            label=f"KTransformers (Docker {version})",
        )


# ----------------------------
# Readiness checks
# ----------------------------


def wait_for_openai_ready(host: str, port: int) -> bool:
    """Readiness check for OpenAI-compatible API (used by vLLM and TensorRT-LLM)."""
    try:
        from openai import OpenAI

        client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key="dummy")
        client.models.list()
        return True
    except Exception:
        return False


def wait_for_llama_ready(host: str, port: int) -> bool:
    """Readiness check for llama.cpp server via /health endpoint."""
    try:
        import requests

        r = requests.get(f"http://{host}:{port}/health", timeout=5)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False


# ----------------------------
# Open WebUI helpers
# ----------------------------


def start_open_webui(
    backend_port: int,
    webui_port: int = 3000,
    image: str = "ghcr.io/open-webui/open-webui:main",
    timeout: int = 60,
) -> str | None:
    """Start Open WebUI container connected to backend.

    Args:
        backend_port: Backend server port (OpenAI-compatible API)
        webui_port: Port for Open WebUI (default: 3000)
        image: Docker image for Open WebUI
        timeout: Startup timeout in seconds

    Returns:
        Container ID if started successfully, None otherwise
    """
    # Build OpenAI base URL for the backend
    openai_base_url = f"http://localhost:{backend_port}/v1"

    cmd = [
        "docker",
        "run",
        "-d",  # Detached mode
        "--rm",
        "--network",
        "host",  # Share host network for localhost access
        "-e",
        f"PORT={webui_port}",  # Set port (default is 8080, we want webui_port)
        "-e",
        "HOST=0.0.0.0",  # Listen on all interfaces for external access
        "-e",
        f"OPENAI_API_BASE_URL={openai_base_url}",
        "-e",
        "OPENAI_API_KEY=dummy",  # Required but not used for local backends
        "-e",
        "WEBUI_AUTH=false",  # Disable auth for local dev
        "-v",
        "open-webui:/app/backend/data",  # Persist data
        image,
    ]

    log("Starting Open WebUI...")
    log(f"+ {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            log(f"Failed to start Open WebUI: {result.stderr}")
            return None

        container_id = result.stdout.strip()

        # Wait for Open WebUI to be ready
        log(f"Waiting for Open WebUI to be ready (timeout: {timeout}s)...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            if port_open("localhost", webui_port):
                return container_id
            time.sleep(1)

        # Timeout - cleanup
        log("Open WebUI timed out during startup")
        stop_container(container_id, label="Open WebUI")
        return None

    except Exception as e:
        log(f"Error starting Open WebUI: {e}")
        return None


def stop_container(container_id: str, label: str = "container") -> None:
    """Stop a Docker container by ID.

    Args:
        container_id: Container ID to stop
        label: Label for log messages
    """
    try:
        log(f"Stopping {label} ({container_id[:12]})...")
        subprocess.run(
            ["docker", "stop", container_id],
            timeout=30,
            capture_output=True,
        )
    except Exception as e:
        log(f"Warning: Failed to stop {label}: {e}")
