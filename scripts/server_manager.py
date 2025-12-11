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
            container_ids = [cid.strip() for cid in result.stdout.strip().split('\n') if cid.strip()]
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
        rebuild: bool = False,
        image_type: str = "build",
        image_override: str | None = None,
    ) -> None:
        """Start vLLM server via Docker with version pinning.

        Args:
            model_path: Path to model directory
            tensor_parallel: Tensor parallel size
            version: vLLM version (release tag like 'v0.8.0' or commit SHA)
            max_model_len: Max context length (optional)
            gpu_memory_utilization: GPU memory fraction (optional)
            max_num_batched_tokens: Max batched tokens (optional)
            rebuild: Force rebuild image even if cached
            image_type: 'prebuilt' to use official images, 'build' to build from source
            image_override: Direct image name to use (highest priority)
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
        extra_args: list[str] | None = None,
        rebuild: bool = False,
    ) -> Path:
        """Start a GGUF backend (llama.cpp or ik_llama.cpp) via Docker.

        Args:
            engine: 'llama' or 'ik_llama'
            model_path: Path to .gguf file or directory containing GGUF files
            version: Backend version (release tag or commit SHA)
            n_gpu_layers: GPU layers to offload (optional)
            ctx: Context length (optional)
            parallel: Parallel sequences (optional)
            mmproj_path: Path to multimodal projector (optional)
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
            extra_args=extra_args,
        )

        # Label based on engine
        label = "ik_llama.cpp" if engine == "ik_llama" else "llama.cpp"
        self.start(
            cmd,
            lambda: wait_for_llama_ready(self.host, self.port),
            label=f"{label} (Docker {version})",
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
