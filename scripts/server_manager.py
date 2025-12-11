"""Server lifecycle management for vLLM and llama.cpp servers."""

import select
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from bench_utils import log, port_open, resolve_local_gguf


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

    def is_running(self) -> bool:
        """Check if server is already running on port."""
        return port_open(self.host, self.port)

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
                errors='replace',  # Handle non-UTF-8 bytes gracefully
            )
            stream = self.proc.stderr
        else:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors='replace',  # Handle non-UTF-8 bytes gracefully
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
                        return True
                except Exception:
                    pass

            time.sleep(2)

            # Check if process died
            if self.proc.poll() is not None:
                read_output()
                output = "".join(self._output_lines[-50:])
                raise SystemExit(
                    f"{label} exited unexpectedly. Last output:\n{output}"
                )

        # Timeout - show recent logs and cleanup
        read_output()
        self.stop()
        output = "".join(self._output_lines[-50:])
        raise SystemExit(
            f"{label} failed to start within {self.timeout}s.\n"
            f"Last output:\n{output}"
        )

    def stop(self):
        """Graceful shutdown: SIGINT → wait → terminate → wait → kill."""
        if not self.proc or not self._we_started_it:
            return

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
        """
        from docker_manager import (
            ensure_image,
            build_vllm_docker_cmd as build_versioned_vllm_cmd,
        )

        # Ensure image exists (builds if needed)
        image_name = ensure_image("vllm", version, rebuild=rebuild)

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

        self.start(
            cmd,
            lambda: wait_for_vllm_ready(self.host, self.port),
            label=f"vLLM (Docker {version})",
        )

    def start_llama(
        self,
        model_path: str,
        version: str,
        n_gpu_layers: int | None = None,
        ctx: int | None = None,
        parallel: int | None = None,
        mmproj_path: Path | None = None,
        extra_args: list[str] | None = None,
        rebuild: bool = False,
    ) -> Path:
        """Start llama-server via Docker with version pinning.

        Args:
            model_path: Path to .gguf file or directory containing GGUF files
            version: llama.cpp version (release tag like 'b4521' or commit SHA)
            n_gpu_layers: GPU layers to offload (optional)
            ctx: Context length (optional)
            parallel: Parallel sequences (optional)
            mmproj_path: Path to multimodal projector (optional)
            extra_args: Extra raw args (optional)
            rebuild: Force rebuild image even if cached

        Returns:
            Path to resolved GGUF file
        """
        return self.start_gguf_backend(
            engine="llama",
            model_path=model_path,
            version=version,
            n_gpu_layers=n_gpu_layers,
            ctx=ctx,
            parallel=parallel,
            mmproj_path=mmproj_path,
            extra_args=extra_args,
            rebuild=rebuild,
        )

    def start_ik_llama(
        self,
        model_path: str,
        version: str,
        n_gpu_layers: int | None = None,
        ctx: int | None = None,
        parallel: int | None = None,
        mmproj_path: Path | None = None,
        extra_args: list[str] | None = None,
        rebuild: bool = False,
    ) -> Path:
        """Start ik_llama-server via Docker with version pinning.

        ik_llama.cpp is ikawrakow's performance-optimized fork of llama.cpp.

        Args:
            model_path: Path to .gguf file or directory containing GGUF files
            version: ik_llama.cpp version (release tag or commit SHA)
            n_gpu_layers: GPU layers to offload (optional)
            ctx: Context length (optional)
            parallel: Parallel sequences (optional)
            mmproj_path: Path to multimodal projector (optional)
            extra_args: Extra raw args (optional)
            rebuild: Force rebuild image even if cached

        Returns:
            Path to resolved GGUF file
        """
        return self.start_gguf_backend(
            engine="ik_llama",
            model_path=model_path,
            version=version,
            n_gpu_layers=n_gpu_layers,
            ctx=ctx,
            parallel=parallel,
            mmproj_path=mmproj_path,
            extra_args=extra_args,
            rebuild=rebuild,
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
            ensure_image,
            build_llama_docker_cmd,
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


# ----------------------------
# Readiness checks
# ----------------------------

def wait_for_vllm_ready(host: str, port: int) -> bool:
    """Readiness check for vLLM server via OpenAI API."""
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
