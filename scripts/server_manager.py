"""Server lifecycle management for vLLM and llama.cpp servers."""

import select
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from bench_utils import log, port_open, get_venv_python, resolve_local_gguf


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
            )
            stream = self.proc.stderr
        else:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
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

    def start_llama(
        self,
        model_path: str,
        llama_server_bin: str,
        n_gpu_layers: int | None = None,
        ctx: int | None = None,
        parallel: int | None = None,
        extra_args: list[str] | None = None,
    ) -> Path:
        """Start llama-server for a GGUF model.

        Resolves GGUF path, builds command, starts server, and waits for readiness.

        Args:
            model_path: Path to .gguf file or directory containing GGUF files
            llama_server_bin: Path to llama-server binary
            n_gpu_layers: GPU layers to offload (optional)
            ctx: Context length (optional)
            parallel: Parallel sequences (optional)
            extra_args: Extra raw args (optional)

        Returns:
            Path to resolved GGUF file

        Raises:
            SystemExit if GGUF not found or server fails to start
        """
        gguf_path = resolve_local_gguf(model_path)
        if not gguf_path:
            raise SystemExit(
                f"No GGUF found at: {Path(model_path).expanduser()}\n"
                "Pass --model with an explicit path, e.g.:\n"
                "  --model ~/models/org/repo/quant-folder\n"
                "  --model ~/models/org/repo/model.gguf"
            )

        cmd = build_llama_cmd(
            gguf_path=gguf_path,
            llama_server_bin=llama_server_bin,
            port=self.port,
            n_gpu_layers=n_gpu_layers,
            ctx=ctx,
            parallel=parallel,
            extra_args=extra_args,
        )

        self.start(
            cmd,
            lambda: wait_for_llama_ready(self.host, self.port),
            stream_stderr=True,
            label="llama-server",
            sigint_wait=2,
            term_wait=2,
        )

        return gguf_path

    def start_vllm(
        self,
        model_path: str,
        tensor_parallel: int,
        use_nightly: bool,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | None = None,
        max_num_batched_tokens: int | None = None,
    ) -> None:
        """Start vLLM server for a safetensors model.

        Builds command, starts server, and waits for readiness.
        For Mistral models, automatically uses Docker if available (FP8 compatibility).

        Args:
            model_path: Path to model directory
            tensor_parallel: Tensor parallel size
            use_nightly: If True, use nightly venv; otherwise stable
            max_model_len: Max context length (optional)
            gpu_memory_utilization: GPU memory fraction (optional)
            max_num_batched_tokens: Max batched tokens (optional)

        Raises:
            SystemExit if server fails to start
        """
        # For Mistral models, prefer Docker if available (FP8 kernel issues on some GPUs)
        if is_mistral_model(model_path) and docker_gpu_available():
            log("Using Docker for Mistral model (FP8 compatibility)")
            return self.start_vllm_docker(
                model_path=model_path,
                tensor_parallel=tensor_parallel,
            )

        cmd = build_vllm_cmd(
            model_path=model_path,
            host=self.host,
            port=self.port,
            tensor_parallel=tensor_parallel,
            use_nightly=use_nightly,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_batched_tokens=max_num_batched_tokens,
        )

        self.start(
            cmd,
            lambda: wait_for_vllm_ready(self.host, self.port),
            label="vLLM",
        )

    def start_vllm_docker(
        self,
        model_path: str,
        tensor_parallel: int,
        image: str = "vllm/vllm-openai:latest",
    ) -> None:
        """Start vLLM server via Docker.

        Args:
            model_path: Path to model directory
            tensor_parallel: Tensor parallel size
            image: Docker image to use

        Raises:
            SystemExit if server fails to start
        """
        cmd = build_vllm_docker_cmd(
            model_path=model_path,
            host=self.host,
            port=self.port,
            tensor_parallel=tensor_parallel,
            image=image,
        )

        self.start(
            cmd,
            lambda: wait_for_vllm_ready(self.host, self.port),
            label="vLLM (Docker)",
        )


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


# ----------------------------
# Model detection helpers
# ----------------------------

def is_glm_vision_model(model_path: str) -> bool:
    """Check if model is a GLM vision variant (GLM-4.5V, GLM-4.6V, etc.)."""
    lower = model_path.lower()
    # Match patterns like glm-4.5v, glm-4.6v, glm4v, etc.
    return "glm" in lower and "v" in lower.split("glm")[-1]


def is_mistral_model(model_path: str) -> bool:
    """Check if model is a Mistral/Devstral model requiring mistral tokenizer mode."""
    lower = model_path.lower()
    return "mistral" in lower or "devstral" in lower or "ministral" in lower


# ----------------------------
# Docker support
# ----------------------------

def docker_gpu_available() -> bool:
    """Check if Docker with GPU support is available."""
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all",
             "nvidia/cuda:12.6.0-base-ubuntu24.04", "nvidia-smi"],
            capture_output=True, timeout=60
        )
        return result.returncode == 0
    except Exception:
        return False


def build_vllm_docker_cmd(
    model_path: str,
    host: str,
    port: int,
    tensor_parallel: int,
    image: str = "vllm/vllm-openai:latest",
) -> list[str]:
    """Build Docker command for vLLM server."""
    model_path_resolved = str(Path(model_path).expanduser().resolve())

    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--shm-size", "16g",
        "-p", f"{port}:{port}",
        "-v", f"{model_path_resolved}:{model_path_resolved}:ro",
        image,
        "--model", model_path_resolved,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel),
    ]

    # Mistral-specific flags
    if is_mistral_model(model_path):
        cmd += [
            "--tokenizer_mode", "mistral",
            "--config_format", "mistral",
            "--load_format", "mistral",
        ]

    return cmd


# ----------------------------
# Command builders
# ----------------------------

def build_vllm_cmd(
    model_path: str,
    host: str,
    port: int,
    tensor_parallel: int,
    use_nightly: bool,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
    max_num_batched_tokens: int | None = None,
) -> list[str]:
    """Build vLLM server command with appropriate flags for model type.

    Args:
        model_path: Path to model
        host: Server host
        port: Server port
        tensor_parallel: Tensor parallel size
        use_nightly: If True, use nightly venv; otherwise stable
        max_model_len: Max context length (optional)
        gpu_memory_utilization: GPU memory fraction (optional)
        max_num_batched_tokens: Max batched tokens (optional)

    Returns:
        Command list for subprocess
    """
    python_path = get_venv_python(nightly=use_nightly)

    cmd = [
        python_path, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel),
    ]

    # GLM vision model specific flags (GLM-4.5V, GLM-4.6V, etc.)
    if is_glm_vision_model(model_path):
        cmd += [
            "--enable-expert-parallel",
            "--allowed-local-media-path", "/",
            "--mm-encoder-tp-mode", "data",
            "--mm_processor_cache_type", "shm",
        ]

    # Mistral/Devstral models require native mistral tokenizer
    # FP8 models need explicit dtype override due to vLLM FP8 kernel issues
    if is_mistral_model(model_path):
        cmd += [
            "--tokenizer_mode", "mistral",
            "--config_format", "mistral",
            "--load_format", "mistral",
            "--dtype", "bfloat16",
        ]

    if max_model_len is not None:
        cmd += ["--max-model-len", str(max_model_len)]

    if gpu_memory_utilization is not None:
        cmd += ["--gpu-memory-utilization", str(gpu_memory_utilization)]

    if max_num_batched_tokens is not None:
        cmd += ["--max-num-batched-tokens", str(max_num_batched_tokens)]

    return cmd


def build_llama_cmd(
    gguf_path: Path,
    llama_server_bin: str,
    port: int,
    n_gpu_layers: int | None = None,
    ctx: int | None = None,
    parallel: int | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build llama-server command with appropriate flags.

    Args:
        gguf_path: Path to GGUF model file
        llama_server_bin: Path to llama-server binary
        port: Server port
        n_gpu_layers: GPU layers to offload (optional)
        ctx: Context length (optional)
        parallel: Parallel sequences (optional)
        extra_args: Extra raw args (optional)

    Returns:
        Command list for subprocess
    """
    if not gguf_path.exists():
        raise SystemExit(f"GGUF not found: {gguf_path}")

    cmd = [llama_server_bin, "-m", str(gguf_path), "--port", str(port)]

    # GPU offload
    if n_gpu_layers is not None:
        cmd += ["-ngl", str(n_gpu_layers)]

    # Context length
    if ctx is not None:
        cmd += ["-c", str(ctx)]

    # Parallel sequences
    if parallel is not None and parallel > 1:
        cmd += ["-np", str(parallel)]

    # Extra raw args
    if extra_args:
        cmd += extra_args

    return cmd
