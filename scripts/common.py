"""Shared constants and utility functions for the model workbench."""

from datetime import datetime
from pathlib import Path

# ----------------------------
# Path Constants
# ----------------------------

ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = Path.home() / "models"
RESULTS_ROOT = ROOT / "perf"
EVAL_RESULTS_ROOT = ROOT / "evals"
CONFIG_PATH = ROOT / "config" / "models.yaml"


# ----------------------------
# Logging
# ----------------------------


def log(msg: str):
    """Print a timestamped log message."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# ----------------------------
# Backend Registry
# ----------------------------

BACKEND_REGISTRY = {
    "vllm": {
        "display_name": "vLLM",
        "formats": ["safetensors"],
        "default_port": 8000,
        "image_prefix": "model-bench-vllm",
        "prebuilt_image": "vllm/vllm-openai",
        "dockerfile": ROOT / "docker" / "Dockerfile.vllm",
        "docker_base": ["--gpus", "all", "--ipc", "host"],
    },
    "llama": {
        "display_name": "llama.cpp",
        "formats": ["gguf"],
        "default_port": 8080,
        "image_prefix": "model-bench-llama",
        "prebuilt_image": None,
        "dockerfile": ROOT / "docker" / "Dockerfile.llama",
        "docker_base": ["--gpus", "all"],
    },
    "ik_llama": {
        "display_name": "ik_llama.cpp",
        "formats": ["gguf"],
        "default_port": 8080,
        "image_prefix": "model-bench-ik-llama",
        "prebuilt_image": None,
        "dockerfile": ROOT / "docker" / "Dockerfile.ik_llama",
        "docker_base": ["--gpus", "all"],
    },
    "trtllm": {
        "display_name": "TensorRT-LLM",
        "formats": ["safetensors"],
        "default_port": 8000,
        "image_prefix": None,  # prebuilt only
        "prebuilt_image": "nvcr.io/nvidia/tensorrt-llm/release",
        "dockerfile": None,
        "docker_base": [
            "--gpus",
            "all",
            "--ipc",
            "host",
            "--ulimit",
            "memlock=-1",
            "--ulimit",
            "stack=67108864",
        ],
    },
    "sglang": {
        "display_name": "SGLang",
        "formats": ["safetensors"],
        "default_port": 30000,
        "image_prefix": None,  # Prebuilt only
        "prebuilt_image": "lmsysorg/sglang",
        "dockerfile": None,  # Prebuilt only, no build-from-source support
        "docker_base": [
            "--gpus",
            "all",
            "--ipc",
            "host",
            "--shm-size",
            "32g",
        ],
    },
    "exl": {
        "display_name": "ExLlamaV3",
        "formats": ["exl3"],  # EXL3 quantized models
        "default_port": 5000,  # TabbyAPI default
        "image_prefix": "model-bench-exl",
        "prebuilt_image": None,  # Build-from-source only
        "dockerfile": ROOT / "docker" / "Dockerfile.exl",
        "docker_base": ["--gpus", "all", "--ipc", "host"],
    },
    "ktransformers": {
        "display_name": "KTransformers",
        "formats": ["safetensors"],  # FP8 safetensors for CPU-GPU hybrid
        "default_port": 10002,
        "image_prefix": "model-bench-ktransformers",
        "prebuilt_image": None,  # Build-from-source only
        "dockerfile": ROOT / "docker" / "Dockerfile.ktransformers",
        "docker_base": ["--gpus", "all", "--ipc", "host"],
    },
}
