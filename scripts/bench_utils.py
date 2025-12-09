"""
Shared utilities for benchmark scripts.
"""

import re
import socket
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


# ----------------------------
# Logging
# ----------------------------

def log(msg: str):
    """Print a timestamped log message."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

# ----------------------------
# Path constants
# ----------------------------

ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = Path.home() / "models"
RESULTS_ROOT = ROOT / "benchmarks"
MODELS_CONFIG = ROOT / "config" / "models.yaml"
VENV_STABLE = ROOT / ".venv"
VENV_NIGHTLY = ROOT / "nightly" / ".venv"

# ----------------------------
# String utilities
# ----------------------------

def sanitize(s: str) -> str:
    """Sanitize a string for use in filenames."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)

# ----------------------------
# GPU info + tag inference
# ----------------------------

def get_gpu_info() -> dict:
    """
    Returns a dict with driver_version and a list of GPUs.
    Uses nvidia-smi. Safe to fail quietly.
    """
    try:
        drv = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True
        ).strip().splitlines()
        driver_version = drv[0].strip() if drv else None

        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            text=True
        ).strip()

        gpus = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mib": int(parts[2]),
                })
        return {"driver_version": driver_version, "gpus": gpus}
    except Exception as e:
        return {"error": str(e), "driver_version": None, "gpus": []}

def get_gpu_count() -> int:
    """Return number of available GPUs. Returns 1 if detection fails."""
    return max(1, len(get_gpu_info().get("gpus", [])))

def infer_tag(cli_tag: str | None, tensor_parallel: int | None = None) -> str:
    """
    Infer output tag from CLI arg or GPU count.

    Args:
        cli_tag: User-provided tag (used if not None)
        tensor_parallel: If provided, use this instead of GPU count
    """
    if cli_tag:
        return cli_tag

    if tensor_parallel is not None:
        if tensor_parallel == 1:
            return "single-gpu"
        if tensor_parallel == 2:
            return "dual-gpu"
        return f"{tensor_parallel}-gpu"

    gpus = get_gpu_info().get("gpus", [])
    n = len(gpus)

    if n == 0:
        return "unknown-gpu"
    if n == 1:
        return "single-gpu"
    if n == 2:
        return "dual-gpu"
    return f"{n}-gpu"

# ----------------------------
# Network utilities
# ----------------------------

def port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """Check if a port is open on the given host."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

# ----------------------------
# Model resolution
# ----------------------------

def resolve_model_path(model_arg: str, require_local: bool = True) -> str:
    """
    Resolve model path - checks local paths first.

    Args:
        model_arg: Model argument (path or HF-style repo id)
        require_local: If True, raise error if not found locally

    Returns:
        Resolved path string

    Raises:
        SystemExit if require_local=True and model not found locally
    """
    # Check if it's an absolute or expandable path
    p = Path(model_arg).expanduser()
    if p.exists():
        return str(p)

    # Check relative to MODELS_ROOT
    if "/" in model_arg:
        local = MODELS_ROOT / model_arg
        if local.exists():
            return str(local)

    if require_local:
        raise SystemExit(
            f"Model not found locally: {model_arg}\n"
            f"Checked:\n"
            f"  - {p}\n"
            f"  - {MODELS_ROOT / model_arg if '/' in model_arg else 'N/A'}\n"
            f"\nPlease download the model first or provide the correct path."
        )

    # Return as-is for HuggingFace download
    return model_arg

# ----------------------------
# Model config utilities
# ----------------------------

def load_models_config() -> list[dict]:
    """Load models.yaml config."""
    if not MODELS_CONFIG.exists():
        return []
    with open(MODELS_CONFIG) as f:
        data = yaml.safe_load(f)
    return data.get("models", [])


def get_model_config(model_arg: str) -> dict | None:
    """
    Find model config entry matching the given model argument.

    Args:
        model_arg: Model path or repo_id (e.g., "zai-org/GLM-4.6V-FP8")

    Returns:
        Model config dict if found, None otherwise
    """
    models = load_models_config()
    # Normalize model_arg for matching
    model_lower = model_arg.lower()

    for m in models:
        repo_id = m.get("repo_id", "")
        if repo_id.lower() in model_lower or model_lower in repo_id.lower():
            return m
    return None


def model_needs_nightly(model_arg: str) -> bool:
    """
    Check if a model requires nightly transformers/tokenizers.

    Args:
        model_arg: Model path or repo_id

    Returns:
        True if model has nightly: true in config
    """
    config = get_model_config(model_arg)
    if config:
        return config.get("nightly", False)
    return False


def get_venv_python(nightly: bool = False) -> str:
    """
    Get path to Python executable for the appropriate venv.

    Args:
        nightly: If True, use nightly venv; otherwise stable

    Returns:
        Path to Python executable
    """
    venv = VENV_NIGHTLY if nightly else VENV_STABLE
    python = venv / "bin" / "python"
    if not python.exists():
        env_name = "nightly" if nightly else "stable"
        raise SystemExit(
            f"Python not found in {env_name} venv: {python}\n"
            f"Run bootstrap.sh to create both environments."
        )
    return str(python)
