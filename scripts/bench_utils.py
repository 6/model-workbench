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

def compact_path(path: str) -> str:
    """Replace home directory with ~ for cleaner paths."""
    home = str(Path.home())
    if path.startswith(home):
        return "~" + path[len(home):]
    return path


def get_gpu_info(include_memory: bool = False) -> dict:
    """
    Returns a dict with driver_version and a list of GPUs with PCIe info.
    Uses nvidia-smi. Safe to fail quietly.

    Args:
        include_memory: If True, also query current memory usage per GPU
    """
    try:
        drv = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True
        ).strip().splitlines()
        driver_version = drv[0].strip() if drv else None

        query_fields = "index,name,memory.total,pcie.link.gen.max,pcie.link.width.current"
        if include_memory:
            query_fields += ",memory.used"

        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query_fields}", "--format=csv,noheader,nounits"],
            text=True
        ).strip()

        gpus = []
        total_mem = 0
        total_used = 0
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpu = {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mib": int(parts[2]),
                    "pcie_gen": int(parts[3]) if parts[3].isdigit() else None,
                    "pcie_width": int(parts[4]) if parts[4].isdigit() else None,
                }
                total_mem += gpu["memory_total_mib"]
                if include_memory and len(parts) >= 6:
                    gpu["memory_used_mib"] = int(parts[5])
                    total_used += gpu["memory_used_mib"]
                gpus.append(gpu)

        result = {"driver_version": driver_version, "gpus": gpus}
        if include_memory:
            result["memory"] = {"used_mib": total_used, "total_mib": total_mem}
        return result
    except Exception as e:
        return {"error": str(e), "driver_version": None, "gpus": []}

def get_gpu_count() -> int:
    """Return number of available GPUs. Returns 1 if detection fails."""
    return max(1, len(get_gpu_info().get("gpus", [])))



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
