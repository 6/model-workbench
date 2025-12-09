"""
Shared utilities for benchmark scripts.
"""

import json
import re
import socket
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


# ----------------------------
# Benchmark prompts
# ----------------------------

TEXT_PROMPTS = {
    "short": "Explain speculative decoding in 2 sentences.",
    "medium": "Summarize key tradeoffs between tensor parallelism and pipeline parallelism.",
    "long": "Write a concise technical overview of KV cache and why it matters for long context.",
}

VISION_PROMPTS = {
    "describe": "Describe this image in detail.",
    "analyze": "Analyze this image and explain what you see.",
    "caption": "Provide a brief caption for this image.",
}

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
RESULTS_ROOT = ROOT / "perf"
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


def extract_repo_id(path: str) -> str:
    """
    Extract repo ID (org/repo) from a model path.

    Examples:
        ~/models/unsloth/Qwen3-GGUF/file.gguf -> unsloth/Qwen3-GGUF
        /home/user/models/org/repo -> org/repo
    """
    p = Path(path).expanduser()
    try:
        rel = p.relative_to(MODELS_ROOT)
        parts = rel.parts
        # Return org/repo (first two parts)
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        elif len(parts) == 1:
            return parts[0]
    except ValueError:
        pass
    # Fallback: return compacted path
    return compact_path(path)


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
            result["memory_used_mib"] = total_used
            result["memory_total_mib"] = total_mem
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

def resolve_model_path(model_arg: str) -> str:
    """
    Resolve model path from explicit filesystem path.

    Args:
        model_arg: Explicit path to model (e.g., ~/models/org/repo)

    Returns:
        Resolved absolute path string

    Raises:
        SystemExit if path does not exist
    """
    p = Path(model_arg).expanduser()
    if p.exists():
        return str(p)

    raise SystemExit(f"Model not found: {p}")


def detect_model_format(model_arg: str) -> str:
    """
    Detect model format based on file extension.

    Args:
        model_arg: Path to model file or directory

    Returns:
        'gguf' if GGUF model, 'safetensors' otherwise
    """
    p = Path(model_arg).expanduser()
    if p.suffix == ".gguf":
        return "gguf"
    if p.is_dir() and any(p.rglob("*.gguf")):
        return "gguf"
    return "safetensors"

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


# ----------------------------
# Result writing
# ----------------------------

def write_benchmark_result(
    results_dir: Path,
    repo_id: str,
    model_ref: str,
    engine: str,
    gpu_info: dict,
    config: dict,
    iterations: list[dict],
    summary: dict,
    extra: dict | None = None,
) -> Path:
    """Write benchmark results with consistent format and naming.

    Args:
        results_dir: Directory to write results to
        repo_id: Model repo ID (e.g., "org/repo")
        model_ref: Compacted model path for reference
        engine: Engine name (e.g., "vllm-server", "llama-server")
        gpu_info: GPU info dict from get_gpu_info()
        config: Benchmark configuration (prompt, max_tokens, etc.)
        iterations: List of per-iteration results
        summary: Summary statistics (median_wall_s, etc.)
        extra: Optional extra fields to include at top level

    Returns:
        Path to written results file
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "repo_id": repo_id,
        "model_ref": model_ref,
        "engine": engine,
        "gpu_info": gpu_info,
        "config": config,
        "iterations": iterations,
        "summary": summary,
    }

    # Merge extra fields at top level
    if extra:
        payload.update(extra)

    # Generate filename: DATE_REPO-ID_ENGINE.json
    date_str = datetime.now().strftime("%Y-%m-%d")
    safe_repo = sanitize(repo_id)
    safe_engine = sanitize(engine)
    filename = f"{date_str}_{safe_repo}_{safe_engine}.json"

    out_path = results_dir / filename
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    log(f"Wrote: {out_path}")
    return out_path
