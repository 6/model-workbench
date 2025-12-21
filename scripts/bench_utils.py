"""
Shared utilities for benchmark scripts.
"""

import base64
import json
import re
import socket
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

import requests
import yaml
from common import BACKEND_REGISTRY, CONFIG_PATH, MODELS_ROOT, ROOT, log
from openai import OpenAI

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

# Combined prompts for CLI choices
ALL_PROMPTS = {**TEXT_PROMPTS, **VISION_PROMPTS}


def get_compatible_backends(model_format: str) -> list[str]:
    """Return backends that support the given format."""
    return [name for name, cfg in BACKEND_REGISTRY.items() if model_format in cfg["formats"]]


def get_default_backend(model_format: str) -> str:
    """Return default backend for format."""
    if model_format == "gguf":
        return "llama"
    return "vllm"


def resolve_backend(model_arg: str, backend_override: str | None) -> str:
    """Resolve backend from explicit choice or auto-detect from format.

    Args:
        model_arg: Path to model
        backend_override: Explicit backend choice (or None for auto-detect)

    Returns:
        Backend name ('vllm', 'llama', 'ik_llama')

    Raises:
        SystemExit if backend incompatible with model format
    """
    fmt = detect_model_format(model_arg)
    compatible = get_compatible_backends(fmt)

    if backend_override:
        if backend_override not in BACKEND_REGISTRY:
            raise SystemExit(
                f"Unknown backend '{backend_override}'.\n"
                f"Available: {', '.join(BACKEND_REGISTRY.keys())}"
            )
        if backend_override not in compatible:
            raise SystemExit(
                f"Backend '{backend_override}' does not support {fmt} models.\n"
                f"Compatible backends: {', '.join(compatible)}"
            )
        return backend_override

    return get_default_backend(fmt)


# Built-in test images
BUILTIN_IMAGES = {
    "example": ROOT / "config" / "example.jpg",
    "grayscale": "https://upload.wikimedia.org/wikipedia/commons/f/fa/Grayscale_8bits_palette_sample_image.png",
}

# ----------------------------
# String utilities
# ----------------------------


def sanitize(s: str) -> str:
    """Sanitize a string for use in filenames."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)


# ----------------------------
# Image utilities
# ----------------------------


def resolve_image_source(image_arg: str | None) -> tuple[str | None, str]:
    """Resolve image argument to path and label.

    Args:
        image_arg: Image path, URL, builtin name ('example', 'grayscale'), or 'none'

    Returns:
        (image_path, label) tuple where image_path is None for text-only
    """
    if image_arg is None or image_arg.lower() == "none":
        return None, "none"

    if image_arg in BUILTIN_IMAGES:
        src = BUILTIN_IMAGES[image_arg]
        return str(src), f"builtin:{image_arg}"

    p = Path(image_arg).expanduser()
    if p.exists():
        return str(p), str(p)

    # Assume it's a URL
    return image_arg, image_arg


def encode_image_base64(image_path: str) -> str:
    """Encode local image to base64 data URL.

    Args:
        image_path: Path to local image file

    Returns:
        Base64 data URL string (e.g., "data:image/jpeg;base64,...")
    """
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    # Determine MIME type from extension
    suffix = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime = mime_types.get(suffix, "image/jpeg")
    return f"data:{mime};base64,{data}"


def build_chat_messages(prompt: str, image_path: str | None = None) -> list[dict]:
    """Build OpenAI-compatible chat messages with optional image.

    Args:
        prompt: Text prompt
        image_path: Optional image path (local file or URL)

    Returns:
        List of message dicts for chat completions API
    """
    if image_path is None:
        return [{"role": "user", "content": prompt}]

    if image_path.startswith("http"):
        image_content = {"type": "image_url", "image_url": {"url": image_path}}
    else:
        data_url = encode_image_base64(image_path)
        image_content = {"type": "image_url", "image_url": {"url": data_url}}

    return [{"role": "user", "content": [image_content, {"type": "text", "text": prompt}]}]


# ----------------------------
# Statistics utilities
# ----------------------------


def med(results: list[dict], key: str) -> float | None:
    """Compute median of a metric across results, ignoring None values.

    Args:
        results: List of result dicts
        key: Key to extract from each result

    Returns:
        Median value or None if no valid values
    """
    vals = [r.get(key) for r in results if r.get(key) is not None]
    return statistics.median(vals) if vals else None


# ----------------------------
# GPU info + tag inference
# ----------------------------


def compact_path(path: str) -> str:
    """Replace home directory with ~ for cleaner paths."""
    home = str(Path.home())
    if path.startswith(home):
        return "~" + path[len(home) :]
    return path


def extract_repo_id(path: str) -> str:
    """
    Extract repo ID from a model path.

    For directories: returns up to org/repo/quant (3 parts)
    For .gguf files: returns org/repo/filename (without .gguf extension)

    Examples:
        ~/models/unsloth/GLM-GGUF/Q4_K_XL -> unsloth/GLM-GGUF/Q4_K_XL
        ~/models/unsloth/Qwen-GGUF/Model-Q4.gguf -> unsloth/Qwen-GGUF/Model-Q4
        ~/models/org/repo -> org/repo
    """
    p = Path(path).expanduser()
    try:
        rel = p.relative_to(MODELS_ROOT)
        parts = rel.parts

        if p.is_dir():
            # Directory: include up to 3 parts (org/repo/quant)
            n = min(len(parts), 3)
            return "/".join(parts[:n])
        else:
            # File: include org/repo + filename without extension
            if len(parts) >= 2:
                repo_parts = parts[:2]  # org/repo
                filename = p.stem  # filename without extension
                return f"{repo_parts[0]}/{repo_parts[1]}/{filename}"
            elif len(parts) == 1:
                return p.stem
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
        drv = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
            )
            .strip()
            .splitlines()
        )
        driver_version = drv[0].strip() if drv else None

        query_fields = "index,name,memory.total,pcie.link.gen.max,pcie.link.width.current"
        if include_memory:
            query_fields += ",memory.used"

        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query_fields}", "--format=csv,noheader,nounits"],
            text=True,
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


def detect_max_position_embeddings(model_path: str) -> int | None:
    """Detect max_position_embeddings from model's config.json.

    Args:
        model_path: Path to model directory

    Returns:
        max_position_embeddings value if found, None otherwise
    """
    config_path = Path(model_path).expanduser() / "config.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path) as f:
            config = json.load(f)
        # Check common field names used by different model architectures
        for field in ["max_position_embeddings", "seq_length", "context_length", "max_seq_len"]:
            if field in config:
                return config[field]
    except Exception:
        pass
    return None


# ----------------------------
# GGUF resolution
# ----------------------------

_SHARD_RE = re.compile(r".*-\d{5}-of-\d{5}\.gguf$")


def is_gguf_file(p: Path) -> bool:
    """Check if path is a GGUF file."""
    return p.is_file() and p.suffix == ".gguf"


def find_shard_entrypoints(dir_path: Path) -> list[Path]:
    """Return all *-00001-of-*.gguf under dir_path (sorted)."""
    return sorted(dir_path.rglob("*-00001-of-*.gguf"))


def list_all_ggufs(dir_path: Path) -> list[Path]:
    """Return all .gguf files under dir_path (sorted)."""
    return sorted(dir_path.rglob("*.gguf"))


def pick_gguf_from_dir(dir_path: Path) -> Path | None:
    """
    Pick a GGUF entrypoint from a directory ONLY if unambiguous.

      1) If exactly one shard entrypoint (*-00001-of-*.gguf) -> return it
      2) Else if exactly one non-sharded gguf anywhere -> return it
      3) Else -> None (caller decides whether to raise ambiguity)
    """
    entrypoints = find_shard_entrypoints(dir_path)
    if len(entrypoints) == 1:
        return entrypoints[0]
    if len(entrypoints) > 1:
        return None

    ggufs = list_all_ggufs(dir_path)
    if len(ggufs) == 1:
        return ggufs[0]

    non_shards = [p for p in ggufs if not _SHARD_RE.match(p.name)]
    if len(non_shards) == 1:
        return non_shards[0]

    return None


def raise_if_multiple_variants(dir_path: Path, model_arg: str):
    """
    Raise helpful errors if GGUF variants exist but are ambiguous:
      - multiple shard entrypoints
      - multiple root-level quant files
    """
    ggufs = list_all_ggufs(dir_path)
    if len(ggufs) <= 1:
        return

    entrypoints = find_shard_entrypoints(dir_path)
    if len(entrypoints) > 1:
        raise SystemExit(
            f"Multiple split GGUF variants found under:\n  {dir_path}\n"
            f"Pass a more specific --model like:\n"
            f"  {model_arg}/UD-Q4_K_XL\n"
            f"  or an exact .gguf file path."
        )

    non_shards = [p for p in ggufs if not _SHARD_RE.match(p.name)]
    if len(non_shards) > 1:
        raise SystemExit(
            f"Multiple GGUF files found under:\n  {dir_path}\n"
            f"This repo appears to store multiple quant files in the root.\n"
            f"Please pass an exact quant file path, e.g.:\n"
            f"  {model_arg}/<model>-UD-Q4_K_XL.gguf"
        )


def resolve_local_gguf(model_arg: str) -> Path | None:
    """
    Resolve a GGUF path from explicit filesystem path.

    Args:
        model_arg: Explicit path to .gguf file or directory containing GGUF files

    Returns:
        Path to GGUF file (prefer shard entrypoint), or None if not found

    Raises:
        SystemExit if directory contains multiple ambiguous GGUF variants
    """
    p = Path(model_arg).expanduser()
    if is_gguf_file(p):
        return p
    if p.is_dir():
        chosen = pick_gguf_from_dir(p)
        if chosen:
            return chosen
        raise_if_multiple_variants(p, model_arg)
    return None


def find_mmproj(gguf_path: Path) -> Path | None:
    """
    Find multimodal projector file (mmproj-*.gguf) for a vision GGUF model.

    Searches in:
      1. Same directory as the GGUF file
      2. Parent directory (for quant subfolders like UD-Q4_K_XL/)
      3. Up to MODELS_ROOT

    Args:
        gguf_path: Path to the main GGUF model file

    Returns:
        Path to mmproj file if found, None otherwise
    """
    # Start from the directory containing the GGUF
    search_dir = gguf_path.parent if gguf_path.is_file() else gguf_path

    # Search up to MODELS_ROOT
    while search_dir != MODELS_ROOT.parent and search_dir != search_dir.parent:
        mmproj_files = list(search_dir.glob("mmproj-*.gguf"))
        if mmproj_files:
            # If multiple, prefer F16 over lower precision
            for f in mmproj_files:
                if "F16" in f.name or "f16" in f.name:
                    return f
            return mmproj_files[0]
        search_dir = search_dir.parent

    return None


# ----------------------------
# Model config utilities
# ----------------------------

# Config cache to avoid repeated file reads
_CONFIG_CACHE: dict | None = None


def _load_config() -> dict:
    """Load and cache config file.

    Returns:
        Full config dict with 'defaults' and 'models' keys
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                _CONFIG_CACHE = yaml.safe_load(f) or {}
        else:
            _CONFIG_CACHE = {}
    return _CONFIG_CACHE


def load_models_config() -> list[dict]:
    """Load models.yaml config."""
    return _load_config().get("models", [])


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


def get_backend_config(engine: str) -> dict:
    """
    Get full config dict for a backend from defaults.backends.{engine}.

    Args:
        engine: 'vllm', 'llama', 'ik_llama', or 'trtllm'

    Returns:
        Backend config dict with keys: version, image_type, args
    """
    config = _load_config()
    defaults = config.get("defaults", {})
    backends = defaults.get("backends", {})

    if engine in backends:
        backend_cfg = backends[engine]
        return {
            "version": backend_cfg.get("version"),
            "image_type": backend_cfg.get("image_type", "build"),
            "args": backend_cfg.get("args", {}),
        }

    return {"version": None, "image_type": "build", "args": {}}


def get_model_backend_config(model_arg: str, engine: str) -> dict:
    """
    Get merged config for a model + backend.

    Resolution order (later overrides earlier):
    1. Global backend defaults (defaults.backends.{engine})
    2. Model-specific overrides (model.backends.{engine})

    Args:
        model_arg: Model path or repo_id
        engine: Backend name

    Returns:
        Merged config dict with keys: version, image_type, args
    """
    # Start with backend defaults
    result = get_backend_config(engine)

    # Check for model-specific overrides
    model_cfg = get_model_config(model_arg)
    if not model_cfg:
        return result

    # Model-specific overrides: model.backends.{engine}
    model_backends = model_cfg.get("backends", {})
    if engine in model_backends:
        model_backend_cfg = model_backends[engine]
        if model_backend_cfg.get("version"):
            result["version"] = model_backend_cfg["version"]
        if model_backend_cfg.get("image_type"):
            result["image_type"] = model_backend_cfg["image_type"]
        if model_backend_cfg.get("args"):
            # Merge args (model-specific overrides defaults)
            result["args"] = {**result["args"], **model_backend_cfg["args"]}

    return result


def get_model_backend_version(model_arg: str, engine: str) -> str | None:
    """
    Get backend version for a specific model.

    Resolution order:
    1. Model's backends.{engine}.version if specified
    2. Global defaults.backends.{engine}.version

    Args:
        model_arg: Model path or repo_id
        engine: Backend name

    Returns:
        Version string or None if not configured
    """
    return get_model_backend_config(model_arg, engine).get("version")


def resolve_run_config(args):
    """Resolve backend config and apply defaults to args.

    This centralizes the common setup pattern used by run_bench, run_server, run_eval.

    Args:
        args: Parsed argparse namespace with:
            - model (required)
            - backend (optional)
            - port (optional)
            - tensor_parallel (optional, for vLLM/trtllm)
            - max_model_len (optional)
            - gpu_memory_utilization (optional)
            - n_gpu_layers (optional, for llama.cpp)
            - image_type (optional)
            - backend_version (optional)

    Returns:
        Tuple of (backend, model_path, backend_cfg) where:
            - backend: Resolved backend name
            - model_path: Resolved model path (safetensors get expanded, GGUF stays as-is)
            - backend_cfg: Full config dict with merged defaults and model-specific settings

    Side effects:
        Modifies args in place to fill in defaults for:
            - port
            - tensor_parallel (for vLLM/trtllm)
            - max_model_len
            - gpu_memory_utilization
            - n_gpu_layers
    """
    from common import BACKEND_REGISTRY

    # Resolve backend (auto-detect or explicit)
    backend = resolve_backend(args.model, getattr(args, "backend", None))
    backend_info = BACKEND_REGISTRY[backend]

    # Get merged config for this model + backend
    backend_cfg = get_model_backend_config(args.model, backend)
    backend_args = backend_cfg.get("args", {})

    # Set default port based on backend
    if getattr(args, "port", None) is None:
        args.port = backend_info["default_port"]

    # Auto-detect tensor parallel for vLLM and trtllm
    if backend in ("vllm", "trtllm") and getattr(args, "tensor_parallel", None) is None:
        args.tensor_parallel = get_gpu_count()

    # Resolve model path (safetensors get expanded, GGUF stays as-is for internal resolution)
    if backend in ("vllm", "trtllm"):
        model_path = resolve_model_path(args.model)
    else:
        model_path = args.model

    # Apply config defaults for args not specified on CLI
    if getattr(args, "max_model_len", None) is None:
        config_default = backend_args.get("max_model_len", 65536)
        if backend in ("vllm", "trtllm"):
            detected = detect_max_position_embeddings(model_path)
            if detected:
                args.max_model_len = min(config_default, detected)
                if args.max_model_len < config_default:
                    log(
                        f"Capping max_model_len to {args.max_model_len} (model's max_position_embeddings)"
                    )
            else:
                args.max_model_len = config_default
        else:
            args.max_model_len = config_default
    if getattr(args, "gpu_memory_utilization", None) is None:
        args.gpu_memory_utilization = backend_args.get("gpu_memory_utilization", 0.95)
    if getattr(args, "n_gpu_layers", None) is None:
        args.n_gpu_layers = backend_args.get("n_gpu_layers", 999)

    return backend, model_path, backend_cfg


# ----------------------------
# Result writing
# ----------------------------


def write_benchmark_result(
    results_dir: Path,
    repo_id: str,
    model_ref: str,
    engine: str,
    mode: str,
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
        mode: Benchmark mode (e.g., "vision", "text-only")
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
        "mode": mode,
        "gpu_info": gpu_info,
        "config": config,
        "iterations": iterations,
        "summary": summary,
    }

    # Merge extra fields at top level
    if extra:
        payload.update(extra)

    # Generate filename: DATE_REPO-ID_ENGINE_MODE.json
    date_str = datetime.now().strftime("%Y-%m-%d")
    safe_repo = sanitize(repo_id)
    safe_engine = sanitize(engine)
    safe_mode = sanitize(mode)
    filename = f"{date_str}_{safe_repo}_{safe_engine}_{safe_mode}.json"

    out_path = results_dir / filename
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    log(f"Wrote: {out_path}")
    return out_path


def warmup_model(
    backend: str,
    host: str,
    port: int,
    api_model: str,
    prompt: str = "Explain the key differences between tensor parallelism and pipeline parallelism in distributed deep learning inference.",
    max_tokens: int = 64,
    timeout: int = 300,
) -> bool:
    """
    Make an inference request to preload model into GPU memory.

    This prevents lazy loading on the first user request by warming up
    the model during server startup. Uses a substantial request to ensure
    the model is fully loaded (not just partially cached).

    Args:
        backend: Backend type ("vllm", "trtllm", "llama", "ik_llama")
        host: Server host
        port: Server port
        api_model: Model name for API requests
        prompt: Warmup prompt text (default: "Hello")
        max_tokens: Max tokens to generate (default: 16)
        timeout: Request timeout in seconds (default: 300)

    Returns:
        True if warmup succeeded, False if it failed
    """
    try:
        if backend in ("vllm", "trtllm"):
            # Use OpenAI-compatible API
            client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key="dummy")
            messages = build_chat_messages(prompt, image_path=None)

            response = client.chat.completions.create(
                model=api_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )

            # Validate response contains completion
            if not response.choices or not response.choices[0].message.content:
                log(f"ERROR: Warmup got empty response from {backend}")
                log(f"Response: {response}")
                return False

            generated = len(response.choices[0].message.content)
            log(f"Warmup generated {generated} characters")
            return True

        elif backend in ("llama", "ik_llama"):
            # Use llama.cpp native /completion endpoint
            url = f"http://{host}:{port}/completion"
            payload = {
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": 0.0,
            }

            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()

            # Validate response contains completion
            data = r.json()
            if "content" not in data or not data["content"]:
                log(f"ERROR: Warmup got unexpected response from {backend}")
                log(f"Response: {data}")
                return False

            generated = len(data["content"])
            log(f"Warmup generated {generated} characters")
            return True

        else:
            log(f"Unknown backend '{backend}' - skipping warmup")
            return False

    except Exception as e:
        # Log error clearly (not as "non-fatal")
        log(f"ERROR: Warmup request failed: {e}")
        return False
