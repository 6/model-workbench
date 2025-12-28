"""
Benchmark prompts and test images.
"""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent

# ----------------------------
# Text prompts
# ----------------------------

TEXT_PROMPTS = {
    "short": "Explain speculative decoding in 2 sentences.",
    "medium": "Summarize key tradeoffs between tensor parallelism and pipeline parallelism.",
    "long": "Write a concise technical overview of KV cache and why it matters for long context.",
    "longctx": None,  # Special: loaded dynamically via get_longctx_prompt()
}

# ----------------------------
# Vision prompts
# ----------------------------

VISION_PROMPTS = {
    "describe": "Describe this image in detail.",
    "analyze": "Analyze this image and explain what you see.",
    "caption": "Provide a brief caption for this image.",
}

# Combined prompts for CLI choices
ALL_PROMPTS = {**TEXT_PROMPTS, **VISION_PROMPTS}

# ----------------------------
# Test images
# ----------------------------

BUILTIN_IMAGES = {
    "example": PROMPTS_DIR / "example.jpg",
    "grayscale": "https://upload.wikimedia.org/wikipedia/commons/f/fa/Grayscale_8bits_palette_sample_image.png",
}


# ----------------------------
# Long-context prompt loader
# ----------------------------


def get_longctx_prompt() -> str:
    """Load long-context code analysis prompt (~13K tokens).

    Uses Python 3.12's collections/__init__.py as a stable, complex code sample.
    """
    code_file = PROMPTS_DIR / "longctx_collections.py"
    if not code_file.exists():
        raise FileNotFoundError(
            f"Long-context prompt file not found: {code_file}\n"
            "Run: curl -sL https://raw.githubusercontent.com/python/cpython/v3.12.0/Lib/collections/__init__.py "
            f"-o {code_file}"
        )
    code = code_file.read_text()
    return f"Analyze this Python code and summarize the key data structures and their use cases:\n\n```python\n{code}\n```"


# ----------------------------
# Secret (encrypted) prompts
# ----------------------------

from prompts.secret import (
    get_secret_prompt as get_secret_prompt,
)
from prompts.secret import (
    list_secret_prompts as list_secret_prompts,
)
from prompts.secret import (
    prompt_hash as prompt_hash,
)
