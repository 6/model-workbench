"""SOPS-encrypted prompt handling for benchmark challenges.

This module provides access to encrypted prompts stored in secret.yaml,
which is encrypted using SOPS with age keys. Keys are visible in the
encrypted file, but values are encrypted.

Setup:
    1. Install: brew install sops age  (or equivalent)
    2. Generate key: age-keygen -o ~/.config/sops/age/keys.txt
    3. Set env: export SOPS_AGE_KEY_FILE="$HOME/.config/sops/age/keys.txt"
    4. Edit prompts: sops prompts/secret.yaml

Usage:
    from prompts.secret import get_secret_prompt
    prompt = get_secret_prompt("CHALLENGE_1")
"""

import hashlib
import subprocess
from pathlib import Path

import yaml

SECRET_FILE = Path(__file__).parent / "secret.yaml"


class SecretPromptError(Exception):
    """Error loading or decrypting secret prompts."""

    pass


def _decrypt_sops() -> dict:
    """Decrypt secret.yaml using SOPS."""
    if not SECRET_FILE.exists():
        raise SecretPromptError(
            f"Secret file not found: {SECRET_FILE}\nCreate it with: sops prompts/secret.yaml"
        )

    result = subprocess.run(
        ["sops", "--decrypt", str(SECRET_FILE)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "executable file not found" in stderr or "not found" in stderr:
            raise SecretPromptError("SOPS not installed. Install with: brew install sops age")
        if "could not decrypt" in stderr.lower() or "no key" in stderr.lower():
            raise SecretPromptError(
                f"SOPS decryption failed (missing key?):\n{stderr}\n\n"
                "Ensure SOPS_AGE_KEY_FILE is set in .envrc"
            )
        raise SecretPromptError(f"SOPS decryption failed: {stderr}")

    return yaml.safe_load(result.stdout)


def get_secret_prompt(key: str) -> str:
    """Get a decrypted prompt by key name.

    Args:
        key: The prompt key (e.g., "CHALLENGE_1")

    Returns:
        The decrypted prompt text.

    Raises:
        SecretPromptError: If decryption fails or key not found.
    """
    data = _decrypt_sops()
    prompts = data.get("prompts", {})

    if key not in prompts:
        available = ", ".join(sorted(prompts.keys())) or "(none)"
        raise SecretPromptError(f"Prompt '{key}' not found. Available: {available}")

    return prompts[key]


def list_secret_prompts() -> list[str]:
    """List available secret prompt keys.

    Requires decryption to read the file.

    Returns:
        List of prompt keys, or empty list if decryption fails.
    """
    try:
        data = _decrypt_sops()
        return sorted(data.get("prompts", {}).keys())
    except Exception:
        return []


def prompt_hash(text: str) -> str:
    """Generate a short hash of a prompt for result storage.

    Used to identify prompts in benchmark results without storing
    the actual prompt text (to prevent contamination).

    Args:
        text: The prompt text.

    Returns:
        First 12 characters of SHA256 hash.
    """
    return hashlib.sha256(text.encode()).hexdigest()[:12]
