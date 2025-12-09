#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== model-workbench bootstrap =="

# Require uv
if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is not installed."
  echo "Install uv first, then re-run bootstrap."
  exit 1
fi

# Require NVIDIA driver
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Error: nvidia-smi not found."
  echo "Install NVIDIA drivers first (you already did on this machine)."
  exit 1
fi

# Ensure models dir
mkdir -p "$HOME/models"

# Sync env from repo lockfile
cd "$ROOT_DIR"
echo "[1/3] Syncing stable Python environment..."
uv sync --all-extras

echo "[2/3] Syncing nightly Python environment (for bleeding-edge models)..."
uv sync --all-extras --config pyproject.nightly.toml --python-venv .venv-nightly

echo "[3/3] Downloading models listed in config/models.yaml..."
uv run python scripts/fetch_models.py

echo
echo "Bootstrap complete."
echo
echo "Environments created:"
echo "  - .venv         (stable transformers/tokenizers)"
echo "  - .venv-nightly (git master transformers/tokenizers)"
echo
echo "Models with 'nightly: true' in config/models.yaml will auto-use .venv-nightly"
