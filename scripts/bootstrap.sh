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
echo "[1/2] Syncing Python environment from uv.lock..."
uv sync --all-extras

echo "[2/2] Downloading models listed in config/models.yaml..."
uv run python scripts/fetch_models.py

echo
echo "Bootstrap complete."
