#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

UPDATE_NIGHTLY=false
for arg in "$@"; do
  case $arg in
    --update-nightly) UPDATE_NIGHTLY=true ;;
  esac
done

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
# Check Docker (optional, for Mistral models with FP8 issues)
if ! command -v docker >/dev/null 2>&1; then
  echo
  echo "Note: Docker not installed. Required for some Mistral models."
  echo "Install with:"
  echo "  sudo apt-get update && sudo apt-get install -y docker.io"
  echo "  # Add yourself to docker group:"
  echo "  sudo usermod -aG docker \$USER"
  echo "  # Log out and back in, then continue below"
  echo
elif ! docker info >/dev/null 2>&1; then
  echo
  echo "Note: Docker installed but not accessible. Either:"
  echo "  - Run: sudo usermod -aG docker \$USER  (then log out/in)"
  echo "  - Or use: sudo docker ..."
  echo
else
  # Check nvidia-container-toolkit
  if ! docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi >/dev/null 2>&1; then
    echo
    echo "Note: nvidia-container-toolkit not configured. Required for GPU Docker."
    echo "Install with:"
    echo "  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    echo "  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \\"
    echo "    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \\"
    echo "    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
    echo "  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
    echo "  sudo nvidia-ctk runtime configure --runtime=docker"
    echo "  sudo systemctl restart docker"
    echo
  else
    echo "Docker with GPU support: OK"
  fi
fi

if $UPDATE_NIGHTLY; then
  echo "[1/3] Syncing stable Python environment..."
else
  echo "[1/2] Syncing stable Python environment..."
fi
uv sync --all-extras

if $UPDATE_NIGHTLY; then
  echo "[2/3] Syncing nightly Python environment (for bleeding-edge models)..."
  # Upgrade git dependencies to latest commits, then sync
  (cd "$ROOT_DIR/nightly" && uv lock --upgrade-package transformers --upgrade-package tokenizers && uv sync)
  echo "[3/3] Downloading models listed in config/models.yaml..."
else
  echo "[2/2] Downloading models listed in config/models.yaml..."
fi
uv run python scripts/fetch_models.py

echo
echo "Bootstrap complete."
echo
echo "Environments created:"
echo "  - .venv          (stable transformers/tokenizers)"
echo "  - nightly/.venv  (git master transformers/tokenizers)"
echo
echo "Models with 'nightly: true' in config/models.yaml will auto-use nightly/.venv"
