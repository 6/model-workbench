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
  echo "Install NVIDIA drivers first."
  exit 1
fi

# Ensure models dir
mkdir -p "$HOME/models"

# Sync env from repo lockfile
cd "$ROOT_DIR"

# ============================================================================
# Docker with GPU support (REQUIRED for benchmarks)
# ============================================================================
# All benchmarks run via Docker for reproducibility with version pinning.
# Each model can specify a backend_version in config/models.yaml.
# ============================================================================

DOCKER_GPU_OK=false

if ! command -v docker >/dev/null 2>&1; then
  echo
  echo "ERROR: Docker not installed."
  echo "Docker is REQUIRED for running benchmarks."
  echo
  echo "Install with:"
  echo "  sudo apt-get update && sudo apt-get install -y docker.io"
  echo "  # Add yourself to docker group:"
  echo "  sudo usermod -aG docker \$USER"
  echo "  # Log out and back in, then continue below"
  echo
  exit 1
elif ! docker info >/dev/null 2>&1; then
  echo
  echo "ERROR: Docker installed but not accessible."
  echo
  echo "Either:"
  echo "  - Run: sudo usermod -aG docker \$USER  (then log out/in)"
  echo "  - Or use: sudo docker ..."
  echo
  exit 1
else
  # Check nvidia-container-toolkit
  if ! docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi >/dev/null 2>&1; then
    echo
    echo "ERROR: nvidia-container-toolkit not configured."
    echo "Docker with GPU support is REQUIRED for benchmarks."
    echo
    echo "Install with:"
    echo "  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    echo "  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \\"
    echo "    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \\"
    echo "    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
    echo "  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
    echo "  sudo nvidia-ctk runtime configure --runtime=docker"
    echo "  sudo systemctl restart docker"
    echo
    exit 1
  else
    echo "Docker with GPU support: OK"
    DOCKER_GPU_OK=true
  fi
fi

echo "[1/2] Syncing Python environment..."
uv sync --all-extras

echo "[2/2] Downloading models listed in config/models.yaml..."
uv run python scripts/fetch_models.py

echo
echo "Bootstrap complete."
echo
echo "Run benchmarks with:"
echo "  uv run python scripts/run_bench.py --model ~/models/..."
echo
echo "Backend versions are configured in config/models.yaml:"
echo "  defaults.vllm_version  - for safetensors models"
echo "  defaults.llama_version - for GGUF models"
