#!/usr/bin/env bash
# Setup alpr API on Linux CPU (Ubuntu 22.04+).
# Usage: bash deploy/linux-cpu/setup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "==> Installing system packages (sudo)..."
sudo apt-get update -y
sudo apt-get install -y python3.12 python3.12-venv python3-pip

echo "==> Creating venv..."
python3.12 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip
pip install -r requirements-linux-cpu.txt

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "==> Wrote .env from .env.example (ALPR_MODE=fast, ALPR_PROVIDERS=cpu)"
fi

echo "==> Smoke check..."
python -c "import onnxruntime as ort; print('providers:', ort.get_available_providers())"
python -m python.cli info || true

echo ""
echo "Setup done. Start with:"
echo "  source .venv/bin/activate"
echo "  set -a; source .env; set +a"
echo "  python -m python.api"
echo ""
echo "Or install systemd:"
echo "  sudo bash deploy/linux-cpu/install-systemd.sh"
