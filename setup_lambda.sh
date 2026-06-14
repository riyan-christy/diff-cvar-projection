#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-shot environment setup for a Lambda Cloud A100 instance.
#
#   Recommended instance : 1x A100 (40 GB SXM4), 30 vCPU, 200 GiB RAM, 0.5 TiB SSD
#   Recommended image    : "Lambda Stack 22.04"  (Ubuntu 22.04, Python 3.10,
#                          NVIDIA driver + CUDA + cuDNN preinstalled and verified)
#
# Usage (on the instance, from the repo root):
#   bash setup_lambda.sh
#   source .venv/bin/activate
#   bash run_all.sh
# ---------------------------------------------------------------------------
set -euo pipefail

echo "==> host / GPU"
uname -a || true
nvidia-smi || { echo "WARNING: nvidia-smi failed — is this a GPU instance with the driver?"; }

PY=python3
echo "==> python: $($PY --version)"

echo "==> creating isolated venv (.venv) for byte-reproducibility"
$PY -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

echo "==> installing dependencies (this compiles diffcp; takes a few min)"
if [ -f requirements.lock.txt ]; then
  echo "    using requirements.lock.txt (byte-exact versions from the reference A100 run)"
  pip install -r requirements.lock.txt
else
  echo "    no lock file found; using requirements.txt (human-readable pins)"
  pip install -r requirements.txt
fi

echo "==> sanity: torch sees the GPU"
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY

echo "==> recording this machine's exact freeze -> results/pip_freeze_thisrun.txt"
mkdir -p results
pip freeze > results/pip_freeze_thisrun.txt   # compare against the committed requirements.lock.txt

echo "==> capturing environment metadata -> results/env.json"
python env_capture.py

echo ""
echo "Setup complete. Next:"
echo "  source .venv/bin/activate"
echo "  bash run_all.sh        # reproduces every figure/table into results/"
