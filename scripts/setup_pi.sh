#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
if [ "$(uname -m)" != aarch64 ]; then
  echo 'Requires 64-bit Raspberry Pi OS (aarch64).'; exit 1
fi
sudo apt-get update
sudo apt-get install -y python3-venv libgl1 libglib2.0-0
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
export PIP_DEFAULT_TIMEOUT=120
export PIP_RETRIES=5
# 必须先安装 CPU 包。只传 device=cpu 不会改变 pip 的依赖选择。
export PIP_CONSTRAINT="$PWD/constraints-cpu.txt"
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.11.0+cpu' 'torchvision==0.26.0+cpu'
.venv/bin/python -m pip install -e '.[pi]'
.venv/bin/python -m pip check
.venv/bin/python - <<'PY'
import torch
if torch.version.cuda is not None or '+cpu' not in torch.__version__:
    raise RuntimeError('Expected CPU-only PyTorch; stopping before model export.')
print('PyTorch:', torch.__version__)
PY
.venv/bin/python -m vision_servo_tracker.models
.venv/bin/python -m vision_servo_tracker.export --imgsz 416
echo 'Setup finished. Start: .venv/bin/python -m vision_servo_tracker.web'
