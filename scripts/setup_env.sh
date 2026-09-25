#!/usr/bin/env bash
# Create .venv with the tested stack. Needs an NVIDIA driver that supports CUDA 12.4 (A5000: driver >= 550).
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v uv >/dev/null 2>&1; then
  uv venv -p 3.12 .venv
  source .venv/bin/activate
  uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  uv pip install -r requirements.txt
else
  python3.12 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  pip install -r requirements.txt
fi

python - <<'EOF'
import torch, spacy
print("torch", torch.__version__, "cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
spacy.load("en_core_sci_sm")
print("environment OK")
EOF
