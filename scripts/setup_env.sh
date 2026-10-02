#!/usr/bin/env bash
# Create .venv with the tested stack. Everything (uv, Python 3.12, package cache, models) stays
# inside the repo. Tested on Ubuntu 26.04 LTS (system Python 3.14 is not used).
#   bash scripts/setup_env.sh          # CUDA 12.4 wheels (NVIDIA driver >= 550)
#   bash scripts/setup_env.sh cu118    # older drivers
set -euo pipefail
source "$(dirname "$0")/env.sh"
cuda="${1:-cu124}"

if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
else
  echo "WARNING: nvidia-smi not found: install the NVIDIA driver (sudo ubuntu-drivers install)" >&2
fi
for tool in curl git; do
  command -v "${tool}" >/dev/null || { echo "missing ${tool}: sudo apt install ${tool}" >&2; exit 1; }
done

if [[ ! -x "${UV}" ]]; then
  echo "Installing uv into ${UV_UNMANAGED_INSTALL}"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
"${UV}" --version

[[ -x .venv/bin/python ]] || "${UV}" venv .venv --python 3.12
"${UV}" pip install --python .venv/bin/python torch==2.6.0 --index-url "https://download.pytorch.org/whl/${cuda}"
"${UV}" pip install --python .venv/bin/python -r requirements.txt

.venv/bin/python - <<'EOF'
import torch, spacy
print("python", __import__("sys").version.split()[0], "| torch", torch.__version__,
      "| cuda", torch.cuda.is_available(),
      "|", ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())))
spacy.load("en_core_sci_sm")
print("environment OK")
EOF
echo "uv cache: $("${UV}" cache dir)"
echo "HF cache: ${HF_HOME}"
