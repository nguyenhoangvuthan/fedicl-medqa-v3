#!/usr/bin/env bash
# Whole pipeline on 120/24/24 questions (~10-15 min on a 6 GB GPU). Writes processed_data_smoke/ and outputs_smoke/.
set -euo pipefail
source "$(dirname "$0")/env.sh"
[[ -x .venv/bin/python ]] || { echo "no .venv: run bash scripts/setup_env.sh first" >&2; exit 1; }

python -m pytest -q tests
bash scripts/prepare_data.sh --config configs/smoke.yaml "$@"
bash scripts/run_all.sh --config configs/smoke.yaml "$@"
