#!/usr/bin/env bash
# Whole pipeline on 120/24/24 questions (~10-15 min on a 6 GB GPU). Writes processed_data_smoke/ and outputs_smoke/.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pytest -q tests
bash scripts/prepare_data.sh --config configs/smoke.yaml "$@"
bash scripts/run_all.sh --config configs/smoke.yaml "$@"
