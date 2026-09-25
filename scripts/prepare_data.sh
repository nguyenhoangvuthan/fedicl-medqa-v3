#!/usr/bin/env bash
# Data pipeline: MedQA -> raw/centralized -> partitions -> retrieval features -> demo assignments -> plots.
# Extra args are passed to every step, e.g.:  bash scripts/prepare_data.sh --config configs/smoke.yaml
set -euo pipefail
cd "$(dirname "$0")/.."

python -m fedicl.data.prepare "$@"
python -m fedicl.data.partition "$@"
python -m fedicl.retrieval.features "$@"
python -m fedicl.retrieval.retrieve "$@"
python -m fedicl.visualize "$@"
