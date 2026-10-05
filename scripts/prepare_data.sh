#!/usr/bin/env bash
# Data pipeline: MedQA (or MedMCQA with --config configs/medmcqa.yaml) -> raw/centralized -> partitions -> retrieval features -> demo assignments -> plots.
# Extra args are passed to every step, e.g.:  bash scripts/prepare_data.sh --config configs/smoke.yaml
set -euo pipefail
source "$(dirname "$0")/env.sh"
[[ -x .venv/bin/python ]] || { echo "no .venv: run bash scripts/setup_env.sh first" >&2; exit 1; }

python -m fedicl.data.prepare "$@"
python -m fedicl.data.partition "$@"
python -m fedicl.retrieval.features "$@"
python -m fedicl.retrieval.retrieve "$@"
python -m fedicl.visualize "$@"
python -m fedicl.audit "$@"      # every FL demo must come from the client's own data
