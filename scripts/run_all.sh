#!/usr/bin/env bash
# Run the 4 arms (completed arms are skipped, interrupted arms resume), then summarize.
# Extra args are passed through, e.g.:  bash scripts/run_all.sh --config configs/smoke.yaml
set -euo pipefail
cd "$(dirname "$0")/.."

# Non-ICL arms first: they are ~4x cheaper and give the baselines early.
for arm in centralized_non_icl federated_non_icl centralized_icl federated_icl; do
  echo "================ ${arm} ================"
  python -m fedicl.run --arm "${arm}" "$@"
done
python -m fedicl.summarize "$@"
