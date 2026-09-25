#!/usr/bin/env bash
# Run the arms (completed arms are skipped, interrupted ones resume), then summarize.
# Extra args are passed through, e.g.:  bash scripts/run_all.sh --config configs/smoke.yaml
# Optional env: FEDICL_GPU=1 (GPU index as in nvidia-smi), FEDICL_ARMS="centralized_non_icl,centralized_icl"
set -euo pipefail
cd "$(dirname "$0")/.."

# Non-ICL arms first: they are ~4x cheaper and give the baselines early.
all_arms=(centralized_non_icl federated_non_icl centralized_icl federated_icl)
arms=("${all_arms[@]}")
if [[ -n "${FEDICL_ARMS:-}" ]]; then
  IFS=', ;' read -r -a arms <<< "${FEDICL_ARMS}"
  for a in "${arms[@]}"; do
    [[ " ${all_arms[*]} " == *" ${a} "* ]] || { echo "unknown arm '${a}' in FEDICL_ARMS" >&2; exit 2; }
  done
fi
echo "GPU: ${FEDICL_GPU:-runtime.gpu from config} | arms: ${arms[*]}"

for arm in "${arms[@]}"; do
  [[ -z "${arm}" ]] && continue
  echo "================ ${arm} ================"
  python -m fedicl.run --arm "${arm}" "$@"
done
python -m fedicl.summarize "$@"
