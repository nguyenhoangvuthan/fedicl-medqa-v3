#!/usr/bin/env bash
# Verification ablations (see scripts/windows/run_ablation.ps1 for details).
#   FEDICL_GPU=0 bash scripts/run_ablation.sh no_licl
#   FEDICL_GPU=1 bash scripts/run_ablation.sh k2
# Extra args after the name go to every step (e.g. --config configs/smoke.yaml for a dry run).
set -euo pipefail
cd "$(dirname "$0")/.."
ab="${1:?usage: run_ablation.sh no_licl|k2 [extra args]}"; shift
[[ "${ab}" == no_licl || "${ab}" == k2 ]] || { echo "unknown ablation '${ab}'" >&2; exit 2; }
cfg="configs/ablations/${ab}.yaml"
ab_root=$(python -m fedicl.config --get save.root "$@" --config "${cfg}" | tail -n 1)
echo "ablation ${ab} -> ${ab_root}"

if [[ "${ab}" == k2 ]]; then
  bash scripts/prepare_data.sh "$@" --config "${cfg}"
  export FEDICL_ARMS="federated_non_icl,federated_icl"
else
  export FEDICL_ARMS="federated_icl"   # non-ICL baseline is unchanged: reuse the main run
fi
bash scripts/run_all.sh "$@" --config "${cfg}"

for ckpt in best round_3; do
  if [[ "${ab}" == k2 ]]; then
    python -m fedicl.compare "$@" --checkpoint "${ckpt}" --out "${ab_root}/compare_${ckpt}.csv" \
      --add "k2_icl=${ab_root}/federated_icl" --add "k2_non_icl=${ab_root}/federated_non_icl" \
      --vs k2_icl,k2_non_icl --vs k2_icl,federated_icl --vs k2_non_icl,federated_non_icl
  else
    python -m fedicl.compare "$@" --checkpoint "${ckpt}" --out "${ab_root}/compare_${ckpt}.csv" \
      --add "no_licl=${ab_root}/federated_icl" \
      --vs no_licl,federated_non_icl --vs no_licl,federated_icl
  fi
done
