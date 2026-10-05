# Data pipeline: MedQA (or MedMCQA with --config configs/medmcqa.yaml) -> raw/centralized -> partitions -> features -> demo assignments -> plots.
#   pwsh -ExecutionPolicy Bypass -File scripts\windows\prepare_data.ps1 [-Gpu 0|1] [--config configs/smoke.yaml] [overrides...]
param([ValidateSet("0", "1")][string]$Gpu = "")   # -Gpu 0 or -Gpu 1 (nvidia-smi index)
. "$PSScriptRoot\env.ps1"
Use-Gpu $Gpu
$Rest = $args

Invoke-Py -m fedicl.data.prepare @Rest
Invoke-Py -m fedicl.data.partition @Rest
Invoke-Py -m fedicl.retrieval.features @Rest
Invoke-Py -m fedicl.retrieval.retrieve @Rest
Invoke-Py -m fedicl.visualize @Rest
Invoke-Py -m fedicl.audit @Rest      # every FL demo must come from the client's own data
