# Whole pipeline on 120/24/24 questions (~10 min). Writes processed_data_smoke\ and outputs_smoke\.
#   pwsh -ExecutionPolicy Bypass -File scripts\windows\smoke_test.ps1 [-Gpu 0|1]
param([ValidateSet("0", "1")][string]$Gpu = "")   # -Gpu 0 or -Gpu 1 (nvidia-smi index)
. "$PSScriptRoot\env.ps1"
Use-Gpu $Gpu
$Rest = $args

Invoke-Py -m pytest -q tests
Invoke-Script "$PSScriptRoot\prepare_data.ps1" --config configs/smoke.yaml @Rest
Invoke-Script "$PSScriptRoot\run_all.ps1" --config configs/smoke.yaml @Rest
