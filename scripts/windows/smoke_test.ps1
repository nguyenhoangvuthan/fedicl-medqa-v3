# Whole pipeline on 120/24/24 questions (~10 min). Writes processed_data_smoke\ and outputs_smoke\.
#   pwsh -ExecutionPolicy Bypass -File scripts\windows\smoke_test.ps1
. "$PSScriptRoot\env.ps1"
$Rest = $args

Invoke-Py -m pytest -q tests
& "$PSScriptRoot\prepare_data.ps1" --config configs/smoke.yaml @Rest
& "$PSScriptRoot\run_all.ps1" --config configs/smoke.yaml @Rest
