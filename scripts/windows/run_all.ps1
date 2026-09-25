# Run the 4 arms (completed arms are skipped, interrupted ones resume), then summarize.
#   pwsh -ExecutionPolicy Bypass -File scripts\windows\run_all.ps1 [--config configs/smoke.yaml] [overrides...]
. "$PSScriptRoot\env.ps1"
$Rest = $args

# Non-ICL arms first: ~4x cheaper, they give the baselines early.
foreach ($arm in @("centralized_non_icl", "federated_non_icl", "centralized_icl", "federated_icl")) {
    Write-Host "================ $arm ================"
    Invoke-Py -m fedicl.run --arm $arm @Rest
}
Invoke-Py -m fedicl.summarize @Rest
