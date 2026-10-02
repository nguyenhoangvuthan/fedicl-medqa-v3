# Run the arms (completed arms are skipped, interrupted ones resume), then summarize.
#   pwsh -ExecutionPolicy Bypass -File scripts\windows\run_all.ps1 [-Gpu 0|1] [--config configs/smoke.yaml] [overrides...]
# Optional, set in the same session before running:
#   $env:FEDICL_GPU  = "1"                                  # same as -Gpu 1 (default: runtime.gpu = 0)
#   $env:FEDICL_ARMS = "centralized_non_icl,centralized_icl"  # subset of arms (default: all 4)
param([ValidateSet("0", "1")][string]$Gpu = "")   # -Gpu 0 or -Gpu 1 (nvidia-smi index)
. "$PSScriptRoot\env.ps1"
Use-Gpu $Gpu
$Rest = $args

# Non-ICL arms first: ~4x cheaper, they give the baselines early.
$AllArms = @("centralized_non_icl", "federated_non_icl", "centralized_icl", "federated_icl")
$Arms = $AllArms
if ($env:FEDICL_ARMS) {
    $Arms = @($env:FEDICL_ARMS -split '[,; ]+' | Where-Object { $_ -ne "" })
    foreach ($a in $Arms) {
        if ($AllArms -notcontains $a) { throw "unknown arm '$a' in FEDICL_ARMS (valid: $($AllArms -join ', '))" }
    }
}
# NB: no local variable may be called $gpu: PowerShell names are case-insensitive, so it would be the
# -Gpu parameter and inherit its ValidateSet. The GPU in use is printed by Use-Gpu above.
Write-Host "arms: $($Arms -join ', ')"

foreach ($arm in $Arms) {
    Write-Host "================ $arm ================"
    Invoke-Py -m fedicl.run --arm $arm @Rest
}
Invoke-Py -m fedicl.summarize @Rest
Invoke-Py -m fedicl.audit @Rest      # re-check, now including what the FL model was shown at eval
