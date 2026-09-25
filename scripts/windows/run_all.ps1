# Run the arms (completed arms are skipped, interrupted ones resume), then summarize.
#   pwsh -ExecutionPolicy Bypass -File scripts\windows\run_all.ps1 [--config configs/smoke.yaml] [overrides...]
# Optional, set in the same session before running:
#   $env:FEDICL_GPU  = "1"                                  # GPU index as in nvidia-smi (default: runtime.gpu = 0)
#   $env:FEDICL_ARMS = "centralized_non_icl,centralized_icl"  # subset of arms (default: all 4)
. "$PSScriptRoot\env.ps1"
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
$gpu = if ($env:FEDICL_GPU) { "$($env:FEDICL_GPU) (FEDICL_GPU)" } else { "runtime.gpu from config" }
Write-Host "GPU: $gpu | arms: $($Arms -join ', ')"

foreach ($arm in $Arms) {
    Write-Host "================ $arm ================"
    Invoke-Py -m fedicl.run --arm $arm @Rest
}
Invoke-Py -m fedicl.summarize @Rest
