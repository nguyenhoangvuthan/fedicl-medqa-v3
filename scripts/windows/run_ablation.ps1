# Verification ablations for the federated ICL gain. Results go to their own folders
# (outputs\ablation_<name>\...), the main results in outputs\qwen3-0.6b\ are never touched.
#
#   no_licl : federated_icl with lambda_ICL = 0       (does L_ICL make the model ignore demos?)
#   k2      : federated_{non_,}icl with 2 clients     (does a larger local demo repository help?)
#
# One ablation per GPU, in two PowerShell windows (about 8 h and 15 h on an A5000):
#   $env:FEDICL_GPU = "0"; pwsh -ExecutionPolicy Bypass -File scripts\windows\run_ablation.ps1 -Ablation no_licl
#   $env:FEDICL_GPU = "1"; pwsh -ExecutionPolicy Bypass -File scripts\windows\run_ablation.ps1 -Ablation k2
# Re-running continues where it stopped (completed arms are skipped).
param([Parameter(Mandatory = $true)][ValidateSet("no_licl", "k2")][string]$Ablation)
. "$PSScriptRoot\env.ps1"

$cfgFile = "configs/ablations/$Ablation.yaml"
$abRoot  = ((Invoke-Py -m fedicl.config --get save.root --config $cfgFile) | Select-Object -Last 1).Trim()
Write-Host "ablation $Ablation -> $abRoot"

if ($Ablation -eq "k2") {
    # demo assignments for the 2-client partition (idempotent; reuses features and partitions)
    & "$PSScriptRoot\prepare_data.ps1" --config $cfgFile
    $env:FEDICL_ARMS = "federated_non_icl,federated_icl"
} else {
    $env:FEDICL_ARMS = "federated_icl"   # non-ICL baseline is unchanged: reuse the main run
}

& "$PSScriptRoot\run_all.ps1" --config $cfgFile

foreach ($ckpt in @("best", "round_3")) {
    if ($Ablation -eq "k2") {
        Invoke-Py -m fedicl.compare --checkpoint $ckpt --out "$abRoot/compare_$ckpt.csv" `
            --add "k2_icl=$abRoot/federated_icl" --add "k2_non_icl=$abRoot/federated_non_icl" `
            --vs "k2_icl,k2_non_icl" --vs "k2_icl,federated_icl" --vs "k2_non_icl,federated_non_icl"
    } else {
        Invoke-Py -m fedicl.compare --checkpoint $ckpt --out "$abRoot/compare_$ckpt.csv" `
            --add "no_licl=$abRoot/federated_icl" `
            --vs "no_licl,federated_non_icl" --vs "no_licl,federated_icl"
    }
}
