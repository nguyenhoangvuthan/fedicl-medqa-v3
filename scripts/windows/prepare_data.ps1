# Data pipeline: MedQA -> raw/centralized -> partitions -> features -> demo assignments -> plots.
#   pwsh -ExecutionPolicy Bypass -File scripts\windows\prepare_data.ps1 [--config configs/smoke.yaml] [overrides...]
. "$PSScriptRoot\env.ps1"
$Rest = $args

Invoke-Py -m fedicl.data.prepare @Rest
Invoke-Py -m fedicl.data.partition @Rest
Invoke-Py -m fedicl.retrieval.features @Rest
Invoke-Py -m fedicl.retrieval.retrieve @Rest
Invoke-Py -m fedicl.visualize @Rest
