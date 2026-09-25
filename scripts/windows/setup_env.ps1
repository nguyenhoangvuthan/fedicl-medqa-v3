# Create .venv with the tested stack; uv + every cache live inside the repo folder.
#   powershell -ExecutionPolicy Bypass -File scripts\windows\setup_env.ps1
# Optional: -Cuda cu124 (default; needs NVIDIA driver >= 550) or cu118 for older drivers.
param([string]$Cuda = "cu124")
. "$PSScriptRoot\env.ps1"

# Deep paths (uv cache, torch headers) can exceed the 260-char MAX_PATH limit.
$lp = (Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -ErrorAction SilentlyContinue).LongPathsEnabled
if ($lp -ne 1) {
    Write-Warning "Windows long paths are disabled. If installing torch fails with a path error, run once as Administrator:"
    Write-Warning '  New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force'
    Write-Warning "or clone the repo close to the drive root (e.g. D:\fedicl-medqa)."
}

& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
if ($LASTEXITCODE -ne 0) { Write-Warning "nvidia-smi failed: is the NVIDIA driver installed?" }

if (-not (Test-Path $Uv)) {
    Write-Host "Installing uv into $($env:UV_UNMANAGED_INSTALL)"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
}
Invoke-Checked $Uv --version

if (-not (Test-Path $Py)) {
    Invoke-Checked $Uv venv .venv --python 3.12
}
Invoke-Checked $Uv pip install --python $Py torch==2.6.0 --index-url "https://download.pytorch.org/whl/$Cuda"
Invoke-Checked $Uv pip install --python $Py -r requirements.txt

Invoke-Py -c "import torch, spacy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''); spacy.load('en_core_sci_sm'); print('environment OK')"
Write-Host "uv cache: $(& $Uv cache dir)"
Write-Host "HF cache: $($env:HF_HOME)"
