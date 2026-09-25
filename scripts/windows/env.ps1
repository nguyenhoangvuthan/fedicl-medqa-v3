# Shared environment for every Windows script. Dot-source it:  . "$PSScriptRoot\env.ps1"
# Keeps ALL caches inside the cloned repo (same drive as the repo), nothing under C:\Users\...
# Compatible with Windows PowerShell 5.1 (keep this file ASCII-only: 5.1 reads BOM-less files as ANSI).

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

# --- uv: binary, package cache, managed Python -----------------------------------------------
$env:UV_UNMANAGED_INSTALL   = Join-Path $Root ".uv-bin"      # uv.exe (no PATH edit, no self-updater)
$env:UV_CACHE_DIR           = Join-Path $Root ".uv-cache"    # wheels/sdists; same drive as .venv => hardlinks
$env:UV_PYTHON_INSTALL_DIR  = Join-Path $Root ".uv-python"   # Python 3.12 downloaded by uv
$env:UV_PYTHON_BIN_DIR      = Join-Path $Root ".uv-python\bin"
$env:UV_PYTHON_PREFERENCE   = "only-managed"                 # never pick a Python installed on C:
$env:UV_LINK_MODE           = "hardlink"

# --- model/data downloads and temp files ----------------------------------------------------
$env:HF_HOME                          = Join-Path $Root ".hf-cache"   # Qwen3, encoders, MedQA
$env:HF_HUB_DISABLE_SYMLINKS_WARNING  = "1"
$env:TMP                              = Join-Path $Root ".tmp"
$env:TEMP                             = $env:TMP
New-Item -ItemType Directory -Force -Path $env:TMP | Out-Null

# --- smaller library caches that otherwise default to C:\Users\<user>\... -----------------------
$Cache = Join-Path $Root ".cache"
$env:XDG_CACHE_HOME   = $Cache                              # generic cache dir used by several libs
$env:MPLCONFIGDIR     = Join-Path $Cache "matplotlib"       # matplotlib font cache (%USERPROFILE%\.matplotlib)
$env:TORCH_HOME       = Join-Path $Cache "torch"            # torch hub / checkpoints
$env:CUDA_CACHE_PATH  = Join-Path $Cache "nv-compute"       # NVIDIA JIT cache (%APPDATA%\NVIDIA\ComputeCache)

# --- Python runtime --------------------------------------------------------------------------
$env:PYTHONUTF8       = "1"        # UTF-8 everywhere regardless of the console code page
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS   = "ignore"

$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Uv = Join-Path $env:UV_UNMANAGED_INSTALL "uv.exe"

# Simple functions on purpose (no [Parameter()] attributes): an advanced function would try to
# bind "-m" / "-c" as its own parameter names instead of passing them through to python.
function Invoke-Checked {
    # Run a native command and stop on a non-zero exit code
    # (PowerShell 5.1 does not do this even with ErrorActionPreference=Stop).
    $exe = $args[0]
    $rest = @($args | Select-Object -Skip 1)
    & $exe @rest
    if ($LASTEXITCODE -ne 0) {
        throw "FAILED (exit $LASTEXITCODE): $exe $($rest -join ' ')"
    }
}

# --- Hugging Face token ---------------------------------------------------------------------
# Default: <repo>\HF_Access_Token or <repo>\HF_Access_Token.txt (Notepad often adds .txt).
# Override with an absolute path:  $env:FEDICL_HF_TOKEN_FILE = "D:\secrets\HF_Access_Token.txt"
# Scanned before every run, then exported as HF_TOKEN for THIS process tree only.
# The token itself is never printed, logged, or passed on a command line.
$HfTokenNames = @("HF_Access_Token", "HF_Access_Token.txt")

function Test-HfToken {
    if ($env:FEDICL_HF_SCANNED -eq "1") { return }      # scan once per session (set only on success)

    if ($env:FEDICL_HF_TOKEN_FILE) {
        if (-not (Test-Path -LiteralPath $env:FEDICL_HF_TOKEN_FILE -PathType Leaf)) {
            throw "[HF token] FEDICL_HF_TOKEN_FILE points to a missing file: $($env:FEDICL_HF_TOKEN_FILE)"
        }
        $HfTokenFile = (Resolve-Path -LiteralPath $env:FEDICL_HF_TOKEN_FILE).Path
    } else {
        $found = @($HfTokenNames | Where-Object { Test-Path (Join-Path $Root $_) })
        if ($found.Count -eq 0) {
            Write-Host "[HF token] scanning $Root for $($HfTokenNames -join ' / ')"
            Write-Warning "[HF token] file not found: downloads run anonymously (slower, rate-limited)."
            $env:FEDICL_HF_SCANNED = "1"
            return
        }
        if ($found.Count -gt 1) {
            throw "[HF token] both HF_Access_Token and HF_Access_Token.txt exist: keep only one."
        }
        $HfTokenFile = Join-Path $Root $found[0]
    }
    $name = Split-Path -Leaf $HfTokenFile
    Write-Host "[HF token] scanning $HfTokenFile"

    # 1) Must never be committed (only relevant when the file lives inside the repo).
    $rootPrefix = $Root.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    if ($HfTokenFile.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        $rel = $HfTokenFile.Substring($rootPrefix.Length) -replace '\\', '/'
        $gitignore = Join-Path $Root ".gitignore"
        $pattern = '^/?' + [regex]::Escape($rel) + '$'
        if (-not ((Test-Path $gitignore) -and (Select-String -LiteralPath $gitignore -Pattern $pattern -Quiet))) {
            throw "[HF token] add a line '$rel' to .gitignore before using the token."
        }
        if ((Test-Path (Join-Path $Root ".git")) -and (Get-Command git -ErrorAction SilentlyContinue)) {
            # "ls-files -- <path>" prints the path only if tracked and writes nothing to stderr.
            # (Windows PowerShell 5.1 turns ANY native stderr line into a terminating error under
            # ErrorActionPreference=Stop, even with 2>$null, so avoid stderr and relax it locally.)
            $eap = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            $tracked = & git -C $Root ls-files -- $rel 2>$null
            $ErrorActionPreference = $eap
            if ($tracked) {
                throw "[HF token] $rel is tracked by git. Run: git rm --cached $rel, then revoke the token on huggingface.co (it is in git history)."
            }
        }
    }

    # 2) Content: one token, optionally written as HF_TOKEN=hf_...; ignore BOM, blank lines, # comments.
    $lines = @(Get-Content -LiteralPath $HfTokenFile | ForEach-Object { $_.Trim([char]0xFEFF).Trim() } |
               Where-Object { $_ -ne "" -and -not $_.StartsWith("#") })
    if ($lines.Count -ne 1) {
        throw "[HF token] expected exactly 1 non-comment line in $name, found $($lines.Count)."
    }
    $token = $lines[0] -replace '^(HF_TOKEN|HUGGING_FACE_HUB_TOKEN)\s*=\s*', ''
    $token = $token.Trim('"', "'", ' ')
    if ($token -notmatch '^hf_[A-Za-z0-9]{30,}$') {
        throw "[HF token] content does not look like a Hugging Face token (expected hf_ + letters/digits, got $($token.Length) chars). Nothing was sent anywhere."
    }

    # 3) Online check against huggingface.co only (read-only whoami endpoint).
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $me = Invoke-RestMethod -Uri "https://huggingface.co/api/whoami-v2" -TimeoutSec 20 `
                                -Headers @{ Authorization = "Bearer $token" }
        $role = $me.auth.accessToken.role
        Write-Host "[HF token] OK: user '$($me.name)', token role '$role'"
        if ($role -and $role -ne "read") {
            Write-Warning "[HF token] role is '$role'. This project only downloads public models: a read-only token is enough and safer."
        }
    } catch {
        $code = $null
        try { $code = [int]$_.Exception.Response.StatusCode } catch { }
        if ($code -eq 401) {
            # Do NOT export a rejected token: huggingface_hub would send it and even public downloads
            # would fail. All models/datasets of this project are public, so continue anonymously.
            Write-Warning "[HF token] rejected by huggingface.co (401): invalid, expired, revoked or incompletely copied."
            Write-Warning "[HF token] token in $name has $($token.Length) chars (a classic HF token is 37: 'hf_' + 34). Create a new READ token at https://huggingface.co/settings/tokens and paste it again."
            Write-Warning "[HF token] continuing WITHOUT a token (anonymous downloads, rate-limited)."
            Remove-Variable token
            $env:FEDICL_HF_SCANNED = "1"
            return
        }
        Write-Warning "[HF token] could not verify online (network?): continuing with the token as-is."
    }

    $env:HF_TOKEN = $token        # read by huggingface_hub / transformers / datasets
    Remove-Variable token
    $env:FEDICL_HF_SCANNED = "1"
}

function Invoke-Py {
    if (-not (Test-Path $Py)) { throw ".venv not found: run scripts\windows\setup_env.ps1 first" }
    Invoke-Checked $Py @args
}

Set-Location $Root
Test-HfToken
