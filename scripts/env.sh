# Shared environment for every Linux script:  source "$(dirname "$0")/env.sh"
# Also usable by hand from the repo root:      source scripts/env.sh
# Keeps uv, Python, the package cache, Hugging Face downloads and temp files inside the repo,
# puts .venv first on PATH, and scans the Hugging Face token before anything runs.

FEDICL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${FEDICL_ROOT}"

# --- uv: binary, package cache, managed Python ---------------------------------------------
export UV_UNMANAGED_INSTALL="${FEDICL_ROOT}/.uv-bin"     # uv binary (no shell-profile edits)
export UV_CACHE_DIR="${FEDICL_ROOT}/.uv-cache"           # same filesystem as .venv => hardlinks
export UV_PYTHON_INSTALL_DIR="${FEDICL_ROOT}/.uv-python" # Python 3.12 (Ubuntu 26.04 ships 3.14)
export UV_PYTHON_BIN_DIR="${FEDICL_ROOT}/.uv-python/bin"
export UV_PYTHON_PREFERENCE="only-managed"
export UV_LINK_MODE="hardlink"
UV="${UV_UNMANAGED_INSTALL}/uv"

# --- downloads, caches, temp ------------------------------------------------------------
export HF_HOME="${FEDICL_ROOT}/.hf-cache"                # Qwen3, encoders, MedQA
export XDG_CACHE_HOME="${FEDICL_ROOT}/.cache"
export MPLCONFIGDIR="${FEDICL_ROOT}/.cache/matplotlib"
export TORCH_HOME="${FEDICL_ROOT}/.cache/torch"
export CUDA_CACHE_PATH="${FEDICL_ROOT}/.cache/nv-compute"
export TMPDIR="${FEDICL_ROOT}/.tmp"
mkdir -p "${TMPDIR}"
export PYTHONWARNINGS="ignore"

if [[ -x "${FEDICL_ROOT}/.venv/bin/python" ]]; then
  export VIRTUAL_ENV="${FEDICL_ROOT}/.venv"
  export PATH="${VIRTUAL_ENV}/bin:${PATH}"
fi

# --- Hugging Face token -------------------------------------------------------------------
# Default <repo>/HF_Access_Token or <repo>/HF_Access_Token.txt; override with an absolute path
# in FEDICL_HF_TOKEN_FILE. Exported as HF_TOKEN for this shell only; never printed, never put
# on a command line (the online check feeds the header to curl through stdin).
fedicl_hf_token_scan() {
  [[ "${FEDICL_HF_SCANNED:-}" == 1 ]] && return 0
  local file="" name rel line token code role user
  if [[ -n "${FEDICL_HF_TOKEN_FILE:-}" ]]; then
    [[ -f "${FEDICL_HF_TOKEN_FILE}" ]] || { echo "[HF token] FEDICL_HF_TOKEN_FILE points to a missing file: ${FEDICL_HF_TOKEN_FILE}" >&2; return 1; }
    file="$(realpath "${FEDICL_HF_TOKEN_FILE}")"
  else
    local found=()
    for name in HF_Access_Token HF_Access_Token.txt; do
      [[ -f "${FEDICL_ROOT}/${name}" ]] && found+=("${FEDICL_ROOT}/${name}")
    done
    if (( ${#found[@]} == 0 )); then
      echo "[HF token] no HF_Access_Token(.txt) in ${FEDICL_ROOT}: downloads run anonymously (slower, rate-limited)." >&2
      export FEDICL_HF_SCANNED=1; return 0
    fi
    (( ${#found[@]} == 1 )) || { echo "[HF token] both HF_Access_Token and HF_Access_Token.txt exist: keep only one." >&2; return 1; }
    file="${found[0]}"
  fi
  echo "[HF token] scanning ${file}" >&2

  # 1) never committed (only relevant inside the repo)
  if [[ "${file}" == "${FEDICL_ROOT}/"* ]]; then
    rel="${file#"${FEDICL_ROOT}/"}"
    grep -qxF -e "${rel}" -e "/${rel}" "${FEDICL_ROOT}/.gitignore" 2>/dev/null \
      || { echo "[HF token] add a line '${rel}' to .gitignore before using the token." >&2; return 1; }
    if [[ -d "${FEDICL_ROOT}/.git" ]] && command -v git >/dev/null \
       && [[ -n "$(git -C "${FEDICL_ROOT}" ls-files -- "${rel}")" ]]; then
      echo "[HF token] ${rel} is tracked by git. Run: git rm --cached ${rel}, then revoke the token on huggingface.co." >&2
      return 1
    fi
  fi
  # Linux only: the token file should not be readable by other users.
  if [[ $(( 8#$(stat -c '%a' "${file}") & 8#077 )) -ne 0 ]]; then
    echo "[HF token] WARNING: ${file} is readable by other users; run: chmod 600 '${file}'" >&2
  fi

  # 2) content: one line hf_..., optionally HF_TOKEN=...; BOM/CRLF/comments/blank lines ignored
  mapfile -t lines < <(sed -e '1s/^\xEF\xBB\xBF//' -e 's/\r$//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' "${file}" \
                       | grep -v -e '^$' -e '^#')
  (( ${#lines[@]} == 1 )) || { echo "[HF token] expected exactly 1 non-comment line, found ${#lines[@]}." >&2; return 1; }
  line="${lines[0]}"
  token="$(sed -E -e 's/^(HF_TOKEN|HUGGING_FACE_HUB_TOKEN)[[:space:]]*=[[:space:]]*//' -e "s/^[\"']//" -e "s/[\"']$//" <<< "${line}")"
  [[ "${token}" =~ ^hf_[A-Za-z0-9]{30,}$ ]] \
    || { echo "[HF token] content does not look like a Hugging Face token (${#token} chars). Nothing was sent anywhere." >&2; return 1; }

  # 3) online check (read-only whoami); header via stdin so the token never appears in `ps`
  local body
  body="$(mktemp)"
  code="$(printf 'Authorization: Bearer %s\n' "${token}" \
          | curl -s -m 20 -o "${body}" -w '%{http_code}' -H @- https://huggingface.co/api/whoami-v2 || true)"
  if [[ "${code}" == 200 ]]; then
    user="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("name",""))' "${body}" 2>/dev/null)"
    role="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("auth",{}).get("accessToken",{}).get("role",""))' "${body}" 2>/dev/null)"
    echo "[HF token] OK: user '${user}', token role '${role}'" >&2
    [[ -n "${role}" && "${role}" != read ]] && echo "[HF token] WARNING: role '${role}': a read-only token is enough here and safer." >&2
    export HF_TOKEN="${token}"
  elif [[ "${code}" == 401 ]]; then
    echo "[HF token] WARNING: rejected by huggingface.co (401): invalid, expired, revoked or incompletely copied." >&2
    echo "[HF token] WARNING: token has ${#token} chars (a classic HF token is 37). Continuing WITHOUT a token." >&2
  else
    echo "[HF token] WARNING: could not verify online (HTTP ${code:-none}): using the token as-is." >&2
    export HF_TOKEN="${token}"
  fi
  rm -f "${body}"
  unset token line lines
  export FEDICL_HF_SCANNED=1
}
fedicl_hf_token_scan || { return 1 2>/dev/null || exit 1; }
