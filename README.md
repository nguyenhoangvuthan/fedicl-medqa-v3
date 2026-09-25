# FedICL-MQA — Federated In-Context Learning for MedQA (Qwen3-0.6B + LoRA)

Implementation of `prompt-fedicl-medqa.md` (the spec). Section numbers below (§) refer to it.

## 1. Windows Server (1x RTX A5000, 24 GB)

Clone to a non-system drive, close to the root (e.g. `D:\fedicl-medqa`). Everything is kept
inside that folder: `uv.exe` (`.uv-bin\`), the uv package cache (`.uv-cache\`), the Python 3.12
that uv downloads (`.uv-python\`), `.venv\`, Hugging Face models + MedQA (`.hf-cache\`, ~3 GB)
temp files (`.tmp\`) and small library caches (`.cache\`: matplotlib, torch hub, NVIDIA JIT).
Nothing is written under `C:\Users\...` as long as commands run through `scripts\windows\*.ps1`
or a session that dot-sourced `scripts\windows\env.ps1`.

```powershell
cd D:\fedicl-medqa
powershell -ExecutionPolicy Bypass -File scripts\windows\setup_env.ps1     # once (~6 GB download)
powershell -ExecutionPolicy Bypass -File scripts\windows\smoke_test.ps1    # optional, ~10 min
powershell -ExecutionPolicy Bypass -File scripts\windows\prepare_data.ps1
powershell -ExecutionPolicy Bypass -File scripts\windows\run_all.ps1 *>&1 | Tee-Object run_all.log
```
- **Hugging Face token (optional, avoids anonymous rate limits):** put a *read* token in a file
  named `HF_Access_Token` at the repo root (one line: `hf_...`, or `HF_TOKEN=hf_...`). Every
  script scans it first: it must be git-ignored and not tracked, contain exactly one well-formed
  token, and be accepted by `huggingface.co/api/whoami-v2`. Otherwise the script stops before
  running anything. The token is exported as `HF_TOKEN` for that process only and is never printed.
- Needs an NVIDIA driver with CUDA 12.4 support (>= 550); for an older driver use
  `setup_env.ps1 -Cuda cu118`.
- If `setup_env.ps1` warns about long paths, enable them once (Administrator PowerShell):
  `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force`.
- The run takes hours: disconnecting from Remote Desktop is fine, **signing out kills it**.
  If it is interrupted, run the same `run_all.ps1` again: completed arms are skipped, the
  interrupted one resumes from its last saved epoch/round.
- To run single commands by hand, first load the environment in that PowerShell session so the
  caches stay in the repo: `. .\scripts\windows\env.ps1`, then e.g.
  `Invoke-Py -m fedicl.run --arm federated_icl` or `Invoke-Py -m fedicl.summarize`.

## 1b. Linux

```bash
bash scripts/setup_env.sh          # .venv with torch 2.6 (CUDA 12.4) + the pinned stack
source .venv/bin/activate
```

## 2. Run (Linux; on Windows use the scripts\windows\*.ps1 equivalents above)

```bash
# 0) optional, ~10 min: whole pipeline on 120/24/24 questions -> processed_data_smoke/, outputs_smoke/
bash scripts/smoke_test.sh

# 1) data: MedQA -> raw/centralized -> partitions -> features -> demo assignments -> plots
bash scripts/prepare_data.sh

# 2) the 4 arms (completed arms are skipped, interrupted ones resume), then summary
nohup bash scripts/run_all.sh > run_all.log 2>&1 &
```
Results: `outputs/qwen3-0.6b/seed42/{summary.csv, headline.csv, curves.png, fl_per_client.png}`.

Single arm / overrides (OmegaConf dotlist, any key of `configs/base.yaml`):
```bash
python -m fedicl.run --arm federated_icl
python -m fedicl.run --arm centralized_icl retrieval.strategy=semantic loss.icl.weight=0   # ablation
```
An override that changes results changes `config_hash`; an existing arm then refuses to resume
(use `--overwrite`, which archives the old run to `_archive/`, never deletes it).

## 3. Re-use results (§5.3)

| Goal | Command |
|------|---------|
| Continue a run that hit the budget while still improving | `python -m fedicl.run --arm centralized_icl --extend_to 5` |
| Rerun an arm from scratch (old run archived) | `python -m fedicl.run --arm centralized_icl --overwrite` |
| Re-evaluate a saved adapter | `python -m fedicl.eval --arm federated_icl --checkpoint round_2` |
| Re-score saved generations with another matcher config | `python -m fedicl.rescore --arm federated_icl eval.match.weights.semantic=0.3 eval.match.weights.lexical=0.7` |
| Rebuild summary tables/plots | `python -m fedicl.summarize` |
| Load an adapter in a notebook | `from fedicl.modeling import load_adapter; model, tok = load_adapter("outputs/qwen3-0.6b/seed42/federated_icl", "best")` |

## 4. Architecture → code

| FedICL-MQA component | Code |
|----------------------|------|
| 1. Question Encoding (`NeuML/pubmedbert-base-embeddings`) | `fedicl/retrieval/features.py`, `encoders.py` |
| 2. Adaptive Demonstration Retrieval: semantic + clinical relevance (scispaCy + SapBERT soft-F1) + MMR diversity, near-duplicate filter | `fedicl/retrieval/retrieve.py` |
| Privacy-aware retrieval: FL demos only from `client_i`, for training **and** evaluation | `retrieve.py` (scopes), `train_common.eval_views` |
| 3. In-Context Prompt Construction (unlabeled, shuffled options) | `fedicl/prompt.py` |
| 4. SLM generation + text → option matching (exact → contains → nearest = token-F1 + SapBERT) | `fedicl/evaluate.py`, `fedicl/matching.py` |
| Local objective `L_QA + λ_ICL·L_ICL + L_REG` | `fedicl/losses.py`, `train_common.StepRunner` |
| 5. FedAvg over LoRA, global model distribution | `fedicl/train_federated.py`, `modeling.fedavg` |
| Centralized baseline with early stopping every 0.25 epoch | `fedicl/train_centralized.py` |

## 5. Arms (§2)

| Arm | Role | Demos | Loss |
|-----|------|-------|------|
| `centralized_non_icl` | Centralized SLM baseline | — | L_QA |
| `federated_non_icl` | FedICL-MQA w/o ICL | — | L_QA + L_REG |
| `centralized_icl` | FedICL-MQA w/o FL | repo = train | L_QA + λ·L_ICL |
| `federated_icl` | **FedICL-MQA** | repo = client_i | L_QA + λ·L_ICL + L_REG |

Headline FL number = `demo_pool = macro_mean` (mean over the per-client views); `min` = worst client.

## 6. Data layout (§3.7)

```
processed_data/MedQA/
  dataset_summary.csv  config.json
  raw/{train,validation,test}.csv                    # MedQA as-is (GBaker/MedQA-USMLE-4-options-hf)
  centralized/{train,validation,test}.csv            # + q_hash, validated; what every run reads
  centralized/features/<encoder_tag>/                # question embeddings, entities, entity embeddings
  centralized/demo_assignment_{train,validation,test}.json, demo_check_report.json
  federated_iid/k{2..5}/client_*.csv, stats.csv
  federated_noniid/alpha_{0.5,0.1}/k{2..5}/client_*.csv, stats.csv
  federated_noniid/alpha_0.5/k3/demo_assignment_client_{1,2,3}_{train,validation,test}.json
  visualizations/*.png
```
Notes: the HF `-hf` release has no `meta` column; `meta` (step1 / step2&3) is joined by question
text from `GBaker/MedQA-USMLE-4-options`, which has no validation split, so validation `meta` is
`unknown` (statistics only).

## 7. Reproducibility

`runtime.deterministic: true` (default) seeds python/numpy/torch **and** forces deterministic
CUDA kernels (incl. the attention backward, which is non-deterministic by default). Verified:
rerunning an arm gives a bit-identical adapter and identical `result.csv`, with no measurable
slowdown in the smoke test. Results can still differ across GPU models or library versions.

## 8. Tests

```bash
python -m pytest -q tests
```
