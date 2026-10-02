# Running FedICL-MQA on Ubuntu 26.04 LTS (1-2x RTX A5000)

Tested on Ubuntu 26.04.1 LTS (kernel 7.0, NVIDIA driver 595). The system Python (3.14) is not
used: `uv` downloads Python 3.12 into the repo.

Everything stays inside the cloned folder: `uv` (`.uv-bin/`), the package cache (`.uv-cache/`),
Python 3.12 (`.uv-python/`), `.venv/`, Hugging Face models + MedQA (`.hf-cache/`, ~3 GB), temp
files (`.tmp/`) and small library caches (`.cache/`). Nothing is written to `~/.cache` or
`~/.bashrc` as long as commands go through `scripts/*.sh` or a shell that ran
`source scripts/env.sh`. Delete the folder to uninstall.

Disk: keep **>= 20 GB free** on the filesystem that holds the clone (environment ~7 GB, models and
data ~3 GB, checkpoints/predictions of the 4 arms several GB more). Do not clone into `/tmp` when
it is a RAM disk (`df -h /tmp` shows `tmpfs`): it fills up while unpacking the CUDA wheels.

## 1. One-time setup

```bash
sudo apt install -y git curl tmux          # tmux keeps long runs alive after SSH disconnects
nvidia-smi                                 # driver must support CUDA 12.4 (>= 550); else: sudo ubuntu-drivers install

git clone git@github.com:nguyenhoangvuthan/fedicl-medqa-v3.git ~/fedicl-medqa-v3
cd ~/fedicl-medqa-v3
bash scripts/setup_env.sh                  # ~6 GB download; older driver: bash scripts/setup_env.sh cu118
```

## 2. Hugging Face token (optional)

Put a **read** token (https://huggingface.co/settings/tokens) in `HF_Access_Token` or
`HF_Access_Token.txt` at the repo root, one line `hf_...`, and restrict it to your user:

```bash
nano HF_Access_Token.txt && chmod 600 HF_Access_Token.txt
```

Every script scans it first (same rules as on Windows): it must be git-ignored and not tracked,
contain exactly one `hf_...` token, and be accepted by `huggingface.co/api/whoami-v2`. A 401
only warns and continues anonymously (all models/datasets here are public). The token is
exported as `HF_TOKEN` for that run only, never printed, and passed to `curl` through stdin so
it never shows up in `ps`. Another location: `export FEDICL_HF_TOKEN_FILE=/abs/path/token`.

## 3. Run

```bash
cd ~/fedicl-medqa-v3
bash scripts/smoke_test.sh                 # optional, ~10-15 min on 120/24/24 questions
bash scripts/prepare_data.sh               # MedQA -> partitions -> features -> demos -> audit
tmux new -s fedicl                         # detach: Ctrl-b d ; re-attach: tmux attach -t fedicl
bash scripts/run_all.sh 2>&1 | tee run_all.log
```

If a run stops, start the same command again: completed arms are skipped, the interrupted one
resumes from its last saved epoch/round. Per-arm logs: `outputs/qwen3-0.6b/seed42/<arm>/logs/run.log`.

### Both GPUs in parallel

```bash
bash scripts/prepare_data.sh                                   # once
tmux new -s gpu0   # window 1
FEDICL_GPU=0 FEDICL_ARMS=centralized_non_icl,centralized_icl bash scripts/run_all.sh 2>&1 | tee run_gpu0.log
tmux new -s gpu1   # window 2
FEDICL_GPU=1 FEDICL_ARMS=federated_non_icl,federated_icl     bash scripts/run_all.sh 2>&1 | tee run_gpu1.log
```

`FEDICL_GPU` is the index shown by `nvidia-smi`. A GPU that CUDA cannot see stops the run
instead of silently using the CPU. Changing the GPU does not change `config_hash`, so an arm can
resume on the other GPU.

## 4. Results and verification

```bash
source scripts/env.sh                      # puts .venv on PATH, caches in the repo
python -m fedicl.summarize                 # headline.csv, summary.csv, curves.png
python -m fedicl.compare                   # paired McNemar / bootstrap between arms (seconds)
python -m fedicl.compare --checkpoint round_3
python -m fedicl.audit                     # every FL demo comes from the client's own data
```

Ablations for the federated ICL gain (own output folders, main results untouched):

```bash
tmux new -s ab0;  FEDICL_GPU=0 bash scripts/run_ablation.sh no_licl 2>&1 | tee ablation_no_licl.log   # ~8 h
tmux new -s ab1;  FEDICL_GPU=1 bash scripts/run_ablation.sh k2      2>&1 | tee ablation_k2.log        # ~15 h
```

## 5. Single commands

After `source scripts/env.sh` in a shell, use `python -m ...` directly:

| Goal | Command |
|------|---------|
| One arm | `python -m fedicl.run --arm federated_icl` |
| One arm on GPU 1 | `python -m fedicl.run --arm federated_icl runtime.gpu=1` |
| Continue a run that hit the budget while improving | `python -m fedicl.run --arm centralized_icl --extend_to 5` |
| Rerun from scratch (old run archived, not deleted) | `python -m fedicl.run --arm centralized_icl --overwrite` |
| Re-evaluate a saved adapter | `python -m fedicl.eval --arm federated_icl --checkpoint round_2` |
| Re-score generations with another matcher config | `python -m fedicl.rescore --arm federated_icl eval.match.weights.semantic=0.3 eval.match.weights.lexical=0.7` |
| Unit tests | `python -m pytest -q tests` |
