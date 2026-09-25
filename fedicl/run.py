"""Run one arm with skip / archive / resume semantics (spec §5.3).

    python -m fedicl.run --arm federated_icl [--config configs/smoke.yaml] [overrides...]
    python -m fedicl.run --arm centralized_icl --overwrite          # archive old run, start fresh
    python -m fedicl.run --arm centralized_icl --extend_to 5        # continue a max_budget_reached run
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from .config import (ROOT, add_config_args, arm_dir, arm_name, centralized_dir, config_from_args,
                     config_hash, partition_dir)
from .utils import read_json, setup_logging, sha1_file, write_csv_df, write_json, write_text

LOG = logging.getLogger("fedicl.run")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _versions() -> dict:
    import peft
    import torch
    import transformers

    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__}


def _demo_files(cfg) -> dict:
    if not cfg.icl.enabled:
        return {}
    files = list(centralized_dir(cfg).glob("demo_assignment_*.json"))
    if cfg.setting == "federated":
        files += list(partition_dir(cfg).glob("demo_assignment_client_*.json"))
    return {str(f.relative_to(ROOT)): sha1_file(f) for f in sorted(files)}


def update_runs_index(cfg, out: Path, meta: dict, result: dict | None) -> None:
    path = out.parent / "runs_index.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    row = {"arm": arm_name(cfg), "status": meta["status"], "path": str(out.relative_to(ROOT)),
           "best_at": (result or {}).get("best_at"), "best_test_acc": (result or {}).get("best_test_acc"),
           "started_at": meta.get("started_at"), "finished_at": meta.get("finished_at")}
    if len(df) and "arm" in df:
        df = df[df["arm"] != row["arm"]]
    write_csv_df(path, pd.concat([df, pd.DataFrame([row])], ignore_index=True).sort_values("arm"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("--overwrite", action="store_true", help="archive the existing arm dir and rerun")
    p.add_argument("--allow_config_change", action="store_true")
    p.add_argument("--extend_to", type=int, default=None,
                   help="continue training up to N epochs/rounds (<= train.max_extend)")
    args = p.parse_args()
    if not args.arm:
        p.error("--arm is required")
    cfg = config_from_args(args)
    out = arm_dir(cfg)
    chash = config_hash(cfg)
    meta_path = out / "run_meta.json"

    if args.extend_to and args.extend_to > int(cfg.train.max_extend):
        p.error(f"--extend_to {args.extend_to} exceeds train.max_extend={cfg.train.max_extend}")

    if meta_path.exists():
        meta = read_json(meta_path)
        if args.overwrite or cfg.save.overwrite:
            dst = out.parent / "_archive" / f"{out.name}_{dt.datetime.now():%Y%m%d-%H%M%S}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(out), dst)
            print(f"archived previous run to {dst}")
        elif meta["status"] == "completed" and not args.extend_to:
            print(f"[skip] {arm_name(cfg)} already completed ({out}); use --overwrite or --extend_to")
            return
        elif meta["config_hash"] != chash and not args.allow_config_change:
            raise SystemExit(f"config changed since this run started ({meta['config_hash']} -> {chash}); "
                             "rerun with --allow_config_change or --overwrite")

    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "logs" / "run.log")
    from .modeling import gpu_name

    meta = read_json(meta_path) if meta_path.exists() else {"started_at": dt.datetime.now().isoformat(timespec="seconds")}
    meta.update({"arm": arm_name(cfg), "status": "running", "config_hash": chash, "seed": int(cfg.seed),
                 "git_commit": _git_commit(), "versions": _versions(), "gpu": gpu_name(),
                 "eval_match": OmegaConf.to_container(cfg.eval.match), "demo_assignment_sha1": _demo_files(cfg),
                 "extend_to": args.extend_to, "finished_at": None})
    write_json(meta_path, meta)
    write_text(out / "config.yaml", OmegaConf.to_yaml(cfg))
    update_runs_index(cfg, out, meta, None)
    LOG.info("arm %s -> %s (config_hash %s)", arm_name(cfg), out, chash)

    try:
        if cfg.setting == "centralized":
            from .train_centralized import run
        else:
            from .train_federated import run
        result = run(cfg, out, extend_to=args.extend_to)
        meta.update(status="completed", finished_at=dt.datetime.now().isoformat(timespec="seconds"), result=result)
    except BaseException:
        meta.update(status="failed", error=traceback.format_exc(limit=5),
                    finished_at=dt.datetime.now().isoformat(timespec="seconds"))
        write_json(meta_path, meta)
        update_runs_index(cfg, out, meta, None)
        raise
    write_json(meta_path, meta)
    update_runs_index(cfg, out, meta, result)
    LOG.info("done: %s", result)


if __name__ == "__main__":
    main()
