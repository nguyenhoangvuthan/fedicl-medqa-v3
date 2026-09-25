"""result.csv rows (spec §5.2) and prediction/metric persistence for one checkpoint."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from .config import arm_name
from .utils import write_csv_df, write_json, write_jsonl

RESULT_COLUMNS = [
    "arm", "model", "setting", "mode", "k_shot", "retrieval_strategy", "seed", "checkpoint",
    "epoch_or_round", "split", "demo_pool", "accuracy", "accuracy_strict", "accuracy_std", "val_loss",
    "n_samples", "exact_rate", "contains_rate", "nearest_rate", "empty_rate", "low_conf_rate",
    "is_best", "adapter_path", "timestamp",
]


def result_rows(cfg, checkpoint: str, epoch_or_round: float | int, metrics: dict,
                adapter_path: str, is_best: bool = False) -> list[dict]:
    """metrics = {split: {demo_pool: {metric: value}}}."""
    ts = dt.datetime.now().isoformat(timespec="seconds")
    rows = []
    for split, pools in metrics.items():
        for pool, m in pools.items():
            rows.append({
                "arm": arm_name(cfg), "model": cfg.model.short_name, "setting": cfg.setting,
                "mode": "icl" if cfg.icl.enabled else "non_icl",
                "k_shot": int(cfg.icl.k_shot) if cfg.icl.enabled else 0,
                "retrieval_strategy": cfg.retrieval.strategy if cfg.icl.enabled else "none",
                "seed": int(cfg.seed), "checkpoint": checkpoint, "epoch_or_round": epoch_or_round,
                "split": split, "demo_pool": pool, **{k: m.get(k) for k in RESULT_COLUMNS if k in m},
                "is_best": is_best, "adapter_path": adapter_path, "timestamp": ts,
            })
    return rows


def append_results(path: Path, rows: list[dict]) -> None:
    new = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    write_csv_df(path, new)


def save_eval_outputs(ckpt_dir: Path, metrics: dict, predictions: dict[tuple[str, str], list[dict]]) -> None:
    """predictions keyed by (split, file_suffix). metrics.json is written LAST: it marks completion."""
    for (split, suffix), preds in predictions.items():
        if preds:
            write_jsonl(ckpt_dir / f"predictions_{split}{suffix}.jsonl", preds)
    write_json(ckpt_dir / "metrics.json", metrics)
