"""Spec §3.5: split centralized/train.csv into the IID and Dirichlet non-IID client grid.

    python -m fedicl.data.partition [--config ...] [overrides...]
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from ..config import add_config_args, centralized_dir, config_from_args, data_dir, partition_dir
from ..utils import read_json, rng_for, setup_logging, write_csv_df, write_json
from .io import LETTERS, load_split, save_split

LOG = logging.getLogger("fedicl.partition")


def iid_split(n: int, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    return [np.sort(a) for a in np.array_split(rng.permutation(n), k)]


def dirichlet_split(labels: np.ndarray, k: int, alpha: float, rng: np.random.Generator,
                    min_size: int, max_tries: int = 1000) -> list[np.ndarray]:
    """Classic label-Dirichlet partition: per class, sample client proportions ~ Dir(alpha)."""
    classes = np.unique(labels)
    for _ in range(max_tries):
        parts: list[list[int]] = [[] for _ in range(k)]
        for c in classes:
            idx = rng.permutation(np.flatnonzero(labels == c))
            props = rng.dirichlet(np.full(k, alpha))
            cuts = (np.cumsum(props)[:-1] * len(idx)).astype(int)
            for i, chunk in enumerate(np.split(idx, cuts)):
                parts[i].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.sort(np.array(p, dtype=int)) for p in parts]
    raise RuntimeError(f"Dirichlet(alpha={alpha}, k={k}) never gave every client >= {min_size} samples")


def client_stats(clients: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for i, df in enumerate(clients, start=1):
        row = {"client_id": i, "n_samples": len(df)}
        for l in LETTERS:
            row[f"answer_{l}"] = int((df["answer"] == l).sum())
        row["meta_step1"] = int((df["meta"] == "step1").sum())
        row["meta_step2&3"] = int((df["meta"] == "step2&3").sum())
        rows.append(row)
    return pd.DataFrame(rows)


def write_partition(train: pd.DataFrame, parts: list[np.ndarray], out_dir) -> pd.DataFrame:
    ids = [set(train["id"].iloc[p]) for p in parts]
    assert set().union(*ids) == set(train["id"]), "union of clients != train"
    assert sum(len(s) for s in ids) == len(train), "clients overlap"
    clients = [train.iloc[p].reset_index(drop=True) for p in parts]
    for i, df in enumerate(clients, start=1):
        save_split(df, out_dir / f"client_{i}.csv")
    stats = client_stats(clients)
    write_csv_df(out_dir / "stats.csv", stats)
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p, with_arm=False)
    args = p.parse_args()
    cfg = config_from_args(args)
    setup_logging()
    dp = cfg.data_prep

    train = load_split(centralized_dir(cfg) / "train.csv")
    labels = train[dp.label_key].to_numpy()
    grid = [("iid", None, k) for k in dp.iid_k] + \
           [("noniid", a, k) for a in dp.noniid_alphas for k in dp.noniid_k]
    summary = {}
    for ptype, alpha, k in grid:
        rng = rng_for(dp.partition_seed, ptype, alpha, k)
        parts = (iid_split(len(train), k, rng) if ptype == "iid"
                 else dirichlet_split(labels, k, float(alpha), rng, dp.min_client_size))
        out = partition_dir(cfg, ptype, alpha, k)
        stats = write_partition(train, parts, out)
        name = str(out.relative_to(data_dir(cfg)))
        summary[name] = stats["n_samples"].tolist()
        LOG.info("%s\n%s", name, stats.to_string(index=False))

    meta_path = data_dir(cfg) / "config.json"
    meta = read_json(meta_path)
    meta["partition"] = {"seed": dp.partition_seed, "label_key": dp.label_key,
                         "iid_k": list(dp.iid_k), "noniid_alphas": list(dp.noniid_alphas),
                         "noniid_k": list(dp.noniid_k), "client_sizes": summary}
    write_json(meta_path, meta)


if __name__ == "__main__":
    main()
