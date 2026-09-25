"""Collect every {arm}/result.csv -> summary.csv + headline table + learning-curve plots.

    python -m fedicl.summarize [--config ...]
"""
from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import ARMS, ROOT, add_config_args, config_from_args
from .utils import setup_logging, write_csv_df

LOG = logging.getLogger("fedicl.summarize")
HEADLINE_POOLS = ("macro_mean", "train", "none")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p, with_arm=False)
    args = p.parse_args()
    cfg = config_from_args(args)
    setup_logging()
    root = ROOT / cfg.save.root
    frames = [pd.read_csv(root / a / "result.csv") for a in ARMS if (root / a / "result.csv").exists()]
    if not frames:
        raise SystemExit(f"no result.csv under {root}")
    df = pd.concat(frames, ignore_index=True)
    write_csv_df(root / "summary.csv", df)

    best = df[(df["checkpoint"] == "best") & df["demo_pool"].isin(HEADLINE_POOLS)]
    table = best.pivot_table(index="arm", columns="split", values=["accuracy", "accuracy_strict"],
                             aggfunc="last")
    worst = df[(df["checkpoint"] == "best") & (df["demo_pool"] == "min") & (df["split"] == "test")]
    table[("worst_client_acc", "test")] = worst.set_index("arm")["accuracy"]
    write_csv_df(root / "headline.csv", table.reset_index())
    print(table.round(4).to_string())

    curves = df[df["checkpoint"].str.match(r"^(epoch|round)_\d+$") & df["demo_pool"].isin(HEADLINE_POOLS)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm, g in curves.groupby("arm"):
        for ax, (split, metric) in zip(axes, (("test", "accuracy"), ("validation", "val_loss"))):
            s = g[g["split"] == split].sort_values("epoch_or_round")
            ax.plot(s["epoch_or_round"], s[metric], marker="o", label=arm)
            ax.set_xlabel("epoch / round")
            ax.set_ylabel(f"{split} {metric}")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "curves.png", dpi=150)

    per_client = df[(df["setting"] == "federated") & df["demo_pool"].str.startswith("client_")
                    & (df["split"] == "test") & df["checkpoint"].str.startswith("round_")]
    if len(per_client):
        fig, ax = plt.subplots(figsize=(6, 4))
        for (arm, pool), g in per_client.groupby(["arm", "demo_pool"]):
            g = g.sort_values("epoch_or_round")
            ax.plot(g["epoch_or_round"], g["accuracy"], marker="o", label=f"{arm}/{pool}")
        ax.set_xlabel("round")
        ax.set_ylabel("test accuracy (local demo repository)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(root / "fl_per_client.png", dpi=150)
    LOG.info("wrote %s/{summary.csv, headline.csv, curves.png}", root)


if __name__ == "__main__":
    main()
