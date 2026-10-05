"""Data visualizations (spec §3.7) -> processed_data/<dataset>/visualizations/.

    python -m fedicl.visualize [--config ...]
"""
from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import add_config_args, centralized_dir, config_from_args, data_dir
from .data.io import LETTERS, SPLITS, load_split, meta_values, to_examples
from .utils import read_json, setup_logging

LOG = logging.getLogger("fedicl.visualize")


def split_distribution(cfg, out) -> None:
    dfs = {s: load_split(centralized_dir(cfg) / f"{s}.csv") for s in SPLITS}
    metas = meta_values(cfg.data.dataset, list(dfs.values()), with_unknown=True)
    fig, axes = plt.subplots(1, 2, figsize=(10 + 0.3 * max(0, len(metas) - 3), 3.5 + 0.8 * (len(metas) > 3)),
                             width_ratios=[1, max(1, len(metas) / 4)])
    for s, df in dfs.items():
        axes[0].plot(LETTERS, [(df["answer"] == l).mean() for l in LETTERS], marker="o", label=s)
        axes[1].plot(metas, [(df["meta"] == m).mean() for m in metas], marker="o", label=s)
    axes[0].set_title("answer position (original file order)")
    axes[1].set_title("meta")
    if len(metas) > 3:
        axes[1].tick_params(axis="x", labelrotation=60, labelsize=7)
        plt.setp(axes[1].get_xticklabels(), ha="right")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "split_distribution.png", dpi=150)
    plt.close(fig)


def client_bars(stats: pd.DataFrame, title: str, path, prefix: str = "answer") -> None:
    cols = [c for c in stats.columns if c.startswith(f"{prefix}_")]
    fig, ax = plt.subplots(figsize=(1.6 + 0.9 * len(stats) + 1.5 * (len(cols) > 6), 3.5))
    bottom = np.zeros(len(stats))
    x = np.arange(len(stats))  # numeric positions: string ids like "1" trigger matplotlib's categorical-units notice
    colors = (plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors if len(cols) > 10
              else plt.get_cmap("tab10").colors)   # MedMCQA has 21 subjects: 40 distinct colors
    for i, c in enumerate(cols):
        v = stats[c].to_numpy()
        ax.bar(x, v, bottom=bottom, label=c[len(prefix) + 1:], color=colors[i % len(colors)])
        bottom += v
    ax.set_xticks(x, [str(c) for c in stats["client_id"]])
    ax.set_title(title)
    ax.set_xlabel("client")
    if len(cols) > 6:
        ax.legend(title=prefix, fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5))
    else:
        ax.legend(title=prefix, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def partitions(cfg, out) -> None:
    root = data_dir(cfg)
    for stats_path in sorted(root.glob("federated_*/**/stats.csv")):
        rel = stats_path.parent.relative_to(root)
        name = "_".join(rel.parts).replace("federated_", "").replace("alpha_", "alpha")
        stats = pd.read_csv(stats_path)
        client_bars(stats, str(rel), out / f"{name}_clients.png")
        if cfg.data_prep.label_key == "meta":   # non-IID by meta (MedMCQA: subject) -> show that too
            client_bars(stats, str(rel), out / f"{name}_clients_meta.png", prefix="meta")


def prompt_lengths(cfg, out) -> None:
    from transformers import AutoTokenizer

    from .prompt import PromptBuilder

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    b = PromptBuilder(tok, cfg)
    cdir = centralized_dir(cfg)
    ex = to_examples(load_split(cdir / "train.csv"))
    path = cdir / "demo_assignment_train.json"
    if not path.exists():
        LOG.warning("no demo assignments yet; skipping prompt_length_hist")
        return
    assign = read_json(path)
    ids = list(assign)[:2000]
    icl = [len(b.prompt_ids(ex[q], [ex[d] for d in assign[q]["demo_ids"]])) for q in ids]
    non = [len(b.prompt_ids(ex[q], [])) for q in ids]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(non, bins=40, alpha=0.6, label="non-ICL")
    ax.hist(icl, bins=40, alpha=0.6, label="ICL (3 demos)")
    ax.axvline(cfg.model.max_seq_len, color="k", ls="--", label="max_seq_len")
    ax.set_xlabel(f"prompt tokens ({cfg.model.short_name})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "prompt_length_hist.png", dpi=150)
    plt.close(fig)
    LOG.info("ICL prompt tokens: p50=%d p99=%d max=%d", np.percentile(icl, 50), np.percentile(icl, 99), max(icl))


def retrieval_similarity(cfg, out) -> None:
    report_path = centralized_dir(cfg) / "demo_check_report.json"
    if not report_path.exists():
        return
    scopes = read_json(report_path)["scopes"]
    names = [n for n in scopes if n.endswith("/test")]
    p10, p50, p90 = zip(*[scopes[n]["sem_selected_p10_p50_p90"] for n in names])
    fig, ax = plt.subplots(figsize=(1.5 + 1.2 * len(names), 3.5))
    x = np.arange(len(names))
    ax.errorbar(x, p50, yerr=[np.subtract(p50, p10), np.subtract(p90, p50)], fmt="o", capsize=4)
    ax.set_xticks(x, [n.replace("/test", "").split("/")[-1] for n in names], rotation=20)
    ax.set_ylabel("sem(query, selected demo)\np10 / p50 / p90")
    ax.set_title("test-query demo similarity by repository")
    fig.tight_layout()
    fig.savefig(out / "retrieval_similarity.png", dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p, with_arm=False)
    args = p.parse_args()
    cfg = config_from_args(args)
    setup_logging()
    out = data_dir(cfg) / "visualizations"
    out.mkdir(parents=True, exist_ok=True)
    split_distribution(cfg, out)
    partitions(cfg, out)
    retrieval_similarity(cfg, out)
    prompt_lengths(cfg, out)
    LOG.info("wrote %s", out)


if __name__ == "__main__":
    main()
