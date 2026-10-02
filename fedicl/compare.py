"""Paired significance tests between arms on the SAME test questions (spec §5.1 reporting).

    python -m fedicl.compare                                   # the 4 main arms, checkpoint best
    python -m fedicl.compare --checkpoint round_3              # FL arms at round 3 (centralized: best)
    python -m fedicl.compare --add no_licl=outputs/ablation_no_licl/qwen3-0.6b/seed42/federated_icl \
                             --vs no_licl,federated_non_icl --vs federated_icl,no_licl

Per run, a question's score is its correctness (0/1); for federated ICL it is the mean over the
client views (so the run's accuracy is the macro_mean reported everywhere else).
For each pair (A, B):
  gain      = acc(A) - acc(B) in percentage points
  bootstrap = paired bootstrap over questions (10k resamples): 95% CI of the gain + two-sided p
  McNemar   = exact two-sided test when both sides are a single 0/1 view
FL ICL vs a single-view run also gets one row per client view (client_i vs B).
Writes <save.root>/compare_<split>_<checkpoint>.csv (or --out).
"""
from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ARMS, ROOT, add_config_args, config_from_args
from .utils import read_jsonl, setup_logging, write_csv_df

LOG = logging.getLogger("fedicl.compare")

DEFAULT_PAIRS = [
    ("ICL gain (centralized)", "centralized_icl", "centralized_non_icl"),
    ("ICL gain (federated)", "federated_icl", "federated_non_icl"),
    ("FL vs centralized (non-ICL)", "federated_non_icl", "centralized_non_icl"),
    ("FL vs centralized (ICL) = privacy cost", "federated_icl", "centralized_icl"),
]


@dataclass
class Run:
    label: str
    views: dict[str, dict[str, bool]]  # pool -> {question id -> correct}

    @property
    def is_multi(self) -> bool:
        return len(self.views) > 1


def pred_dir(arm_dir: Path, checkpoint: str) -> Path:
    if checkpoint == "best":
        return arm_dir / "best"
    if checkpoint.startswith("round_"):
        return arm_dir / "rounds" / checkpoint
    return arm_dir / "checkpoints" / checkpoint


def load_run(label: str, arm_dir: Path, checkpoint: str, split: str = "test") -> Run | None:
    d = pred_dir(arm_dir, checkpoint)
    if not d.exists():  # e.g. --checkpoint round_3 for a centralized arm: fall back to best
        d = arm_dir / "best"
    files = sorted(d.glob(f"predictions_{split}*.jsonl"))
    files = [f for f in files if not f.name.endswith((".rescored.jsonl", "_train_diagnostic.jsonl"))]
    if not files:
        return None
    views = {}
    for f in files:
        rows = read_jsonl(f)
        views[rows[0]["demo_pool"]] = {r["id"]: bool(r["correct"]) for r in rows}
    return Run(label, views)


def score_vector(run: Run, ids: list[str], pool: str | None = None) -> np.ndarray:
    pools = [pool] if pool else list(run.views)
    return np.mean([[float(run.views[p][q]) for q in ids] for p in pools], axis=0)


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Exact two-sided McNemar on paired 0/1 vectors -> (A-only correct, B-only correct, p)."""
    n10 = int(np.sum((a == 1) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    tail = sum(math.comb(n, k) for k in range(min(n10, n01) + 1)) / 2 ** n
    return n10, n01, min(1.0, 2 * tail)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0):
    """95% CI of mean(a - b) and a two-sided bootstrap p-value (questions resampled jointly)."""
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = min(1.0, 2 * min(np.mean(boots <= 0), np.mean(boots >= 0)))
    return float(lo), float(hi), float(p)


def compare(name: str, A: Run, B: Run, pool_a: str | None = None, pool_b: str | None = None) -> dict:
    ids = sorted(set.intersection(*[set(v) for v in A.views.values()], *[set(v) for v in B.views.values()]))
    a, b = score_vector(A, ids, pool_a), score_vector(B, ids, pool_b)
    lo, hi, p_boot = paired_bootstrap(a, b)
    row = {"comparison": name, "A": A.label + (f"[{pool_a}]" if pool_a else ""),
           "B": B.label + (f"[{pool_b}]" if pool_b else ""), "n": len(ids),
           "acc_A": a.mean(), "acc_B": b.mean(), "gain_pp": 100 * (a.mean() - b.mean()),
           "ci95_low_pp": 100 * lo, "ci95_high_pp": 100 * hi, "p_bootstrap": p_boot,
           "a_only": None, "b_only": None, "p_mcnemar": None}
    binary_a = pool_a is not None or not A.is_multi
    binary_b = pool_b is not None or not B.is_multi
    if binary_a and binary_b:
        row["a_only"], row["b_only"], row["p_mcnemar"] = mcnemar_exact(a, b)
    p = row["p_mcnemar"] if row["p_mcnemar"] is not None else p_boot
    row["significant_5pct"] = bool(p < 0.05)
    return row


def run_pair(name: str, A: Run, B: Run) -> list[dict]:
    rows = [compare(name, A, B)]
    if A.is_multi and not B.is_multi:  # FL ICL vs single view: also each client view
        rows += [compare(f"  {name} / {pool}", A, B, pool_a=pool) for pool in sorted(A.views)]
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p, with_arm=False)
    p.add_argument("--checkpoint", default="best", help="best | round_N | epoch_N (missing -> best)")
    p.add_argument("--split", default="test", choices=["test", "validation"])
    p.add_argument("--add", action="append", default=[], metavar="LABEL=ARM_DIR",
                   help="extra run, e.g. an ablation arm directory")
    p.add_argument("--vs", action="append", default=[], metavar="A,B",
                   help="extra comparison between run labels (main arms or --add labels)")
    p.add_argument("--out", help="CSV path (default <save.root>/compare_<split>_<checkpoint>.csv)")
    args = p.parse_args()
    cfg = config_from_args(args)
    setup_logging()
    root = ROOT / cfg.save.root

    runs: dict[str, Run] = {}
    for arm in ARMS:
        r = load_run(arm, root / arm, args.checkpoint, args.split)
        if r:
            runs[arm] = r
    for spec in args.add:
        label, _, path = spec.partition("=")
        r = load_run(label, (ROOT / path) if not Path(path).is_absolute() else Path(path),
                     args.checkpoint, args.split)
        if r is None:
            raise SystemExit(f"--add {spec}: no predictions_{args.split}*.jsonl found")
        runs[label] = r
    missing = [a for a in ARMS if a not in runs]
    if missing:
        LOG.warning("no predictions for %s under %s (skipped)", missing, root)

    pairs = [(n, a, b) for n, a, b in DEFAULT_PAIRS if a in runs and b in runs]
    for spec in args.vs:
        a, _, b = spec.partition(",")
        if a not in runs or b not in runs:
            raise SystemExit(f"--vs {spec}: unknown run label (have {sorted(runs)})")
        pairs.append((f"{a} vs {b}", a, b))
    rows = [row for n, a, b in pairs for row in run_pair(n, runs[a], runs[b])]
    if not rows:
        raise SystemExit("nothing to compare")

    df = pd.DataFrame(rows)
    out = Path(args.out) if args.out else root / f"compare_{args.split}_{args.checkpoint}.csv"
    if not out.is_absolute():
        out = ROOT / out
    write_csv_df(out, df)
    show = df.copy()
    for c in ("acc_A", "acc_B"):
        show[c] = (100 * show[c]).round(2)
    for c in ("gain_pp", "ci95_low_pp", "ci95_high_pp"):
        show[c] = show[c].round(2)
    for c in ("p_bootstrap", "p_mcnemar"):
        show[c] = show[c].map(lambda v: "" if v is None or pd.isna(v) else f"{v:.4f}")
    pd.set_option("display.width", 220)
    print(show[["comparison", "A", "B", "n", "acc_A", "acc_B", "gain_pp", "ci95_low_pp", "ci95_high_pp",
                "p_mcnemar", "p_bootstrap", "significant_5pct"]].to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
