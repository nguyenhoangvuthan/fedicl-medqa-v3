"""Re-run match_answer() on saved raw_output with the current eval.match config (no generation).

    python -m fedicl.rescore --arm federated_icl eval.match.weights.semantic=0.3 eval.match.weights.lexical=0.7
Writes predictions_*.rescored.jsonl + metrics.rescored.json next to each original file.
"""
from __future__ import annotations

import argparse
import logging

from omegaconf import OmegaConf

from .config import add_config_args, arm_dir, config_from_args
from .evaluate import metrics_from_predictions
from .matching import Matcher
from .data.io import LETTERS
from .train_common import load_examples
from .utils import read_jsonl, setup_logging, write_json, write_jsonl

LOG = logging.getLogger("fedicl.rescore")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    args = p.parse_args()
    cfg = config_from_args(args)
    setup_logging()
    out = arm_dir(cfg)
    ex, _ = load_examples(cfg)
    matcher = Matcher.from_config(cfg)
    summary = {}
    for f in sorted(out.rglob("predictions_*.jsonl")):
        if f.name.endswith(".rescored.jsonl"):
            continue
        rows = read_jsonl(f)
        displayed = [[ex[r["id"]].options[LETTERS.index(l)] for l in r["option_order"]] for r in rows]
        matcher.prefetch([r["raw_output"] for r in rows] + [o for d in displayed for o in d])
        for r, opts in zip(rows, displayed):
            m = matcher.match(r["raw_output"], opts)
            r["pred"] = r["option_order"][m.index] if m.index is not None else None
            r["pred_text"] = opts[m.index] if m.index is not None else None
            r.update(match_type=m.match_type, match_score=round(m.score, 4), margin=round(m.margin, 4),
                     low_confidence=m.low_confidence, correct=r["pred"] == r["gold"])
        write_jsonl(f.with_suffix(".rescored.jsonl"), rows)
        summary[str(f.relative_to(out))] = metrics_from_predictions(rows)
        LOG.info("%s: acc=%.4f", f.relative_to(out), summary[str(f.relative_to(out))]["accuracy"])
    write_json(out / "metrics.rescored.json", {"eval_match": OmegaConf.to_container(cfg.eval.match), "files": summary})


if __name__ == "__main__":
    main()
