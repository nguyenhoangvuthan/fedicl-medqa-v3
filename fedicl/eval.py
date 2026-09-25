"""Eval-only (spec §5.3): load a saved adapter, evaluate validation + test, append to result.csv.

    python -m fedicl.eval --arm federated_icl --checkpoint round_2
    python -m fedicl.eval --arm centralized_icl --checkpoint best
    python -m fedicl.eval --arm centralized_icl --adapter_path path/to/adapter
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging

import torch

from .config import add_config_args, arm_dir, config_from_args, require_gpu
from .evaluate import Evaluator
from .matching import Matcher
from .modeling import load_adapter_state, load_lora_model, load_tokenizer, set_adapter_state
from .prompt import PromptBuilder
from .results import append_results, result_rows, save_eval_outputs
from .train_common import eval_views, load_examples, run_evaluation
from .utils import setup_logging

LOG = logging.getLogger("fedicl.evalonly")


def adapter_dir_for(out, checkpoint: str):
    if checkpoint == "best":
        return out / "best" / "adapter"
    if checkpoint.startswith(("epoch_", "step_")):
        return out / "checkpoints" / checkpoint / "adapter"
    if checkpoint.startswith("round_"):
        return out / "rounds" / checkpoint / "global_adapter"
    raise ValueError(f"bad checkpoint {checkpoint!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", help="epoch_N | step_N | round_N | best")
    g.add_argument("--adapter_path")
    args = p.parse_args()
    cfg = config_from_args(args)
    require_gpu(cfg)
    out = arm_dir(cfg)
    setup_logging(out / "logs" / "eval.log")
    adir = adapter_dir_for(out, args.checkpoint) if args.checkpoint else args.adapter_path

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer(cfg)
    model = load_lora_model(cfg, dev)
    set_adapter_state(model, load_adapter_state(adir))
    ex, ids = load_examples(cfg)
    ev = Evaluator(model, tok, PromptBuilder(tok, cfg), Matcher.from_config(cfg, dev), ex, cfg, dev)
    metrics, preds = run_evaluation(ev, eval_views(cfg, ids))
    tag = args.checkpoint or "adapter"
    dest = out / "evals" / f"{tag}_{dt.datetime.now():%Y%m%d-%H%M%S}"
    save_eval_outputs(dest, metrics, preds)
    append_results(out / "result.csv", result_rows(cfg, f"{tag}(re-eval)", -1, metrics, str(adir)))
    LOG.info("wrote %s", dest)


if __name__ == "__main__":
    main()
