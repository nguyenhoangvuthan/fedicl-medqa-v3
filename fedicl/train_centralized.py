"""Centralized arms: up to `train.epochs` epochs on full train, early stopping every eval_every epoch."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import torch

from .config import centralized_dir
from .evaluate import Evaluator
from .matching import Matcher
from .modeling import (load_adapter_state, load_lora_model, load_tokenizer,
                       save_adapter, set_adapter_state, trainable_params)
from .prompt import PromptBuilder
from .results import append_results, result_rows, save_eval_outputs
from .train_common import (EarlyStopper, StepRunner, append_log, build_train_items, eval_views,
                           headline_accuracy, load_assign, load_examples, lr_at, make_optimizer,
                           micro_batches, monitored_loss, run_evaluation, steps_per_pass)
from .utils import read_json, rmtree, rng_for, set_seed, write_json

LOG = logging.getLogger("fedicl.centralized")


def _latest_epoch(ckpt_root: Path) -> int:
    done = [int(p.name.split("_")[1]) for p in ckpt_root.glob("epoch_*")
            if (p / "metrics.json").exists() and (p / "trainer_state" / "state.pt").exists()]
    return max(done, default=0)


def _prune_steps(ckpt_root: Path, keep_best: str | None, keep_last: int) -> None:
    steps = sorted(ckpt_root.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    keep = {p.name for p in steps[-keep_last:]} if keep_last > 0 else set()
    for p in steps:
        if p.name not in keep and str(p) != keep_best:
            rmtree(p)


def _prune_states(ckpt_root: Path, best_path: str | None, keep_last: int) -> None:
    """Optimizer states are large: keep the last k epochs + the best one; adapters are never deleted."""
    epochs = sorted(ckpt_root.glob("epoch_*"), key=lambda p: int(p.name.split("_")[1]))
    keep = {p.name for p in epochs[-keep_last:]}
    for p in epochs:
        if p.name not in keep and str(p) != best_path and (p / "trainer_state").exists():
            rmtree(p / "trainer_state")


def run(cfg, out: Path, extend_to: int | None = None) -> dict:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(int(cfg.seed), bool(cfg.runtime.deterministic))
    tok = load_tokenizer(cfg)
    model = load_lora_model(cfg, dev)
    builder = PromptBuilder(tok, cfg)
    ex, ids = load_examples(cfg)
    assign = load_assign(centralized_dir(cfg) / "demo_assignment_train.json") if cfg.icl.enabled else None

    items = build_train_items(cfg, builder, ex, ids["train"], assign)
    LOG.info("train items %d (demos dropped to fit max_seq_len: %d)", len(items), builder.truncated)
    evaluator = Evaluator(model, tok, builder, Matcher.from_config(cfg, dev), ex, cfg, dev)
    views = eval_views(cfg, ids)
    val_views = [v for v in views if v.split == "validation"]

    ckpt_root = out / "checkpoints"
    params = trainable_params(model)
    opt = make_optimizer(cfg, params)
    runner = StepRunner(model, cfg, dev, tok.pad_token_id, params, use_reg=False)
    es_cfg = cfg.train.early_stopping
    stopper = EarlyStopper(int(es_cfg.patience), float(es_cfg.min_delta))
    max_epochs = int(extend_to or cfg.train.epochs)
    spe = steps_per_pass(len(items), cfg)
    horizon = spe * max_epochs
    check_every = max(1, round(float(es_cfg.eval_every) * spe))
    global_step, start_epoch = 0, 1

    last = _latest_epoch(ckpt_root)
    if last:
        st = torch.load(ckpt_root / f"epoch_{last}" / "trainer_state" / "state.pt", weights_only=False)
        set_adapter_state(model, load_adapter_state(ckpt_root / f"epoch_{last}" / "adapter"))
        opt.load_state_dict(st["optimizer"])
        torch.set_rng_state(st["torch_rng"])
        if st.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        global_step, start_epoch = st["global_step"], last + 1
        stopper = EarlyStopper.from_state(st["early_stop"])
        if extend_to and stopper.stopped and stopper.reason != "overfit":
            stopper.stopped = False
        LOG.info("resumed after epoch %d (global_step %d)", last, global_step)

    def check(at: str, epoch_float: float, epoch_end: bool, metrics: dict | None = None) -> None:
        if metrics is None:
            metrics, _ = run_evaluation(evaluator, val_views,
                                        with_generation=bool(cfg.eval.accuracy_on_checks))
        vl = monitored_loss(metrics)
        path = str(ckpt_root / at)
        improved = stopper.update(vl, at, path)
        append_log(out / "logs" / "val_curve.csv", [{
            "at": at, "step": global_step, "epoch": round(epoch_float, 4), "demo_pool": pool,
            "val_loss": m["val_loss"], "val_accuracy": m.get("accuracy"),
            "train_loss": recent_loss()} for pool, m in metrics["validation"].items()])
        if improved and not epoch_end:
            save_adapter(model, ckpt_root / at / "adapter")
            _prune_steps(ckpt_root, stopper.best_path, int(cfg.save.keep_last_k_states))
        LOG.info("check %s: val_loss=%.4f best=%.4f (%s) bad=%d", at, vl, stopper.best,
                 stopper.best_at, stopper.bad)

    recent: list[float] = []

    def recent_loss() -> float | None:
        return float(np.mean(recent[-20:])) if recent else None

    for epoch in range(start_epoch, max_epochs + 1):
        if stopper.stopped:
            break
        order = rng_for(int(cfg.seed), "epoch", epoch).permutation(len(items))
        logs = []
        steps = list(micro_batches(items, order, int(cfg.train.batch_size), int(cfg.train.grad_accum)))
        for i, micro in enumerate(steps):
            log = runner.step(micro, opt, lr_at(cfg, global_step / horizon))
            global_step += 1
            recent.append(log["loss_total"])
            logs.append({"step": global_step, "epoch": epoch, "client_id": None, **log})
            if global_step % 20 == 0:
                LOG.info("epoch %d step %d/%d loss=%.4f (qa %.4f icl %.4f) lr=%.2e", epoch, i + 1,
                         len(steps), log["loss_total"], log["loss_qa"], log["loss_icl"], log["lr"])
            if (i + 1) % check_every == 0 and i + 1 < len(steps):
                append_log(out / "logs" / "train_log.csv", logs)
                logs = []
                check(f"step_{global_step}", epoch - 1 + (i + 1) / len(steps), epoch_end=False)
                if stopper.stopped:
                    break
        append_log(out / "logs" / "train_log.csv", logs)
        if stopper.stopped:
            LOG.info("early stop inside epoch %d: no epoch checkpoint", epoch)
            break

        # ---- epoch checkpoint: adapter, trainer state, full val+test eval, result rows
        ep_dir = ckpt_root / f"epoch_{epoch}"
        save_adapter(model, ep_dir / "adapter")
        metrics, preds = run_evaluation(evaluator, views)
        check(f"epoch_{epoch}", float(epoch), epoch_end=True, metrics=metrics)
        torch.save({"optimizer": opt.state_dict(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "global_step": global_step, "early_stop": stopper.state()},
                   _mk(ep_dir / "trainer_state") / "state.pt")
        save_eval_outputs(ep_dir, metrics, preds)
        append_results(out / "result.csv", result_rows(
            cfg, f"epoch_{epoch}", epoch, metrics, str((ep_dir / "adapter").relative_to(out))))
        _prune_states(ckpt_root, stopper.best_path, int(cfg.save.keep_last_k_states))

    return finalize(cfg, out, model, evaluator, views, stopper)


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def finalize(cfg, out: Path, model, evaluator, views, stopper: EarlyStopper) -> dict:
    """Restore the best checkpoint into best/, evaluate it on validation + test, write early_stop.json."""
    if stopper.best_path is None:
        raise RuntimeError("no checkpoint was evaluated")
    src = Path(stopper.best_path)
    adapter_src = src / "adapter" if (src / "adapter").exists() else src / "global_adapter"
    best = out / "best"
    set_adapter_state(model, load_adapter_state(adapter_src))
    save_adapter(model, best / "adapter")
    write_json(best / "pointer.json", {"source": str(src.relative_to(out)), "best_at": stopper.best_at,
                                       "best_val_loss": stopper.best})
    if (src / "metrics.json").exists():  # epoch/round checkpoint: reuse its evaluation
        metrics = read_json(src / "metrics.json")
        for f in src.glob("predictions_*.jsonl"):
            shutil.copy(f, best / f.name)
        write_json(best / "metrics.json", metrics)
    else:                                # mid-epoch step checkpoint: evaluate now
        metrics, preds = run_evaluation(evaluator, views)
        save_eval_outputs(best, metrics, preds)
    append_results(out / "result.csv", result_rows(cfg, "best", stopper.best_at, metrics,
                                                   "best/adapter", is_best=True))
    summary = stopper.summary()
    if not stopper.stopped:
        LOG.warning("budget exhausted while val_loss still improving: consider `python -m fedicl.run --arm <arm> --extend_to %d`",
                    int(cfg.train.max_extend))
    write_json(out / "early_stop.json", summary)
    return {"best_at": stopper.best_at, "best_test_acc": headline_accuracy(metrics, "test")}
