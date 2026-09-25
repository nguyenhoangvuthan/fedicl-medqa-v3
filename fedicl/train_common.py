"""Pieces shared by the centralized trainer and the FL simulator."""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import centralized_dir, partition_dir
from .data.io import SPLITS, Example, load_split, to_examples
from .evaluate import EvalView, Evaluator, aggregate_clients, collate
from .losses import answer_logits, icl_consistency_loss, proximal_loss, qa_loss
from .prompt import Encoded, PromptBuilder
from .utils import read_json, write_csv_df

LOG = logging.getLogger("fedicl.train")


# ------------------------------------------------------------------ data
def load_examples(cfg) -> tuple[dict[str, Example], dict[str, list[str]]]:
    ex: dict[str, Example] = {}
    ids: dict[str, list[str]] = {}
    for s in SPLITS:
        df = load_split(centralized_dir(cfg) / f"{s}.csv")
        ex.update(to_examples(df))
        ids[s] = df["id"].tolist()
    return ex, ids


def load_assign(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"{path} missing: run `python -m fedicl.retrieval.retrieve` first")
    return read_json(path)


@dataclass
class TrainItem:
    main: Encoded
    alt: Encoded | None  # context for L_ICL (alt demos, or no demos for context_distill)


def build_train_items(cfg, builder: PromptBuilder, ex: dict[str, Example], qids: list[str],
                      assign: dict | None) -> list[TrainItem]:
    use_icl_loss = cfg.icl.enabled and cfg.loss.icl.weight > 0
    items = []
    for q in qids:
        demos = [ex[d] for d in assign[q]["demo_ids"]] if assign else []
        main = builder.encode(ex[q], demos, with_target=True)
        alt = None
        if use_icl_loss:
            if cfg.loss.icl.variant == "demo_consistency":
                alt_demos = [ex[d] for d in assign[q]["alt_demo_ids"]]
            elif cfg.loss.icl.variant == "context_distill":
                alt_demos = []
            else:
                raise ValueError(f"unknown loss.icl.variant {cfg.loss.icl.variant!r}")
            alt = builder.encode(ex[q], alt_demos, with_target=True)
        items.append(TrainItem(main, alt))
    return items


def eval_views(cfg, ids: dict[str, list[str]], splits=("validation", "test")) -> list[EvalView]:
    """Which (split, repository) pairs a checkpoint is evaluated on (spec §3.2)."""
    views = []
    for s in splits:
        if not cfg.icl.enabled:
            views.append(EvalView(s, "none", ids[s], None))
        elif cfg.setting == "centralized":
            views.append(EvalView(s, "train", ids[s],
                                  load_assign(centralized_dir(cfg) / f"demo_assignment_{s}.json")))
        else:
            pdir = partition_dir(cfg)
            for i in range(1, int(cfg.federated.num_clients) + 1):
                views.append(EvalView(s, f"client_{i}", ids[s],
                                      load_assign(pdir / f"demo_assignment_client_{i}_{s}.json")))
            if cfg.eval.fl_global_pool_diagnostic:
                views.append(EvalView(s, "train_diagnostic", ids[s],
                                      load_assign(centralized_dir(cfg) / f"demo_assignment_{s}.json")))
    return views


def run_evaluation(evaluator: Evaluator, views: list[EvalView], with_generation: bool = True):
    """-> (metrics {split: {pool: m}}, predictions {(split, suffix): preds}, monitored val_loss)."""
    metrics: dict[str, dict] = {}
    preds: dict[tuple[str, str], list[dict]] = {}
    for v in views:
        m, p = evaluator.evaluate(v, with_generation=with_generation)
        metrics.setdefault(v.split, {})[v.pool] = m
        preds[(v.split, v.suffix)] = p
        LOG.info("  eval %-10s %-16s val_loss=%.4f acc=%s", v.split, v.pool, m["val_loss"],
                 f"{m['accuracy']:.4f}" if "accuracy" in m else "-")
    for split, pools in metrics.items():
        clients = {k: v for k, v in pools.items() if k.startswith("client_")}
        if clients:
            pools.update(aggregate_clients(clients))
    return metrics, preds


def monitored_loss(metrics: dict, split: str = "validation") -> float:
    """Early-stopping signal: FL ICL -> macro mean over client views; else the single view."""
    pools = metrics[split]
    for key in ("macro_mean", "train", "none"):
        if key in pools:
            return float(pools[key]["val_loss"])
    raise KeyError(f"no monitored view in {list(pools)}")


def headline_accuracy(metrics: dict, split: str) -> float | None:
    pools = metrics.get(split, {})
    for key in ("macro_mean", "train", "none"):
        if key in pools and "accuracy" in pools[key]:
            return float(pools[key]["accuracy"])
    return None


# ------------------------------------------------------------------ optimisation
def lr_at(cfg, progress: float) -> float:
    """Linear warmup then linear decay to min_lr_ratio * lr; progress in [0, 1] over the horizon."""
    t = cfg.train
    base, w, floor = float(t.lr), float(t.warmup_ratio), float(t.min_lr_ratio)
    if w > 0 and progress < w:
        return base * max(progress / w, 0.01)
    frac = min(max((progress - w) / max(1 - w, 1e-8), 0.0), 1.0)
    return base * (1 - (1 - floor) * frac)


def make_optimizer(cfg, params) -> torch.optim.Optimizer:
    return torch.optim.AdamW(params, lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))


def micro_batches(items: list, order: np.ndarray, batch_size: int, grad_accum: int):
    """Yield one optimizer step = list of grad_accum micro-batches (last step may be shorter)."""
    per_step = batch_size * grad_accum
    for s in range(0, len(order), per_step):
        idx = order[s:s + per_step]
        yield [[items[j] for j in idx[m:m + batch_size]] for m in range(0, len(idx), batch_size)]


def steps_per_pass(n: int, cfg) -> int:
    return math.ceil(n / (int(cfg.train.batch_size) * int(cfg.train.grad_accum)))


class StepRunner:
    """One optimizer step of L_QA + lambda_ICL * L_ICL + L_REG."""

    def __init__(self, model, cfg, dev: str, pad_id: int, params: list[torch.nn.Parameter],
                 use_reg: bool):
        self.model, self.cfg, self.dev, self.pad_id, self.params = model, cfg, dev, pad_id, params
        self.lam_icl = float(cfg.loss.icl.weight) if cfg.icl.enabled else 0.0
        self.mu = float(cfg.loss.reg.mu) if use_reg else 0.0
        self.global_params: list[torch.Tensor] | None = None

    def set_global(self) -> None:
        """Snapshot theta^(t) at the start of an FL round for the proximal term."""
        self.global_params = [p.detach().clone() for p in self.params] if self.mu > 0 else None

    def step(self, micro: list[list[TrainItem]], optimizer, lr: float) -> dict[str, float]:
        # from_pretrained returns eval mode: without this, LoRA dropout AND gradient checkpointing
        # (which only runs when module.training) are silently off.
        self.model.train()
        for g in optimizer.param_groups:
            g["lr"] = lr
        n_total = sum(len(m) for m in micro)
        logs = {"loss_qa": 0.0, "loss_icl": 0.0, "loss_reg": 0.0, "loss_total": 0.0}
        for mb in micro:
            w = len(mb) / n_total
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev == "cuda"):
                logits, tgt = answer_logits(self.model, collate([it.main for it in mb], self.pad_id, self.dev))
                l_qa = qa_loss(logits, tgt)
                l_icl = torch.zeros((), device=self.dev)
                if self.lam_icl > 0 and mb[0].alt is not None:
                    alt_logits, alt_tgt = answer_logits(
                        self.model, collate([it.alt for it in mb], self.pad_id, self.dev))
                    if not torch.equal(alt_tgt, tgt):
                        raise RuntimeError("answer tokens differ between main and alt contexts")
                    l_icl = icl_consistency_loss(logits, alt_logits)
            l_reg = (proximal_loss(self.params, self.global_params, self.mu)
                     if self.global_params is not None else torch.zeros((), device=self.dev))
            loss = l_qa + self.lam_icl * l_icl + l_reg
            (loss * w).backward()
            for k, v in (("loss_qa", l_qa), ("loss_icl", l_icl), ("loss_reg", l_reg), ("loss_total", loss)):
                logs[k] += float(v.detach()) * w
        gn = torch.nn.utils.clip_grad_norm_(self.params, float(self.cfg.train.max_grad_norm))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        logs["lr"], logs["grad_norm"] = lr, float(gn)
        return logs


# ------------------------------------------------------------------ early stopping
@dataclass
class EarlyStopper:
    patience: int
    min_delta: float
    best: float = float("inf")
    best_at: str | None = None
    best_path: str | None = None
    bad: int = 0
    stopped: bool = False
    reason: str | None = None
    stop_at: str | None = None
    history: list = field(default_factory=list)

    def update(self, value: float, at: str, path: str | None) -> bool:
        """Returns True when `value` is a new best."""
        self.history.append({"at": at, "val_loss": value})
        if value < self.best - self.min_delta:
            self.best, self.best_at, self.best_path, self.bad = value, at, path, 0
            return True
        self.bad += 1
        if self.bad >= self.patience:
            self.stopped, self.reason, self.stop_at = True, "overfit", at
        return False

    def state(self) -> dict:
        return asdict(self)

    @classmethod
    def from_state(cls, d: dict) -> "EarlyStopper":
        return cls(**d)

    def summary(self) -> dict:
        return {"stopped": self.stopped, "reason": self.reason or "max_budget_reached",
                "stop_at": self.stop_at, "best_at": self.best_at, "best_val_loss": self.best,
                "history": self.history}


def append_log(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    write_csv_df(path, df)
