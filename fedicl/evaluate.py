"""Spec §5.1: val_loss (L_QA only) + generation + text->option matching, per demo-pool view."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import torch

from .data.io import LETTERS, Example
from .losses import answer_logits
from .matching import Matcher
from .prompt import Encoded, PromptBuilder

LOG = logging.getLogger("fedicl.eval")


@dataclass
class EvalView:
    """One way of evaluating a split: which repository the demos come from.

    pool: 'train' (centralized repo), 'client_i' (FL local repo), 'none' (non-ICL),
          'train_diagnostic' (FL global model with the full train repo; never an FL result).
    """
    split: str
    pool: str
    qids: list[str]
    assign: dict | None

    @property
    def suffix(self) -> str:
        return "" if self.pool in ("train", "none") else f"_{self.pool}"


def collate(encs: list[Encoded], pad_id: int, dev: str) -> dict[str, torch.Tensor]:
    """Right-padded training/val batch; labels = -100 except the answer tokens."""
    L = max(len(e.input_ids) for e in encs)
    ids = torch.full((len(encs), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(encs), L), dtype=torch.long)
    lab = torch.full((len(encs), L), -100, dtype=torch.long)
    for i, e in enumerate(encs):
        n = len(e.input_ids)
        ids[i, :n] = torch.tensor(e.input_ids)
        att[i, :n] = 1
        lab[i, e.n_prompt:n] = ids[i, e.n_prompt:n]
    return {"input_ids": ids.to(dev), "attention_mask": att.to(dev), "labels": lab.to(dev)}


class Evaluator:
    def __init__(self, model, tok, builder: PromptBuilder, matcher: Matcher,
                 examples: dict[str, Example], cfg, dev: str):
        self.model, self.tok, self.b, self.matcher = model, tok, builder, matcher
        self.ex, self.cfg, self.dev = examples, cfg, dev
        self.gen = cfg.eval.generation
        self._cache: dict[tuple, tuple[list[Encoded], list[Encoded]]] = {}

    def demos(self, view: EvalView, qid: str) -> list[Example]:
        if view.assign is None:
            return []
        return [self.ex[d] for d in view.assign[qid]["demo_ids"]]

    def encodings(self, view: EvalView) -> tuple[list[Encoded], list[Encoded]]:
        """(teacher-forced encodings for val_loss, prompt-only encodings for generation), cached."""
        key = (view.split, view.pool)
        if key not in self._cache:
            tf, gen = [], []
            for q in view.qids:
                d = self.demos(view, q)
                tf.append(self.b.encode(self.ex[q], d, with_target=True))
                gen.append(self.b.encode(self.ex[q], d, with_target=False,
                                         reserve=int(self.gen.max_new_tokens)))
            self._cache[key] = (tf, gen)
        return self._cache[key]

    # ------------------------------------------------------------------ loss
    @torch.no_grad()
    def val_loss(self, view: EvalView) -> float:
        tf, _ = self.encodings(view)
        self.model.eval()
        total, n_tok = 0.0, 0
        bs = int(self.cfg.eval.batch_size)
        order = np.argsort([-len(e.input_ids) for e in tf])
        for i in range(0, len(order), bs):
            batch = collate([tf[j] for j in order[i:i + bs]], self.tok.pad_token_id, self.dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev == "cuda"):
                logits, tgt = answer_logits(self.model, batch)
            total += torch.nn.functional.cross_entropy(logits, tgt, reduction="sum").item()
            n_tok += tgt.numel()
        self.model.train()
        return total / max(n_tok, 1)

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, view: EvalView) -> list[str]:
        _, encs = self.encodings(view)
        self.model.eval()
        eos = [self.b.end_id, self.tok.eos_token_id]
        pad = self.tok.pad_token_id
        out: list[str] = [""] * len(encs)
        bs = int(self.cfg.eval.gen_batch_size)
        order = np.argsort([-len(e.input_ids) for e in encs])  # similar lengths => little padding
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            L = max(len(encs[j].input_ids) for j in idx)
            ids = torch.full((len(idx), L), pad, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for r, j in enumerate(idx):  # left padding for decoder-only generation
                n = len(encs[j].input_ids)
                ids[r, L - n:] = torch.tensor(encs[j].input_ids)
                att[r, L - n:] = 1
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev == "cuda"):
                gen = self.model.generate(
                    input_ids=ids.to(self.dev), attention_mask=att.to(self.dev),
                    max_new_tokens=int(self.gen.max_new_tokens), do_sample=bool(self.gen.do_sample),
                    eos_token_id=eos, pad_token_id=pad, use_cache=True,
                    temperature=None, top_p=None, top_k=None)
            texts = self.tok.batch_decode(gen[:, L:], skip_special_tokens=True)
            for r, j in enumerate(idx):
                out[j] = self._clean(texts[r])
        self.model.train()
        return out

    def _clean(self, text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        if self.gen.stop_at_newline:
            text = text.split("\n", 1)[0].strip()
        return text

    # ------------------------------------------------------------------ scoring
    def score(self, view: EvalView, raw: list[str]) -> tuple[dict, list[dict]]:
        _, encs = self.encodings(view)
        preds = []
        displayed = [[self.ex[q].options[j] for j in e.order] for q, e in zip(view.qids, encs)]
        self.matcher.prefetch(raw + [o for opts in displayed for o in opts])
        for q, e, text, opts in zip(view.qids, encs, raw, displayed):
            ex = self.ex[q]
            m = self.matcher.match(text, opts)
            pred = LETTERS[e.order[m.index]] if m.index is not None else None
            preds.append({
                "id": q, "raw_output": text, "pred": pred,
                "pred_text": ex.options[e.order[m.index]] if m.index is not None else None,
                "match_type": m.match_type, "match_score": round(m.score, 4),
                "margin": round(m.margin, 4), "low_confidence": m.low_confidence,
                "gold": ex.answer, "gold_text": ex.gold_text, "correct": pred == ex.answer,
                "demo_ids": [d.id for d in self.demos(view, q)][:e.n_demos], "demo_pool": view.pool,
                "option_order": [LETTERS[j] for j in e.order],
            })
        return metrics_from_predictions(preds), preds

    def evaluate(self, view: EvalView, with_generation: bool = True) -> tuple[dict, list[dict]]:
        metrics = {"val_loss": self.val_loss(view)}
        preds: list[dict] = []
        if with_generation:
            m, preds = self.score(view, self.generate(view))
            metrics.update(m)
        return metrics, preds


def metrics_from_predictions(preds: list[dict]) -> dict:
    n = len(preds)
    if n == 0:
        return {"n_samples": 0}
    mt = [p["match_type"] for p in preds]
    strict = sum(p["correct"] and p["match_type"] in ("exact", "contains") for p in preds)
    return {
        "n_samples": n,
        "accuracy": sum(p["correct"] for p in preds) / n,
        "accuracy_strict": strict / n,
        **{f"{t}_rate": mt.count(t) / n for t in ("exact", "contains", "nearest", "empty")},
        "low_conf_rate": sum(p["low_confidence"] for p in preds) / n,
    }


def aggregate_clients(per_client: dict[str, dict]) -> dict[str, dict]:
    """FL ICL views -> macro_mean (headline FL number), min (worst client) and std rows."""
    accs = [m["accuracy"] for m in per_client.values() if "accuracy" in m]
    losses = [m["val_loss"] for m in per_client.values()]
    macro = {"val_loss": float(np.mean(losses)), "n_samples": next(iter(per_client.values()))["n_samples"]}
    if accs:
        keys = [k for k in next(iter(per_client.values())) if k.endswith(("_rate", "accuracy", "accuracy_strict"))]
        macro.update({k: float(np.mean([m[k] for m in per_client.values()])) for k in keys})
        macro["accuracy_std"] = float(np.std(accs))
    out = {"macro_mean": macro}
    if accs:
        out["min"] = {"accuracy": float(min(accs)), "n_samples": macro["n_samples"],
                      "val_loss": float(max(losses))}
    return out
