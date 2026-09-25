"""Spec §4.2 Local Training Objective: L_i = L_QA + lambda_ICL * L_ICL + lambda_REG * L_REG.

Logits are computed ONLY at answer positions: running lm_head on every position would cost
batch x seq x 152k-vocab floats (~5 GB at 4 x 2048) per forward, and L_ICL needs two forwards.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _backbone_and_head(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return getattr(base, base.base_model_prefix), base.get_output_embeddings()


def answer_logits(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """(logits, targets) flattened over answer tokens, in batch order then position order."""
    backbone, head = _backbone_and_head(model)
    hidden = backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
    labels = batch["labels"][:, 1:]
    mask = labels != -100
    logits = head(hidden[:, :-1][mask]).float()
    return logits, labels[mask]


def qa_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """L_QA: token-level cross-entropy on the gold option text + end-of-turn."""
    return F.cross_entropy(logits, targets)


def icl_consistency_loss(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    """L_ICL = mean_t KL( stopgrad(p(.|retrieved demos)) || p(.|alternative context) ).

    Both contexts are teacher-forced on the same answer tokens, so positions align 1:1.
    """
    t = F.log_softmax(teacher_logits.detach(), dim=-1)
    s = F.log_softmax(student_logits, dim=-1)
    return F.kl_div(s, t, log_target=True, reduction="batchmean")


def proximal_loss(params: list[torch.Tensor], global_params: list[torch.Tensor], mu: float) -> torch.Tensor:
    """L_REG (FedProx): mu/2 * ||theta_LoRA - theta_LoRA^global||^2."""
    sq = sum(((p - g) ** 2).sum() for p, g in zip(params, global_params))
    return 0.5 * mu * sq
