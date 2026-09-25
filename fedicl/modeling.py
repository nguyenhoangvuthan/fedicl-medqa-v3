"""Qwen3 + LoRA loading, adapter (de)serialization and FedAvg over LoRA weights."""
from __future__ import annotations

import os
from pathlib import Path

import torch

from .utils import _retry, rmtree



def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_tokenizer(cfg):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(cfg, dev: str):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name, dtype=getattr(torch, cfg.model.dtype),
        attn_implementation=cfg.model.attn_implementation)
    model.config.use_cache = False
    return model.to(dev)


def load_lora_model(cfg, dev: str):
    """Frozen bf16 base + trainable fp32 LoRA (trained under bf16 autocast)."""
    from peft import LoraConfig, get_peft_model

    model = load_base_model(cfg, dev)
    if cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lc = cfg.lora
    model = get_peft_model(model, LoraConfig(
        r=lc.r, lora_alpha=lc.alpha, lora_dropout=lc.dropout,
        target_modules=list(lc.target_modules), bias="none", task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model


def trainable_params(model) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def get_adapter_state(model) -> dict[str, torch.Tensor]:
    from peft import get_peft_model_state_dict

    return {k: v.detach().float().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}


def set_adapter_state(model, state: dict[str, torch.Tensor]) -> None:
    from peft import set_peft_model_state_dict

    res = set_peft_model_state_dict(model, state)
    unexpected = getattr(res, "unexpected_keys", [])
    if unexpected:
        raise RuntimeError(f"adapter keys not in model: {unexpected[:5]}")


def fedavg(states: list[dict[str, torch.Tensor]], weights: list[float]) -> dict[str, torch.Tensor]:
    """Weighted average of LoRA A/B matrices (weights = client sample counts)."""
    total = float(sum(weights))
    return {k: sum(w / total * s[k] for s, w in zip(states, weights)) for k in states[0]}


def save_adapter(model, out_dir: Path, state: dict[str, torch.Tensor] | None = None) -> None:
    """Atomic save as adapter_model.safetensors + adapter_config.json (PEFT format)."""
    out_dir = Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    rmtree(tmp)
    if state is None:
        model.save_pretrained(tmp)
    else:
        from safetensors.torch import save_file

        tmp.mkdir(parents=True)
        model.peft_config["default"].save_pretrained(tmp)
        save_file({k: v.contiguous() for k, v in state.items()}, tmp / "adapter_model.safetensors")
    rmtree(out_dir)
    _retry(os.replace, tmp, out_dir)


def load_adapter_state(adapter_dir: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    # clone: detach from the memory-mapped file so it can later be replaced/deleted (Windows)
    return {k: v.clone() for k, v in load_file(Path(adapter_dir) / "adapter_model.safetensors").items()}


def load_adapter(arm_or_dir: str | Path, checkpoint: str = "best", cfg=None):
    """Spec §5.3 helper: base model with a saved adapter attached, ready for inference.

    load_adapter("outputs/qwen3-0.6b/seed42/federated_icl", "round_2")
    load_adapter("outputs/.../centralized_icl/best/adapter")
    """
    from peft import PeftModel

    from .config import load_config

    path = Path(arm_or_dir)
    if not (path / "adapter_model.safetensors").exists():
        sub = {"best": "best/adapter"}.get(checkpoint)
        if sub is None:
            sub = (f"checkpoints/{checkpoint}/adapter" if checkpoint.startswith(("epoch_", "step_"))
                   else f"rounds/{checkpoint}/global_adapter")
        path = path / sub
    cfg = cfg or load_config()
    base = load_base_model(cfg, device())
    model = PeftModel.from_pretrained(base, path).eval()
    return model, load_tokenizer(cfg)


def gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

