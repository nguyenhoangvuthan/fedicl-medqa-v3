"""Config loading: configs/base.yaml <- extra yaml files <- configs/arms/{arm}.yaml <- CLI dotlist."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .utils import slug

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
ARMS = ("centralized_icl", "centralized_non_icl", "federated_icl", "federated_non_icl")

# Keys that change bookkeeping only, never results -> excluded from config_hash.
# runtime.gpu is excluded so an arm can resume on the other (identical) GPU.
_HASH_EXCLUDE = {("save", "overwrite"), ("eval", "batch_size"), ("eval", "gen_batch_size"),
                 ("runtime", "gpu")}


def load_config(arm: str | None = None, extra: list[str] | None = None,
                overrides: list[str] | None = None) -> DictConfig:
    cfg = OmegaConf.load(CONFIG_DIR / "base.yaml")
    for path in extra or []:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    if arm is not None:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
        cfg = OmegaConf.merge(cfg, OmegaConf.load(CONFIG_DIR / "arms" / f"{arm}.yaml"))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    return cfg


def arm_name(cfg: DictConfig) -> str:
    return f"{cfg.setting}_{'icl' if cfg.icl.enabled else 'non_icl'}"


def arm_dir(cfg: DictConfig) -> Path:
    return ROOT / cfg.save.root / arm_name(cfg)


def data_dir(cfg: DictConfig) -> Path:
    return ROOT / cfg.data.dir


def centralized_dir(cfg: DictConfig) -> Path:
    return ROOT / cfg.data.centralized_dir


def partition_dir(cfg: DictConfig, ptype: str | None = None, alpha: float | None = None,
                  k: int | None = None) -> Path:
    ptype = ptype or cfg.federated.partition.type
    k = k or cfg.federated.num_clients
    if ptype == "iid":
        return data_dir(cfg) / "federated_iid" / f"k{k}"
    if ptype == "noniid":
        alpha = cfg.federated.partition.alpha if alpha is None else alpha
        return data_dir(cfg) / "federated_noniid" / f"alpha_{alpha}" / f"k{k}"
    raise ValueError(f"unknown partition type {ptype!r}")


def feature_tag(cfg: DictConfig) -> str:
    r = cfg.retrieval
    return "__".join([
        slug(r.question_encoder.name.split("/")[-1]),
        slug(r.entity_extractor),
        slug(r.entity_encoder.name.split("/")[-1]),
    ])


def features_dir(cfg: DictConfig) -> Path:
    return centralized_dir(cfg) / "features" / feature_tag(cfg)


def config_hash(cfg: DictConfig) -> str:
    c = OmegaConf.to_container(cfg, resolve=True)
    for a, b in _HASH_EXCLUDE:
        c.get(a, {}).pop(b, None)
    return hashlib.sha1(json.dumps(c, sort_keys=True).encode()).hexdigest()[:16]


def add_config_args(p: argparse.ArgumentParser, with_arm: bool = True) -> None:
    if with_arm:
        p.add_argument("--arm", choices=ARMS, help="experiment arm")
    p.add_argument("--config", action="append", default=[],
                   help="extra yaml merged over base.yaml (repeatable), e.g. configs/smoke.yaml")
    p.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. train.lr=1e-4")


def config_from_args(args: argparse.Namespace) -> DictConfig:
    cfg = load_config(getattr(args, "arm", None), args.config, args.overrides)
    select_gpu(cfg)
    return cfg


def requested_gpu(cfg: DictConfig) -> str | None:
    """GPU id(s) to use: env FEDICL_GPU overrides runtime.gpu. None/'all' = leave visibility as is."""
    gpu = os.environ.get("FEDICL_GPU")
    if gpu is None:
        gpu = cfg.get("runtime", {}).get("gpu")
    if gpu is None or str(gpu).strip().lower() in ("", "none", "null", "all"):
        return None
    ids = str(gpu).replace(" ", "")
    if not re.fullmatch(r"\d+(,\d+)*", ids):
        raise SystemExit(f"invalid GPU selection {gpu!r}: use an index like 0 or 1 (see nvidia-smi)")
    return ids


def select_gpu(cfg: DictConfig) -> None:
    """Pin the process to the requested GPU. Must run before CUDA initializes (it does: every CLI
    calls config_from_args first). PCI_BUS_ID order makes index N the same GPU as in nvidia-smi."""
    ids = requested_gpu(cfg)
    if ids is None:
        return
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ids


def require_gpu(cfg: DictConfig) -> None:
    """Fail loudly instead of silently training on CPU when the requested GPU is not visible."""
    ids = requested_gpu(cfg)
    if ids is None:
        return
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(f"GPU {ids} requested (runtime.gpu / FEDICL_GPU) but CUDA sees no device: "
                         "check the index with nvidia-smi and the NVIDIA driver")
