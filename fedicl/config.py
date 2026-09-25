"""Config loading: configs/base.yaml <- extra yaml files <- configs/arms/{arm}.yaml <- CLI dotlist."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .utils import slug

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
ARMS = ("centralized_icl", "centralized_non_icl", "federated_icl", "federated_non_icl")

# Keys that change bookkeeping only, never results -> excluded from config_hash.
_HASH_EXCLUDE = {("save", "overwrite"), ("eval", "batch_size"), ("eval", "gen_batch_size")}


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
    return load_config(getattr(args, "arm", None), args.config, args.overrides)
