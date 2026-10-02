"""Privacy audit for federated ICL (spec §3.2-3.3): every demonstration a client sees must come from
that client's own data, for training (demo + alt demos) AND evaluation. Read-only; exits 1 on any
violation.

    python -m fedicl.audit [--config ...] [overrides...]

Checks, for the partition selected by federated.partition / federated.num_clients:
  1. partition: clients are disjoint and their union is exactly centralized/train.csv
  2. demo_assignment_client_{i}_{train,validation,test}.json: demo_ids / alt_demo_ids are in
     client_i, never share q_hash with the query, and train queries are client_i's own questions
  3. outputs/<federated_icl>/**/predictions_*_client_{i}.jsonl (if the arm has run): every demo
     actually placed in an evaluation prompt belongs to client_i
"""
from __future__ import annotations

import argparse
import logging
import re
from collections import Counter

from .config import ROOT, add_config_args, arm_dir, centralized_dir, config_from_args, partition_dir
from .data.io import SPLITS, load_split
from .utils import read_json, read_jsonl, setup_logging

LOG = logging.getLogger("fedicl.audit")


def _rel(path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p, with_arm=False)
    args = p.parse_args()
    cfg = config_from_args(args)
    setup_logging()
    problems: list[str] = []
    stats: Counter = Counter()

    train = load_split(centralized_dir(cfg) / "train.csv")
    q_hash = {r.id: r.q_hash for r in train.itertuples()}
    for s in ("validation", "test"):
        q_hash.update({r.id: r.q_hash for r in load_split(centralized_dir(cfg) / f"{s}.csv").itertuples()})

    pdir = partition_dir(cfg)
    K = int(cfg.federated.num_clients)
    clients = {i: set(load_split(pdir / f"client_{i}.csv")["id"]) for i in range(1, K + 1)}

    # 1. partition
    union = set().union(*clients.values())
    if union != set(train["id"]):
        problems.append(f"partition: union of clients != train ({len(union)} vs {len(train)})")
    for i in clients:
        for j in clients:
            if i < j and clients[i] & clients[j]:
                problems.append(f"partition: client_{i} and client_{j} share {len(clients[i] & clients[j])} ids")

    # 2. demo assignments
    for i, own in clients.items():
        for s in SPLITS:
            path = pdir / f"demo_assignment_client_{i}_{s}.json"
            if not path.exists():
                problems.append(f"missing {_rel(path)} (run fedicl.retrieval.retrieve)")
                continue
            for qid, rec in read_json(path).items():
                if s == "train" and qid not in own:
                    problems.append(f"client_{i}/{s}: query {qid} is not client_{i}'s own question")
                for key in ("demo_ids", "alt_demo_ids"):
                    for d in rec.get(key, []):
                        stats[f"assign_{key}"] += 1
                        if d not in own:
                            owner = next((f"client_{j}" for j, ids in clients.items() if d in ids), "outside train")
                            problems.append(f"client_{i}/{s}: {key} {d} of {qid} belongs to {owner}")
                        elif q_hash[d] == q_hash[qid]:
                            problems.append(f"client_{i}/{s}: {key} {d} duplicates query {qid}")

    # 3. what the trained federated ICL model was actually shown at evaluation time
    fl_out = arm_dir(load_config_for_arm(cfg))
    pred_files = sorted(fl_out.rglob("predictions_*_client_*.jsonl")) if fl_out.exists() else []
    for f in pred_files:
        i = int(re.search(r"_client_(\d+)\.jsonl$", f.name).group(1))
        for r in read_jsonl(f):
            stats["eval_prompt_demos"] += len(r["demo_ids"])
            if r["demo_pool"] != f"client_{i}":
                problems.append(f"{_rel(f)}: row {r['id']} has demo_pool {r['demo_pool']}")
            bad = [d for d in r["demo_ids"] if d not in clients[i]]
            if bad:
                problems.append(f"{_rel(f)}: {r['id']} shown demos {bad} not in client_{i}")

    print(f"partition       : {_rel(pdir)}  sizes {[len(clients[i]) for i in clients]}")
    print(f"assignments     : {stats['assign_demo_ids']} demos + {stats['assign_alt_demo_ids']} alt demos checked")
    print(f"eval predictions: {len(pred_files)} files, {stats['eval_prompt_demos']} demos shown to the "
          f"model checked" + ("" if pred_files else " (federated_icl has not produced predictions yet)"))
    if problems:
        print(f"FAILED: {len(problems)} violation(s)")
        for msg in problems[:50]:
            print("  -", msg)
        raise SystemExit(1)
    print("PASSED: every federated ICL demo comes from the client's own local data")


def load_config_for_arm(cfg):
    """Same data/partition config, but resolved for the federated_icl arm's output directory."""
    from omegaconf import OmegaConf

    c = OmegaConf.merge(cfg, {"setting": "federated", "icl": {"enabled": True}})
    return c


if __name__ == "__main__":
    main()
