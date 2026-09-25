"""Spec §3.2, §3.3, §3.6: Adaptive Demonstration Retrieval -> demo_assignment_*.json + check report.

    python -m fedicl.retrieval.retrieve [--config ...] [overrides...]

Scopes written (repository in brackets, see §3.2):
  centralized/demo_assignment_{train[train], validation[train], test[train]}.json
  <partition>/demo_assignment_client_{i}_{train, validation, test}.json   [client_i]
The partition is the one selected by federated.partition / federated.num_clients.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (add_config_args, centralized_dir, config_from_args, data_dir, features_dir,
                      partition_dir)
from ..data.io import SPLITS, load_split
from ..utils import read_json, read_jsonl, rng_for, setup_logging, sha1_file, write_json

LOG = logging.getLogger("fedicl.retrieve")
STRATEGIES = ("random", "semantic", "semantic_clinical", "adaptive")


@dataclass
class Features:
    """Question embeddings + entity index lists for every id of the dataset."""
    emb: dict[str, np.ndarray]
    ents: dict[str, list[int]]
    entity_emb: np.ndarray

    @classmethod
    def load(cls, fdir: Path) -> "Features":
        emb, ents = {}, {}
        for s in SPLITS:
            ids = pd.read_csv(fdir / f"question_ids_{s}.csv", dtype=str, keep_default_na=False)["id"]
            mat = np.load(fdir / f"question_emb_{s}.npy").astype(np.float32)
            emb.update(zip(ids, mat))
            ents.update((r["id"], r["entities"]) for r in read_jsonl(fdir / f"entities_{s}.jsonl"))
        return cls(emb, ents, np.load(fdir / "entity_emb.npy").astype(np.float32))

    def matrix(self, ids: list[str]) -> np.ndarray:
        return np.stack([self.emb[i] for i in ids])


def soft_entity_f1(q_ents, d_ents: list[int], entity_emb: np.ndarray) -> float:
    """clin(q, d): F1 of best-match cosine between the two entity sets (0 if either is empty).

    q_ents may be an index list or an already-gathered (n, dim) matrix (hoisted per query).
    """
    if len(q_ents) == 0 or not d_ents:
        return 0.0
    q = q_ents if isinstance(q_ents, np.ndarray) and q_ents.ndim == 2 else entity_emb[q_ents]
    sim = np.clip(q @ entity_emb[d_ents].T, 0.0, 1.0)
    recall, precision = sim.max(1).mean(), sim.max(0).mean()
    return float(2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0


@dataclass
class RetrievalStats:
    n_queries: int = 0
    excluded_q_hash: int = 0
    excluded_near_dup: int = 0
    short_of_k: int = 0
    queries_without_entities: int = 0
    sem_selected: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "sem_selected"}
        s = np.array(self.sem_selected) if self.sem_selected else np.zeros(1)
        d["sem_selected_mean"] = round(float(s.mean()), 4)
        d["sem_selected_p10_p50_p90"] = [round(float(x), 4) for x in np.percentile(s, [10, 50, 90])]
        return d


class Retriever:
    def __init__(self, cfg, feats: Features, seed: int):
        r = cfg.retrieval
        if r.strategy not in STRATEGIES:
            raise ValueError(f"retrieval.strategy must be one of {STRATEGIES}")
        self.r, self.k, self.f, self.seed = r, int(cfg.icl.k_shot), feats, seed

    def assign(self, queries: pd.DataFrame, pool: pd.DataFrame, with_alt: bool,
               stats: RetrievalStats) -> dict[str, dict]:
        f = self.f
        pool_ids = pool["id"].tolist()
        pool_hash = pool["q_hash"].to_numpy()
        pool_emb = f.matrix(pool_ids)
        out: dict[str, dict] = {}
        chunk = 512
        for start in range(0, len(queries), chunk):
            qdf = queries.iloc[start:start + chunk]
            sem_all = f.matrix(qdf["id"].tolist()) @ pool_emb.T
            for row, sem in zip(qdf.itertuples(index=False), sem_all):
                out[row.id] = self._one(row.id, row.q_hash, sem, pool_ids, pool_hash, pool_emb,
                                        with_alt, stats)
        return out

    def _one(self, qid, qhash, sem, pool_ids, pool_hash, pool_emb, with_alt, stats) -> dict:
        r, f, k = self.r, self.f, self.k
        stats.n_queries += 1
        same_hash = pool_hash == qhash
        near_dup = (sem >= r.near_dup_threshold) & ~same_hash
        stats.excluded_q_hash += int(same_hash.sum())
        stats.excluded_near_dup += int(near_dup.sum())
        valid = np.flatnonzero(~same_hash & ~near_dup)
        rng = rng_for(self.seed, qid)
        q_ents = f.ents.get(qid, [])
        if not q_ents:
            stats.queries_without_entities += 1

        if r.strategy == "random":
            order = rng.permutation(valid)
            chosen = self._take_distinct(order, pool_hash, k)
            scores = {"sem": [float(sem[j]) for j in chosen]}
        else:
            n = min(int(r.candidate_top_n), len(valid))
            # random permutation first => argmax/argsort ties break by rng(seed, query_id)
            perm = rng.permutation(valid)
            cand = perm[np.argsort(-sem[perm], kind="stable")[:n]]
            cand_sem = sem[cand]
            if r.strategy == "semantic":
                rel = cand_sem
                clin = np.zeros(len(cand))
            else:
                q_mat = f.entity_emb[q_ents] if q_ents else np.zeros((0, 1), np.float32)
                clin = np.array([soft_entity_f1(q_mat, f.ents.get(pool_ids[j], []), f.entity_emb)
                                 for j in cand])
                rel = r.weights.semantic * cand_sem + r.weights.clinical * clin
            if r.strategy == "adaptive":
                picked = self._mmr(rel, pool_emb[cand], pool_hash[cand], k)
            else:
                picked = self._take_distinct(np.argsort(-rel, kind="stable"), pool_hash[cand], k)
            chosen = [int(cand[i]) for i in picked]
            scores = {"sem": [float(cand_sem[i]) for i in picked],
                      "clin": [float(clin[i]) for i in picked],
                      "rel": [float(rel[i]) for i in picked]}

        if len(chosen) < k:
            stats.short_of_k += 1
        stats.sem_selected.extend(float(sem[j]) for j in chosen)
        rec = {"demo_ids": [pool_ids[j] for j in chosen], **scores}
        if with_alt:
            alt_rng = rng_for(self.seed, qid, "alt")
            rest = np.setdiff1d(valid, np.array(chosen, dtype=int))
            rec["alt_demo_ids"] = [pool_ids[j] for j in
                                   self._take_distinct(alt_rng.permutation(rest), pool_hash, k)]
        return rec

    @staticmethod
    def _take_distinct(order, hashes, k) -> list[int]:
        """First k indices of `order` whose q_hash differs pairwise (rule 3 of §3.3)."""
        seen, out = set(), []
        for j in order:
            if hashes[j] not in seen:
                seen.add(hashes[j])
                out.append(int(j))
                if len(out) == k:
                    break
        return out

    def _mmr(self, rel: np.ndarray, emb: np.ndarray, hashes: np.ndarray, k: int) -> list[int]:
        lam = float(self.r.mmr_lambda)
        sim = emb @ emb.T
        picked: list[int] = []
        used_hash: set = set()
        max_sim = np.full(len(rel), -np.inf)
        for _ in range(min(k, len(rel))):
            score = lam * rel - (1 - lam) * np.where(np.isinf(max_sim), 0.0, max_sim)
            score[picked] = -np.inf
            score[[i for i in range(len(rel)) if hashes[i] in used_hash]] = -np.inf
            if not np.isfinite(score).any():
                break
            best = int(np.argmax(score))  # candidates are already rng-permuted => fair tie-break
            picked.append(best)
            used_hash.add(hashes[best])
            max_sim = np.maximum(max_sim, sim[best])
        return picked


# ------------------------------------------------------------------------------ check (§3.3)
def check_assignment(assign: dict, queries: pd.DataFrame, pool: pd.DataFrame, near_dup: float,
                     feats: Features, k: int) -> dict:
    """Re-verify the 4 hard rules; raises AssertionError on the first violation."""
    pool_hash = dict(zip(pool["id"], pool["q_hash"]))
    qh = dict(zip(queries["id"], queries["q_hash"]))
    assert set(assign) == set(queries["id"]), "every query must have an assignment"
    for qid, rec in assign.items():
        for key in ("demo_ids", "alt_demo_ids"):
            demos = rec.get(key, [])
            assert all(d in pool_hash for d in demos), f"{qid}: {key} outside its repository"
            assert all(pool_hash[d] != qh[qid] for d in demos), f"{qid}: {key} shares q_hash with query"
            assert len({pool_hash[d] for d in demos}) == len(demos), f"{qid}: {key} q_hash not distinct"
            for d in demos:
                s = float(feats.emb[qid] @ feats.emb[d])
                assert s < near_dup + 1e-6, f"{qid}: {key} near-duplicate demo {d} (sem={s:.3f})"
        assert len(rec["demo_ids"]) <= k
    return {"n_queries": len(assign), "passed": True}


def run_scope(name: str, retriever: Retriever, queries: pd.DataFrame, pool: pd.DataFrame,
              with_alt: bool, out_path: Path, cfg, feats: Features, report: dict) -> None:
    stats = RetrievalStats()
    assign = retriever.assign(queries, pool, with_alt, stats)
    check = check_assignment(assign, queries, pool, cfg.retrieval.near_dup_threshold, feats,
                             int(cfg.icl.k_shot))
    write_json(out_path, assign)
    report["scopes"][name] = {"file": str(out_path.relative_to(data_dir(cfg))),
                              "repository_size": len(pool), **stats.as_dict(), "check": check,
                              "sha1": sha1_file(out_path)}
    LOG.info("%s: %s", name, report["scopes"][name])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p, with_arm=False)
    args = p.parse_args()
    cfg = config_from_args(args)
    setup_logging()

    feats = Features.load(features_dir(cfg))
    retriever = Retriever(cfg, feats, int(cfg.seed))
    cdir = centralized_dir(cfg)
    splits = {s: load_split(cdir / f"{s}.csv") for s in SPLITS}
    train_hash = set(splits["train"]["q_hash"])
    report: dict = {
        "strategy": cfg.retrieval.strategy, "k_shot": int(cfg.icl.k_shot),
        "near_dup_threshold": cfg.retrieval.near_dup_threshold,
        "train_overlap_with": {s: int(splits[s]["q_hash"].isin(train_hash).sum())
                               for s in ("validation", "test")},
        "feature_stats": read_json(features_dir(cfg) / "feature_stats.json"),
        "scopes": {},
    }

    # Centralized: repository = full train (train queries also get alt demos for L_ICL).
    for s in SPLITS:
        run_scope(f"centralized/{s}", retriever, splits[s], splits["train"], s == "train",
                  cdir / f"demo_assignment_{s}.json", cfg, feats, report)

    # Federated: repository = client_i only, for both training and evaluation (privacy-aware).
    pdir = partition_dir(cfg)
    if not pdir.exists():
        raise SystemExit(f"{pdir} missing: run `python -m fedicl.data.partition` first")
    for i in range(1, int(cfg.federated.num_clients) + 1):
        client = load_split(pdir / f"client_{i}.csv")
        for s in SPLITS:
            queries = client if s == "train" else splits[s]
            run_scope(f"{pdir.relative_to(data_dir(cfg))}/client_{i}/{s}", retriever, queries, client,
                      s == "train", pdir / f"demo_assignment_client_{i}_{s}.json", cfg, feats, report)

    write_json(cdir / "demo_check_report.json", report)
    LOG.info("all assignment checks passed")


if __name__ == "__main__":
    main()
