import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from fedicl.data.partition import dirichlet_split, iid_split
from fedicl.retrieval.retrieve import Features, Retriever, RetrievalStats, check_assignment, soft_entity_f1
from fedicl.utils import q_hash, rng_for


def test_iid_split_is_disjoint_cover():
    parts = iid_split(103, 4, rng_for(0, "iid"))
    allidx = np.concatenate(parts)
    assert sorted(allidx.tolist()) == list(range(103))
    assert max(map(len, parts)) - min(map(len, parts)) <= 1


def test_dirichlet_split_cover_and_skew():
    labels = np.array(list("ABCD") * 250)
    parts = dirichlet_split(labels, 3, 0.1, rng_for(0, "d"), min_size=10)
    allidx = np.concatenate(parts)
    assert sorted(allidx.tolist()) == list(range(1000))
    shares = [np.bincount(np.searchsorted(list("ABCD"), labels[p]), minlength=4) / len(p) for p in parts]
    assert max(s.max() for s in shares) > 0.4  # alpha=0.1 must be visibly non-IID


def _toy(n=40, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n, dim)).astype(np.float32)
    emb[1] = emb[0] + 1e-3 * rng.normal(size=dim)        # near-duplicate of item 0
    emb[2] = emb[0] + 0.5 * rng.normal(size=dim)          # similar (cos ~0.9), not duplicate
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    ids = [f"q{i}" for i in range(n)]
    questions = [f"question {i}" for i in range(n)]
    questions[3] = questions[0].upper()                   # same q_hash as q0 (exact duplicate)
    df = pd.DataFrame({"id": ids, "q_hash": [q_hash(q) for q in questions]})
    ent_emb = np.eye(8, dtype=np.float32)
    ents = {i: [j % 8, (j + 1) % 8] for j, i in enumerate(ids)}
    return df, Features(dict(zip(ids, emb)), ents, ent_emb)


@pytest.mark.parametrize("strategy", ["random", "semantic", "semantic_clinical", "adaptive"])
def test_retrieval_rules(strategy):
    df, feats = _toy()
    cfg = OmegaConf.create({"icl": {"k_shot": 3}, "retrieval": {
        "strategy": strategy, "weights": {"semantic": 0.7, "clinical": 0.3},
        "candidate_top_n": 10, "mmr_lambda": 0.7, "near_dup_threshold": 0.95}})
    stats = RetrievalStats()
    assign = Retriever(cfg, feats, 42).assign(df, df, with_alt=True, stats=stats)
    check_assignment(assign, df, df, 0.95, feats, 3)       # raises on any violated rule
    q0 = assign["q0"]
    assert "q0" not in q0["demo_ids"] and "q3" not in q0["demo_ids"]   # self + same q_hash
    assert "q1" not in q0["demo_ids"]                                  # near-duplicate
    assert not set(q0["demo_ids"]) & set(q0["alt_demo_ids"])
    assert all(len(r["demo_ids"]) == 3 for r in assign.values())
    # deterministic
    again = Retriever(cfg, feats, 42).assign(df, df, with_alt=True, stats=RetrievalStats())
    assert again == assign


def test_semantic_prefers_similar_item():
    df, feats = _toy()
    cfg = OmegaConf.create({"icl": {"k_shot": 3}, "retrieval": {
        "strategy": "semantic", "weights": {"semantic": 1.0, "clinical": 0.0},
        "candidate_top_n": 10, "mmr_lambda": 0.7, "near_dup_threshold": 0.95}})
    assign = Retriever(cfg, feats, 42).assign(df.iloc[:1], df, with_alt=False, stats=RetrievalStats())
    assert assign["q0"]["demo_ids"][0] == "q2"


def test_repository_restriction():
    df, feats = _toy()
    cfg = OmegaConf.create({"icl": {"k_shot": 3}, "retrieval": {
        "strategy": "adaptive", "weights": {"semantic": 0.7, "clinical": 0.3},
        "candidate_top_n": 10, "mmr_lambda": 0.7, "near_dup_threshold": 0.95}})
    client = df.iloc[20:]
    assign = Retriever(cfg, feats, 1).assign(df.iloc[:5], client, with_alt=False, stats=RetrievalStats())
    assert all(d in set(client["id"]) for r in assign.values() for d in r["demo_ids"])


def test_soft_entity_f1():
    e = np.eye(4, dtype=np.float32)
    assert soft_entity_f1([0, 1], [0, 1], e) == pytest.approx(1.0)
    assert soft_entity_f1([0], [2], e) == 0.0
    assert soft_entity_f1([], [1], e) == 0.0
