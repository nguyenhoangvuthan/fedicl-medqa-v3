import numpy as np
import pandas as pd

from fedicl.config import ARMS, CONFIG_DIR, arm_dir, config_hash, load_config
from fedicl.data.io import LETTERS, meta_values
from fedicl.data.partition import client_stats
from fedicl.data.prepare import build_medmcqa, centralize, stratified_sample
from fedicl.utils import q_hash, rng_for

MEDMCQA = str(CONFIG_DIR / "medmcqa.yaml")
SMOKE = str(CONFIG_DIR / "smoke.yaml")


def hf_rows(n, prefix, subjects=("Anatomy", "Pharmacology", "Dental"), cop=None):
    """Rows in the HF MedMCQA schema."""
    return pd.DataFrame({
        "id": [f"uuid-{prefix}-{i}" for i in range(n)],
        "question": [f"{prefix} question {i}?" for i in range(n)],
        "opa": [f"a{i}" for i in range(n)], "opb": [f"b{i}" for i in range(n)],
        "opc": [f"c{i}" for i in range(n)], "opd": [f"d{i}" for i in range(n)],
        "cop": cop if cop is not None else [i % 4 for i in range(n)],
        "choice_type": "single", "exp": "",
        "subject_name": [subjects[i % len(subjects)] for i in range(n)],
        "topic_name": None,
    })


def test_stratified_sample_exact_size_and_shares():
    labels = np.array(["a"] * 600 + ["b"] * 300 + ["c"] * 100)
    idx = stratified_sample(labels, 200, rng_for(0, "s"))
    assert len(idx) == 200 and len(set(idx)) == 200
    counts = pd.Series(labels[idx]).value_counts()
    assert abs(counts["a"] - 120) <= 1 and abs(counts["b"] - 60) <= 1 and abs(counts["c"] - 20) <= 1
    assert np.array_equal(idx, stratified_sample(labels, 200, rng_for(0, "s")))
    assert len(stratified_sample(labels, 5000, rng_for(0, "s"))) == len(labels)


def test_build_medmcqa_filters_dedups_and_splits():
    train = hf_rows(300, "tr")
    train.loc[0, "opb"] = " A0 "                           # identical to opa after normalization
    train.loc[1, "question"] = "dev question 5?"           # leaks into the test split
    train.loc[2, "question"] = train.loc[3, "question"]    # duplicate within train
    train.loc[4, "subject_name"] = None                    # missing subject -> "Unknown"
    dev = hf_rows(40, "dev", cop=[-1] + [i % 4 for i in range(39)])   # first row has no gold

    out, info = build_medmcqa(train, dev, {"train": 200, "validation": 50}, seed=42)
    assert info["dropped"] == {"dev_unscorable": 1, "train_unscorable": 1,
                               "train_q_hash_in_test": 1, "train_q_hash_duplicate": 1}
    assert [len(out[s]) for s in ("train", "validation", "test")] == [200, 50, 39]
    for s, df in out.items():
        cent, dropped = centralize(df)
        assert dropped == 0 and len(cent) == len(df), s
        assert df["answer"].isin(LETTERS).all()
    hashes = {s: set(df["question"].map(q_hash)) for s, df in out.items()}
    assert not hashes["train"] & hashes["validation"]
    assert not (hashes["train"] | hashes["validation"]) & hashes["test"]
    assert "medmcqa-train-000000" not in set(out["train"]["id"]) | set(out["validation"]["id"])
    assert out["test"]["id"].iloc[0] == "medmcqa-dev-000001"
    sampled = pd.concat([out["train"], out["validation"]]).set_index("id")["meta"]
    assert sampled.get("medmcqa-train-000004", "Unknown") == "Unknown"
    shares = out["train"]["meta"].value_counts()
    assert shares.max() - shares[["Anatomy", "Pharmacology", "Dental"]].min() <= 3   # stratified
    again, _ = build_medmcqa(train, dev, {"train": 200, "validation": 50}, seed=42)
    assert all(out[s].equals(again[s]) for s in out)


def test_meta_columns_keep_medqa_schema():
    df = pd.DataFrame({"answer": ["A", "B"], "meta": ["step1", "unknown"]})
    assert meta_values("MedQA", [df]) == ["step1", "step2&3"]
    assert meta_values("MedQA", [df], with_unknown=True) == ["step1", "step2&3", "unknown"]
    stats = client_stats([df], meta_values("MedQA", [df]))
    assert list(stats.columns) == ["client_id", "n_samples", *[f"answer_{l}" for l in LETTERS],
                                   "meta_step1", "meta_step2&3"]
    sub = pd.DataFrame({"answer": ["A", "B", "C"], "meta": ["Surgery", "Anatomy", "Surgery"]})
    assert meta_values("MedMCQA", [sub]) == ["Anatomy", "Surgery"]
    stats = client_stats([sub], meta_values("MedMCQA", [sub]))
    assert stats[["meta_Anatomy", "meta_Surgery"]].iloc[0].tolist() == [1, 2]


def test_medmcqa_config_same_protocol_separate_paths():
    assert load_config("federated_icl").save.root == "outputs/qwen3-0.6b/seed42"   # MedQA unchanged
    for arm in ARMS:
        mcq, mqa = load_config(arm, [MEDMCQA]), load_config(arm, [])
        for key in ("model", "prompt", "icl", "retrieval", "loss", "eval", "lora", "train", "federated"):
            assert mcq[key] == mqa[key], key
        assert mcq.data.dir == "processed_data/MedMCQA" and mcq.data_prep.label_key == "meta"
        assert arm_dir(mcq) != arm_dir(mqa) and config_hash(mcq) != config_hash(mqa)
        assert mcq.save.root == "outputs/MedMCQA/qwen3-0.6b/seed42"
    for extra in ([SMOKE, MEDMCQA], [MEDMCQA, SMOKE]):
        smoke = load_config("federated_icl", extra)
        assert smoke.data.dir == "processed_data_smoke/MedMCQA"
        assert smoke.save.root == "outputs_smoke/MedMCQA/qwen3-0.6b/seed42"
    bio = load_config("federated_icl", [MEDMCQA, str(CONFIG_DIR / "biogpt.yaml")])
    assert bio.save.root == "outputs/MedMCQA/biogpt/seed42"
    ab = load_config("federated_icl", [str(CONFIG_DIR / "ablations" / "k2.yaml"), MEDMCQA])
    assert ab.save.root == "outputs/ablation_k2/MedMCQA/qwen3-0.6b/seed42"
