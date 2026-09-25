"""Spec §3.1 + §3.7: MedQA (HF) -> raw/*.csv -> centralized/*.csv, config.json, dataset_summary.csv.

    python -m fedicl.data.prepare [--config configs/smoke.yaml] [overrides...]
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from ..config import add_config_args, centralized_dir, config_from_args, data_dir
from ..utils import normalize_question, q_hash, setup_logging, sha1_file, write_csv_df, write_json
from .io import COLUMNS, LETTERS, OPTION_COLS, RAW_COLUMNS, SPLITS, save_split

LOG = logging.getLogger("fedicl.prepare")


def load_medqa(source: str, meta_source: str | None) -> dict[str, pd.DataFrame]:
    from datasets import load_dataset

    ds = load_dataset(source)
    meta_map: dict[str, str] = {}
    if meta_source:
        try:
            mds = load_dataset(meta_source)
            for split in mds:
                for q, m in zip(mds[split]["question"], mds[split]["meta_info"]):
                    meta_map[normalize_question(q)] = m
        except Exception as e:  # meta is only used for statistics; never block the pipeline on it
            LOG.warning("could not load meta source %s (%s); meta = 'unknown'", meta_source, e)

    out = {}
    for split in SPLITS:
        d = ds[split]
        rows = []
        for i, r in enumerate(d):
            question = " ".join(s.strip() for s in (r["sent1"], r["sent2"]) if s and s.strip())
            rows.append({
                "id": f"medqa-{split}-{i:05d}",
                "question": question,
                **{c: r[f"ending{j}"] for j, c in enumerate(OPTION_COLS)},
                "answer": LETTERS[int(r["label"])],
                "meta": meta_map.get(normalize_question(question), "unknown"),
            })
        out[split] = pd.DataFrame(rows, columns=RAW_COLUMNS)
        n_meta = (out[split]["meta"] != "unknown").sum()
        LOG.info("%s: %d rows, meta matched for %d", split, len(rows), n_meta)
    return out


def centralize(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Validate rows (answer in A-D, no empty option, unique id) and add q_hash. Duplicates stay."""
    ok = df["answer"].isin(LETTERS)
    for c in OPTION_COLS:
        ok &= df[c].str.strip() != ""
    ok &= df["question"].str.strip() != ""
    ok &= ~df["id"].duplicated(keep="first")
    out = df[ok].copy()
    out["q_hash"] = out["question"].map(q_hash)
    return out[COLUMNS].reset_index(drop=True), int((~ok).sum())


def summarize(layer: str, split: str, df: pd.DataFrame, train_hashes: set[str], n_dropped: int,
              tok, system_prompt: str) -> dict:
    hashes = df["question"].map(q_hash)
    row = {
        "layer": layer, "split": split, "n_samples": len(df), "n_dropped_invalid": n_dropped,
        "n_unique_q_hash": hashes.nunique(),
        "n_dup_within_split": int(len(df) - hashes.nunique()),
        "n_overlap_with_train": 0 if split == "train" else int(hashes.isin(train_hashes).sum()),
    }
    for l in LETTERS:
        row[f"answer_{l}"] = int((df["answer"] == l).sum())
    row["meta_step1"] = int((df["meta"] == "step1").sum())
    row["meta_step2&3"] = int((df["meta"] == "step2&3").sum())
    row["meta_unknown"] = int((df["meta"] == "unknown").sum())
    if tok is not None and len(df):
        q_tok = np.array([len(x) for x in tok(df["question"].tolist(), add_special_tokens=False).input_ids])
        opts = df[list(OPTION_COLS)].to_numpy().ravel().tolist()
        o_tok = np.array([len(x) for x in tok(opts, add_special_tokens=False).input_ids])
        per_q = q_tok + o_tok.reshape(-1, 4).sum(1) + 12  # "Question:/Options:/- " scaffolding
        sys_tok = len(tok(system_prompt, add_special_tokens=False).input_ids) + 20
        row["avg_question_tokens"] = round(float(q_tok.mean()), 1)
        row["max_option_tokens"] = int(o_tok.max())
        # Estimate: system + 3 demos + query, each at the split's 99th percentile length.
        row["est_max_prompt_tokens_icl"] = int(sys_tok + 4 * np.percentile(per_q, 99))
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p, with_arm=False)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    cfg = config_from_args(args)
    setup_logging()

    out_root = data_dir(cfg)
    if (centralized_dir(cfg) / "train.csv").exists() and not args.overwrite:
        LOG.info("%s already prepared; use --overwrite to rebuild", out_root)
        return

    raw = load_medqa(cfg.data.hf_source, cfg.data.hf_meta_source)
    limit = cfg.data_prep.max_samples_per_split
    if limit:
        raw = {s: df.head(int(limit[s])).reset_index(drop=True) for s, df in raw.items()}
        LOG.warning("SMOKE subset: %s", {s: len(d) for s, d in raw.items()})

    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(cfg.model.name)
    except Exception as e:
        LOG.warning("tokenizer unavailable (%s); token columns skipped", e)
        tok = None

    cent, dropped = {}, {}
    for split, df in raw.items():
        save_split(df, out_root / "raw" / f"{split}.csv", RAW_COLUMNS)
        cent[split], dropped[split] = centralize(df)
        save_split(cent[split], centralized_dir(cfg) / f"{split}.csv")

    train_hashes = set(cent["train"]["q_hash"])
    rows = []
    for layer, dfs in (("raw", raw), ("centralized", cent)):
        for split in SPLITS:
            rows.append(summarize(layer, split, dfs[split], train_hashes,
                                  dropped[split] if layer == "centralized" else 0,
                                  tok, cfg.prompt.system))
    write_csv_df(out_root / "dataset_summary.csv", pd.DataFrame(rows))

    files = {f"raw/{s}.csv": sha1_file(out_root / "raw" / f"{s}.csv") for s in SPLITS}
    files.update({f"centralized/{s}.csv": sha1_file(centralized_dir(cfg) / f"{s}.csv") for s in SPLITS})
    write_json(out_root / "config.json", {
        "name": cfg.data.dataset,
        "source": {"hf_dataset": cfg.data.hf_source, "meta_source": cfg.data.hf_meta_source,
                   "note": "MedQA USMLE 4-options (Jin et al., 2021), official splits"},
        "language": "en",
        "num_options": 4,
        "option_letters": list(LETTERS),
        "splits": {s: f"centralized/{s}.csv" for s in SPLITS},
        "columns": COLUMNS,
        "normalize_rule": "lowercase, collapse whitespace, strip punctuation at both ends",
        "hash_algo": "sha1(normalize(question))",
        "subset": OmegaConf.to_container(limit) if limit else None,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "sha1": files,
    })
    LOG.info("wrote %s", out_root)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
