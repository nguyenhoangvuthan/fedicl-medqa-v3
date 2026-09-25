"""Spec §3.6 steps 1 + 2a (offline): question embeddings, scispaCy entities, SapBERT entity embeddings.

    python -m fedicl.retrieval.features [--config ...] [overrides...]

Encoders are frozen and every question is encoded independently, so computing features once for
the whole train split and slicing per client is identical to each client computing its own.
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from ..config import add_config_args, centralized_dir, config_from_args, features_dir, require_gpu
from ..data.io import SPLITS, load_split
from ..utils import save_numpy, setup_logging, write_csv_df, write_json, write_jsonl
from .encoders import TextEncoder

LOG = logging.getLogger("fedicl.features")


def extract_entities(texts: list[str], model_name: str) -> list[list[str]]:
    import spacy

    nlp = spacy.load(model_name, disable=["lemmatizer"])
    out = []
    for doc in nlp.pipe(texts, batch_size=64):
        ents = dict.fromkeys(e.text.lower().strip() for e in doc.ents if e.text.strip())
        out.append(list(ents))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p, with_arm=False)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    cfg = config_from_args(args)
    require_gpu(cfg)
    setup_logging()
    r = cfg.retrieval
    out = features_dir(cfg)
    if (out / "entity_emb.npy").exists() and not args.overwrite:
        LOG.info("features exist at %s; use --overwrite to rebuild", out)
        return

    dfs = {s: load_split(centralized_dir(cfg) / f"{s}.csv") for s in SPLITS}

    qenc = TextEncoder(r.question_encoder.name, r.question_encoder.pooling, r.question_encoder.max_length)
    for s, df in dfs.items():
        texts = df[list(r.question_encoder.fields)].agg("\n".join, axis=1).tolist()
        emb = qenc.encode(texts, r.encode_batch_size)
        save_numpy(out / f"question_emb_{s}.npy", emb.astype(np.float16))
        write_csv_df(out / f"question_ids_{s}.csv", df[["id"]])
        LOG.info("question_emb_%s: %s", s, emb.shape)
    qenc.close()

    ents = {s: extract_entities(df["question"].tolist(), r.entity_extractor) for s, df in dfs.items()}
    vocab = list(dict.fromkeys(e for s in SPLITS for lst in ents[s] for e in lst))
    index = {e: i for i, e in enumerate(vocab)}
    for s, df in dfs.items():
        write_jsonl(out / f"entities_{s}.jsonl",
                    ({"id": i, "entities": [index[e] for e in lst]} for i, lst in zip(df["id"], ents[s])))

    eenc = TextEncoder(r.entity_encoder.name, r.entity_encoder.pooling, r.entity_encoder.max_length)
    eemb = eenc.encode(vocab, batch_size=256)
    eenc.close()
    write_csv_df(out / "entity_vocab.csv", pd.DataFrame({"entity": vocab}))
    save_numpy(out / "entity_emb.npy", eemb.astype(np.float16))

    stats = {s: {"n_questions": len(ents[s]),
                 "avg_entities_per_question": float(np.mean([len(x) for x in ents[s]])) if ents[s] else 0.0,
                 "n_without_entities": int(sum(len(x) == 0 for x in ents[s]))} for s in SPLITS}
    write_json(out / "feature_stats.json", {
        "question_encoder": dict(r.question_encoder), "entity_extractor": r.entity_extractor,
        "entity_encoder": dict(r.entity_encoder), "entity_vocab_size": len(vocab), "splits": stats})
    LOG.info("entity vocab %d, stats %s", len(vocab), stats)


if __name__ == "__main__":
    main()
