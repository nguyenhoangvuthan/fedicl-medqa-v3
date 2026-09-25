"""Spec §5.1: map a free-text generation to the closest option (never to a letter).

Cascade: empty -> exact -> contains (longest option wins) -> nearest (always argmax of
w_lex * token-F1 + w_sem * SapBERT cosine). `low_confidence` is a diagnostic flag only.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass

import numpy as np

_PUNCT = string.punctuation + "“”‘’«»…–—"
_PREFIXES = re.compile(
    r"^(?:(?:the\s+)?(?:correct\s+|final\s+|best\s+)?answer(?:\s+is)?\s*[:\-]?\s*|"
    r"answer\s*[:\-]\s*|[-*•]\s+)+"
)
_TOKEN = re.compile(r"\w+")


def normalize(text: str) -> str:
    t = re.sub(r"\s+", " ", text.lower()).strip()
    t = t.strip(_PUNCT + " ")
    prev = None
    while prev != t:  # strip stacked prefixes like "- The answer is: ..."
        prev = t
        t = _PREFIXES.sub("", t).strip(_PUNCT + " ")
    return t


def token_f1(a: str, b: str) -> float:
    ta, tb = _TOKEN.findall(a), _TOKEN.findall(b)
    if not ta or not tb:
        return 0.0
    common = sum((Counter(ta) & Counter(tb)).values())
    if common == 0:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


@dataclass
class Match:
    index: int | None       # index into the options list given (displayed order)
    match_type: str         # exact | contains | nearest | empty
    score: float
    margin: float
    low_confidence: bool


class Matcher:
    def __init__(self, w_lex: float = 0.5, w_sem: float = 0.5, encoder=None,
                 min_score: float = 0.5, min_margin: float = 0.05):
        if w_sem > 0 and encoder is None:
            raise ValueError("semantic weight > 0 needs an encoder")
        self.w_lex, self.w_sem, self.encoder = w_lex, w_sem, encoder
        self.min_score, self.min_margin = min_score, min_margin

    @classmethod
    def from_config(cls, cfg, device: str | None = None) -> "Matcher":
        m = cfg.eval.match
        enc = None
        if m.weights.semantic > 0:
            from .retrieval.encoders import TextEncoder

            e = m.semantic_encoder
            enc = TextEncoder(e.name, e.pooling, e.max_length, device=device)
        return cls(m.weights.lexical, m.weights.semantic, enc, m.low_conf.min_score,
                   m.low_conf.min_margin)

    def prefetch(self, texts: list[str]) -> None:
        """Batch-encode normalized texts once so match() is cheap."""
        if self.encoder is not None:
            self.encoder.encode_cached([t for t in dict.fromkeys(map(normalize, texts)) if t])

    def match(self, generated: str, options: list[str]) -> Match:
        gen = normalize(generated)
        if not gen:
            return Match(None, "empty", 0.0, 0.0, True)
        opts = [normalize(o) for o in options]

        exact = [i for i, o in enumerate(opts) if o == gen]
        if exact:
            return Match(exact[0], "exact", 1.0, 1.0, False)

        contained = [i for i, o in enumerate(opts)
                     if o and re.search(r"(?<!\w)" + re.escape(o) + r"(?!\w)", gen)]
        # Keep only maximal matches: "aspirin" is dropped when "aspirin and clopidogrel" also matches.
        maximal = [i for i in contained
                   if not any(j != i and opts[i] in opts[j] for j in contained)]
        if len(maximal) == 1:
            return Match(maximal[0], "contains", 1.0, 1.0, False)
        ambiguous = len(maximal) > 1  # e.g. "aspirin or heparin": never break the tie by position

        scores = np.array([self.w_lex * token_f1(gen, o) for o in opts])
        if self.w_sem > 0:
            vecs = self.encoder.encode_cached([gen, *opts])
            scores = scores + self.w_sem * np.clip(vecs[1:] @ vecs[0], 0.0, 1.0)
        best = int(np.argmax(scores))
        top2 = np.sort(scores)[-2:]
        margin = float(top2[1] - top2[0]) if len(scores) > 1 else float(scores[best])
        low = bool(ambiguous or scores[best] < self.min_score or margin < self.min_margin)
        return Match(best, "nearest", float(scores[best]), margin, low)
