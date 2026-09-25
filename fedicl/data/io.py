"""CSV schema of spec §3.1 and helpers to read it. Every CSV (centralized/*, client_*) shares it."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..utils import write_csv_df

LETTERS = ("A", "B", "C", "D")
OPTION_COLS = tuple(f"option_{l}" for l in LETTERS)
RAW_COLUMNS = ["id", "question", *OPTION_COLS, "answer", "meta"]
COLUMNS = [*RAW_COLUMNS, "q_hash"]
SPLITS = ("train", "validation", "test")


def load_split(path: str | Path) -> pd.DataFrame:
    # keep_default_na=False: an option literally named "None"/"NA" must stay a string, not NaN.
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def save_split(df: pd.DataFrame, path: str | Path, columns: list[str] = COLUMNS) -> None:
    write_csv_df(path, df[columns])


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    options: tuple[str, str, str, str]  # original order A..D
    answer: str                          # original letter
    meta: str
    q_hash: str

    @property
    def gold_index(self) -> int:
        return LETTERS.index(self.answer)

    @property
    def gold_text(self) -> str:
        return self.options[self.gold_index]


def to_examples(df: pd.DataFrame) -> dict[str, Example]:
    out: dict[str, Example] = {}
    for r in df.itertuples(index=False):
        out[r.id] = Example(
            id=r.id, question=r.question,
            options=(r.option_A, r.option_B, r.option_C, r.option_D),
            answer=r.answer, meta=r.meta, q_hash=r.q_hash,
        )
    return out
