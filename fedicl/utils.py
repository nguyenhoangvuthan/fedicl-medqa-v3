"""Small shared helpers: seeding, deterministic per-item RNG, atomic file IO, logging."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import random
import re
import shutil
import stat
import string
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

LOG = logging.getLogger("fedicl")


def setup_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python/numpy/torch. deterministic=True also forces deterministic CUDA kernels
    (atomic-add backward ops such as index_put otherwise differ run to run). Call it before the
    first CUDA op: CUBLAS_WORKSPACE_CONFIG is read when the cuBLAS handle is created."""
    random.seed(seed)
    np.random.seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def stable_int(*keys: Any) -> int:
    """Hash arbitrary keys into a stable 64-bit int (Python's hash() is salted per process)."""
    h = hashlib.sha256("\x1f".join(str(k) for k in keys).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def rng_for(seed: int, *keys: Any) -> np.random.Generator:
    """Deterministic RNG for one item, e.g. rng_for(seed, query_id, "alt")."""
    return np.random.default_rng([seed, stable_int(*keys)])


# ---------------------------------------------------------------- text / hashing
_PUNCT = string.punctuation + "“”‘’«»…–—"


def normalize_question(text: str) -> str:
    text = re.sub(r"\s+", " ", text.lower()).strip()
    return text.strip(_PUNCT + " ")


def q_hash(question: str) -> str:
    return hashlib.sha1(normalize_question(question).encode("utf-8")).hexdigest()


def sha1_file(path: str | Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


# ---------------------------------------------------------------- atomic IO
def _retry(fn, *args, attempts: int = 6):
    """Windows: antivirus / search indexer can hold a just-written file for a moment."""
    for i in range(attempts):
        try:
            return fn(*args)
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.25 * (i + 1))


def _atomic_write(path: str | Path, data: str | bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    # newline="": never translate "\n" -> "\r\n" on Windows (it would also alter newlines
    # inside quoted CSV fields, i.e. change question text vs. a Linux run).
    kw = {} if isinstance(data, bytes) else {"encoding": "utf-8", "newline": ""}
    with open(tmp, "wb" if isinstance(data, bytes) else "w", **kw) as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    _retry(os.replace, tmp, path)


def rmtree(path: str | Path) -> None:
    """shutil.rmtree that survives read-only files and short-lived locks (Windows)."""
    def _onerror(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)

    if Path(path).exists():
        kw = {"onexc": _onerror} if sys.version_info >= (3, 12) else {"onerror": _onerror}
        _retry(lambda: shutil.rmtree(path, **kw))


def write_text(path: str | Path, text: str) -> None:
    _atomic_write(path, text)


def write_json(path: str | Path, obj: Any) -> None:
    _atomic_write(path, json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default))


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    buf = io.StringIO()
    for r in rows:
        buf.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")
    _atomic_write(path, buf.getvalue())


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv_df(path: str | Path, df) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    _atomic_write(path, buf.getvalue())


def save_numpy(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, arr)
    _retry(os.replace, tmp, path)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(o, (DictConfig, ListConfig)):
            return OmegaConf.to_container(o, resolve=True)
    except ImportError:
        pass
    raise TypeError(f"not JSON serializable: {type(o)}")
