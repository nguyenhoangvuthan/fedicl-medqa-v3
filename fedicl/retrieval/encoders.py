"""Frozen text encoders shared by retrieval (§3.6) and answer matching (§5.1)."""
from __future__ import annotations

import numpy as np
import torch


class TextEncoder:
    """HF encoder + pooling ('mean' | 'cls'), L2-normalized float32 output.

    Works for sentence encoders (pubmedbert-base-embeddings, mean pooling) and entity encoders
    (SapBERT, CLS pooling) alike, and for raw MLMs such as BioBERT/ClinicalBERT (ablation).
    """

    def __init__(self, name: str, pooling: str = "mean", max_length: int = 512,
                 device: str | None = None):
        from transformers import AutoModel, AutoTokenizer

        if pooling not in ("mean", "cls"):
            raise ValueError(f"pooling must be 'mean' or 'cls', got {pooling!r}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        self.pooling, self.max_length = pooling, max_length
        self._cache: dict[str, np.ndarray] = {}

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = self.tok(texts[i:i + batch_size], padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.autocast(self.device if self.device != "cpu" else "cpu",
                                dtype=torch.bfloat16, enabled=self.device == "cuda"):
                h = self.model(**batch).last_hidden_state.float()
            if self.pooling == "cls":
                v = h[:, 0]
            else:
                m = batch["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
        dim = self.model.config.hidden_size
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, dim), np.float32)

    def encode_cached(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if missing:
            for t, v in zip(missing, self.encode(missing, batch_size)):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts]) if texts else np.zeros((0, 1), np.float32)

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
