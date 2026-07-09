"""Sentence-transformer embeddings, L2-normalized."""
from __future__ import annotations
import numpy as np

_MODEL_CACHE: dict[str, object] = {}


def _get_model(name: str):
    if name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        _MODEL_CACHE[name] = SentenceTransformer(name)
    return _MODEL_CACHE[name]


def embed_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    model = _get_model(model_name)
    vecs = np.asarray(model.encode(texts, show_progress_bar=False), dtype=np.float64)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms
