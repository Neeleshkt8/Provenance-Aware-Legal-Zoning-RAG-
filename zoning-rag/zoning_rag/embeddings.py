"""Dense embedders.

Production: sentence-transformers (all-MiniLM-L6-v2).
Fallback:   a deterministic hashing embedder (word + char n-grams, sklearn) so the
            system and its tests run fully offline with no model download.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class HashingEmbedder:
    name = "hashing-ngrams-2048"

    def __init__(self, dim: int = 2048):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.word = HashingVectorizer(
            n_features=dim, alternate_sign=False, norm=None, ngram_range=(1, 2), stop_words="english"
        )
        self.char = HashingVectorizer(
            n_features=dim, alternate_sign=False, norm=None, analyzer="char_wb", ngram_range=(3, 5)
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        w = self.word.transform(texts).toarray()
        c = self.char.transform(texts).toarray()
        v = np.hstack([np.log1p(w) * 2.0, np.log1p(c)])
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.where(n == 0, 1, n)


class SentenceTransformerEmbedder:
    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)
        self.name = model

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True))


def get_embedder(kind: str = "auto", st_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    if kind in ("auto", "sentence-transformers"):
        try:
            return SentenceTransformerEmbedder(st_model)
        except Exception as e:  # not installed or no network to fetch weights
            if kind == "sentence-transformers":
                raise
            log.warning("sentence-transformers unavailable (%s); using hashing embedder", e)
    return HashingEmbedder()
