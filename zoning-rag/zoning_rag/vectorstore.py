"""Vector store: ChromaDB (persistent, cosine HNSW) with an in-memory numpy fallback."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .ingest import Chunk

log = logging.getLogger(__name__)


class NumpyStore:
    backend = "numpy"

    def __init__(self):
        self.ids: list[str] = []
        self.mat: np.ndarray | None = None

    def reset(self):
        self.ids, self.mat = [], None

    def add(self, chunks: list[Chunk], embeddings: np.ndarray):
        self.ids.extend(c.chunk_id for c in chunks)
        self.mat = embeddings if self.mat is None else np.vstack([self.mat, embeddings])

    def query(self, q: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self.mat is None:
            return []
        sims = self.mat @ q
        idx = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in idx]


class ChromaStore:
    backend = "chromadb"

    def __init__(self, path: Path, collection: str):
        import chromadb

        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self.name = collection
        self.col = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def reset(self):
        self.client.delete_collection(self.name)
        self.col = self.client.get_or_create_collection(self.name, metadata={"hnsw:space": "cosine"})

    def add(self, chunks: list[Chunk], embeddings: np.ndarray):
        self.col.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[c.to_meta() for c in chunks],
        )

    def query(self, q: np.ndarray, k: int) -> list[tuple[str, float]]:
        r = self.col.query(query_embeddings=[q.tolist()], n_results=k)
        # cosine distance -> similarity
        return [(i, 1.0 - float(d)) for i, d in zip(r["ids"][0], r["distances"][0])]


def get_store(path: Path, collection: str):
    try:
        return ChromaStore(path, collection)
    except Exception as e:
        log.warning("ChromaDB unavailable (%s); using in-memory numpy store", e)
        return NumpyStore()
