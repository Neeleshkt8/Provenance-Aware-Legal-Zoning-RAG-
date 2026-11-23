"""Hybrid retrieval: BM25 (lexical) + dense (ChromaDB) fused with Reciprocal Rank Fusion,
plus a calibrated retrieval-confidence score used for abstention."""
from __future__ import annotations

from dataclasses import dataclass

from .bm25 import BM25
from .config import Settings
from .embeddings import get_embedder
from .ingest import Chunk, load_corpus
from .text import content_tokens
from .vectorstore import get_store


@dataclass
class Hit:
    chunk: Chunk
    score: float  # fused RRF score
    bm25: float
    dense: float
    bm25_rank: int | None
    dense_rank: int | None

    def to_dict(self) -> dict:
        return {
            **self.chunk.to_meta(),
            "span": self.chunk.span,
            "text": self.chunk.text,
            "score": round(self.score, 5),
            "bm25": round(self.bm25, 4),
            "dense": round(self.dense, 4),
            "bm25_rank": self.bm25_rank,
            "dense_rank": self.dense_rank,
        }


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit]
    confidence: float
    signals: dict


class HybridRetriever:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.embedder = get_embedder(cfg.embedder, cfg.st_model)
        self.store = get_store(cfg.chroma_dir, cfg.collection)
        self.chunks: list[Chunk] = []
        self.by_id: dict[str, Chunk] = {}
        self.bm25: BM25 | None = None

    def index(self) -> int:
        self.chunks = load_corpus(self.cfg.corpus_dir, self.cfg.chunk_max_chars)
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.bm25 = BM25([f"{c.section} {c.text}" for c in self.chunks])
        self.store.reset()
        self.store.add(self.chunks, self.embedder.embed([c.text for c in self.chunks]))
        return len(self.chunks)

    def retrieve(self, query: str, k: int | None = None) -> RetrievalResult:
        if self.bm25 is None:
            self.index()
        k = k or self.cfg.top_k
        pool = min(len(self.chunks), max(k * 4, 20))

        bm = self.bm25.scores(query)
        bm_order = sorted(range(len(bm)), key=lambda i: -bm[i])[:pool]
        bm_rank = {self.chunks[i].chunk_id: r for r, i in enumerate(bm_order) if bm[i] > 0}
        bm_score = {self.chunks[i].chunk_id: bm[i] for i in range(len(bm))}

        qv = self.embedder.embed([query])[0]
        dense = self.store.query(qv, pool)
        dn_rank = {cid: r for r, (cid, _) in enumerate(dense)}
        dn_score = dict(dense)

        fused: dict[str, float] = {}
        for cid, r in bm_rank.items():
            fused[cid] = fused.get(cid, 0) + self.cfg.bm25_weight / (self.cfg.rrf_k + r + 1)
        for cid, r in dn_rank.items():
            fused[cid] = fused.get(cid, 0) + self.cfg.dense_weight / (self.cfg.rrf_k + r + 1)

        top = sorted(fused, key=lambda c: -fused[c])[:k]
        hits = [
            Hit(self.by_id[c], fused[c], bm_score.get(c, 0.0), dn_score.get(c, 0.0), bm_rank.get(c), dn_rank.get(c))
            for c in top
        ]
        conf, signals = self._confidence(query, hits)
        return RetrievalResult(query, hits, conf, signals)

    def idf(self) -> dict[str, float]:
        """Term weights; unseen terms get the max weight (rare = informative, e.g. 'Chicago')."""
        if self.bm25 is None:
            self.index()
        return self.bm25.idf

    def weighted_coverage(self, q: set[str], doc: set[str]) -> float:
        idf = self.idf()
        mx = max(idf.values())
        w = {t: idf.get(t, mx) for t in q}
        total = sum(w.values())
        return sum(v for t, v in w.items() if t in doc) / total if total else 0.0

    def _confidence(self, query: str, hits: list[Hit]) -> tuple[float, dict]:
        """Blend of (a) IDF-weighted query-term coverage by the best chunk and (b) top dense
        similarity. Coverage catches out-of-domain questions; dense sim catches paraphrases."""
        q = content_tokens(query)
        if not hits or not q:
            return 0.0, {"coverage": 0.0, "dense_top": 0.0, "agreement": False}
        coverage = max(self.weighted_coverage(q, content_tokens(f"{h.chunk.section} {h.chunk.text}"))
                       for h in hits)
        dense_top = max(h.dense for h in hits)
        agreement = hits[0].bm25_rank is not None and hits[0].dense_rank is not None
        conf = 0.6 * coverage + 0.4 * max(0.0, dense_top) + (0.05 if agreement else 0.0)
        return min(1.0, conf), {
            "coverage": round(coverage, 3),
            "dense_top": round(dense_top, 3),
            "agreement": agreement,
        }
