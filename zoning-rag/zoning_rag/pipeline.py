"""End-to-end RAG pipeline: retrieve -> (abstain gate) -> generate -> ground/validate."""
from __future__ import annotations

import logging
import time

from .config import Settings, settings as default_settings
from .grounding import SYSTEM_PROMPT, build_prompt, extractive_answer, ground_answer
from .ingest import read_span
from .llm import LLMRouter, LLMUnavailable
from .retriever import HybridRetriever

log = logging.getLogger(__name__)

DISCLAIMER = "General information from the NYC Zoning Resolution, not legal advice."


class ZoningRAG:
    def __init__(self, cfg: Settings | None = None, llm: LLMRouter | None = None):
        self.cfg = cfg or default_settings
        self.retriever = HybridRetriever(self.cfg)
        self.n_chunks = self.retriever.index()
        self.llm = llm if llm is not None else LLMRouter(self.cfg)

    # ---- public API -------------------------------------------------------
    def search(self, query: str, k: int | None = None) -> dict:
        r = self.retriever.retrieve(query, k)
        return {"query": query, "confidence": round(r.confidence, 3), "signals": r.signals,
                "hits": [h.to_dict() for h in r.hits]}

    def read_source(self, source: str, start_line: int, end_line: int) -> str:
        return read_span(self.cfg.corpus_dir, source, start_line, end_line)

    def ask(self, question: str, k: int | None = None, provider: str | None = None) -> dict:
        t0 = time.perf_counter()
        r = self.retriever.retrieve(question, k)
        retrieval = {"confidence": round(r.confidence, 3), "signals": r.signals,
                     "hits": [{"span": h.chunk.span, "score": round(h.score, 5)} for h in r.hits]}

        # Gate 1: abstain before generation if retrieval is weak
        if r.confidence < self.cfg.min_retrieval_confidence:
            return self._finish(t0, question, {
                "answer": "", "abstained": True, "provider": None,
                "reason": f"low retrieval confidence ({r.confidence:.2f} < {self.cfg.min_retrieval_confidence})",
                "citations": [], "sentences": [], "dropped_sentences": [], "grounded_ratio": 0.0,
            }, retrieval)

        sources = [h.chunk for h in r.hits]
        raw, used = self._generate(question, sources, provider)

        # Gate 2: validate the generated answer against its cited sources
        g = ground_answer(raw, sources, self.cfg.min_sentence_support, self.cfg.min_grounded_ratio)
        out = {**g.to_dict(), "provider": used, "raw_answer": raw}
        return self._finish(t0, question, out, retrieval)

    # ---- internals --------------------------------------------------------
    def _generate(self, question, sources, provider):
        if provider != "extractive" and self.llm.available:
            try:
                return self.llm.complete(SYSTEM_PROMPT, build_prompt(question, sources))
            except LLMUnavailable as e:
                log.warning("LLM failed, falling back to extractive: %s", e)
        return extractive_answer(question, sources, idf=self.retriever.idf()), "extractive"

    def _finish(self, t0, question, out, retrieval):
        out.update(question=question, retrieval=retrieval, disclaimer=DISCLAIMER,
                   latency_ms=int((time.perf_counter() - t0) * 1000))
        if out["abstained"]:
            out["answer"] = ("I can't answer that reliably from the indexed Zoning Resolution text. "
                             "Try rephrasing, or consult the official text or NYC Department of City Planning.")
        return out
