"""Citation grounding: every answer sentence must cite a retrieved source and be
lexically supported by it; each citation is narrowed to exact file/line spans.
Unsupported sentences are dropped, and the whole answer abstains if too little survives."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .ingest import Chunk
from .text import content_tokens, split_sentences, tokenize

SYSTEM_PROMPT = """You are a careful assistant answering questions about the New York City Zoning Resolution.
Rules:
1. Use ONLY the numbered sources provided. Do not use outside knowledge.
2. End EVERY sentence with one or more citations in the form [S1] or [S1][S3].
3. Stay close to the wording of the sources; do not add numbers, dates or section references that are not in them.
4. If the sources do not contain the answer, reply exactly: INSUFFICIENT_CONTEXT
5. Be concise: at most 5 sentences. This is general information, not legal advice."""

_CITE_RE = re.compile(r"\[S(\d+)\]")
ABSTAIN_TOKEN = "INSUFFICIENT_CONTEXT"


def build_prompt(question: str, sources: list[Chunk]) -> str:
    blocks = [f"[S{i}] ({c.span}; {c.section})\n{c.text}" for i, c in enumerate(sources, 1)]
    return "SOURCES:\n\n" + "\n\n".join(blocks) + f"\n\nQUESTION: {question}\n\nANSWER:"


@dataclass
class Citation:
    source_id: str  # "S1"
    source: str
    start_line: int
    end_line: int
    quote: str

    @property
    def span(self) -> str:
        return f"{self.source}:L{self.start_line}" + (f"-L{self.end_line}" if self.end_line != self.start_line else "")


@dataclass
class GroundedSentence:
    text: str
    citations: list[Citation]
    support: float
    supported: bool


@dataclass
class GroundingResult:
    answer: str
    abstained: bool
    reason: str | None
    sentences: list[GroundedSentence] = field(default_factory=list)
    grounded_ratio: float = 0.0
    dropped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "abstained": self.abstained,
            "reason": self.reason,
            "grounded_ratio": round(self.grounded_ratio, 3),
            "citations": [
                {"id": c.source_id, "span": c.span, "source": c.source, "start_line": c.start_line,
                 "end_line": c.end_line, "quote": c.quote}
                for c in _unique_citations(self.sentences)
            ],
            "sentences": [
                {"text": s.text, "support": round(s.support, 3), "supported": s.supported,
                 "citations": [c.span for c in s.citations]}
                for s in self.sentences
            ],
            "dropped_sentences": self.dropped,
        }


def _unique_citations(sents: list[GroundedSentence]) -> list[Citation]:
    seen, out = set(), []
    for s in sents:
        for c in s.citations:
            if c.span not in seen:
                seen.add(c.span)
                out.append(c)
    return out


def _best_lines(sentence_toks: set[str], chunk: Chunk) -> tuple[int, int, str, float]:
    """Narrow a chunk citation to the contiguous line window (1-2 lines) best supporting the sentence."""
    best = (chunk.start_line, chunk.end_line, chunk.text, 0.0)
    lines = chunk.lines
    # single lines first, then 2-line windows; a wider window must be strictly better
    for width in (1, 2):
        for i in range(len(lines) - width + 1):
            j = i + width - 1
            text = " ".join(t for _, t in lines[i : j + 1])
            score = len(sentence_toks & content_tokens(text)) / max(1, len(sentence_toks))
            if score > best[3] + 1e-9:
                best = (lines[i][0], lines[j][0], text, score)
    return best


def ground_answer(raw: str, sources: list[Chunk], min_support: float = 0.5,
                  min_grounded_ratio: float = 0.6) -> GroundingResult:
    raw = (raw or "").strip()
    if not raw or ABSTAIN_TOKEN in raw:
        return GroundingResult("", True, "model reported insufficient context")

    # "text. [S1] Next" -> "text [S1]. Next" so citations stay with their sentence
    raw = re.sub(r"([.!?])((?:\s*\[S\d+\])+)", r"\2\1", raw)
    raw = re.sub(r"^\s*[-*•]\s*", "", raw, flags=re.M)  # strip bullet markers

    sentences: list[GroundedSentence] = []
    dropped: list[str] = []
    for sent in split_sentences(raw):
        ids = [int(m) for m in _CITE_RE.findall(sent)]
        clean = _CITE_RE.sub("", sent).strip()
        clean = re.sub(r"\s+([.,;:])", r"\1", clean)
        if not clean:
            continue
        valid = [i for i in ids if 1 <= i <= len(sources)]
        toks = content_tokens(clean)
        cites, best_support = [], 0.0
        for i in dict.fromkeys(valid):  # dedupe, keep order
            s, e, quote, sup = _best_lines(toks, sources[i - 1])
            # support is measured against the whole cited chunk, span is narrowed for display
            chunk_sup = len(toks & content_tokens(sources[i - 1].text)) / max(1, len(toks))
            best_support = max(best_support, chunk_sup)
            cites.append(Citation(f"S{i}", sources[i - 1].source, s, e, quote))
        supported = bool(cites) and best_support >= min_support
        gs = GroundedSentence(clean, cites, best_support, supported)
        if supported:
            sentences.append(gs)
        else:
            dropped.append(clean)

    total = len(sentences) + len(dropped)
    ratio = len(sentences) / total if total else 0.0
    if not sentences:
        return GroundingResult("", True, "no answer sentence was supported by a cited source",
                               dropped=dropped, grounded_ratio=ratio)
    if ratio < min_grounded_ratio:
        return GroundingResult("", True, f"only {ratio:.0%} of sentences were grounded",
                               sentences=sentences, dropped=dropped, grounded_ratio=ratio)
    answer = " ".join(f"{s.text} " + "".join(f"[{c.span}]" for c in s.citations) for s in sentences)
    return GroundingResult(answer.strip(), False, None, sentences, ratio, dropped)


def extractive_answer(question: str, sources: list[Chunk], max_sentences: int = 3,
                      idf: dict[str, float] | None = None) -> str:
    """Offline fallback 'generator': pick the source sentences that best match the question
    (IDF-weighted), emitted in the same [S#]-cited format an LLM would produce."""
    q = content_tokens(question)
    idf = idf or {}
    mx = max(idf.values(), default=1.0)
    w = {t: idf.get(t, mx) for t in q}
    total = sum(w.values()) or 1.0
    scored = []
    for si, c in enumerate(sources, 1):
        for pos, (_, line) in enumerate(c.lines):
            for sent in split_sentences(line):
                st = content_tokens(sent)
                if not st:
                    continue
                overlap = sum(w[t] for t in q & st) / total
                if overlap:
                    # definitional boost: query terms appearing at the start of the sentence
                    # ("A zoning lot is ...") usually mark the sentence that answers the question
                    head = set(tokenize(sent)[:5])
                    overlap += 0.3 * sum(w[t] for t in q & head) / total
                    # rank by weighted overlap, then source rank, then position
                    scored.append((-overlap, si, pos, sent))
    if not scored:
        return ABSTAIN_TOKEN
    scored.sort()
    best = -scored[0][0]
    picked = [x for x in scored[:max_sentences] if -x[0] >= 0.6 * best]
    picked.sort(key=lambda x: (x[1], x[2]))  # read in document order
    return " ".join(f"{s.rstrip()} [S{si}]" if not s.rstrip().endswith(".") else f"{s.rstrip()[:-1]} [S{si}]."
                    for _, si, _, s in picked)
