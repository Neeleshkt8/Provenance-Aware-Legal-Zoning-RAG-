"""Load corpus files and chunk them while keeping exact file/line provenance."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

SUPPORTED = {".txt", ".md"}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str  # path relative to corpus dir
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    section: str  # nearest heading above the chunk
    text: str
    lines: tuple[tuple[int, str], ...]  # (line_no, text) for line-level citation

    @property
    def span(self) -> str:
        return f"{self.source}:L{self.start_line}-L{self.end_line}"

    def to_meta(self) -> dict:
        d = asdict(self)
        d.pop("lines")
        d.pop("text")
        return d


def _paragraphs(lines: list[str]):
    """Yield (heading, [(line_no, text), ...]) for blank-line separated paragraphs."""
    heading, buf = "", []
    for i, raw in enumerate(lines, start=1):
        line = raw.rstrip()
        if line.startswith("#"):
            if buf:
                yield heading, buf
                buf = []
            # heading lines become section context (skip the demo-corpus disclaimer line)
            if "demo corpus" not in line.lower():
                heading = line.lstrip("# ").strip()
            continue
        if not line.strip():
            if buf:
                yield heading, buf
                buf = []
            continue
        buf.append((i, line.strip()))
    if buf:
        yield heading, buf


def chunk_file(path: Path, corpus_dir: Path, max_chars: int = 700) -> list[Chunk]:
    rel = path.relative_to(corpus_dir).as_posix()
    lines = path.read_text(encoding="utf-8").splitlines()
    chunks: list[Chunk] = []
    cur: list[tuple[int, str]] = []
    cur_heading = ""

    def flush():
        nonlocal cur
        if not cur:
            return
        text = " ".join(t for _, t in cur)
        cid = hashlib.sha1(f"{rel}:{cur[0][0]}:{cur[-1][0]}:{text}".encode()).hexdigest()[:16]
        chunks.append(Chunk(cid, rel, cur[0][0], cur[-1][0], cur_heading, text, tuple(cur)))
        cur = []

    for heading, para in _paragraphs(lines):
        para_len = sum(len(t) for _, t in para)
        cur_len = sum(len(t) for _, t in cur)
        # merge small adjacent paragraphs; split on heading change or size limit
        if cur and (heading != cur_heading or cur_len + para_len > max_chars):
            flush()
        cur_heading = heading
        cur.extend(para)
    flush()
    return chunks


def load_corpus(corpus_dir: Path, max_chars: int = 700) -> list[Chunk]:
    corpus_dir = Path(corpus_dir)
    files = sorted(p for p in corpus_dir.rglob("*") if p.suffix.lower() in SUPPORTED)
    if not files:
        raise FileNotFoundError(f"No .txt/.md files found in {corpus_dir}")
    out: list[Chunk] = []
    for f in files:
        out.extend(chunk_file(f, corpus_dir, max_chars))
    return out


def read_span(corpus_dir: Path, source: str, start: int, end: int) -> str:
    """Return the exact lines of a source file (used to verify/display citations)."""
    path = (Path(corpus_dir) / source).resolve()
    if Path(corpus_dir).resolve() not in path.parents:
        raise ValueError("source path escapes corpus directory")
    lines = path.read_text(encoding="utf-8").splitlines()
    start, end = max(1, start), min(len(lines), end)
    return "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
