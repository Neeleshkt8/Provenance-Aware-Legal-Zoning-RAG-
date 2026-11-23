"""Tokenisation and sentence helpers shared by retrieval and grounding."""
from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an and are as at be been by can could do does for from has have how i if in into is it its
    may might must of on or so such than that the their them then there these this those to under
    up was were what when where which who whom why will with would you your about any also each
    other only same some not no nor own very just should s t""".split()
)

# Keep zoning-style tokens intact: "r6a", "c1-2", "10.0", "72-21"
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*")


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    toks = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    out: list[str] = []
    for t in toks:
        t = stem(t)
        out.append(t)
        # also index sub-parts so "c1-2" matches "c1" and "m1-2/r6" matches "r6"
        if any(c in t for c in ".-/") and not re.fullmatch(r"[\d.]+", t):
            out.extend(p for p in re.split(r"[.\-/]", t) if p and p not in STOPWORDS)
    return out


def stem(t: str) -> str:
    """Very light plural stemming: 'rezonings'->'rezoning', 'districts'->'district'."""
    if len(t) > 4 and t.isalpha():
        if t.endswith("ies"):
            return t[:-3] + "y"
        if t.endswith("s") and not t.endswith(("ss", "us", "is")):
            return t[:-1]
    return t


def content_tokens(text: str) -> set[str]:
    return set(tokenize(text, drop_stopwords=True))


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[(\"'])")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    # don't split inside numbers like "10.0" (regex requires whitespace after punctuation)
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]
