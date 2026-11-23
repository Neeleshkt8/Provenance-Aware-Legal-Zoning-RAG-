"""SQLite data layer: query log, citations, and user feedback."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    mode          TEXT    NOT NULL,            -- 'rag' | 'agent'
    question      TEXT    NOT NULL,
    answer        TEXT,
    abstained     INTEGER NOT NULL,
    reason        TEXT,
    provider      TEXT,
    confidence    REAL,
    grounded_ratio REAL,
    latency_ms    INTEGER,
    payload       TEXT                          -- full JSON response
);
CREATE TABLE IF NOT EXISTS citations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id   INTEGER NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    source     TEXT    NOT NULL,
    start_line INTEGER NOT NULL,
    end_line   INTEGER NOT NULL,
    quote      TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id   INTEGER NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    created_at TEXT    NOT NULL DEFAULT (datetime('now')),
    rating     INTEGER NOT NULL CHECK (rating IN (-1, 1)),
    comment    TEXT
);
CREATE INDEX IF NOT EXISTS idx_citations_query ON citations(query_id);
"""


class Database:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)

    @contextmanager
    def tx(self):
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def log_query(self, mode: str, question: str, result: dict) -> int:
        with self.tx() as c:
            cur = c.execute(
                "INSERT INTO queries (mode, question, answer, abstained, reason, provider, confidence,"
                " grounded_ratio, latency_ms, payload) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (mode, question, result.get("answer"), int(bool(result.get("abstained"))), result.get("reason"),
                 result.get("provider"), result.get("retrieval", {}).get("confidence"),
                 result.get("grounded_ratio"), result.get("latency_ms"), json.dumps(result)),
            )
            qid = cur.lastrowid
            c.executemany(
                "INSERT INTO citations (query_id, source, start_line, end_line, quote) VALUES (?,?,?,?,?)",
                [(qid, x["source"], x["start_line"], x["end_line"], x.get("quote")) for x in result.get("citations", [])],
            )
        return qid

    def add_feedback(self, query_id: int, rating: int, comment: str | None = None) -> int:
        with self.tx() as c:
            if not c.execute("SELECT 1 FROM queries WHERE id=?", (query_id,)).fetchone():
                raise KeyError(query_id)
            return c.execute("INSERT INTO feedback (query_id, rating, comment) VALUES (?,?,?)",
                             (query_id, rating, comment)).lastrowid

    def history(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, created_at, mode, question, answer, abstained, reason, provider, confidence,"
            " grounded_ratio, latency_ms FROM queries ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get(self, query_id: int) -> dict | None:
        r = self._conn.execute("SELECT payload FROM queries WHERE id=?", (query_id,)).fetchone()
        return json.loads(r["payload"]) if r else None

    def stats(self) -> dict:
        r = self._conn.execute(
            "SELECT COUNT(*) n, AVG(abstained) abstain_rate, AVG(latency_ms) avg_latency_ms,"
            " AVG(grounded_ratio) avg_grounded FROM queries"
        ).fetchone()
        fb = self._conn.execute("SELECT AVG(rating) FROM feedback").fetchone()[0]
        return {**dict(r), "avg_feedback": fb}
