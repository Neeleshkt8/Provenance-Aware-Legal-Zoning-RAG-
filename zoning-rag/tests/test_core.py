import json
from pathlib import Path

import pytest

from zoning_rag.agent import ZoningAgent
from zoning_rag.bm25 import BM25
from zoning_rag.config import Settings
from zoning_rag.db import Database
from zoning_rag.grounding import ground_answer
from zoning_rag.ingest import load_corpus, read_span
from zoning_rag.llm import LLMRouter
from zoning_rag.pipeline import ZoningRAG

CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus"


class FakeRouter(LLMRouter):
    """Stands in for Groq/Gemini: returns a scripted answer."""

    def __init__(self, text):
        self.clients, self.text = [], text

    @property
    def available(self):
        return ["fake"]

    def complete(self, system, prompt, temperature=0.0):
        return self.text, "fake"


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    d = tmp_path_factory.mktemp("rag")
    return Settings(corpus_dir=CORPUS, chroma_dir=d / "chroma", sqlite_path=d / "t.db", llm_providers=["extractive"])


@pytest.fixture(scope="module")
def rag(cfg):
    return ZoningRAG(cfg)


# ---------- ingestion / provenance ----------
def test_chunks_have_exact_line_provenance():
    chunks = load_corpus(CORPUS)
    assert chunks
    for c in chunks:
        lines = (CORPUS / c.source).read_text().splitlines()
        for ln, txt in c.lines:
            assert lines[ln - 1].strip() == txt
        assert c.start_line == c.lines[0][0] and c.end_line == c.lines[-1][0]


def test_read_span_blocks_path_traversal():
    with pytest.raises(ValueError):
        read_span(CORPUS, "../../etc/passwd", 1, 2)


def test_bm25_ranks_exact_term_first():
    b = BM25(["the cat sat", "MX districts pair M1 with residence", "zoning lot definition"])
    s = b.scores("MX district")
    assert max(range(3), key=lambda i: s[i]) == 1


# ---------- retrieval + abstention ----------
def test_hybrid_retrieval_finds_gold_source(rag):
    r = rag.retriever.retrieve("How many findings must the BSA make for a variance?")
    assert r.hits[0].chunk.source == "05_administration.txt"


@pytest.mark.parametrize("q", ["What is the capital of France?", "Zoning rules in Chicago for parking?"])
def test_abstains_out_of_domain(rag, q):
    r = rag.ask(q)
    assert r["abstained"] and r["citations"] == []


def test_answers_in_domain_with_valid_citations(rag):
    r = rag.ask("What is a non-complying building?")
    assert not r["abstained"]
    for c in r["citations"]:
        lines = (CORPUS / c["source"]).read_text().splitlines()
        assert " ".join(l.strip() for l in lines[c["start_line"] - 1 : c["end_line"]]) == c["quote"]


# ---------- grounding validator ----------
def test_grounding_drops_hallucinated_sentence(rag):
    srcs = [h.chunk for h in rag.retriever.retrieve("BSA variance findings").hits]
    raw = ("Under ZR 72-21, the BSA must make five findings before granting a variance [S1]. "
           "Variances in Queens are automatically approved within ten days by the mayor [S1].")
    g = ground_answer(raw, srcs, min_grounded_ratio=0.5)
    assert not g.abstained
    assert len(g.sentences) == 1 and len(g.dropped) == 1
    assert "five findings" in g.answer and "mayor" not in g.answer


def test_grounding_rejects_uncited_and_fake_ids(rag):
    srcs = [h.chunk for h in rag.retriever.retrieve("floor area ratio").hits]
    g = ground_answer("FAR is total floor area divided by lot area. Also see [S99].", srcs)
    assert g.abstained


def test_grounding_handles_citation_after_period(rag):
    srcs = [h.chunk for h in rag.retriever.retrieve("floor area ratio definition").hits]
    sid = next(i for i, c in enumerate(srcs, 1) if "divided by the lot area" in c.text)
    g = ground_answer(f"Floor area ratio is the total floor area on a zoning lot divided by the lot area. [S{sid}]", srcs)
    assert not g.abstained and g.sentences[0].citations


def test_model_abstain_token(rag):
    srcs = [h.chunk for h in rag.retriever.retrieve("floor area").hits]
    assert ground_answer("INSUFFICIENT_CONTEXT", srcs).abstained


def test_pipeline_validates_llm_output(cfg):
    hallucinating = FakeRouter("The maximum FAR everywhere in Manhattan is 50.0 by mayoral decree [S1].")
    r = ZoningRAG(cfg, llm=hallucinating).ask("What is floor area ratio?")
    assert r["provider"] == "fake" and r["abstained"]


def test_pipeline_accepts_grounded_llm_output(cfg):
    rag = ZoningRAG(cfg, llm=FakeRouter("x"))
    srcs = [h.chunk for h in rag.retriever.retrieve("What is floor area ratio?").hits]
    sid = next(i for i, c in enumerate(srcs, 1) if "divided by the lot area" in c.text)
    rag.llm.text = f"Floor area ratio is the total floor area on a zoning lot divided by its lot area [S{sid}]."
    r = rag.ask("What is floor area ratio?")
    assert not r["abstained"] and r["citations"][0]["source"] == "01_definitions.txt"


# ---------- agent ----------
def test_rule_based_agent_answers_and_traces(rag):
    r = ZoningAgent(rag).run("What is an MX district and what does M1 allow?")
    tools = [s["tool"] for s in r["agent"]["steps"]]
    assert "search_zoning" in tools and tools[-1] == "grounded_answer"
    assert not r["abstained"]


def test_agent_read_source_tool(rag):
    a = ZoningAgent(rag)
    a.run("show 05_administration.txt lines 4 to 5")
    assert a.trace[0]["tool"] == "read_source" and "72-21" in a.trace[0]["output_preview"]


# ---------- SQLite ----------
def test_db_roundtrip(rag):
    db = Database(":memory:")
    res = rag.ask("What is a zoning lot?")
    qid = db.log_query("rag", res["question"], res)
    assert db.get(qid)["question"] == "What is a zoning lot?"
    db.add_feedback(qid, 1, "good")
    assert db.stats()["n"] == 1 and db.stats()["avg_feedback"] == 1
    n_cites = db._conn.execute("SELECT COUNT(*) FROM citations WHERE query_id=?", (qid,)).fetchone()[0]
    assert n_cites == len(res["citations"]) > 0
    with pytest.raises(KeyError):
        db.add_feedback(9999, 1)


def test_eval_harness_runs(tmp_path):
    from eval.run_eval import check_gates, run

    r = run(Path(__file__).resolve().parent.parent / "eval" / "dataset.jsonl", "extractive", False, tmp_path)
    assert (tmp_path / "summary.json").exists()
    assert check_gates(r["summary"]) == []
    json.loads((tmp_path / "summary.json").read_text())
