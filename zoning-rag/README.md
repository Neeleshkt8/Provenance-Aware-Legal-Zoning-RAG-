# Provenance-Aware Legal Zoning RAG (NYC)

A RAG system for questions about the NYC Zoning Resolution. **Every answer sentence is traced to an exact `file:line` span, and the system abstains instead of guessing** when retrieval is weak or the generated text isn't supported by its sources.

```
question
   │
   ▼
Hybrid retrieval ── BM25 (lexical) ─┐
   (retriever.py)    Dense (ChromaDB)┴─► Reciprocal Rank Fusion ─► top-k chunks + confidence
   │
   ├─ Gate 1: confidence < threshold ──► ABSTAIN (no LLM call)
   ▼
Generation (llm.py) ── Groq → Gemini → offline extractive (ordered failover)
   │                   prompt forces [S#] citation on every sentence
   ▼
Grounding validator (grounding.py)
   • each sentence must cite a real retrieved source
   • lexical support vs cited chunk ≥ MIN_SENTENCE_SUPPORT, else sentence dropped
   • citation narrowed to the exact 1–2 source lines
   • Gate 2: grounded ratio < MIN_GROUNDED_RATIO ──► ABSTAIN
   ▼
FastAPI (api.py) ── SQLite log of queries, citations, feedback (db.py)
```

The agent layer (`agent.py`) is a LangChain tool-calling loop (Groq/Gemini chat models with `bind_tools`) over three tools: `search_zoning`, `read_source`, `grounded_answer`. The agent's final answer **must** come from `grounded_answer`; free-form agent text is never returned unvalidated.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # add GROQ_API_KEY and/or GEMINI_API_KEY (optional)
make test                     # unit + API tests
make eval-offline             # eval harness, no API keys needed
make serve                    # http://localhost:8000/docs
python -m zoning_rag.cli "What is the max commercial FAR for an overlay in an R7 district?"
```

With no keys, the system runs fully offline: an extractive generator produces `[S#]`-cited answers that go through the same grounding validator.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/ask` | Grounded RAG answer `{question, top_k?, provider?}` |
| POST | `/agent` | Agentic tool-calling answer with a step trace |
| GET | `/search?q=` | Raw hybrid retrieval with BM25/dense ranks |
| GET | `/source?file=&start=&end=` | Read exact cited lines (path-traversal safe) |
| GET | `/history`, `/history/{id}` | Logged queries from SQLite |
| POST | `/feedback` | Thumbs up or down on a logged query |
| GET | `/stats` | Abstain rate, latency, grounded ratio, feedback |
| POST | `/reindex` | Re-chunk and re-embed the corpus |

Example `/ask` response (trimmed):
```json
{
  "answer": "Under ZR 72-21, the BSA must make five findings before granting a variance. [05_administration.txt:L5]",
  "abstained": false,
  "grounded_ratio": 1.0,
  "citations": [{"span": "05_administration.txt:L5", "quote": "Under ZR 72-21, the BSA must make five findings ..."}],
  "dropped_sentences": [],
  "retrieval": {"confidence": 0.736, "signals": {"coverage": 0.853, "dense_top": 0.435, "agreement": true}},
  "provider": "groq"
}
```

## Evaluation harness (`eval/run_eval.py`)

| Tier | Metrics |
|---|---|
| Deterministic | retrieval hit@k, MRR, abstention accuracy (false-abstain and false-answer counts), **citation validity** (the quoted text must match the cited file lines exactly), citation precision vs gold sources |
| Heuristic | expected-keyword recall, **number faithfulness** (every number in the answer appears in a cited quote), grounded-sentence ratio, length bounds |
| LLM-judge | faithfulness and relevance (1–5), scored by Groq/Gemini with JSON output |

`--gate` exits non-zero if thresholds fail, so you can use it as a CI check. Current offline result on the 20-question set (15 answerable, 5 out-of-domain): hit@k 1.0, MRR 0.92, abstention accuracy 1.0, citation validity 1.0, keyword recall 1.0, number faithfulness 1.0.

## Design notes

- **Retrieval confidence** = 0.6 × IDF-weighted query-term coverage + 0.4 × top dense cosine (+ a small bonus when BM25 and dense agree on the top hit). IDF weighting means a rare unknown term ("Chicago", "Tokyo") pulls confidence down, while a common one ("zoning") doesn't prop it up.
- **Tokeniser** keeps zoning identifiers intact (`r6a`, `c1-2`, `72-21`, `10.0`) and also indexes their parts.
- **Graceful degradation:** ChromaDB → in-memory numpy store; sentence-transformers → hashing n-gram embedder; LangChain → rule-based planner; Groq → Gemini → extractive. Every fallback is logged.

## Corpus

`data/corpus/` holds a **small, paraphrased demo corpus** of NYC Zoning Resolution concepts. It is *not* the official text and may be out of date (City of Yes, 2024, changed many rules). For real use, replace it with text exported from the official Zoning Resolution at zr.planning.nyc.gov and run `/reindex`; provenance spans will then point to the official text. Answers are general information, not legal advice.

## Layout
```
zoning_rag/  config · text · ingest · bm25 · embeddings · vectorstore · retriever
             llm · grounding · pipeline · agent · db · api · cli
eval/        dataset.jsonl · run_eval.py
tests/       test_core.py · test_api.py
data/corpus/ demo Zoning Resolution text
```
