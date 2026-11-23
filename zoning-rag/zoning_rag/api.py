"""FastAPI service.  Run:  uvicorn zoning_rag.api:app --reload"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .agent import ZoningAgent
from .config import settings
from .db import Database
from .pipeline import ZoningRAG

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["rag"] = ZoningRAG(settings)
    state["db"] = Database(settings.sqlite_path)
    yield
    state.clear()


app = FastAPI(title="Provenance-Aware NYC Zoning RAG", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int | None = Field(None, ge=1, le=20)
    provider: str | None = Field(None, description="force 'extractive' to skip LLM calls")


class FeedbackRequest(BaseModel):
    query_id: int
    rating: int = Field(..., description="1 = helpful, -1 = not helpful")
    comment: str | None = None


@app.get("/health")
def health():
    rag: ZoningRAG = state["rag"]
    return {"status": "ok", "chunks": rag.n_chunks, "vector_store": rag.retriever.store.backend,
            "embedder": rag.retriever.embedder.name, "llm_providers": rag.llm.available or ["extractive"]}


@app.post("/ask")
def ask(req: AskRequest):
    res = state["rag"].ask(req.question, req.top_k, req.provider)
    res["query_id"] = state["db"].log_query("rag", req.question, res)
    return res


@app.post("/agent")
def agent(req: AskRequest):
    res = ZoningAgent(state["rag"]).run(req.question)
    res["query_id"] = state["db"].log_query("agent", req.question, res)
    return res


@app.get("/search")
def search(q: str = Query(..., min_length=2), k: int = Query(5, ge=1, le=20)):
    return state["rag"].search(q, k)


@app.get("/source")
def source(file: str, start: int = Query(..., ge=1), end: int = Query(..., ge=1)):
    if end < start or end - start > 200:
        raise HTTPException(400, "invalid line range")
    try:
        return {"file": file, "start": start, "end": end, "text": state["rag"].read_source(file, start, end)}
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404, str(e))


@app.get("/history")
def history(limit: int = Query(20, ge=1, le=200)):
    return state["db"].history(limit)


@app.get("/history/{query_id}")
def history_item(query_id: int):
    r = state["db"].get(query_id)
    if r is None:
        raise HTTPException(404, "not found")
    return r


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if req.rating not in (-1, 1):
        raise HTTPException(422, "rating must be 1 or -1")
    try:
        return {"feedback_id": state["db"].add_feedback(req.query_id, req.rating, req.comment)}
    except KeyError:
        raise HTTPException(404, "query not found")


@app.get("/stats")
def stats():
    return state["db"].stats()


@app.post("/reindex")
def reindex():
    rag: ZoningRAG = state["rag"]
    rag.n_chunks = rag.retriever.index()
    return {"chunks": rag.n_chunks}
