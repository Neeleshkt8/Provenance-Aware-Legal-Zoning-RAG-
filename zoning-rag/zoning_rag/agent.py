"""Agentic tool-calling layer.

With LangChain + an API key, an LLM (Groq or Gemini via LangChain chat models) chooses
among three tools in a bounded loop. The agent's *final* answer must come from the
`grounded_answer` tool, so it is always citation-validated; if the agent never calls
it, we fall back to the grounded pipeline rather than trusting free-form agent text.

Without LangChain/keys, a deterministic rule-based planner drives the same tools.
"""
from __future__ import annotations

import json
import logging
import re

from .pipeline import ZoningRAG

log = logging.getLogger(__name__)

AGENT_SYSTEM = """You are a NYC zoning research agent. You have tools:
- search_zoning(query): hybrid search over the Zoning Resolution; returns chunks with file:line spans.
- read_source(source, start_line, end_line): read exact lines of a source file.
- grounded_answer(question): produce a citation-validated answer (abstains if unsupported).
Plan briefly, use search_zoning/read_source if the question is multi-part or needs disambiguation,
and ALWAYS finish by calling grounded_answer with a focused question. Never answer from memory."""

MAX_STEPS = 5


class ZoningAgent:
    def __init__(self, rag: ZoningRAG):
        self.rag = rag
        self.trace: list[dict] = []
        self.last_grounded: dict | None = None

    # ---- tools (plain functions, wrapped for LangChain below) -------------
    def search_zoning(self, query: str) -> str:
        r = self.rag.search(query, k=4)
        return json.dumps({"confidence": r["confidence"],
                           "hits": [{"span": h["span"], "section": h["section"], "text": h["text"][:500]}
                                    for h in r["hits"]]})

    def read_source(self, source: str, start_line: int, end_line: int) -> str:
        try:
            return self.rag.read_source(source, int(start_line), int(end_line))
        except Exception as e:
            return f"ERROR: {e}"

    def grounded_answer(self, question: str) -> str:
        r = self.rag.ask(question)
        self.last_grounded = r
        return json.dumps({k: r[k] for k in ("answer", "abstained", "reason", "citations")})

    def _call(self, name: str, args: dict) -> str:
        fn = {"search_zoning": self.search_zoning, "read_source": self.read_source,
              "grounded_answer": self.grounded_answer}.get(name)
        out = fn(**args) if fn else f"ERROR: unknown tool {name}"
        self.trace.append({"tool": name, "args": args, "output_preview": out[:300]})
        return out

    # ---- entry point ------------------------------------------------------
    def run(self, question: str) -> dict:
        self.trace, self.last_grounded = [], None
        mode = "rule-based"
        try:
            chat = self._langchain_model()
        except Exception as e:
            log.info("LangChain agent unavailable (%s); using rule-based planner", e)
            chat = None
        if chat is not None:
            try:
                self._run_langchain(chat, question)
                mode = "langchain"
            except Exception as e:
                log.warning("LangChain agent failed (%s); using rule-based planner", e)
                self.trace.append({"tool": "error", "args": {}, "output_preview": str(e)[:300]})
        if mode == "rule-based":
            self._run_rules(question)

        result = self.last_grounded or self.rag.ask(question)  # never return unvalidated text
        return {**result, "agent": {"mode": mode, "steps": self.trace}}

    # ---- LangChain loop ---------------------------------------------------
    def _langchain_model(self):
        from langchain_core.tools import StructuredTool

        cfg = self.rag.cfg
        chat = None
        for p in cfg.llm_providers:
            p = p.strip()
            if p == "groq" and cfg.groq_api_key:
                from langchain_groq import ChatGroq
                chat = ChatGroq(model=cfg.groq_model, temperature=0, api_key=cfg.groq_api_key)
            elif p == "gemini" and cfg.gemini_api_key:
                from langchain_google_genai import ChatGoogleGenerativeAI
                chat = ChatGoogleGenerativeAI(model=cfg.gemini_model, temperature=0, google_api_key=cfg.gemini_api_key)
            if chat is not None:
                break
        if chat is None:
            raise RuntimeError("no LLM key configured for agent")
        tools = [
            StructuredTool.from_function(self.search_zoning, name="search_zoning",
                                         description="Hybrid BM25+dense search over the NYC Zoning Resolution."),
            StructuredTool.from_function(self.read_source, name="read_source",
                                         description="Read exact lines (1-based, inclusive) of a corpus source file."),
            StructuredTool.from_function(self.grounded_answer, name="grounded_answer",
                                         description="Citation-validated answer to a focused zoning question."),
        ]
        return chat.bind_tools(tools)

    def _run_langchain(self, chat, question: str):
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

        msgs = [SystemMessage(AGENT_SYSTEM), HumanMessage(question)]
        for _ in range(MAX_STEPS):
            ai = chat.invoke(msgs)
            msgs.append(ai)
            if not ai.tool_calls:
                break
            for tc in ai.tool_calls:
                out = self._call(tc["name"], tc["args"])
                msgs.append(ToolMessage(out, tool_call_id=tc["id"]))
            if self.last_grounded is not None:
                break

    # ---- deterministic fallback planner -----------------------------------
    _SPAN_RE = re.compile(r"([\w\-./]+\.(?:txt|md))\D+(\d+)\D+(\d+)", re.I)

    def _run_rules(self, question: str):
        m = self._SPAN_RE.search(question)
        if m:
            self._call("read_source", {"source": m.group(1), "start_line": int(m.group(2)),
                                       "end_line": int(m.group(3))})
            return
        # multi-part questions: search each part, then answer the whole
        parts = [p.strip() for p in re.split(r"\?|\band\b(?=\s+(?:what|how|is|are|does|can|which))", question, flags=re.I)
                 if p and len(p.strip()) > 8]
        for p in parts[:3]:
            self._call("search_zoning", {"query": p})
        self._call("grounded_answer", {"question": question})
