"""End-to-end evaluation harness with three tiers:

  1. Deterministic  - retrieval hit@k / MRR, abstention accuracy, citation validity
                      (every cited span exists and its quote matches the file byte-for-byte),
                      citation precision against gold sources.
  2. Heuristic      - expected-keyword recall, grounded-sentence ratio, number faithfulness
                      (every number in the answer appears in a cited quote), length bounds.
  3. LLM-judge      - faithfulness + relevance (1-5) scored by Groq/Gemini; skipped if no key.

Usage:
  python -m eval.run_eval                     # full run, writes eval/results/
  python -m eval.run_eval --provider extractive --no-judge
  python -m eval.run_eval --gate              # non-zero exit if thresholds fail (CI)
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from zoning_rag.config import settings  # noqa: E402
from zoning_rag.llm import LLMUnavailable  # noqa: E402
from zoning_rag.pipeline import ZoningRAG  # noqa: E402

GATES = {  # minimum acceptable values for --gate
    "retrieval_hit_at_k": 0.90,
    "abstention_accuracy": 0.90,
    "citation_validity": 1.00,
    "keyword_recall": 0.75,
    "number_faithfulness": 0.95,
}

JUDGE_SYSTEM = "You are a strict evaluator of retrieval-augmented answers. Respond with JSON only."
JUDGE_PROMPT = """Score the ANSWER to the QUESTION using only the SOURCES.
faithfulness: 5 = every claim is directly supported by SOURCES, 1 = mostly unsupported.
relevance:    5 = fully answers the question, 1 = does not address it.
Return JSON: {{"faithfulness": int, "relevance": int, "rationale": "<one sentence>"}}

QUESTION: {q}

SOURCES:
{sources}

ANSWER: {a}"""

_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


# ---------------- deterministic -------------------------------------------
def citation_valid(c: dict) -> bool:
    """The cited lines must exist and contain exactly the quoted text."""
    p = settings.corpus_dir / c["source"]
    if not p.exists():
        return False
    lines = p.read_text(encoding="utf-8").splitlines()
    if not (1 <= c["start_line"] <= c["end_line"] <= len(lines)):
        return False
    actual = " ".join(l.strip() for l in lines[c["start_line"] - 1 : c["end_line"]] if l.strip())
    return actual == c["quote"]


def deterministic(item: dict, res: dict) -> dict:
    gold = set(item.get("gold_sources", []))
    hit_sources = [h["span"].split(":")[0] for h in res["retrieval"]["hits"]]
    rank = next((i + 1 for i, s in enumerate(hit_sources) if s in gold), None)
    cites = res.get("citations", [])
    return {
        "abstention_correct": res["abstained"] == (not item["answerable"]),
        "hit_at_k": (rank is not None) if item["answerable"] else None,
        "rr": (1 / rank if rank else 0.0) if item["answerable"] else None,
        "citations_valid": all(citation_valid(c) for c in cites) if cites else None,
        "citation_precision": (mean(c["source"] in gold for c in cites) if cites and gold else None),
    }


# ---------------- heuristic -----------------------------------------------
def heuristic(item: dict, res: dict) -> dict:
    if res["abstained"] or not item["answerable"]:
        return {"keyword_recall": None, "number_faithful": None, "grounded_ratio": None, "length_ok": None}
    ans = res["answer"]
    body = re.sub(r"\[[^\]]+:L\d+(?:-L\d+)?\]", "", ans)  # strip citation markers
    kws = item.get("expected_keywords", [])
    quotes = " ".join(c["quote"] for c in res.get("citations", []))
    nums = _NUM_RE.findall(body)
    return {
        "keyword_recall": mean(k.lower() in body.lower() for k in kws) if kws else None,
        "number_faithful": all(n in quotes for n in nums) if nums else True,
        "grounded_ratio": res.get("grounded_ratio"),
        "length_ok": 20 <= len(body) <= 1500,
    }


# ---------------- LLM judge -----------------------------------------------
def llm_judge(rag: ZoningRAG, item: dict, res: dict) -> dict:
    if res["abstained"] or not item["answerable"] or not rag.llm.available:
        return {"faithfulness": None, "relevance": None}
    srcs = "\n".join(f"- ({c['span']}) {c['quote']}" for c in res["citations"])
    try:
        raw, prov = rag.llm.complete(JUDGE_SYSTEM, JUDGE_PROMPT.format(q=item["question"], sources=srcs, a=res["answer"]))
        j = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        return {"faithfulness": int(j["faithfulness"]), "relevance": int(j["relevance"]),
                "judge_rationale": j.get("rationale"), "judge": prov}
    except (LLMUnavailable, AttributeError, ValueError, KeyError) as e:
        return {"faithfulness": None, "relevance": None, "judge_error": str(e)[:200]}


# ---------------- runner --------------------------------------------------
def _avg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(mean(float(v) for v in vals), 3) if vals else None


def run(dataset: Path, provider: str | None, judge: bool, out_dir: Path) -> dict:
    rag = ZoningRAG(settings)
    items = [json.loads(l) for l in dataset.read_text().splitlines() if l.strip()]
    rows = []
    for it in items:
        res = rag.ask(it["question"], provider=provider)
        row = {"id": it["id"], "question": it["question"], "answerable": it["answerable"],
               "abstained": res["abstained"], "provider": res["provider"],
               "confidence": res["retrieval"]["confidence"], "answer": res["answer"],
               **deterministic(it, res), **heuristic(it, res)}
        if judge:
            row.update(llm_judge(rag, it, res))
        rows.append(row)

    summary = {
        "n": len(rows),
        "providers": sorted({r["provider"] or "none" for r in rows}),
        "deterministic": {
            "retrieval_hit_at_k": _avg(rows, "hit_at_k"),
            "retrieval_mrr": _avg(rows, "rr"),
            "abstention_accuracy": _avg(rows, "abstention_correct"),
            "false_abstain": sum(r["answerable"] and r["abstained"] for r in rows),
            "false_answer": sum((not r["answerable"]) and (not r["abstained"]) for r in rows),
            "citation_validity": _avg(rows, "citations_valid"),
            "citation_precision": _avg(rows, "citation_precision"),
        },
        "heuristic": {
            "keyword_recall": _avg(rows, "keyword_recall"),
            "number_faithfulness": _avg(rows, "number_faithful"),
            "grounded_ratio": _avg(rows, "grounded_ratio"),
            "length_ok": _avg(rows, "length_ok"),
        },
        "llm_judge": {"faithfulness": _avg(rows, "faithfulness"), "relevance": _avg(rows, "relevance")}
        if judge else "skipped",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_item.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return {"summary": summary, "rows": rows}


def check_gates(summary: dict) -> list[str]:
    flat = {**summary["deterministic"], **summary["heuristic"]}
    return [f"{k}={flat.get(k)} < {v}" for k, v in GATES.items() if flat.get(k) is None or flat[k] < v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=ROOT / "eval" / "dataset.jsonl")
    ap.add_argument("--provider", default=None, help="'extractive' forces the offline generator")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results")
    ap.add_argument("--gate", action="store_true", help="exit 1 if quality gates fail")
    a = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)

    r = run(a.dataset, a.provider, not a.no_judge, a.out)
    print(json.dumps(r["summary"], indent=2))
    bad = [x for x in r["rows"] if not x["abstention_correct"] or x.get("citations_valid") is False
           or (x.get("keyword_recall") is not None and x["keyword_recall"] < 1)]
    for x in bad:
        print(f"  ! {x['id']}: abstained={x['abstained']} kw={x.get('keyword_recall')} "
              f"cites_valid={x.get('citations_valid')} conf={x['confidence']}")
    if a.gate:
        fails = check_gates(r["summary"])
        print("GATES:", "PASS" if not fails else "FAIL -> " + "; ".join(fails))
        sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
