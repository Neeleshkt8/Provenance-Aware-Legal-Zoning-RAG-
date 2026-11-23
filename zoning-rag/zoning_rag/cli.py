"""CLI:  python -m zoning_rag.cli "What is floor area ratio?" [--agent] [--json]"""
import argparse
import json
import logging

from .agent import ZoningAgent
from .pipeline import ZoningRAG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--agent", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--provider", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)
    rag = ZoningRAG()
    r = ZoningAgent(rag).run(a.question) if a.agent else rag.ask(a.question, provider=a.provider)
    if a.json:
        print(json.dumps(r, indent=2))
        return
    print(f"\n{r['answer']}\n")
    if not r["abstained"]:
        print("Sources:")
        for c in r["citations"]:
            print(f"  [{c['span']}] {c['quote']}")
    else:
        print(f"(abstained: {r['reason']})")
    print(f"\nprovider={r['provider']} confidence={r['retrieval']['confidence']} latency={r['latency_ms']}ms")
    print(r["disclaimer"])


if __name__ == "__main__":
    main()
