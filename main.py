"""Demo entrypoint: ingest a 10-K, index it, and answer a question.

    python main.py AAPL "What are the company's main business segments?"

Requires a running Ollama server with the gen_model pulled
(`ollama pull llama3.2`).
"""

from __future__ import annotations

import sys

from config import RagConfig
from ingest import fetch_10k_text
from index import build_index
from rag import answer


def main() -> None:
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    question = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "What are the company's main business segments?"
    )

    config = RagConfig()  # default single config for the foundation demo

    print(f"Fetching latest 10-K for {ticker} ...")
    text = fetch_10k_text(ticker)
    print(f"  {len(text):,} chars")

    print(f"Indexing into collection '{config.name}' ...")
    build_index(doc_id=ticker.upper(), text=text, config=config)

    print(f"Q: {question}")
    result = answer(question, config)
    print(f"A: {result.answer}")
    print(f"\n(grounded in {len(result.contexts)} retrieved chunks)")


if __name__ == "__main__":
    main()
