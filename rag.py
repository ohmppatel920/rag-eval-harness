"""Retrieve-then-generate over an indexed 10-K, using a local Ollama model.

The RagResult bundles the answer with the retrieved contexts, because the eval
half of the project (RAGAS faithfulness / context recall) needs both the
generated answer and the contexts it was grounded in.
"""

from __future__ import annotations

from dataclasses import dataclass

import ollama

from config import RagConfig
from index import get_collection, _model

SYSTEM_PROMPT = (
    "You answer questions about a company's SEC 10-K filing. "
    "Use ONLY the provided context. If the answer is not in the context, "
    "say you don't have enough information. Be concise and factual."
)


@dataclass
class RagResult:
    question: str
    answer: str
    contexts: list[str]


def retrieve(question: str, config: RagConfig, n: int | None = None) -> list[str]:
    """Return chunks for a question using the config's embedder.

    `n` overrides how many are returned (defaults to config.top_k). The
    reranker uses a larger `n` to build a candidate pool before re-scoring.
    """
    collection = get_collection(config)
    query_emb = _model(config.embedding_model).encode([question]).tolist()
    res = collection.query(
        query_embeddings=query_emb, n_results=n or config.top_k
    )
    return res["documents"][0]


def generate(question: str, contexts: list[str], config: RagConfig) -> str:
    """Generate a grounded answer from retrieved contexts via Ollama."""
    context_block = "\n\n---\n\n".join(
        f"[{i + 1}] {c}" for i, c in enumerate(contexts)
    )
    user = f"Context:\n{context_block}\n\nQuestion: {question}"
    resp = ollama.chat(
        model=config.gen_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0.0},
    )
    return resp["message"]["content"].strip()


def answer(question: str, config: RagConfig) -> RagResult:
    """Full RAG: retrieve top-k, then generate a grounded answer."""
    contexts = retrieve(question, config)
    text = generate(question, contexts, config)
    return RagResult(question=question, answer=text, contexts=contexts)
