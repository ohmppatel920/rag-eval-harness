"""Cross-encoder reranking.

Bi-encoder retrieval (ChromaDB) is fast but coarse. A cross-encoder scores each
(question, chunk) pair jointly, which is slower but sharper. We retrieve a wide
candidate pool, then keep the top-k after reranking — the single biggest
retrieval-quality lever in the benchmark.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from config import RagConfig
from rag import retrieve

_RERANKER: CrossEncoder | None = None
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _reranker() -> CrossEncoder:
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoder(_MODEL_NAME)
    return _RERANKER


def rerank(question: str, chunks: list[str], top_k: int) -> list[str]:
    """Re-score candidate chunks against the question and keep the top-k."""
    scores = _reranker().predict([(question, c) for c in chunks])
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_k]]


def retrieve_reranked(
    question: str, config: RagConfig, pool: int = 20
) -> list[str]:
    """Retrieve a `pool` of candidates, rerank, and return the top-k."""
    candidates = retrieve(question, config, n=pool)
    return rerank(question, candidates, config.top_k)
