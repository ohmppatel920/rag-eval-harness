"""Chunk filing text, embed it, and store vectors in ChromaDB.

Each RagConfig gets its own collection (named by config.name) so the benchmark
grid can compare chunk size / embedding model without collisions. Embeddings
are computed with sentence-transformers and passed to Chroma directly, so the
same model is used at index and query time.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from config import RagConfig

CHROMA_DIR = Path(__file__).parent / "chroma_db"
_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def _model(name: str) -> SentenceTransformer:
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = SentenceTransformer(name)
    return _MODEL_CACHE[name]


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping fixed-size character windows."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
    return chunks


def _client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def build_index(doc_id: str, text: str, config: RagConfig):
    """Embed `text` under `doc_id` into the collection for `config`.

    Returns the populated Chroma collection. Idempotent per (doc_id, config):
    re-running deletes the doc's old chunks before re-adding.
    """
    client = _client()
    collection = client.get_or_create_collection(config.name)

    # Clear any prior chunks for this document so re-indexing is clean.
    existing = collection.get(where={"doc_id": doc_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    chunks = chunk_text(text, config.chunk_size, config.chunk_overlap)
    embeddings = _model(config.embedding_model).encode(
        chunks, show_progress_bar=False
    ).tolist()

    collection.add(
        ids=[f"{doc_id}-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"doc_id": doc_id, "chunk": i} for i in range(len(chunks))],
    )
    return collection


def get_collection(config: RagConfig):
    """Open the existing collection for a config (read path for the RAG step)."""
    return _client().get_collection(config.name)
