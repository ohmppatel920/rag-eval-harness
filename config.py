"""Pipeline configuration.

A single RagConfig defines one point in the benchmark grid: which embedding
model encodes chunks, how big the chunks are, and how many are retrieved. The
eval half of the project (RAGAS / DeepEval) sweeps over a list of these.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RagConfig:
    chunk_size: int = 800          # characters per chunk
    chunk_overlap: int = 100       # characters shared between adjacent chunks
    embedding_model: str = "all-MiniLM-L6-v2"  # sentence-transformers model
    top_k: int = 4                 # chunks retrieved per query
    gen_model: str = "llama3.2"    # local Ollama model for generation

    @property
    def name(self) -> str:
        """Stable identifier used to name the ChromaDB collection."""
        model = self.embedding_model.replace("/", "-")
        return f"{model}_cs{self.chunk_size}_ov{self.chunk_overlap}"


# A small benchmark grid. The foundation half only needs one config to run;
# the eval half iterates over the whole list.
GRID = [
    RagConfig(chunk_size=500, top_k=4),
    RagConfig(chunk_size=800, top_k=4),
    RagConfig(chunk_size=1200, top_k=6),
]
