"""RAGAS benchmark sweep over the config grid.

For each RagConfig (optionally with the cross-encoder reranker), answer the eval
set and score it with RAGAS faithfulness + answer relevancy. Both metrics are
ground-truth-free, so the eval set stays portable across companies. RAGAS is
pointed at a local Ollama model so the sweep runs without API rate limits.

Run:
    python evaluate.py AAPL                 # baseline grid (first 8 questions)
    python evaluate.py AAPL --rerank        # grid with reranking
    python evaluate.py AAPL --limit 30      # full eval set (slow on CPU)

Notes for local models:
  * RAGAS embeds with a dedicated embedding model (nomic-embed-text), not the
    chat model -- chat models 501 on the embeddings endpoint.
  * Scoring runs serially with a long timeout (RunConfig) so a slow local LLM
    on CPU doesn't trip RAGAS's default 180s/job concurrent timeout.
    First: `ollama pull llama3.2 && ollama pull nomic-embed-text`
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate as ragas_evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, faithfulness
from ragas.run_config import RunConfig

from config import GRID, RagConfig
from index import build_index
from ingest import fetch_10k_text
from rag import generate, retrieve
from rerank import retrieve_reranked

EVAL_SET = Path(__file__).parent / "eval_set.json"
METRICS = [faithfulness, answer_relevancy]
EMBED_MODEL = "nomic-embed-text"          # dedicated Ollama embedding model
# RAGAS scores by asking the judge for strict JSON. Small models (llama3.2:3b)
# can't follow that format -> parse errors and nan scores. Use a model good at
# structured output as the judge; generation still uses config.gen_model.
JUDGE_MODEL = os.environ.get("RAGAS_JUDGE_MODEL", "qwen2.5:7b")
DEFAULT_LIMIT = 8                         # questions per run; raise with --limit
# Serialize scoring and give each call room; local CPU generation is slow.
RUN_CONFIG = RunConfig(max_workers=1, timeout=600)


def _ragas_models():
    """Wrap a JSON-capable Ollama judge + embedding model for RAGAS."""
    llm = LangchainLLMWrapper(ChatOllama(model=JUDGE_MODEL, temperature=0.0))
    emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=EMBED_MODEL))
    return llm, emb


def _limit(argv: list[str]) -> int:
    if "--limit" in argv:
        return int(argv[argv.index("--limit") + 1])
    return DEFAULT_LIMIT


def run_config(ticker: str, config: RagConfig, questions: list[str], use_rerank: bool):
    """Build the eval dataset for one config by running the RAG pipeline."""
    rows = {"question": [], "answer": [], "contexts": []}
    for i, q in enumerate(questions, 1):
        print(f"  [{config.name}] generating answer {i}/{len(questions)} ...",
              flush=True)
        contexts = (
            retrieve_reranked(q, config) if use_rerank else retrieve(q, config)
        )
        rows["question"].append(q)
        rows["answer"].append(generate(q, contexts, config))
        rows["contexts"].append(contexts)
    print(f"  [{config.name}] scoring with RAGAS (this is the slow part) ...",
          flush=True)
    return Dataset.from_dict(rows)


def main() -> None:
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    use_rerank = "--rerank" in sys.argv

    questions = json.loads(EVAL_SET.read_text())["questions"][: _limit(sys.argv)]
    # --quick: one config instead of the full grid, for fast smoke runs.
    configs = GRID[:1] if "--quick" in sys.argv else GRID
    text = fetch_10k_text(ticker)
    print(f"Scoring {len(questions)} questions x {len(configs)} configs "
          f"(rerank={use_rerank}) ...")

    results = []
    for config in configs:
        build_index(doc_id=ticker.upper(), text=text, config=config)
        dataset = run_config(ticker, config, questions, use_rerank)
        llm, emb = _ragas_models()
        scores = ragas_evaluate(
            dataset, metrics=METRICS, llm=llm, embeddings=emb,
            run_config=RUN_CONFIG,
        )
        results.append((config.name, scores))

    # Compact results table.
    print(f"\n{'config':<32} {'faithfulness':>14} {'answer_rel':>12}")
    print("-" * 60)
    for name, scores in results:
        df = scores.to_pandas()
        print(
            f"{name:<32} {df['faithfulness'].mean():>14.3f} "
            f"{df['answer_relevancy'].mean():>12.3f}"
        )


if __name__ == "__main__":
    main()
