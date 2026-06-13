# RAG Evaluation Harness

A retrieval-augmented-generation pipeline over SEC 10-K filings, built to
benchmark retrieval configurations (chunk size, embedding model, top-k). Local
generation runs on **Ollama** so the eval grid can run without API rate limits.

It covers the full loop: ingestion and indexing, a RAG pipeline, a cross-encoder
reranker, a RAGAS benchmark sweep, and DeepEval/Pytest gates for CI.

## Pipeline

```
ingest.py            ticker -> CIK -> latest 10-K -> clean text   (SEC EDGAR)
index.py             text -> overlapping chunks -> embeddings -> ChromaDB
rag.py               question -> top-k retrieval -> Ollama generation
rerank.py            wide candidate pool -> cross-encoder rerank -> top-k
config.py            RagConfig + benchmark GRID
eval_set.json        30 portable 10-K questions
evaluate.py          RAGAS sweep (faithfulness + answer relevancy) over the grid
test_rag_quality.py  DeepEval/Pytest thresholds for CI regression detection
main.py              end-to-end demo
```

A `RagConfig` is one point in the benchmark grid; each config gets its own
ChromaDB collection so configurations never collide.

## Setup

```bash
pip install -r requirements.txt

# Local generation model
ollama pull llama3.2

# SEC requires a descriptive User-Agent with a contact email
export SEC_USER_AGENT="rag-eval-harness you@example.com"
```

## Run

```bash
# Single-question demo
python main.py AAPL "What are the company's main business segments?"

# RAGAS benchmark sweep over the config grid
python evaluate.py AAPL            # baseline
python evaluate.py AAPL --rerank   # with cross-encoder reranking

# CI gates (point DeepEval at Ollama first: deepeval set-ollama llama3.2)
pytest test_rag_quality.py
```

First run downloads and caches the filing under `data/`, indexes it into
`chroma_db/`, then retrieves and answers.

## Status

| Component | State |
|---|---|
| SEC EDGAR ingestion | done |
| Configurable ChromaDB indexing | done |
| RAG retrieval + Ollama generation | done |
| 30-question eval set | done |
| Cross-encoder reranker | done |
| RAGAS metrics sweep | done |
| DeepEval / Pytest CI | done |
