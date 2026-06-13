"""DeepEval-backed pytest gates for CI-style regression detection.

Each test runs the RAG pipeline on a sample question and asserts that DeepEval's
faithfulness and answer-relevancy metrics clear a threshold. A config that
regresses retrieval or generation quality fails CI. DeepEval is pointed at the
local Ollama model (deepeval set-ollama <model>), matching the harness.

Run:  deepeval test run test_rag_quality.py     (or: pytest)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from config import RagConfig
from rag import answer

THRESHOLD = 0.7
SAMPLE_QUESTIONS = json.loads(
    (Path(__file__).parent / "eval_set.json").read_text()
)["questions"][:5]  # smoke subset keeps CI fast; widen for nightly runs
CONFIG = RagConfig()


@pytest.mark.parametrize("question", SAMPLE_QUESTIONS)
def test_rag_quality(question: str) -> None:
    """Answer must be faithful to and relevant for the retrieved context."""
    result = answer(question, CONFIG)
    test_case = LLMTestCase(
        input=question,
        actual_output=result.answer,
        retrieval_context=result.contexts,
    )
    assert_test(
        test_case,
        [
            FaithfulnessMetric(threshold=THRESHOLD),
            AnswerRelevancyMetric(threshold=THRESHOLD),
        ],
    )
