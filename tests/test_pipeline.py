from pathlib import Path

from metarag.corpus import load_corpus, load_seed_qas
from metarag.pipeline import RAGPipeline


def test_pipeline_returns_grounded_answer_for_seed_question():
    pipeline = RAGPipeline(
        load_corpus(Path("data/nist_ai_rmf_corpus.md")),
        load_seed_qas(Path("data/seed_qa.yaml")),
    )

    answer = pipeline.query("What is the main purpose of the Govern function?")

    assert "policies" in answer.answer
    assert answer.contexts
    assert answer.matched_seed_id == "q01"


def test_pipeline_can_corrupt_context_for_validation():
    pipeline = RAGPipeline(
        load_corpus(Path("data/nist_ai_rmf_corpus.md")),
        load_seed_qas(Path("data/seed_qa.yaml")),
    )

    answer = pipeline.query("What is the main purpose of the Govern function?", corrupt_context=True)

    assert answer.corrupted is True
    assert all(chunk.section != "Govern Function" for chunk in answer.contexts)
