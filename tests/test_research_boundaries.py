import pytest

from metarag.corpus import Chunk, load_corpus, load_seed_qas
from metarag.evaluation import ROOT
from metarag.oracles import NLIOracle
from metarag.pipeline import RAGPipeline


@pytest.mark.parametrize(
    "probabilities,grounded",
    [
        ([0.8, 0.1, 0.1], False),
        ([0.1, 0.8, 0.1], True),
        ([0.1, 0.1, 0.8], False),
        ([0.5, 0.5, 0], False),
    ],
)
def test_nli_uses_entailment_not_maximum_class(monkeypatch, probabilities, grounded):
    class ControlledModel:
        def predict(self, pairs, apply_softmax=False):
            assert apply_softmax is True
            return [probabilities]

    oracle = NLIOracle()
    monkeypatch.setattr(oracle, "_load", lambda: ControlledModel())
    result = oracle.judge(
        "question", "supported answer", [Chunk("section", "supported answer")]
    )
    assert result.provider == "nli" and result.grounded is grounded
    assert result.score == probabilities[1]


@pytest.mark.parametrize(
    "probabilities", [[1], [float("nan"), 0.5, 0.5], [-0.1, 0.6, 0.5], [0.8, 0.8, 0.8]]
)
def test_invalid_nli_output_has_explicit_lexical_provider(monkeypatch, probabilities):
    class ControlledModel:
        def predict(self, pairs, apply_softmax=False):
            return [probabilities]

    oracle = NLIOracle()
    monkeypatch.setattr(oracle, "_load", lambda: ControlledModel())
    result = oracle.judge(
        "question", "supported answer", [Chunk("section", "supported answer")]
    )
    assert result.provider == "nli-lexical-fallback"


def test_requested_llm_failure_cannot_masquerade_as_gold_answer(monkeypatch):
    pipeline = RAGPipeline(
        load_corpus(ROOT / "data/nist_ai_rmf_corpus.md"),
        load_seed_qas(ROOT / "data/seed_qa.yaml"),
    )
    monkeypatch.setattr(pipeline, "_openai_answer", lambda question, contexts: None)
    with pytest.raises(ValueError, match="refusing to substitute"):
        pipeline.query("What does Govern do?", use_openai=True)
