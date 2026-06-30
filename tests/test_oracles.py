from metarag.corpus import Chunk
from metarag.oracles import NLIOracle, ragas_faithfulness_proxy


def test_nli_oracle_fallback_distinguishes_supported_answer():
    oracle = NLIOracle()
    contexts = [Chunk("Govern Function", "Govern establishes policies roles responsibilities and accountability practices.")]

    result = oracle.judge(
        "What does govern do?",
        "Govern establishes policies, roles, responsibilities, and accountability.",
        contexts,
    )

    assert result.grounded is True
    assert result.score > 0


def test_ragas_proxy_returns_bounded_score():
    contexts = [Chunk("Safety", "Safety addresses physical and societal harms.")]

    result = ragas_faithfulness_proxy("Safety addresses physical harms.", contexts)

    assert 0 <= result["score"] <= 1
    assert "method" in result
