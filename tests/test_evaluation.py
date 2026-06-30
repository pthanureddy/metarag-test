from metarag.evaluation import evaluate


def test_evaluation_produces_required_metrics():
    result = evaluate(use_openai_answers=False, include_openai_judge=False)

    summary = result["summary"]
    assert summary["total_seed_questions"] == 20
    assert 0 <= summary["groundedness_accuracy"] <= 1
    assert 0 <= summary["hallucination_catch_rate"] <= 1
    assert 0 <= summary["metamorphic_violation_rate"] <= 1
    assert len(result["metamorphic"]) == 60
