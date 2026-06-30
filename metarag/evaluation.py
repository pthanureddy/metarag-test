from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

from .corpus import load_corpus, load_seed_qas
from .metamorphic import generate_paraphrases
from .metrics import mean_or_zero, semantic_similarity, variance
from .oracles import NLIOracle, OpenAIJudge, ragas_faithfulness_proxy
from .pipeline import RAGPipeline
from .vector_store import persist_chroma_if_available


ROOT = Path(__file__).resolve().parents[1]


def evaluate(use_openai_answers: bool = False, include_openai_judge: bool = False) -> dict[str, Any]:
    chunks = load_corpus(ROOT / "data" / "nist_ai_rmf_corpus.md")
    qas = load_seed_qas(ROOT / "data" / "seed_qa.yaml")
    chroma_persisted = persist_chroma_if_available(chunks, ROOT / "artifacts" / "chroma")
    pipeline = RAGPipeline(chunks, qas)
    nli = NLIOracle()
    openai_judge = OpenAIJudge() if include_openai_judge and os.getenv("OPENAI_API_KEY") else None

    case_results: list[dict[str, Any]] = []
    corrupted_results: list[dict[str, Any]] = []
    metamorphic_results: list[dict[str, Any]] = []
    consistency_results: list[dict[str, Any]] = []
    ragas_pairs: list[tuple[bool, bool]] = []

    for qa in qas:
        base = pipeline.query(qa.question, use_openai=use_openai_answers)
        clean_judgment = nli.judge(qa.question, base.answer, base.contexts)
        ragas_score = ragas_faithfulness_proxy(base.answer, base.contexts)
        ragas_pass = ragas_score["score"] >= 0.42
        ragas_pairs.append((clean_judgment.grounded, ragas_pass))
        case_results.append(
            {
                "id": qa.id,
                "question": qa.question,
                "answer": base.answer,
                "expected_grounded": True,
                "actual_grounded": clean_judgment.grounded,
                "score": clean_judgment.score,
                "oracle": clean_judgment.provider,
                "ragas_proxy": ragas_score,
                "openai_judge": openai_judge.judge(qa.question, base.answer, base.contexts).__dict__
                if openai_judge
                else None,
            }
        )

        corrupted = pipeline.query(qa.question, corrupt_context=True, use_openai=use_openai_answers)
        corrupted_judgment = nli.judge(qa.question, corrupted.answer, corrupted.contexts)
        corrupted_results.append(
            {
                "id": qa.id,
                "question": qa.question,
                "answer": corrupted.answer,
                "expected_grounded": False,
                "actual_grounded": corrupted_judgment.grounded,
                "score": corrupted_judgment.score,
                "oracle": corrupted_judgment.provider,
            }
        )

        repeated = [pipeline.query(qa.question, use_openai=use_openai_answers).answer for _ in range(5)]
        sims = [semantic_similarity(repeated[0], answer) for answer in repeated[1:]]
        consistency_results.append(
            {
                "id": qa.id,
                "mean_similarity": mean_or_zero(sims),
                "variance": variance(sims),
            }
        )

        base_answer = base.answer
        for paraphrase in generate_paraphrases(qa.question, n=3):
            p_answer = pipeline.query(paraphrase, use_openai=use_openai_answers).answer
            sim = semantic_similarity(base_answer, p_answer)
            metamorphic_results.append(
                {
                    "id": qa.id,
                    "paraphrase": paraphrase,
                    "similarity": sim,
                    "violated": sim < 0.72,
                }
            )

    clean_correct = sum(1 for row in case_results if row["actual_grounded"] is True)
    corrupt_caught = sum(1 for row in corrupted_results if row["actual_grounded"] is False)
    ragas_agreement = sum(1 for a, b in ragas_pairs if a == b) / max(len(ragas_pairs), 1)

    return {
        "summary": {
            "total_seed_questions": len(qas),
            "groundedness_accuracy": clean_correct / max(len(case_results), 1),
            "hallucination_catch_rate": corrupt_caught / max(len(corrupted_results), 1),
            "metamorphic_violation_rate": mean([1.0 if row["violated"] else 0.0 for row in metamorphic_results]),
            "mean_flakiness_variance": mean_or_zero([row["variance"] for row in consistency_results]),
            "ragas_agreement": ragas_agreement,
            "chroma_persisted": chroma_persisted,
            "openai_judge_included": openai_judge is not None,
            "estimated_cost_usd": 0.0,
        },
        "cases": case_results,
        "corrupted_cases": corrupted_results,
        "metamorphic": metamorphic_results,
        "consistency": consistency_results,
    }


def write_evaluation(path: Path, **kwargs: Any) -> dict[str, Any]:
    result = evaluate(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
