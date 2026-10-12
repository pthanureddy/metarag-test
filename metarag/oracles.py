from __future__ import annotations

import math
import os
from dataclasses import dataclass

from .corpus import Chunk, tokens


@dataclass(frozen=True)
class GroundednessResult:
    grounded: bool
    score: float
    reasoning: str
    provider: str


def context_text(contexts: list[Chunk]) -> str:
    return "\n\n".join(f"{chunk.section}: {chunk.text}" for chunk in contexts)


class NLIOracle:
    provider = "nli"

    def __init__(self) -> None:
        self.model_name = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-base")
        self.fallback_model_name = os.getenv(
            "NLI_FALLBACK_MODEL", "cross-encoder/nli-deberta-v3-xsmall"
        )
        self._model = None
        self._model_loaded = False

    def _load(self):
        if self._model_loaded:
            return self._model
        self._model_loaded = True
        try:
            from sentence_transformers import CrossEncoder

            try:
                self._model = CrossEncoder(self.model_name)
            except Exception:
                self._model = CrossEncoder(self.fallback_model_name)
        except Exception:
            self._model = None
        return self._model

    def judge(
        self, question: str, answer: str, contexts: list[Chunk]
    ) -> GroundednessResult:
        text = context_text(contexts)
        model = self._load()
        if model is not None:
            try:
                scores = model.predict([(text, answer)], apply_softmax=True)
                if hasattr(scores, "tolist"):
                    scores = scores.tolist()
                if (
                    not isinstance(scores, list)
                    or len(scores) != 1
                    or len(scores[0]) != 3
                ):
                    raise ValueError("NLI expects three class probabilities")
                probabilities = [float(value) for value in scores[0]]
                if not all(
                    math.isfinite(value) and 0 <= value <= 1 for value in probabilities
                ) or not math.isclose(sum(probabilities), 1, abs_tol=1e-5):
                    raise ValueError("Invalid NLI probabilities")
                # DeBERTa NLI order: contradiction, entailment, neutral.
                entailment_score = probabilities[1]
                return GroundednessResult(
                    grounded=entailment_score >= 0.5
                    and entailment_score > max(probabilities[0], probabilities[2]),
                    score=entailment_score,
                    reasoning="NLI entailment probability (class 1), distinguished from contradiction/neutral",
                    provider=self.provider,
                )
            except Exception:
                pass

        answer_tokens = tokens(answer)
        context_tokens = tokens(text)
        overlap = len(answer_tokens & context_tokens) / max(len(answer_tokens), 1)
        return GroundednessResult(
            grounded=overlap >= 0.42,
            score=overlap,
            reasoning="Lexical fallback for NLI when local model is unavailable",
            provider="nli-lexical-fallback",
        )


class OpenAIJudge:
    provider = "openai-judge"

    def judge(
        self, question: str, answer: str, contexts: list[Chunk]
    ) -> GroundednessResult:
        if not os.getenv("OPENAI_API_KEY"):
            return GroundednessResult(
                False, 0.0, "OPENAI_API_KEY is not set", self.provider
            )
        try:
            from openai import OpenAI

            response = OpenAI().responses.create(
                model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
                input=(
                    "Judge whether the answer is fully grounded in the provided context. "
                    'Return JSON only: {"grounded": true|false, "score": 0.0, "reasoning": "..."}.\n\n'
                    f"Question: {question}\nAnswer: {answer}\nContext:\n{context_text(contexts)}"
                ),
            )
            import json
            import re

            match = re.search(r"\{.*\}", response.output_text, flags=re.DOTALL)
            payload = json.loads(match.group(0)) if match else {}
            return GroundednessResult(
                grounded=bool(payload.get("grounded")),
                score=float(payload.get("score", 0.0)),
                reasoning=str(payload.get("reasoning", "")),
                provider=self.provider,
            )
        except Exception as exc:
            return GroundednessResult(
                False, 0.0, f"OpenAI judge failed: {exc}", self.provider
            )


def ragas_faithfulness_proxy(answer: str, contexts: list[Chunk]) -> dict:
    text = context_text(contexts)
    answer_tokens = tokens(answer)
    context_tokens = tokens(text)
    score = len(answer_tokens & context_tokens) / max(len(answer_tokens), 1)
    try:
        import ragas  # noqa: F401

        available = True
    except Exception:
        available = False
    return {
        "score": score,
        "available": available,
        "method": "ragas-installed-proxy"
        if available
        else "lexical-faithfulness-proxy",
    }
