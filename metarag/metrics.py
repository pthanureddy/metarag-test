from __future__ import annotations

from difflib import SequenceMatcher
from statistics import mean, pvariance

from .corpus import tokens


def semantic_similarity(a: str, b: str) -> float:
    try:
        from sentence_transformers import SentenceTransformer, util

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode([a, b], convert_to_tensor=True)
        return float(util.cos_sim(embeddings[0], embeddings[1]).item())
    except Exception:
        a_tokens = tokens(a)
        b_tokens = tokens(b)
        if not a_tokens or not b_tokens:
            return SequenceMatcher(None, a, b).ratio()
        jaccard = len(a_tokens & b_tokens) / len(a_tokens | b_tokens)
        return max(jaccard, SequenceMatcher(None, a.lower(), b.lower()).ratio() * 0.8)


def variance(values: list[float]) -> float:
    return pvariance(values) if len(values) > 1 else 0.0


def mean_or_zero(values: list[float]) -> float:
    return mean(values) if values else 0.0
