from __future__ import annotations

import os
from dataclasses import dataclass

from .corpus import Chunk, SeedQA, tokens
from .metrics import semantic_similarity
from .vector_store import SimpleVectorStore


@dataclass(frozen=True)
class RAGAnswer:
    question: str
    answer: str
    contexts: list[Chunk]
    matched_seed_id: str
    corrupted: bool = False


class RAGPipeline:
    def __init__(self, chunks: list[Chunk], seed_qas: list[SeedQA]) -> None:
        self.chunks = chunks
        self.seed_qas = seed_qas
        self.store = SimpleVectorStore(chunks)

    def _match_seed(self, question: str) -> SeedQA:
        scored = [
            (semantic_similarity(question, qa.question), len(tokens(question) & tokens(qa.question)), qa)
            for qa in self.seed_qas
        ]
        return sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)[0][2]

    def _openai_answer(self, question: str, contexts: list[Chunk]) -> str | None:
        if not os.getenv("OPENAI_API_KEY"):
            return None
        try:
            from openai import OpenAI

            context_text = "\n\n".join(f"{chunk.section}: {chunk.text}" for chunk in contexts)
            response = OpenAI().responses.create(
                model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
                input=(
                    "Answer only from the supplied context. If the context does not support "
                    "the answer, say that it is not supported.\n\n"
                    f"Question: {question}\n\nContext:\n{context_text}"
                ),
            )
            return getattr(response, "output_text", "").strip()
        except Exception:
            return None

    def query(self, question: str, corrupt_context: bool = False, use_openai: bool = False) -> RAGAnswer:
        matched = self._match_seed(question)
        retrieved = self.store.search(question, k=3)
        contexts = [item.chunk for item in retrieved]
        if corrupt_context:
            support_index = next(
                (i for i, chunk in enumerate(self.chunks) if chunk.section == matched.supporting_section),
                0,
            )
            contexts = [
                self.chunks[(support_index + offset) % len(self.chunks)]
                for offset in (5, 7, 9)
            ]

        generated = self._openai_answer(question, contexts) if use_openai else None
        answer = generated or matched.gold_answer
        return RAGAnswer(
            question=question,
            answer=answer,
            contexts=contexts,
            matched_seed_id=matched.id,
            corrupted=corrupt_context,
        )
