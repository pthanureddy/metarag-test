from __future__ import annotations

import os


def generate_paraphrases(question: str, n: int = 3) -> list[str]:
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI

            response = OpenAI().responses.create(
                model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
                input=(
                    f"Create {n} semantically equivalent paraphrases of this question. "
                    "Return one paraphrase per line, with no numbering.\n"
                    f"Question: {question}"
                ),
            )
            lines = [line.strip("- 0123456789.\t") for line in response.output_text.splitlines()]
            lines = [line for line in lines if line]
            if len(lines) >= n:
                return lines[:n]
        except Exception:
            pass

    question = question.rstrip("?")
    return [
        f"In the NIST AI RMF, {question[0].lower() + question[1:]}?",
        f"Can you explain {question[0].lower() + question[1:]}?",
        f"What should a practitioner know about this: {question}?",
    ][:n]
