from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Chunk:
    section: str
    text: str


@dataclass(frozen=True)
class SeedQA:
    id: str
    question: str
    gold_answer: str
    supporting_section: str


def load_corpus(path: Path) -> list[Chunk]:
    text = path.read_text(encoding="utf-8")
    chunks: list[Chunk] = []
    current_section = "Preamble"
    current_lines: list[str] = []
    for line in text.splitlines():
        heading = re.match(r"^##\s+(.+)$", line)
        if heading:
            if current_lines:
                chunks.append(Chunk(current_section, "\n".join(current_lines).strip()))
            current_section = heading.group(1).strip()
            current_lines = []
        elif line.strip() and not line.startswith("# "):
            current_lines.append(line.strip())
    if current_lines:
        chunks.append(Chunk(current_section, "\n".join(current_lines).strip()))
    return chunks


def load_seed_qas(path: Path) -> list[SeedQA]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [SeedQA(**item) for item in raw]


def tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in {"the", "and", "for", "with", "that"}
    }
