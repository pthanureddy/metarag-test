from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .corpus import Chunk, tokens


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


class SimpleVectorStore:
    """A deterministic lexical store used in CI when Chroma embeddings are unavailable."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def search(self, query: str, k: int = 3) -> list[RetrievedChunk]:
        q_tokens = tokens(query)
        scored = []
        for chunk in self.chunks:
            c_tokens = tokens(chunk.section + " " + chunk.text)
            overlap = len(q_tokens & c_tokens)
            score = overlap / max(len(q_tokens), 1)
            scored.append(RetrievedChunk(chunk, score))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:k]


def persist_chroma_if_available(chunks: list[Chunk], persist_dir: Path) -> bool:
    try:
        import chromadb
    except Exception:
        return False

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection("nist_ai_rmf")
    existing = set(collection.get().get("ids", []))
    for index, chunk in enumerate(chunks):
        doc_id = f"chunk-{index:03d}"
        if doc_id in existing:
            continue
        collection.add(
            ids=[doc_id],
            documents=[f"{chunk.section}\n{chunk.text}"],
            metadatas=[{"section": chunk.section}],
        )
    return True
