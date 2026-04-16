"""Text chunking logic for retrieval."""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(cleaned), step):
        part = cleaned[start : start + chunk_size]
        if part:
            chunks.append(part)
        if start + chunk_size >= len(cleaned):
            break
    return chunks

