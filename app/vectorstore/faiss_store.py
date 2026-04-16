"""FAISS vector store manager."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


class FaissStore:
    """Persist and query vectors with a simple metadata sidecar."""

    def __init__(self, index_dir: str | Path) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "index.faiss"
        self.meta_path = self.index_dir / "metadata.json"
        self.index: faiss.Index | None = None
        self.metadata: list[dict] = []

    def build(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D")
        if embeddings.shape[0] != len(metadata):
            raise ValueError("metadata length must match embeddings rows")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self.index = index
        self.metadata = metadata

    def save(self) -> None:
        if self.index is None:
            raise RuntimeError("index is not initialized")
        faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self.metadata, ensure_ascii=True, indent=2), encoding="utf-8")

    def load(self) -> None:
        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError("Persisted FAISS index/metadata not found")
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[dict]:
        if self.index is None:
            raise RuntimeError("index is not initialized")
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        scores, indices = self.index.search(query, top_k)
        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            results.append(item)
        return results

