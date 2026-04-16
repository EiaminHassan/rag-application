"""Top-K retriever built on FAISS store."""

from __future__ import annotations

from app.embeddings.embedder import Embedder
from app.vectorstore.faiss_store import FaissStore


class Retriever:
    """Retrieve most relevant chunks for a user query."""

    def __init__(self, embedder: Embedder, store: FaissStore, top_k: int = 4) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict]:
        query_vec = self.embedder.encode_query(query)
        return self.store.search(query_vec, top_k=self.top_k)

