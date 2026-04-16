"""End-to-end RAG orchestration."""

from __future__ import annotations

from pathlib import Path

from app.config.settings import Settings, VECTORSTORE_DIR
from app.embeddings.embedder import Embedder
from app.ingestion.chunking import chunk_text
from app.ingestion.pdf_loader import extract_text_from_pdf
from app.llm.groq_client import GroqClient
from app.retriever.retriever import Retriever
from app.vectorstore.faiss_store import FaissStore


class RAGPipeline:
    """Build index from PDFs and answer questions."""

    def __init__(self, settings: Settings | None = None, index_dir: Path | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.embedder = Embedder(self.settings.embedding_model_name)
        self.store = FaissStore(index_dir or VECTORSTORE_DIR)
        self.retriever = Retriever(self.embedder, self.store, top_k=self.settings.top_k)
        self.llm = GroqClient(api_key=self.settings.groq_api_key, model=self.settings.groq_model)

    def ingest_pdfs(self, pdf_paths: list[str | Path]) -> int:
        docs: list[dict] = []
        for path in pdf_paths:
            text = extract_text_from_pdf(path)
            chunks = chunk_text(
                text,
                chunk_size=self.settings.chunk_size,
                overlap=self.settings.chunk_overlap,
            )
            for idx, chunk in enumerate(chunks):
                docs.append({"text": chunk, "source": str(path), "chunk_id": idx})

        if not docs:
            raise ValueError("No text chunks were produced from provided PDFs")

        embeddings = self.embedder.encode([d["text"] for d in docs])
        self.store.build(embeddings, docs)
        self.store.save()
        return len(docs)

    def load_index(self) -> None:
        self.store.load()

    def answer(self, question: str) -> dict:
        results = self.retriever.retrieve(question)
        context = "\n\n".join([item["text"] for item in results])
        answer = self.llm.generate(question=question, context=context)
        return {"answer": answer, "contexts": results}

