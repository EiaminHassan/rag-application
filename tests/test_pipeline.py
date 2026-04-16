"""Basic tests for RAG pipeline components."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from app.ingestion.chunking import chunk_text
from app.vectorstore.faiss_store import FaissStore


def test_chunk_text_overlap_behavior() -> None:
    text = "A" * 2200
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    assert len(chunks) == 3
    assert len(chunks[0]) == 1000
    assert chunks[0][-200:] == chunks[1][:200]


def test_faiss_roundtrip_search() -> None:
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    metadata = [
        {"text": "first"},
        {"text": "second"},
        {"text": "third"},
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
        store = FaissStore(Path(temp_dir))
        store.build(vectors, metadata)
        store.save()
        loaded = FaissStore(Path(temp_dir))
        loaded.load()
        results = loaded.search(np.array([0.9, 0.1, 0.0], dtype=np.float32), top_k=2)
        assert len(results) == 2
        assert results[0]["text"] == "first"

