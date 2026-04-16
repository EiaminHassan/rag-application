"""Streamlit entry point."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

# Allow launching via either:
# - streamlit run app/main.py (repo root)
# - cd app && streamlit run main.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import DATA_RAW_DIR
from app.pipeline.rag_pipeline import RAGPipeline


st.set_page_config(page_title="Groq + FAISS RAG", page_icon="📄", layout="wide")
st.title("Groq + FAISS RAG Application")


def save_uploaded_files(uploaded_files: list) -> list[Path]:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    file_paths: list[Path] = []
    for uploaded in uploaded_files:
        path = DATA_RAW_DIR / uploaded.name
        path.write_bytes(uploaded.getbuffer())
        file_paths.append(path)
    return file_paths


if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()

pipeline: RAGPipeline = st.session_state.pipeline

with st.sidebar:
    st.header("Ingestion")
    uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Build / Rebuild Index", use_container_width=True):
        if not uploaded:
            st.warning("Please upload at least one PDF first.")
        else:
            with st.spinner("Processing PDFs and building index..."):
                paths = save_uploaded_files(uploaded)
                count = pipeline.ingest_pdfs(paths)
            st.success(f"Indexed {count} chunks.")

    if st.button("Load Existing Index", use_container_width=True):
        try:
            pipeline.load_index()
            st.success("Loaded existing FAISS index.")
        except FileNotFoundError:
            st.error("No persisted index found. Build one first.")

st.subheader("Ask a question")
question = st.text_input("Question", placeholder="What does the document say about ...?")

if st.button("Get Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            result = pipeline.answer(question.strip())
            st.markdown("### Answer")
            st.write(result["answer"])
            st.markdown("### Retrieved Context")
            for idx, item in enumerate(result["contexts"], start=1):
                with st.expander(f"Chunk {idx} | score={item.get('score', 0):.4f}"):
                    st.caption(item.get("source", "Unknown source"))
                    st.write(item.get("text", ""))
        except RuntimeError as exc:
            st.error(f"Index is not loaded: {exc}")
        except Exception as exc:  # pragma: no cover - defensive UI error handling
            st.error(f"Failed to answer question: {exc}")

