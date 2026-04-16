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

from app.config.settings import DATA_RAW_DIR, Settings
from app.llm.gemini_client import validate_gemini_api_key
from app.pipeline.rag_pipeline import RAGPipeline


st.set_page_config(page_title="Gemini + FAISS RAG", page_icon="📄", layout="wide")
st.title("Gemini + FAISS RAG Application")


def save_uploaded_files(uploaded_files: list) -> list[Path]:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    file_paths: list[Path] = []
    for uploaded in uploaded_files:
        path = DATA_RAW_DIR / uploaded.name
        path.write_bytes(uploaded.getbuffer())
        file_paths.append(path)
    return file_paths


settings = Settings.from_env()

if "gemini_key_status" not in st.session_state:
    st.session_state.gemini_key_status = validate_gemini_api_key(
        settings.gemini_api_key,
        settings.gemini_model,
    )

if "pipeline" not in st.session_state:
    try:
        st.session_state.pipeline = RAGPipeline(settings=settings)
    except Exception as exc:
        st.session_state.pipeline = None
        st.session_state.pipeline_error = str(exc)

pipeline: RAGPipeline | None = st.session_state.pipeline
is_key_valid, key_message = st.session_state.gemini_key_status

with st.sidebar:
    st.subheader("Gemini API Status")
    if is_key_valid:
        st.success(key_message)
    else:
        st.error(key_message)
        st.caption("Tip: update .env with a valid key, then restart Streamlit.")

    st.header("Ingestion")
    uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Build / Rebuild Index", use_container_width=True):
        if pipeline is None:
            st.error("Pipeline failed to initialize. Check your .env configuration.")
            if "pipeline_error" in st.session_state:
                st.caption(st.session_state.pipeline_error)
        elif not uploaded:
            st.warning("Please upload at least one PDF first.")
        else:
            with st.spinner("Processing PDFs and building index..."):
                paths = save_uploaded_files(uploaded)
                count = pipeline.ingest_pdfs(paths)
            st.success(f"Indexed {count} chunks.")

    if st.button("Load Existing Index", use_container_width=True):
        if pipeline is None:
            st.error("Pipeline failed to initialize. Check your .env configuration.")
            if "pipeline_error" in st.session_state:
                st.caption(st.session_state.pipeline_error)
        elif not is_key_valid:
            st.error("Cannot proceed: GEMINI_API_KEY is missing/invalid.")
        else:
            try:
                pipeline.load_index()
                st.success("Loaded existing FAISS index.")
            except FileNotFoundError:
                st.error("No persisted index found. Build one first.")

st.subheader("Ask a question")
question = st.text_input("Question", placeholder="What does the document say about ...?")

if st.button("Get Answer", type="primary", disabled=not is_key_valid):
    if not question.strip():
        st.warning("Please enter a question.")
    elif not is_key_valid:
        st.error("GEMINI_API_KEY is missing/invalid. Fix .env and restart Streamlit.")
    elif pipeline is None:
        st.error("Pipeline failed to initialize. Check your .env configuration.")
        if "pipeline_error" in st.session_state:
            st.caption(st.session_state.pipeline_error)
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

