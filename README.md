# Gemini + FAISS RAG Application

A Python RAG starter that combines:
- **Retrieval**: Sentence Transformers embeddings + FAISS
- **Generation**: Google Gemini API (default: `gemini-2.0-flash`)
- **UI**: Streamlit

## Tech Stack
- Python
- Streamlit
- Gemini API (`google-genai` SDK)
- `sentence-transformers` (`all-MiniLM-L6-v2`)
- FAISS (`faiss-cpu`)
- PDF parsing (`pdfplumber` with `PyPDF2` fallback)

## Project Structure
```text
RAG-APPLICATION/
├── app/
│   ├── main.py
│   ├── config/settings.py
│   ├── ingestion/{pdf_loader.py,chunking.py}
│   ├── embeddings/embedder.py
│   ├── vectorstore/faiss_store.py
│   ├── retriever/retriever.py
│   ├── llm/gemini_client.py
│   ├── pipeline/rag_pipeline.py
│   └── utils/logger.py
├── data/{raw,processed}
├── vectorstore/faiss_index
├── tests/test_pipeline.py
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Setup
1. Create/update `.env`:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-2.0-flash
   EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   TOP_K=4
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run app:
   ```bash
   streamlit run app/main.py
   ```

## Usage
1. Upload one or more PDFs in the sidebar.
2. Click **Build / Rebuild Index**.
3. Ask a question in the main panel.
4. Inspect retrieved chunks and similarity scores.

## Tests
```bash
pytest -q
```

## Docker
```bash
docker compose up --build
```
