# Groq + FAISS RAG Application

A Python RAG starter that combines:
- **Retrieval**: Sentence Transformers embeddings + FAISS
- **Generation**: Groq-hosted LLMs (default: `llama3-8b-8192`)
- **UI**: Streamlit

## Tech Stack
- Python
- Streamlit
- Groq API (`groq` SDK)
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
│   ├── llm/groq_client.py
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
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=llama3-8b-8192
   GROQ_FALLBACK_MODEL=mixtral-8x7b-32768
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
