"""PDF loading logic with parser fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pdfplumber
from PyPDF2 import PdfReader


def _extract_with_pdfplumber(pdf_path: Path) -> str:
    chunks: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                chunks.append(text)
    return "\n".join(chunks).strip()


def _extract_with_pypdf2(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    chunks: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            chunks.append(text)
    return "\n".join(chunks).strip()


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract text from a single PDF path."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    try:
        text = _extract_with_pdfplumber(path)
    except Exception:
        text = ""
    if not text:
        text = _extract_with_pypdf2(path)
    return text


def load_texts_from_pdfs(pdf_paths: Iterable[str | Path]) -> list[str]:
    """Extract text from multiple PDFs."""
    return [extract_text_from_pdf(path) for path in pdf_paths]

