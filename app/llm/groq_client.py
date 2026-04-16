"""Groq API wrapper for text generation."""

from __future__ import annotations

from typing import Tuple

from groq import Groq


class GroqClient:
    """Thin wrapper around Groq chat completions."""

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("GROQ_API_KEY is required")
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, question: str, context: str) -> str:
        prompt = (
            "You are a helpful assistant for question answering over documents.\n"
            "Use only the supplied context. If the answer is not present, say so clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""


def validate_groq_api_key(api_key: str) -> Tuple[bool, str]:
    """
    Validate Groq API key with format and a lightweight API call.
    """
    key = (api_key or "").strip().strip("\"'")
    if not key:
        return False, "Missing GROQ_API_KEY. Add it to .env and restart Streamlit."
    if not key.startswith("gsk_"):
        return False, "GROQ_API_KEY format looks invalid (expected to start with 'gsk_')."
    try:
        client = Groq(api_key=key)
        client.models.list()
    except Exception as exc:  # pragma: no cover - network/auth dependent
        return False, f"Invalid GROQ_API_KEY or Groq authentication failed: {exc}"
    return True, "Groq API key is valid."

