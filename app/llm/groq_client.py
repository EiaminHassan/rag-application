"""Groq API wrapper for text generation."""

from __future__ import annotations

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

