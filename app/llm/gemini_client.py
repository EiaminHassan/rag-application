"""Gemini API wrapper for text generation."""

from __future__ import annotations

from typing import Tuple

from google import genai


DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
MODEL_FALLBACKS = (
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
)


def _normalize_model_name(model_name: str) -> str:
    if model_name.startswith("models/"):
        return model_name.split("/", 1)[1]
    return model_name


def _supports_generate_content(model_obj: object) -> bool:
    supported = getattr(model_obj, "supported_actions", None) or getattr(
        model_obj, "supported_generation_methods", None
    )
    if not supported:
        return True
    supported_lower = {str(item).lower() for item in supported}
    return "generatecontent" in supported_lower or "generate_content" in supported_lower


def _list_available_models(client: genai.Client) -> list[str]:
    names: list[str] = []
    for model_obj in client.models.list():
        raw_name = getattr(model_obj, "name", "")
        if not raw_name or not _supports_generate_content(model_obj):
            continue
        names.append(_normalize_model_name(raw_name))
    return names


def _resolve_model(client: genai.Client, requested_model: str) -> str:
    try:
        available = _list_available_models(client)
    except Exception:
        return requested_model or DEFAULT_GEMINI_MODEL

    if not available:
        return requested_model or DEFAULT_GEMINI_MODEL

    requested_clean = _normalize_model_name(requested_model) if requested_model else ""
    candidates = [requested_clean, *MODEL_FALLBACKS]
    for candidate in candidates:
        if candidate and candidate in available:
            return candidate

    for model in available:
        if "flash" in model:
            return model
    return available[0]


class GeminiClient:
    """Thin wrapper around Gemini text generation."""

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")
        self.client = genai.Client(api_key=api_key)
        self.model = _resolve_model(self.client, model)

    def generate(self, question: str, context: str) -> str:
        prompt = (
            "You are a helpful assistant for question answering over documents.\n"
            "Use only the supplied context. If the answer is not present, say so clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={"temperature": 0.2},
            )
        except Exception as exc:
            error_text = str(exc)
            if "RESOURCE_EXHAUSTED" in error_text or "quota" in error_text.lower():
                raise RuntimeError(
                    "Gemini quota exceeded. Check AI Studio quota/billing or wait and retry."
                ) from exc
            raise
        return (response.text or "").strip()


def validate_gemini_api_key(api_key: str, model: str) -> Tuple[bool, str]:
    """Validate Gemini API key with a lightweight API call."""
    key = (api_key or "").strip().strip("\"'")
    if not key:
        return False, "Missing GEMINI_API_KEY. Add it to .env and restart Streamlit."
    try:
        client = genai.Client(api_key=key)
        resolved_model = _resolve_model(client, model)
        _ = list(client.models.list())
    except Exception as exc:  # pragma: no cover - network/auth dependent
        return False, f"Invalid GEMINI_API_KEY or Gemini authentication failed: {exc}"
    if resolved_model != _normalize_model_name(model):
        return True, f"Gemini API key is valid. Using available model: {resolved_model}"
    return True, "Gemini API key is valid."
