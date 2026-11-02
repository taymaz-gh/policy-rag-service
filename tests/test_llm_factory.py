"""Tests for LLM provider configuration."""

import pytest

from agentic_rag.generation.llm_factory import create_chat_model


def test_missing_gemini_key(monkeypatch):
    monkeypatch.setenv("RAG_LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        create_chat_model()


def test_unsupported_provider(monkeypatch):
    monkeypatch.setenv("RAG_LLM_PROVIDER", "unsupported")

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_chat_model()
