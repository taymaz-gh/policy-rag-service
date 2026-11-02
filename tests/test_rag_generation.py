"""Tests for the guarded RAG generation service."""

import pytest
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai.chat_models import GoogleModelNotFoundError

from agentic_rag.generation.rag_service import RAGService
from agentic_rag.security.guardrails import screen_question


class StaticRetriever(BaseRetriever):
    """Return a predefined set of documents."""

    documents: list[Document]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return self.documents


def make_service(
    documents: list[Document],
    response: str = "Only a supervisor may approve the waiver [S1].",
) -> RAGService:
    return RAGService(
        retriever=StaticRetriever(documents=documents),
        llm=FakeListChatModel(responses=[response]),
    )


def sample_document() -> Document:
    return Document(
        page_content="Only supervisors may approve fee waivers.",
        metadata={"filename": "fee_schedule_2026.pdf", "page": 0},
    )


def test_grounded_answer_with_citation():
    service = make_service([sample_document()])

    result = service.ask("Who approves fee waivers?")

    assert result.status == "answered"
    assert "[S1]" in result.answer
    assert result.sources[0].filename == "fee_schedule_2026.pdf"


def test_empty_retrieval_abstains():
    service = make_service([])

    result = service.ask("What is the applicable fee?")

    assert result.status == "insufficient_evidence"
    assert result.sources == []


def test_invalid_citation_is_rejected():
    service = make_service(
        [sample_document()],
        response="A manager may approve it [S9].",
    )

    result = service.ask("Who approves fee waivers?")

    assert result.status == "citation_validation_failed"


def test_sensitive_question_is_blocked():
    service = make_service([sample_document()])

    result = service.ask("My email is customer@example.com. Can I get a waiver?")

    assert result.status == "blocked"
    assert result.sources == []


def test_iban_is_detected():
    result = screen_question("Please check NL91 ABNA 0417 1643 00.")

    assert not result.allowed


def test_non_sensitive_question_passes():
    result = screen_question("Who can authorize a repayment extension?")

    assert result.allowed


def test_retrieval_only_without_llm():
    """Return retrieved evidence when no LLM is configured."""
    service = RAGService(
        retriever=StaticRetriever(documents=[sample_document()]),
        llm=None,
    )

    result = service.ask("Who approves fee waivers?")

    assert result.status == "retrieval_only"
    assert "fee_schedule_2026.pdf" in result.answer
    assert "[S1]" in result.answer
    assert len(result.sources) == 1


def test_model_unavailable_falls_back(monkeypatch):
    """Return retrieved evidence when the provider model is missing."""
    service = make_service([sample_document()])

    def unavailable(*args, **kwargs):
        raise GoogleModelNotFoundError("Test model unavailable")

    monkeypatch.setattr(type(service.llm), "invoke", unavailable)

    result = service.ask("Who approves fee waivers?")

    assert result.status == "retrieval_only"
    assert "fee_schedule_2026.pdf" in result.answer


def test_unavailable_model_raises_when_fallback_disabled(monkeypatch):
    """Preserve provider errors when automatic fallback is disabled."""
    service = make_service([sample_document()])
    service.fallback_enabled = False

    def unavailable(*args, **kwargs):
        raise GoogleModelNotFoundError("Test model unavailable")

    monkeypatch.setattr(type(service.llm), "invoke", unavailable)

    with pytest.raises(GoogleModelNotFoundError):
        service.ask("Who approves fee waivers?")
