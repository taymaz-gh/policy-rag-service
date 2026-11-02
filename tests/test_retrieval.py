"""Tests for policy-date retrieval filtering."""

from datetime import date
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from agentic_rag.retrieval.vector_store import (
    build_vector_store,
    create_policy_retriever,
    policy_date_filter,
)


def test_policy_filter_2026():
    result = policy_date_filter(date(2026, 11, 1))

    assert result == {
        "$and": [
            {"effective_from_int": {"$lte": 20261101}},
            {"effective_to_int": {"$gte": 20261101}},
        ]
    }


@pytest.mark.parametrize(
    ("as_of", "expected"),
    [
        (date(2025, 1, 1), 20250101),
        (date(2025, 12, 31), 20251231),
        (date(2026, 1, 1), 20260101),
    ],
)
def test_policy_filter_dates(as_of, expected):
    result = policy_date_filter(as_of)

    assert result["$and"][0] == {"effective_from_int": {"$lte": expected}}


"""Integration tests for policy-aware Chroma retrieval."""


class TestEmbeddings(Embeddings):
    """Small deterministic embeddings for testing retrieval logic."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        text = text.lower()
        return [
            float("fee" in text),
            float("waiver" in text),
            1.0,
        ]


def test_chroma_excludes_superseded_policies():
    documents = [
        Document(
            page_content="The late-payment fee was EUR 20.",
            metadata={
                "filename": "fee_schedule_2025.pdf",
                "version": "2025",
                "effective_from_int": 20250101,
                "effective_to_int": 20251231,
            },
        ),
        Document(
            page_content="The late-payment fee is EUR 12.",
            metadata={
                "filename": "fee_schedule_2026.pdf",
                "version": "2026",
                "effective_from_int": 20260101,
                "effective_to_int": 99991231,
            },
        ),
    ]

    store = build_vector_store(documents, TestEmbeddings())

    from datetime import date

    cases = [
        (date(2025, 6, 15), {"fee_schedule_2025.pdf"}),
        (date(2026, 11, 1), {"fee_schedule_2026.pdf"}),
        (date(2024, 6, 15), set()),
    ]

    for as_of, expected_filenames in cases:
        retriever = create_policy_retriever(
            store,
            as_of=as_of,
            k=3,
        )

        results = retriever.invoke("What is the late-payment fee?")

        retrieved = {document.metadata["filename"] for document in results}

        assert retrieved == expected_filenames


class DeterministicEmbeddings(Embeddings):
    """Simple embeddings for testing retrieval infrastructure."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        text = text.lower()
        return [
            float("fee" in text),
            float("waiver" in text),
            1.0,
        ]


def test_chroma_policy_date_filter():
    """Verify that Chroma retrieves only date-applicable policies."""

    documents = [
        Document(
            page_content="The late-payment fee was EUR 20.",
            metadata={
                "filename": "fee_schedule_2025.pdf",
                "version": "2025",
                "effective_from_int": 20250101,
                "effective_to_int": 20251231,
            },
        ),
        Document(
            page_content="The late-payment fee is EUR 12.",
            metadata={
                "filename": "fee_schedule_2026.pdf",
                "version": "2026",
                "effective_from_int": 20260101,
                "effective_to_int": 99991231,
            },
        ),
    ]

    store = build_vector_store(documents, DeterministicEmbeddings())

    cases = [
        (date(2025, 6, 15), {"fee_schedule_2025.pdf"}),
        (date(2026, 11, 1), {"fee_schedule_2026.pdf"}),
        (date(2024, 6, 15), set()),
    ]

    for as_of, expected in cases:
        retriever = create_policy_retriever(
            store,
            as_of=as_of,
            k=3,
        )

        results = retriever.invoke("What is the late-payment fee?")

        actual = {doc.metadata["filename"] for doc in results}

        assert actual == expected, f"Date {as_of}: expected {expected}, got {actual}"


def test_retriever_without_threshold_uses_similarity() -> None:
    """The existing baseline must remain unchanged by default."""
    store = MagicMock()

    create_policy_retriever(
        vector_store=store,
        as_of=date(2026, 11, 2),
        k=3,
    )

    store.as_retriever.assert_called_once()

    kwargs = store.as_retriever.call_args.kwargs

    assert kwargs["search_type"] == "similarity"
    assert kwargs["search_kwargs"]["k"] == 3
    assert "score_threshold" not in kwargs["search_kwargs"]
    assert "filter" in kwargs["search_kwargs"]


def test_retriever_with_threshold_uses_score_filtering() -> None:
    """Enabling a threshold must preserve the policy-date filter."""
    store = MagicMock()

    create_policy_retriever(
        vector_store=store,
        as_of=date(2026, 11, 2),
        k=3,
        score_threshold=0.75,
    )

    kwargs = store.as_retriever.call_args.kwargs

    assert kwargs["search_type"] == "similarity_score_threshold"
    assert kwargs["search_kwargs"]["score_threshold"] == 0.75
    assert kwargs["search_kwargs"]["k"] == 3
    assert kwargs["search_kwargs"]["filter"] == policy_date_filter(date(2026, 11, 2))


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_invalid_relevance_threshold_is_rejected(
    threshold: float,
) -> None:
    store = MagicMock()

    with pytest.raises(ValueError, match="score_threshold"):
        create_policy_retriever(
            vector_store=store,
            as_of=date(2026, 11, 2),
            score_threshold=threshold,
        )

    store.as_retriever.assert_not_called()
