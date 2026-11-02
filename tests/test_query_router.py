"""Tests for deterministic conversational query routing."""

import pytest

from agentic_rag.generation.query_router import conversational_response


@pytest.mark.parametrize(
    "question",
    [
        "Hi",
        "hi!",
        "HELLO",
        "Hello.",
        "What can you do?",
    ],
)
def test_recognized_conversation(question: str) -> None:
    response = conversational_response(question)

    assert isinstance(response, str)
    assert response.strip()


@pytest.mark.parametrize(
    "question",
    [
        "Who approves late-payment fee waivers?",
        "What is the late-payment fee?",
        "What are FinServe's mortgage rates?",
        "What's the weather tomorrow?",
        "Hi, what is the repayment fee?",
        "Please explain how to request a payment extension.",
    ],
)
def test_nonconversational_questions_are_not_intercepted(
    question: str,
) -> None:
    assert conversational_response(question) is None


def test_whitespace_and_punctuation() -> None:
    assert conversational_response("  HI!!!  ") is not None


def test_recognized_response_has_no_fabricated_citation() -> None:
    response = conversational_response("Hello")

    assert response is not None
    assert "[S1]" not in response
