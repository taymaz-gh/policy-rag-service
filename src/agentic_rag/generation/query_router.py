"""Deterministic routing for simple conversational messages."""

import re


def conversational_response(question: str) -> str | None:
    """Return a fixed response for recognized conversational messages."""
    normalized = re.sub(r"[.!?]+$", "", question.strip().lower())

    responses = {
        "hi": "Hello! I can help you find information in FinServe's sample financial policies.",
        "hello": "Hello! Ask me about FinServe's repayment rules, fees, or support procedures.",
        "thanks": "You're welcome.",
        "thank you": "You're welcome.",
        "what can you do": (
            "I can answer questions about the available synthetic "
            "FinServe policies and cite the supporting documents."
        ),
    }

    return responses.get(normalized)
