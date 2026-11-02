"""Tests for retrieval threshold calibration."""

from datetime import date

import pytest

from scripts.evaluate_retrieval import (
    EvaluationCase,
    ScoredCase,
    balanced_accuracy,
    confusion_counts,
    select_threshold,
)


def make_row(score: float, answerable: bool) -> ScoredCase:
    return ScoredCase(
        case=EvaluationCase(
            question="Synthetic test question",
            as_of=date(2026, 11, 2),
            category="supported" if answerable else "unsupported",
            answerable=answerable,
        ),
        score=score,
        source="test.pdf",
    )


def test_confusion_counts() -> None:
    rows = [
        make_row(0.90, True),
        make_row(0.40, True),
        make_row(0.60, False),
        make_row(0.10, False),
    ]

    assert confusion_counts(rows, threshold=0.50) == (1, 1, 1, 1)


def test_balanced_accuracy() -> None:
    assert balanced_accuracy((4, 0, 6, 0)) == pytest.approx(1.0)
    assert balanced_accuracy((2, 3, 3, 2)) == pytest.approx(0.5)


def test_threshold_separates_clear_examples() -> None:
    rows = [
        make_row(0.90, True),
        make_row(0.80, True),
        make_row(0.20, False),
        make_row(0.10, False),
    ]

    threshold = select_threshold(rows)

    assert confusion_counts(rows, threshold) == (2, 0, 2, 0)


def test_no_retrieved_document_means_rejection() -> None:
    row = ScoredCase(
        case=EvaluationCase(
            question="Question with no date-applicable policy",
            as_of=date(2024, 6, 15),
            category="unsupported",
            answerable=False,
        ),
        score=None,
        source=None,
    )

    assert confusion_counts([row], threshold=0.50) == (0, 0, 1, 0)
