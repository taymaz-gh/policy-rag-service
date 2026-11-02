"""Calibrate a retrieval threshold using development and held-out questions."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from agentic_rag.ingestion.pdf_loader import load_and_chunk_pdfs
from agentic_rag.retrieval.vector_store import (
    build_vector_store,
    create_embeddings,
    policy_date_filter,
)

ROOT = Path(__file__).resolve().parents[1]

CURRENT = date(2026, 11, 2)
HISTORICAL = date(2025, 6, 15)


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    as_of: date
    category: str
    # A positive label means the knowledge base contains the answer.
    answerable: bool


@dataclass(frozen=True)
class ScoredCase:
    case: EvaluationCase
    score: float | None
    source: str | None


# Development data: the 12 questions used in the initial experiment.
DEVELOPMENT_CASES = [
    EvaluationCase(
        "Who can approve late-payment fee waivers?",
        CURRENT,
        "supported",
        True,
    ),
    EvaluationCase(
        "What is the late-payment administration fee?",
        CURRENT,
        "supported",
        True,
    ),
    EvaluationCase(
        "What is the late-payment administration fee?",
        HISTORICAL,
        "supported",
        True,
    ),
    EvaluationCase(
        "What happens after two consecutive missed installments?",
        CURRENT,
        "supported",
        True,
    ),
    EvaluationCase(
        "Can support agents approve payment extensions?",
        CURRENT,
        "supported",
        True,
    ),
    EvaluationCase(
        "What are FinServe's mortgage interest rates?",
        CURRENT,
        "unsupported",
        False,
    ),
    EvaluationCase(
        "What is the maximum personal loan amount?",
        CURRENT,
        "unsupported",
        False,
    ),
    EvaluationCase(
        "Does FinServe provide investment advice?",
        CURRENT,
        "unsupported",
        False,
    ),
    EvaluationCase(
        "What is the weather tomorrow?",
        CURRENT,
        "out_of_scope",
        False,
    ),
    EvaluationCase(
        "Write a Python function to sort numbers.",
        CURRENT,
        "out_of_scope",
        False,
    ),
    EvaluationCase(
        "Who won the football World Cup?",
        CURRENT,
        "out_of_scope",
        False,
    ),
    EvaluationCase(
        "What is the capital of France?",
        CURRENT,
        "out_of_scope",
        False,
    ),
]


# Held-out data: not used to choose the threshold.
TEST_CASES = [
    EvaluationCase(
        "Are agents allowed to promise customers a fee waiver?",
        CURRENT,
        "supported",
        True,
    ),
    EvaluationCase(
        "Who handles cases involving disputed charges?",
        CURRENT,
        "supported",
        True,
    ),
    EvaluationCase(
        "Can support staff guarantee loan approval?",
        CURRENT,
        "supported",
        True,
    ),
    EvaluationCase(
        "May agents ask customers for their PIN in chat?",
        CURRENT,
        "supported",
        True,
    ),
    EvaluationCase(
        "How much did a missed-installment charge cost in 2025?",
        HISTORICAL,
        "supported",
        True,
    ),
    EvaluationCase(
        "What is the fee for an international transfer?",
        CURRENT,
        "unsupported",
        False,
    ),
    EvaluationCase(
        "How many days can a payment extension last?",
        CURRENT,
        "unsupported",
        False,
    ),
    EvaluationCase(
        "What is the maximum number of fee waivers per year?",
        CURRENT,
        "unsupported",
        False,
    ),
    EvaluationCase(
        "Explain the theory of general relativity.",
        CURRENT,
        "out_of_scope",
        False,
    ),
    EvaluationCase(
        "Recommend a recipe for dinner.",
        CURRENT,
        "out_of_scope",
        False,
    ),
    EvaluationCase(
        "Translate this sentence into French.",
        CURRENT,
        "out_of_scope",
        False,
    ),
    EvaluationCase(
        "How do I install a Python package?",
        CURRENT,
        "out_of_scope",
        False,
    ),
]


def score_cases(store, cases: list[EvaluationCase]) -> list[ScoredCase]:
    """Retrieve the closest date-applicable chunk for each question."""
    scored = []

    for case in cases:
        results = store.similarity_search_with_score(
            query=case.question,
            k=1,
            filter=policy_date_filter(case.as_of),
        )

        if results:
            document, cosine_distance = results[0]
            scored.append(
                ScoredCase(
                    case=case,
                    score=1.0 - float(cosine_distance),
                    source=document.metadata.get("filename"),
                )
            )
        else:
            scored.append(ScoredCase(case=case, score=None, source=None))

    return scored


def confusion_counts(
    rows: list[ScoredCase],
    threshold: float,
) -> tuple[int, int, int, int]:
    """Return TP, FP, TN, FN for retrieval acceptance."""
    tp = fp = tn = fn = 0

    for row in rows:
        predicted = row.score is not None and row.score >= threshold
        actual = row.case.answerable

        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and not actual:
            tn += 1
        else:
            fn += 1

    return tp, fp, tn, fn


def balanced_accuracy(counts: tuple[int, int, int, int]) -> float:
    """Average supported-question recall and unsupported rejection."""
    tp, fp, tn, fn = counts

    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    return (sensitivity + specificity) / 2


def select_threshold(rows: list[ScoredCase]) -> float:
    """Select threshold using development data only.

    Maximize balanced accuracy, then favor fewer false acceptances.
    """
    candidates = [i / 100 for i in range(101)]

    def objective(threshold: float) -> tuple[float, int, float]:
        counts = confusion_counts(rows, threshold)
        return (
            balanced_accuracy(counts),
            -counts[1],  # Prefer fewer false positives.
            threshold,  # Deterministic tie breaker.
        )

    return max(candidates, key=objective)


def print_scores(name: str, rows: list[ScoredCase]) -> None:
    print(f"\n{name} — individual scores")
    print("-" * 90)

    for row in rows:
        value = "N/A" if row.score is None else f"{row.score:.4f}"

        print(
            f"{row.case.category:13} "
            f"score={value:>7} "
            f"answerable={row.case.answerable!s:5} | "
            f"{row.case.question}"
        )


def print_metrics(
    name: str,
    rows: list[ScoredCase],
    threshold: float,
) -> None:
    tp, fp, tn, fn = confusion_counts(rows, threshold)

    print(f"\n{name} at threshold={threshold:.2f}")
    print("-" * 50)
    print(f"True positives:  {tp}")
    print(f"False positives: {fp}")
    print(f"True negatives:  {tn}")
    print(f"False negatives: {fn}")
    print(f"Balanced accuracy: {balanced_accuracy((tp, fp, tn, fn)):.3f}")


def main() -> None:
    chunks = load_and_chunk_pdfs(
        ROOT / "data" / "sample_documents",
        manifest_path=ROOT / "data" / "document_manifest.json",
    )

    store = build_vector_store(chunks, create_embeddings())

    development = score_cases(store, DEVELOPMENT_CASES)
    held_out = score_cases(store, TEST_CASES)

    print_scores("Development", development)
    print_scores("Held-out test", held_out)

    threshold = select_threshold(development)

    print(f"\nSelected development threshold: {threshold:.2f}")

    print_metrics("Development", development, threshold)
    print_metrics("Held-out test", held_out, threshold)

    print("\nErrors on held-out questions:")
    errors = 0

    for row in held_out:
        predicted = row.score is not None and row.score >= threshold

        if predicted != row.case.answerable:
            errors += 1
            print(
                f"  expected={row.case.answerable}, "
                f"score={row.score}, "
                f"source={row.source}, "
                f"question={row.case.question}"
            )

    if errors == 0:
        print("  No classification errors on this small test set.")


if __name__ == "__main__":
    main()
