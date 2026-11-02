"""Evaluate live RAG responses against labeled synthetic policy questions."""

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal

from dotenv import load_dotenv

from agentic_rag.generation.llm_factory import create_chat_model
from agentic_rag.generation.rag_service import RAGService
from agentic_rag.ingestion.pdf_loader import load_and_chunk_pdfs
from agentic_rag.retrieval.vector_store import (
    build_vector_store,
    create_embeddings,
    create_policy_retriever,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "evaluation" / "runs"

Category = Literal["supported", "unsupported", "out_of_scope"]


@dataclass(frozen=True)
class AnswerCase:
    case_id: str
    question: str
    as_of: date
    category: Category
    expected_behavior: str


CASES = [
    AnswerCase(
        "SUP01",
        "Who can approve late-payment fee waivers?",
        date(2026, 11, 2),
        "supported",
        "Authorized supervisor; agents may submit requests.",
    ),
    AnswerCase(
        "SUP02",
        "What is the late-payment administration fee?",
        date(2026, 11, 2),
        "supported",
        "EUR 12 per qualifying missed installment.",
    ),
    AnswerCase(
        "SUP03",
        "What was the late-payment administration fee in 2025?",
        date(2025, 6, 15),
        "supported",
        "EUR 20 per qualifying missed installment.",
    ),
    AnswerCase(
        "SUP04",
        "Who handles cases involving disputed charges?",
        date(2026, 11, 2),
        "supported",
        "Referral to the appropriate specialist team.",
    ),
    AnswerCase(
        "SUP05",
        "What happens after two consecutive missed installments?",
        date(2026, 11, 2),
        "supported",
        "Referral to the repayment assistance team.",
    ),
    AnswerCase(
        "UNS01",
        "How many days can a payment extension last?",
        date(2026, 11, 2),
        "unsupported",
        "Abstain: no extension duration specified.",
    ),
    AnswerCase(
        "UNS02",
        "What are FinServe's mortgage interest rates?",
        date(2026, 11, 2),
        "unsupported",
        "Abstain: no mortgage interest rates specified.",
    ),
    AnswerCase(
        "UNS03",
        "What is the maximum number of fee waivers per year?",
        date(2026, 11, 2),
        "unsupported",
        "Abstain: no annual maximum specified.",
    ),
    AnswerCase(
        "UNS04",
        "How much does an international bank transfer cost?",
        date(2026, 11, 2),
        "unsupported",
        "Abstain: no international-transfer fee specified.",
    ),
    AnswerCase(
        "OOS01",
        "What is the capital of France?",
        date(2026, 11, 2),
        "out_of_scope",
        "Decline or abstain: unrelated to the policy corpus.",
    ),
    AnswerCase(
        "OOS02",
        "How do I install a Python package?",
        date(2026, 11, 2),
        "out_of_scope",
        "Decline or abstain: unrelated to the policy corpus.",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate grounded RAG responses.")
    parser.add_argument(
        "--cases",
        nargs="+",
        help="Optional evaluation case IDs to run, e.g. UNSUP02 OOS01",
    )
    args = parser.parse_args()

    selected_cases = (
        [case for case in CASES if case.case_id in args.cases] if args.cases else CASES
    )

    if args.cases:
        unknown = set(args.cases) - {case.case_id for case in CASES}
        if unknown:
            parser.error(f"Unknown evaluation case IDs: {sorted(unknown)}")

    load_dotenv(ROOT / ".env")

    chunks = load_and_chunk_pdfs(
        ROOT / "data" / "sample_documents",
        manifest_path=ROOT / "data" / "document_manifest.json",
    )

    store = build_vector_store(chunks, create_embeddings())

    # Fail explicitly if the live model is not configured.
    llm = create_chat_model()
    records = []

    for case in selected_cases:
        retriever = create_policy_retriever(
            store,
            as_of=case.as_of,
            k=3,
            # Keep the threshold disabled for baseline evaluation.
            score_threshold=None,
        )

        service = RAGService(
            retriever=retriever,
            llm=llm,
            fallback_enabled=False,
        )

        started = perf_counter()
        error_type = None
        error_message = None

        try:
            result = service.ask(case.question)
            elapsed = perf_counter() - started
            status = result.status
            answer = result.answer
            sources = [asdict(source) for source in result.sources]
        except Exception as exc:  # noqa: BLE001
            # Evaluation boundary: record unexpected failures so that
            # subsequent test cases can still be evaluated.
            # Never classify these failures as successful responses.
            elapsed = perf_counter() - started
            error_type = type(exc).__name__
            error_message = str(exc)[:500]
            status = "provider_or_runtime_error"
            answer = ""
            sources = []

        record = {
            "case_id": case.case_id,
            "question": case.question,
            "as_of": case.as_of.isoformat(),
            "category": case.category,
            "expected_behavior": case.expected_behavior,
            "status": status,
            "answer": answer,
            "sources": sources,
            "latency_seconds": round(elapsed, 3),
            "error_type": error_type,
            "error_message": error_message,
            "manual_review": {
                "behavior_correct": None,
                "claims_supported": None,
                "citations_support_claims": None,
                "notes": "",
            },
        }
        records.append(record)

        print(f"\n[{case.case_id}] {case.category} |status={status} | {elapsed:.2f}s")
        print(f"Q: {case.question}")
        print(f"A: {answer or '(no answer)'}")
        if error_type:
            print(f"Error type: {error_type}")
            print(f"Error message: {error_message}")

    run_timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    run_type = "targeted" if args.cases else "full"

    output = OUTPUT_DIR / f"rag_answers_{run_type}_{run_timestamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSaved {len(records)} records to: {output}")
    print("Manual review fields remain unset.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
