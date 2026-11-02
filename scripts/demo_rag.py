"""Run an end-to-end RAG question using synthetic FinServe policies."""

import logging

logger = logging.getLogger(__name__)

from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from agentic_rag.generation.llm_factory import create_chat_model
from agentic_rag.generation.rag_service import RAGService
from agentic_rag.ingestion.pdf_loader import load_and_chunk_pdfs
from agentic_rag.retrieval.vector_store import (
    build_vector_store,
    create_embeddings,
    create_policy_retriever,
)
from agentic_rag.security.guardrails import screen_question

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    load_dotenv(ROOT / ".env")

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    question = "Who is authorized to approve a late-payment fee waiver?"

    # Screen before initializing retrieval or contacting a provider.
    screening = screen_question(question)

    if not screening.allowed:
        print(f"Request blocked: {screening.reason}")
        return

    chunks = load_and_chunk_pdfs(
        ROOT / "data" / "sample_documents",
        manifest_path=ROOT / "data" / "document_manifest.json",
    )

    embeddings = create_embeddings()
    store = build_vector_store(chunks, embeddings)

    retriever = create_policy_retriever(
        store,
        as_of=date(2026, 11, 1),
        k=3,
    )

    try:
        llm = create_chat_model()
    except ValueError as exc:
        if "GOOGLE_API_KEY is required" not in str(exc):
            raise

        logger.warning(
            "LLM credentials are not configured. Starting in retrieval-only mode."
        )
        llm = None

    service = RAGService(
        retriever=retriever,
        llm=llm,
        fallback_enabled=True,
    )

    result = service.ask(question)

    print(f"\nStatus: {result.status}")
    print(f"\nAnswer:\n{result.answer}")

    print("\nRetrieved sources:")
    for source in result.sources:
        print(f"[{source.id}] {source.filename}, page {source.page}")


if __name__ == "__main__":
    main()
