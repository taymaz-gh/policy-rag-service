"""Tests for the PDF ingestion pipeline."""

from pathlib import Path

import pytest

from agentic_rag.ingestion.pdf_loader import load_and_chunk_pdfs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample_documents"


def test_pdf_ingestion_returns_documents():
    chunks = load_and_chunk_pdfs(SAMPLE_DIR)

    assert len(chunks) > 0
    assert all(chunk.page_content.strip() for chunk in chunks)


def test_pdf_chunks_preserve_filename():
    chunks = load_and_chunk_pdfs(SAMPLE_DIR)

    filenames = {chunk.metadata["filename"] for chunk in chunks}

    assert filenames == {
        "repayment_policy_2026.pdf",
        "fee_schedule_2026.pdf",
        "customer_support_policy_2026.pdf",
        "fee_schedule_2025.pdf",
    }


def test_pdf_ingestion_rejects_missing_directory(tmp_path):
    missing = tmp_path / "nonexistent"

    with pytest.raises(NotADirectoryError):
        load_and_chunk_pdfs(missing)


def test_pdf_ingestion_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        load_and_chunk_pdfs(
            SAMPLE_DIR,
            chunk_size=100,
            chunk_overlap=100,
        )
