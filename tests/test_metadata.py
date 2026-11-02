"""Tests for policy metadata enrichment."""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from agentic_rag.ingestion.metadata import (
    enrich_document_metadata,
    load_document_manifest,
)
from agentic_rag.ingestion.pdf_loader import load_and_chunk_pdfs

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "document_manifest.json"
PDF_DIR = ROOT / "data" / "sample_documents"


def test_manifest_loads():
    manifest = load_document_manifest(MANIFEST)

    assert len(manifest) == 4
    assert manifest["fee_schedule_2025.pdf"]["version"] == "2025"


def test_enrichment_preserves_source_and_page():
    manifest = load_document_manifest(MANIFEST)

    source = Document(
        page_content="A fictional fee policy.",
        metadata={
            "filename": "fee_schedule_2026.pdf",
            "page": 0,
            "source": "example.pdf",
        },
    )

    result = enrich_document_metadata([source], manifest)

    assert result[0].metadata["page"] == 0
    assert result[0].metadata["source"] == "example.pdf"
    assert result[0].metadata["policy_id"] == "fee_schedule"
    assert result[0].metadata["effective_from"] == "2026-01-01"
    assert "effective_to" not in result[0].metadata

    # Original input remains unchanged.
    assert "policy_id" not in source.metadata


def test_missing_manifest_entry_raises():
    manifest = load_document_manifest(MANIFEST)

    source = Document(
        page_content="Unknown document",
        metadata={"filename": "unknown.pdf"},
    )

    with pytest.raises(ValueError, match="missing from manifest"):
        enrich_document_metadata([source], manifest)


def test_policy_metadata_survives_chunking():
    chunks = load_and_chunk_pdfs(
        PDF_DIR,
        manifest_path=MANIFEST,
    )

    old_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata["filename"] == "fee_schedule_2025.pdf"
    ]

    assert old_chunks
    assert all(chunk.metadata["effective_to"] == "2025-12-31" for chunk in old_chunks)
    assert all(chunk.metadata["policy_id"] == "fee_schedule" for chunk in old_chunks)
