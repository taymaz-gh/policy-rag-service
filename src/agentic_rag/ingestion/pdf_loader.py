"""Load and chunk PDF documents for retrieval."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agentic_rag.ingestion.metadata import (
    enrich_document_metadata,
    load_document_manifest,
)


def load_and_chunk_pdfs(
    directory: str | Path,
    chunk_size: int = 500,
    chunk_overlap: int = 75,
    manifest_path: str | Path | None = None,
) -> list[Document]:
    """Load PDFs and return chunks with source metadata."""

    directory = Path(directory)

    if not directory.is_dir():
        raise NotADirectoryError(f"PDF directory not found: {directory}")

    pdf_files = sorted(directory.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in: {directory}")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")

    pages: list[Document] = []

    for pdf_path in pdf_files:
        loader = PyMuPDF4LLMLoader(
            file_path=str(pdf_path),
            mode="page",
        )

        loaded_pages = loader.load()

        for page in loaded_pages:
            page.metadata["filename"] = pdf_path.name

        pages.extend(loaded_pages)

    if manifest_path is not None:
        manifest = load_document_manifest(manifest_path)
        pages = enrich_document_metadata(pages, manifest)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    chunks = splitter.split_documents(pages)

    # Report ingestion results.
    print("\nPDF ingestion report")
    print("-" * 40)
    print(f"PDF files:       {len(pdf_files)}")
    print(f"Loaded pages:    {len(pages)}")
    print(f"Text chunks:     {len(chunks)}")
    print(f"Chunk size:      {chunk_size}")
    print(f"Chunk overlap:   {chunk_overlap}")
    print(f"Total chars:     {sum(len(p.page_content) for p in pages)}")
    print("-" * 40)

    return chunks
