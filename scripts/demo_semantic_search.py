"""Demonstrate semantic search over fictional financial policies."""

from datetime import date
from pathlib import Path

from agentic_rag.ingestion.pdf_loader import load_and_chunk_pdfs
from agentic_rag.retrieval.vector_store import (
    build_vector_store,
    create_embeddings,
    create_policy_retriever,
)

ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    # Load PDF chunks with policy metadata.
    chunks = load_and_chunk_pdfs(
        ROOT / "data" / "sample_documents",
        manifest_path=ROOT / "data" / "document_manifest.json",
    )

    # Initialize a pretrained sentence embedding model on CPU.
    embeddings = create_embeddings()

    # Verify the embedding representation.
    sample_vector = embeddings.embed_query("Who can approve a late-payment fee waiver?")
    print(f"\nEmbedding dimension: {len(sample_vector)}")

    # Build an in-memory Chroma collection for this experiment.
    vector_store = build_vector_store(chunks, embeddings)

    # Create the LangChain retrieval interface.
    retriever = create_policy_retriever(
        vector_store,
        as_of=date(2026, 11, 1),
        k=3,
    )

    question = "Who is authorized to approve a late payment fee waiver?"

    results = retriever.invoke(question)

    print(f"\nQuestion: {question}")
    print(f"Retrieved chunks: {len(results)}")

    for rank, document in enumerate(results, start=1):
        metadata = document.metadata

        print(f"\n--- Result {rank} ---")
        print(f"File: {metadata['filename']}")
        print(f"Policy: {metadata['policy_id']}")
        print(f"Version: {metadata['version']}")
        print(f"Page: {metadata['page'] + 1}")
        print(f"Text:\n{document.page_content}")


if __name__ == "__main__":
    main()
