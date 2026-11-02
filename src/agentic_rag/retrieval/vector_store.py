"""Vector indexing and date-aware policy retrieval."""

from datetime import date

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def create_embeddings() -> HuggingFaceEmbeddings:
    """Initialize the sentence embedding model on CPU."""
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
) -> Chroma:
    """Build an in-memory Chroma collection."""
    if not chunks:
        raise ValueError("Cannot index an empty document collection")

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="finserve_policies",
        collection_configuration={
            "hnsw": {"space": "cosine"},
        },
    )


def policy_date_filter(as_of: date) -> dict:
    """Build Chroma metadata filters for policy applicability."""
    date_number = int(as_of.strftime("%Y%m%d"))

    return {
        "$and": [
            {"effective_from_int": {"$lte": date_number}},
            {"effective_to_int": {"$gte": date_number}},
        ]
    }


def create_policy_retriever(
    vector_store: Chroma,
    as_of: date,
    k: int = 3,
):
    """Create a LangChain retriever with date filtering."""
    if k <= 0:
        raise ValueError("k must be positive")

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": policy_date_filter(as_of),
        },
    )
