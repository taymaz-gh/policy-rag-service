"""FastAPI application for the FinServe RAG service."""

import logging
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from agentic_rag.generation.llm_factory import create_chat_model
from agentic_rag.generation.rag_service import RAGService
from agentic_rag.ingestion.pdf_loader import load_and_chunk_pdfs
from agentic_rag.retrieval.vector_store import (
    build_vector_store,
    create_embeddings,
    create_policy_retriever,
)
from agentic_rag.security.guardrails import screen_question

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]

Status = Literal[
    "answered",
    "retrieval_only",
    "insufficient_evidence",
    "blocked",
    "citation_validation_failed",
    "conversational",
]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    as_of: date = Field(default_factory=date.today)


class SourceResponse(BaseModel):
    id: str
    filename: str
    page: int


class AskResponse(BaseModel):
    status: Status
    answer: str
    sources: list[SourceResponse]


class ReadinessResponse(BaseModel):
    ready: bool
    generation_mode: Literal["gemini", "retrieval_only", "unavailable"]


def initialize_components(app: FastAPI) -> None:
    """Load retrieval resources once per server process."""
    chunks = load_and_chunk_pdfs(
        ROOT / "data" / "sample_documents",
        manifest_path=ROOT / "data" / "document_manifest.json",
    )

    embeddings = create_embeddings()
    app.state.vector_store = build_vector_store(chunks, embeddings)

    try:
        app.state.llm = create_chat_model()
    except ValueError as exc:
        if "GOOGLE_API_KEY is required" not in str(exc):
            raise

        logger.warning("Gemini key missing; retrieval-only mode enabled.")
        app.state.llm = None


def create_app(initialize: bool = True) -> FastAPI:
    """Construct an API application with optional startup initialization."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.vector_store = None
        app.state.llm = None

        if initialize:
            try:
                await run_in_threadpool(initialize_components, app)
            except Exception:
                logger.exception("RAG initialization failed.")
                app.state.vector_store = None

        yield

    app = FastAPI(
        title="FinServe Agentic RAG API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", response_model=ReadinessResponse)
    def ready() -> ReadinessResponse:
        store = app.state.vector_store

        if store is None:
            return ReadinessResponse(
                ready=False,
                generation_mode="unavailable",
            )

        return ReadinessResponse(
            ready=True,
            generation_mode=(
                "gemini" if app.state.llm is not None else "retrieval_only"
            ),
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        if app.state.vector_store is None:
            raise HTTPException(
                status_code=503,
                detail="Retrieval service is not ready.",
            )

        retriever = create_policy_retriever(
            app.state.vector_store,
            as_of=request.as_of,
            k=3,
        )
        service = RAGService(
            retriever=retriever,
            llm=app.state.llm,
            fallback_enabled=True,
        )

        result = service.ask(request.question)

        # Conservative output screening. Do not return sources if blocked.
        if result.status in {"answered", "retrieval_only"}:
            output_screening = screen_question(result.answer)
            if not output_screening.allowed:
                logger.warning("RAG response failed output screening.")
                return AskResponse(
                    status="blocked",
                    answer="The response was withheld by content screening.",
                    sources=[],
                )

        return AskResponse(
            status=result.status,
            answer=result.answer,
            sources=[
                SourceResponse(
                    id=source.id,
                    filename=source.filename,
                    page=source.page,
                )
                for source in result.sources
            ],
        )

    return app


load_dotenv(ROOT / ".env")
app = create_app()
