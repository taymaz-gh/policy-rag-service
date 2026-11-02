"""Evidence-grounded question answering over retrieved policies."""

import logging
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai.chat_models import (
    GoogleModelNotFoundError,
    GoogleRateLimitError,
)

from agentic_rag.generation.query_router import conversational_response
from agentic_rag.security.guardrails import screen_question

logger = logging.getLogger(__name__)

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "If the source excerpts do not contain enough information "
                "to answer the question, respond with exactly "
                "'INSUFFICIENT_EVIDENCE' and nothing else. "
                "Do not add a citation, explanation, or punctuation "
                "to this abstention signal. "
                "Otherwise, answer using only the supplied evidence "
                "and cite every factual claim with source IDs such as [S1]. "
                "Never use the abstention signal as part of a factual answer. "
            ),
        ),
        (
            "human",
            ("Question:\n{question}\n\nSource excerpts:\n{context}"),
        ),
    ]
)


CITATION_PATTERN = re.compile(r"\[S(\d+)\]")
ABSTENTION_SIGNAL = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class SourceReference:
    id: str
    filename: str
    page: int
    text: str


@dataclass(frozen=True)
class RAGResult:
    answer: str
    status: str
    sources: list[SourceReference]


class RAGService:
    """Coordinate screening, retrieval, and grounded generation."""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseChatModel | None = None,
        fallback_enabled: bool = True,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.fallback_enabled = fallback_enabled

    def ask(self, question: str) -> RAGResult:
        """Answer a question using applicable retrieved evidence."""
        screening = screen_question(question)

        if not screening.allowed:
            return RAGResult(
                answer=screening.reason or "Request blocked.",
                status="blocked",
                sources=[],
            )

        reply = conversational_response(question)

        if reply is not None:
            return RAGResult(
                answer=reply,
                status="conversational",
                sources=[],
            )

        documents = self.retriever.invoke(question)

        if not documents:
            return RAGResult(
                answer="No applicable policy evidence was found.",
                status="insufficient_evidence",
                sources=[],
            )

        sources = self._make_sources(documents)

        if self.llm is None:
            return self._retrieval_only_result(sources)

        context = "\n\n".join(
            f"[{source.id}] {source.filename}, page {source.page}\n"
            f"<source_excerpt>\n{source.text}\n</source_excerpt>"
            for source in sources
        )

        messages = PROMPT.format_messages(
            question=question,
            context=context,
        )

        try:
            response = self.llm.invoke(messages)
        except (GoogleModelNotFoundError, GoogleRateLimitError) as exc:
            if not self.fallback_enabled:
                raise

            logger.warning(
                "LLM generation unavailable; using retrieval-only fallback. "
                "Error type: %s",
                type(exc).__name__,
            )
            return self._retrieval_only_result(sources)

        answer = self._response_text(response)

        # Only the exact reserved signal is accepted as an abstention.
        # Other generated text must pass normal citation validation.
        if answer == ABSTENTION_SIGNAL:
            return RAGResult(
                answer=(
                    "The available policy documents do not establish "
                    "an answer to this question."
                ),
                status="insufficient_evidence",
                sources=sources,
            )

        referenced_ids = {f"S{number}" for number in CITATION_PATTERN.findall(answer)}

        valid_ids = {source.id for source in sources}

        if not referenced_ids or not referenced_ids <= valid_ids:
            return RAGResult(
                answer=(
                    "The generated response could not be validated "
                    "against the retrieved source identifiers."
                ),
                status="citation_validation_failed",
                sources=sources,
            )

        return RAGResult(
            answer=answer,
            status="answered",
            sources=sources,
        )

    @staticmethod
    def _make_sources(
        documents: list[Document],
    ) -> list[SourceReference]:
        return [
            SourceReference(
                id=f"S{index}",
                filename=str(doc.metadata.get("filename", "unknown")),
                page=int(doc.metadata.get("page", 0)) + 1,
                text=doc.page_content,
            )
            for index, doc in enumerate(documents, start=1)
        ]

    @staticmethod
    def _response_text(response: Any) -> str:
        """Extract text from a LangChain chat-model response."""
        if isinstance(response.content, str):
            return response.content.strip()

        return str(response.text).strip()

    @staticmethod
    def _retrieval_only_result(
        sources: list[SourceReference],
    ) -> RAGResult:
        """Return retrieved evidence without a generated answer."""
        excerpts = "\n\n".join(
            (f"[{source.id}] {source.filename}, page {source.page}\n{source.text}")
            for source in sources
        )

        return RAGResult(
            answer=(
                "AI answer generation is unavailable. "
                "The following relevant policy excerpts were retrieved. "
                "They are not a generated answer:\n\n"
                f"{excerpts}"
            ),
            status="retrieval_only",
            sources=sources,
        )
