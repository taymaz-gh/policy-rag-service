"""Offline tests for the FinServe FastAPI interface."""

from fastapi.testclient import TestClient
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever

from agentic_rag.api import main as api_module


class StaticRetriever(BaseRetriever):
    documents: list[Document]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return self.documents


def sample_retriever() -> StaticRetriever:
    return StaticRetriever(
        documents=[
            Document(
                page_content="Only an authorized supervisor can approve waivers.",
                metadata={
                    "filename": "fee_schedule_2026.pdf",
                    "page": 0,
                },
            )
        ]
    )


def test_health():
    app = api_module.create_app(initialize=False)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_not_ready():
    app = api_module.create_app(initialize=False)

    with TestClient(app) as client:
        readiness = client.get("/ready")
        result = client.post(
            "/ask",
            json={"question": "Who approves fee waivers?"},
        )

    assert readiness.json()["ready"] is False
    assert result.status_code == 503


def test_answered_with_source(monkeypatch):
    app = api_module.create_app(initialize=False)

    monkeypatch.setattr(
        api_module,
        "create_policy_retriever",
        lambda store, as_of, k: sample_retriever(),
    )

    with TestClient(app) as client:
        app.state.vector_store = object()
        app.state.llm = FakeListChatModel(
            responses=["Only an authorized supervisor may approve the waiver [S1]."]
        )

        result = client.post(
            "/ask",
            json={
                "question": "Who approves fee waivers?",
                "as_of": "2026-11-02",
            },
        )

    assert result.status_code == 200
    body = result.json()
    assert body["status"] == "answered"
    assert body["sources"][0]["filename"] == "fee_schedule_2026.pdf"
    assert body["sources"][0]["page"] == 1


def test_retrieval_only_without_llm(monkeypatch):
    app = api_module.create_app(initialize=False)

    monkeypatch.setattr(
        api_module,
        "create_policy_retriever",
        lambda store, as_of, k: sample_retriever(),
    )

    with TestClient(app) as client:
        app.state.vector_store = object()
        app.state.llm = None

        assert client.get("/ready").json() == {
            "ready": True,
            "generation_mode": "retrieval_only",
        }

        result = client.post(
            "/ask",
            json={"question": "Who approves fee waivers?"},
        )

    assert result.status_code == 200
    assert result.json()["status"] == "retrieval_only"
    assert "[S1]" in result.json()["answer"]


def test_sensitive_input_blocked(monkeypatch):
    app = api_module.create_app(initialize=False)

    monkeypatch.setattr(
        api_module,
        "create_policy_retriever",
        lambda store, as_of, k: sample_retriever(),
    )

    with TestClient(app) as client:
        app.state.vector_store = object()
        app.state.llm = None

        result = client.post(
            "/ask",
            json={
                "question": "My email is customer@example.com. Who approves waivers?"
            },
        )

    assert result.status_code == 200
    assert result.json()["status"] == "blocked"
    assert result.json()["sources"] == []


def test_output_screening_blocks_sensitive_response(monkeypatch):
    app = api_module.create_app(initialize=False)

    monkeypatch.setattr(
        api_module,
        "create_policy_retriever",
        lambda store, as_of, k: sample_retriever(),
    )

    with TestClient(app) as client:
        app.state.vector_store = object()
        app.state.llm = FakeListChatModel(
            responses=["Contact customer@example.com [S1]."]
        )

        result = client.post(
            "/ask",
            json={"question": "Who approves fee waivers?"},
        )

    assert result.status_code == 200
    assert result.json()["status"] == "blocked"
    assert result.json()["sources"] == []
    assert "customer@example.com" not in result.json()["answer"]
