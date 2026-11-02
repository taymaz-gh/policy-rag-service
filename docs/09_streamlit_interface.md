# 9. Streamlit Interface and Local Development

## Overview

The FinServe Policy RAG Service provides a Streamlit chatbot frontend connected to a FastAPI backend.

The application answers questions using synthetic financial-policy documents, with date-aware retrieval, grounded LLM generation, source citations, and retrieval-only fallback.

Streamlit and FastAPI run as separate local processes and communicate over HTTP.

## Application Architecture

```text
                  User / Browser
                        |
                        v
            Streamlit UI (port 8501)
           +-------------------------+
           | Chat messages           |
           | Policy-date selection   |
           | Retrieved citations     |
           | Developer diagnostics   |
           +-------------------------+
                        |
                        | GET /ready
                        | POST /ask
                        v
            FastAPI API (port 8000)
           +-------------------------+
           | Pydantic validation     |
           | Request orchestration   |
           | Response screening      |
           +-------------------------+
                        |
                        v
                    RAGService
                        |
           +------------+------------+
           |                         |
           v                         v
    Sensitive screening      Policy-aware retrieval
                                     |
                                     v
                              Chroma + MiniLM
                                     |
                                     v
                             Relevant evidence
                                     |
                              +------+------+
                              |             |
                              v             v
                            Gemini     Retrieval-only
                              |             |
                              +------+------+
                                     |
                                     v
                            Citation validation
                                     |
                                     v
                             Structured JSON
                                     |
                                     v
                               Streamlit UI
```

## Implementation Modules

| Module | Responsibility |
|---|---|
| `app/streamlit_app.py` | Chat UI, backend HTTP client, chat history, citations, and diagnostics |
| `src/agentic_rag/api/main.py` | FastAPI endpoints, validation, startup initialization, and response screening |
| `src/agentic_rag/generation/rag_service.py` | Screening, retrieval orchestration, grounded generation, and fallback |
| `src/agentic_rag/generation/llm_factory.py` | Configured LLM creation |
| `src/agentic_rag/retrieval/vector_store.py` | Embeddings, Chroma indexing, and date-aware retrieval |
| `src/agentic_rag/security/guardrails.py` | Initial sensitive-identifier screening |

## Running the Application Locally

### Prerequisites

- Python 3.12.
- Repository installed in an activated virtual environment.
- Dependencies installed using `python -m pip install -e ".[dev]"`.
- Synthetic policy PDFs and document manifest available.
- Optional Gemini API key configured in `.env`.

Both servers must be launched from the repository root, using separate terminals.

### Terminal 1 — FastAPI Backend

```powershell
python -m uvicorn agentic_rag.api.main:app --host 127.0.0.1 --port 8000
```

The FastAPI startup process loads the PDF documents, enriches policy metadata, creates sentence embeddings, builds the Chroma index, and configures the generation client.

The retrieval resources are initialized once per server process rather than per request.

Useful local endpoints:

| URL | Purpose |
|---|---|
| [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) | API process health |
| [http://127.0.0.1:8000/ready](http://127.0.0.1:8000/ready) | Retrieval readiness and generation mode |
| [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) | Interactive API documentation |
| [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json) | Machine-readable OpenAPI specification |

### Terminal 2 — Streamlit Frontend

```powershell
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open the chatbot:

**[http://127.0.0.1:8501](http://127.0.0.1:8501)**

The Streamlit process uses `RAG_API_URL` to identify the backend. Its default is `http://127.0.0.1:8000`.

Both processes can be stopped using `Ctrl+C`.

## Chatbot Request Lifecycle

1. Streamlit collects the question and selected policy-applicability date.
2. Streamlit sends an HTTP `POST /ask` request to FastAPI.
3. FastAPI validates the JSON request using Pydantic.
4. The RAG service screens the question for selected sensitive identifiers.
5. Chroma retrieves policy passages satisfying the effective-date filter.
6. Retrieved chunks are assigned source IDs.
7. Gemini generates an evidence-grounded answer, or the service returns retrieved passages in fallback mode.
8. Generated citation identifiers are checked against the retrieved sources.
9. FastAPI applies basic sensitive-output screening.
10. Streamlit displays the response and its source metadata.

Each question is processed independently. Previous chat messages are retained for display but are not currently supplied as conversation history to the RAG backend.

## User Interface

### Chat History

Streamlit stores displayed messages in `st.session_state`.

This history is maintained within a Streamlit session and is not persisted to a database.

The **Clear conversation** control resets the displayed chat history.

### Policy Applicability Date

The sidebar date selector determines the `as_of` value sent to the API.

This date is used to filter policy documents before semantic retrieval.

The selector is distinct from automatic interpretation of historical dates in natural-language questions, which is not yet implemented.

### Source References

Generated answers can contain citation identifiers such as `[S1]`.

Streamlit matches these identifiers against the source metadata returned by FastAPI and displays cited filenames and page numbers.

The backend may return additional retrieved sources that were not cited in the answer.

The current API does not provide document-download links or full retrieval scores.

### Developer Diagnostics

The optional diagnostics panel displays:

- Backend response status.
- Client-observed request latency.
- Selected applicability date.
- Number of retrieved sources.
- Cited source identifiers.
- Retrieved source metadata.

Client-observed latency includes HTTP communication and server processing. It does not independently measure retrieval time or model inference time.

## Response Statuses

| Status | Interface behavior |
|---|---|
| `answered` | Display the LLM-generated answer and citations |
| `retrieval_only` | Warn that generation is unavailable and display retrieved evidence |
| `insufficient_evidence` | Indicate that no applicable evidence was retrieved |
| `blocked` | Display the screening outcome |
| `citation_validation_failed` | Indicate that generated citation identifiers failed validation |

HTTP errors and invalid responses are reported separately from application-level RAG statuses.

## Troubleshooting

### Backend Unavailable

Check whether the FastAPI terminal is running and whether [http://127.0.0.1:8000/ready](http://127.0.0.1:8000/ready) reports `ready: true`.

A successful `/health` response does not guarantee retrieval readiness.

### HTTP 500 from `/ask`

Inspect the FastAPI terminal traceback. Potential causes include provider errors, unexpected exceptions, and application failures not handled by the retrieval-only fallback.

### Retrieval-only Response

The RAG backend may return retrieved evidence instead of an LLM-generated answer when the model is unconfigured or an expected provider failure occurs.

Retrieved passages are not equivalent to a verified generated answer.

### Incorrect Historical Policy

Verify the sidebar's policy-applicability date. Historical date expressions inside a question are not automatically converted into metadata filters.

## Limitations

- The application is intended for local development and synthetic demonstration data.
- No authentication, role-based authorization, or request rate limiting is implemented.
- The chatbot does not support conversational follow-up resolution.
- The UI does not independently verify factual support for citations.
- Sensitive-content screening is incomplete.
- Detailed retrieval scores and model traces are not exposed.
- Streamlit session history is not durable storage.
- The frontend and backend are not configured for public deployment.