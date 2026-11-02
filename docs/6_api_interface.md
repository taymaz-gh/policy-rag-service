# 6. FastAPI Interface

## Overview

FinServe exposes its Retrieval-Augmented Generation (RAG) service through a REST API implemented with FastAPI.

The API separates the retrieval and generation backend from client applications such as Streamlit. It provides structured request validation, policy-aware document retrieval, LLM-generated answers, source references, and retrieval-only fallback.

**Implementation:** `src/agentic_rag/api/main.py`

**Application factory:** `create_app()`

**Backend service:** `src/agentic_rag/generation/rag_service.py`

## Architecture

```text
                    HTTP Client
                   (Swagger / UI)
                         |
                         v
                     FastAPI
                 Request Validation
                         |
                         v
                Date-aware Retriever
                         |
                         v
                     Chroma DB
                  (MiniLM vectors)
                         |
                         v
                    RAGService
                         |
                  Input Screening
                         |
                         v
                 Retrieved Evidence
                         |
                    +----+----+
                    |         |
                    v         v
                Gemini     Retrieval-only
                    |         |
                    v         |
              Grounded Answer
                    |         |
                    v         |
            Citation Validation
                    |         |
                    +----+----+
                         |
                         v
                 Output Screening
                         |
                         v
                  JSON Response
```

## Starting the API

### Prerequisites

- Python virtual environment activated.
- Project installed in editable mode with required dependencies.
- Synthetic sample PDFs and document manifest available.
- A configured Gemini API key for generated answers, or retrieval-only operation without credentials.

From the repository root, install the project if necessary:

```powershell
python -m pip install -e ".[dev]"
```

Start the FastAPI server by running the Uvicorn ASGI server:

```powershell
python -m uvicorn agentic_rag.api.main:app --host 127.0.0.1 --port 8000
```

The command assumes the application package is installed in the active Python environment.

Successful startup includes:

```text
INFO: Waiting for application startup.
INFO: Application startup complete.
INFO: Uvicorn running on http://127.0.0.1:8000
```

Keep the terminal running while interacting with the API. Stop the server with `Ctrl+C`.

### Development Reloading

For development, automatic restarting after code changes can be enabled:

```powershell
python -m uvicorn agentic_rag.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Reloading restarts the application and reruns its initialization, including embedding-model loading and vector indexing.

## Application Initialization

The API uses FastAPI's lifespan mechanism to initialize shared retrieval resources during application startup.

**Function:** `initialize_components()`

Initialization proceeds as follows:

1. Load the synthetic PDF policies and their metadata manifest.
2. Extract text and split documents into chunks.
3. Initialize the MiniLM embedding model.
4. Embed and index the document chunks in an in-memory Chroma collection.
5. Initialize the configured LLM client, or enable retrieval-only mode when credentials are missing.

The embedding model and Chroma index are initialized once per server process rather than once per request.

The initial vector collection is in memory and is recreated after a server restart.

If retrieval initialization fails, the API process may remain running, but `/ready` reports that the service is unavailable.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check process liveness |
| GET | `/ready` | Check retrieval readiness and configured generation mode |
| POST | `/ask` | Submit a policy question and receive a structured response |
| GET | `/docs` | Interactive Swagger API documentation |
| GET | `/openapi.json` | Machine-readable OpenAPI schema |

### Health Check

URL: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

Method: `GET`

Expected response:

```json
{
  "status": "ok"
}
```

A successful health check indicates that the API process is responding. It does not guarantee that the retrieval index or LLM is available.

### Readiness Check

URL: [http://127.0.0.1:8000/ready](http://127.0.0.1:8000/ready)

Method: `GET`

Example response:

```json
{
  "ready": true,
  "generation_mode": "gemini"
}
```

The readiness response distinguishes retrieval availability from generation configuration.

**Generation modes**

| Mode | Meaning |
|---|---|
| `gemini` | Retrieval initialized and a Gemini client configured |
| `retrieval_only` | Retrieval initialized but no LLM configured |
| `unavailable` | Retrieval service is not initialized |

The `gemini` mode indicates that an LLM client was configured. It does not independently verify live provider availability, remaining quota, or network connectivity.

### Question-Answering Endpoint

URL: [http://127.0.0.1:8000/ask](http://127.0.0.1:8000/ask)

Method: `POST`

The endpoint accepts a question and an optional policy-applicability date.

**Request schema**

| Field | Type | Description |
|---|---|---|
| `question` | String | User question, between 1 and 4000 characters |
| `as_of` | ISO date | Policy applicability date; defaults to the server's current date |

Example request:

```json
{
  "question": "Who is authorized to approve a late-payment fee waiver?",
  "as_of": "2026-11-02"
}
```

**Example response from the live demonstration**

```json
{
  "status": "answered",
  "answer": "Based on the provided documents, only an authorized supervisor is authorized to approve (or reject) a fee-waiver request [S1]. Customer support agents are permitted to submit the request, but they cannot approve it [S1].",
  "sources": [
    {
      "id": "S1",
      "filename": "fee_schedule_2026.pdf",
      "page": 1
    },
    {
      "id": "S2",
      "filename": "repayment_policy_2026.pdf",
      "page": 1
    },
    {
      "id": "S3",
      "filename": "repayment_policy_2026.pdf",
      "page": 1
    }
  ]
}
```

The `sources` array lists retrieved passages, not necessarily sources explicitly cited in the answer.

In this example, the answer cites `[S1]`, which corresponds to the applicable fee schedule. `[S2]` and `[S3]` were retrieved but were