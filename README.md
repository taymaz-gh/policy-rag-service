## Current status: MVP Completed

# Policy RAG Service

A financial-policy question-answering application built with LangChain, Chroma, Gemini, FastAPI, and Streamlit.

The project demonstrates Retrieval-Augmented Generation (RAG) over versioned financial-policy documents, with date-aware filtering, source references, basic sensitive-data screening, and retrieval-only fallback.

**FinServe is a fictional financial-services company. All example policies, documents, and fees are synthetic. This application is not intended for real financial advice or production customer data.**

## Features

- PDF ingestion and recursive character-based chunking.
- Manifest-based policy metadata and effective-date filtering.
- Dense semantic retrieval using `all-MiniLM-L6-v2` and Chroma.
- Gemini-generated answers grounded in retrieved policy excerpts.
- Source-identifier validation and filename/page references.
- Basic sensitive-input and output screening.
- Retrieval-only operation when LLM generation is unavailable.
- FastAPI REST endpoints and Streamlit chat interface.
- Offline automated testing using pytest and Ruff.

## Architecture

```text
Streamlit Chatbot (:8501)
          |
          | POST /ask
          v
FastAPI Backend (:8000)
          |
          v
     RAGService
          |
          +--- Sensitive-input screening
          |
          +--- Date-filtered Chroma retrieval
          |          |
          |          +--- MiniLM embeddings
          |          +--- Policy metadata
          |
          +--- Gemini generation / retrieval-only
          |
          +--- Citation validation
          |
          +--- Sensitive-output screening
          |
          v
  Answer + Status + Sources
```

## Quick Start

### 1. Clone the Repository

```powershell
git clone https://github.com/taymaz-gh/policy-rag-service.git
cd policy-rag-service
```

### 2. Create a Virtual Environment

Python 3.12 is required.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

The first embedding-model execution may download pretrained weights from Hugging Face.

### 3. Configure Gemini

Create a local `.env` file from `.env.example` and set:

```dotenv
GOOGLE_API_KEY=your_api_key
RAG_LLM_PROVIDER=gemini
RAG_LLM_MODEL=your_available_gemini_model
```

To enable Gemini-powered answers, create a free API key in Google AI Studio and set GOOGLE_API_KEY in your local .env file.

Free-tier usage is subject to model-specific quotas. Check your project's current limits and usage in Google AI Studio — Rate Limits.

The selected model must be available to the configured API key.

Without a configured Gemini key, the application supports retrieval-only operation as a fallback.

Do not commit `.env`, API key, or real credentials to Git.

### 4. Start FastAPI

In the first PowerShell terminal:

```powershell
python -m uvicorn agentic_rag.api.main:app --host 127.0.0.1 --port 8000
```

Backend links:

- [Health check](http://127.0.0.1:8000/health)
- [Readiness check](http://127.0.0.1:8000/ready)
- [Interactive API documentation](http://127.0.0.1:8000/docs)

### 5. Start Streamlit

In a second PowerShell terminal, activate the same virtual environment and run:

```powershell
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open the chatbot:

**[http://127.0.0.1:8501](http://127.0.0.1:8501)**

Both servers must remain running while using the chatbot.

## Example Questions

The sample knowledge base contains synthetic fee schedules, repayment policies, and customer-support instructions.

| Policy date | Example question | Expected policy evidence |
|---|---|---|
| 2026-11-02 | Who can approve a late-payment fee waiver? | Only an authorized supervisor can approve or reject the request |
| 2026-11-02 | What is the late-payment administration fee? | €12 per qualifying missed installment |
| 2025-06-15 | What is the late-payment administration fee? | €20 per qualifying missed installment |
| 2026-11-02 | Can a customer-support agent approve a payment extension? | Support agents can explain the procedure but cannot approve an extension |
| 2026-11-02 | What happens after two consecutive missed installments? | Referral to the repayment assistance team |

These are demonstration expectations based on the synthetic policies, not guarantees of identical LLM wording.

**Important:** Change the sidebar's policy applicability date when testing historical documents. The current application does not automatically infer this date from the question.

## API Example

Submit a question through `POST http://127.0.0.1:8000/ask`.

```json
{
  "question": "Who can approve a late-payment fee waiver?",
  "as_of": "2026-11-02"
}
```

An answered response contains an answer, a response status, and source metadata. Source identifiers such as `[S1]` refer to retrieved policy passages.

## Testing

Run:

```powershell
python -m pip check
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

The offline test suite uses fake LLM responses and does not require Gemini credentials.

## Technical Documentation

| Document | Description |
|---|---|
| [Document Ingestion](docs/02_document_ingestion.md) | PDF extraction and text chunking |
| [Policy Versioning](docs/04_policy_versioning.md) | Policy metadata and temporal applicability |
| [Retrieval Strategy](docs/05_retrieval_strategy.md) | Embeddings, Chroma, and similarity search |
| [RAG Generation](docs/06_rag_generation.md) | Grounded generation, citations, and fallback |
| [Security Guardrails](docs/07_security_guardrails.md) | Sensitive-data screening and limitations |
| [FastAPI Interface](docs/08_api_interface.md) | API lifecycle, endpoints, and local testing |
| [Streamlit Interface](docs/09_streamlit_interface.md) | Chatbot architecture and local development |
| [Developer Architecture](docs/03_architecture.md) | System component relationships |
| [Retrieval Threshold Calibration](docs/10_retrieval_threshold_calibration.md) | Similarity scoring, threshold selection, held-out evaluation, and failure analysis |

## Current Limitations

- Small synthetic knowledge base.
- No automatic interpretation of dates embedded in questions.
- Chat history is displayed but not used as LLM conversation context.
- No guarantee of factual correctness or citation support.
- No authentication, persistent conversation storage, or production-grade privacy controls.
- Advanced agentic workflows, monitoring, and retrieval evaluation remain future extensions.


## Development Roadmap

- GitHub Actions CI for automated linting, testing, and pull-request validation.
- Continuous Deployment (CD) for automated application deployment after successful CI checks.
- Expanded retrieval evaluation, factual-grounding tests, and citation-support assessment.
- Stronger sensitive-content moderation, PII protection, and prompt-injection defenses.
- Persistent vector indexing, retrieval diagnostics, and relevance-based abstention.
- LangGraph workflows for query routing, self-query interpretation, and agentic orchestration.
- **PostHog** for privacy-conscious product analytics and chatbot usage metrics.
- **Datadog** for application monitoring, structured logs, performance metrics, and distributed tracing.