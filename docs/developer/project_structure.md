# Project Structure

The project uses a Python `src/` layout with modular components.

```text
agentic-rag-platform/
|
+-- src/agentic_rag/
|   |
|   +-- ingestion/             # PDF loading, chunking, metadata
|   |   +-- pdf_loader.py      # load_and_chunk_pdfs()
|   |   +-- metadata.py        # Planned: policy metadata enrichment
|   |
|   +-- retrieval/             # Embeddings and vector search
|   |   +-- vector_store.py    # Planned: vector indexing/search
|   |
|   +-- generation/            # LLM integration and RAG answers
|   |   +-- rag_service.py     # Planned: RAG question answering
|   |
|   +-- security/              # Input validation and guardrails
|   |   +-- guardrails.py      # Planned: input checks
|   |
|   +-- policies/              # Planned: compliance rules
|   +-- workflows/             # Planned: LangGraph orchestration
|   +-- evaluation/            # Planned: retrieval/answer evaluation
|   +-- observability/         # Planned: logging and telemetry
|   |
|   +-- api/
|       +-- main.py            # Planned: FastAPI endpoints
|
+-- app/
|   +-- streamlit_app.py       # Planned: chatbot and dev panel
|
+-- scripts/
|   +-- generate_sample_pdfs.py
|   +-- ingest_documents.py    # Planned: indexing CLI
|
+-- data/
|   +-- sample_documents/      # Synthetic PDF knowledge base
|   +-- evaluation/            # Future benchmark questions
|
+-- tests/                      # pytest test suite
+-- docs/                       # Technical design decisions
|   +-- developer/             # Developer architecture and workflow
|
+-- pyproject.toml              # Packaging and dependencies
+-- .gitignore                  # Git exclusions
+-- .env.example                # Planned configuration template
+-- README.md                   # Project overview
```

## Component Responsibilities

| Component | Responsibility |
|---|---|
| `ingestion` | Convert PDFs into text chunks with source metadata |
| `retrieval` | Embed and retrieve relevant document passages |
| `generation` | Build prompts and generate cited answers |
| `security` | Apply input validation and security guardrails |
| `policies` | Apply company-policy and compliance checks |
| `workflows` | Coordinate conditional workflows using LangGraph |
| `evaluation` | Measure retrieval relevance and answer quality |
| `observability` | Instrument application analytics and monitoring |
| `api` | Expose backend functionality through HTTP |
| `app` | Provide the user and developer interfaces |

## Implemented Functions

| Function | Module | Responsibility |
|---|---|---|
| `load_and_chunk_pdfs()` | `ingestion/pdf_loader.py` | PDF extraction, splitting, metadata preservation |
| `generate_pdf()` | `scripts/generate_sample_pdfs.py` | Generate one synthetic PDF |
| `main()` | `scripts/generate_sample_pdfs.py` | Generate the synthetic knowledge base |

This document should be updated whenever a new module or major function is added.