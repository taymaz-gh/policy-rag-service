
# Agentic RAG Platform

An end-to-end Retrieval-Augmented Generation (RAG) application combining semantic search, LLM-powered question answering, agentic workflows, product analytics, and application observability.

## Project Objectives

- Implement document ingestion, text embeddings, and semantic search.
- Build a RAG pipeline using LangChain and a vector database.
- Develop stateful, agentic workflows with LangGraph.
- Evaluate retrieval relevance and generated-answer quality.
- Track user behavior and experiments using PostHog.
- Monitor application performance using Datadog.
- Apply software engineering and MLOps practices, including testing, CI/CD, and containerization.

## Planned Technology Stack

| Category | Technologies |
|---|---|
| Language | Python 3.12 |
| LLM and RAG | LangChain, LangGraph |
| Embeddings | Sentence Transformers |
| Vector database | Chroma |
| Backend | FastAPI |
| Product analytics | PostHog |
| Observability | Datadog |
| Testing | pytest, Ruff |
| Deployment | Docker, GitHub Actions |

## Development Roadmap

- [x] Initialize the Python package and development environment.
- [ ] Implement sentence embeddings and semantic search.
- [ ] Integrate a vector database and LangChain.
- [ ] Build an LLM-powered RAG pipeline.
- [ ] Implement LangGraph-based workflows.
- [ ] Add retrieval and generation evaluation.
- [ ] Integrate PostHog analytics and experimentation.
- [ ] Integrate Datadog monitoring.
- [ ] Implement API, CI/CD, and containerization.

## Getting Started

### Requirements

- Python 3.12
- Git

### Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Note: The activation command above is for PowerShell. For Windows Command Prompt, use `.venv\Scripts\activate.bat`

Install the project and development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

Run the tests:

```powershell
python -m pytest -v
```

## Project Status

Under active development. The current milestone establishes the Python package structure and automated testing.

## Data

The project will use public or synthetic example documents. No confidential company data or API credentials will be committed to the repository.



## Documentation

### Technical Design

- [1. Document Ingestion](docs/1_document_ingestion.md)
- [2. Policy Versioning](docs/2_policy_versioning.md)

### Developer Guides

- [Architecture](docs/developer/architecture.md)
- [Project Structure](docs/developer/project_structure.md)
- [Development Workflow](docs/developer/development_workflow.md)
