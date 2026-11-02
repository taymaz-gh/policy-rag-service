# Application Architecture

## Overview

Agentic RAG Platform is a modular Python application for financial customer-support question answering using company knowledge documents.

The project starts with a basic Retrieval-Augmented Generation (RAG) chatbot and progressively adds policy-aware workflows, evaluation, security controls, and observability.

## Target Architecture

```text
                  +-----------------------+
                  |     Streamlit UI      |
                  |   Chat + Dev Panel    |
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  |      FastAPI          |
                  |  Request / Response   |
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  |      RAG Service      |
                  |                       |
                  | Retrieval + Generation|
                  +-----+-----------+-----+
                        |           |
                        v           v
                 +-----------+  +-----------+
                 | Chroma DB |  |  LLM API  |
                 | Vector    |  | Answer    |
                 | Search    |  | Generation|
                 +-----------+  +-----------+

                 DOCUMENT INGESTION
                 ------------------
                  PDF Knowledge Base
                         |
                         v
                   PDF Extraction
                         |
                         v
                  Metadata Enrichment
                         |
                         v
                    Chunking
                         |
                         v
                  Text Embeddings
                         |
                         v
                     Chroma DB
```

## Retrieval

```text
                 USER QUESTION + as_of
                          |
                          v
              LangChain VectorStoreRetriever
                          |
             +------------+------------+
             |                         |
             v                         v
    Policy Date Filter          MiniLM Embeddings
   (effective_from/to)              384D
             |                         |
             +------------+------------+
                          |
                          v
                    Chroma DB
                 Cosine Distance
                          |
                          v
                   Top-k Chunks
                          |
                          v
                 RAG Generation
                    (planned)
                          |
                          v
                   Chat Answer
                    (planned)
```



## Future Extensions

```text
                  LangGraph
                     |
          +----------+----------+
          |          |          |
          v          v          v
       Policy      Query      Human
       Checks     Routing     Review

       PostHog  ---> Product analytics
       Datadog  ---> Logs, traces, metrics
```

## Architectural Principles

- **Modular monolith:** Keep the initial application simple while separating component responsibilities.
- **Separation of concerns:** UI, API, ingestion, retrieval, and generation have distinct responsibilities.
- **Provider flexibility:** Avoid unnecessary coupling to a single LLM provider.
- **Traceability:** Preserve source and policy metadata to support citations and applicability checks.
- **Testability:** Implement and test core components independently.
- **Incremental development:** Introduce advanced components only when needed.

## Current Implementation Status

**Implemented:**
- Python package structure and automated smoke test
- Synthetic PDF generation
- PDF extraction and text chunking
- Basic ingestion tests

**Planned:**
- Structured policy metadata enrichment
- Embeddings and Chroma vector retrieval
- RAG generation and citations
- FastAPI and Streamlit integration
- Guardrails and LangGraph workflows
- PostHog and Datadog integration

The architecture diagram represents the target system, not functionality that is already deployed.