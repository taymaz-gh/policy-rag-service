# 4. Policy Versioning and Applicability

## Problem

Financial institutions may maintain multiple versions of product terms, fee schedules, and internal policies.

A RAG retriever based only on semantic similarity may retrieve a document that is relevant to a question but no longer applicable.

The synthetic FinServe knowledge base contains:

| Document | Late-payment fee | Effective from | Effective to |
|---|---:|---|---|
| `fee_schedule_2025.pdf` | €20 | 2025-01-01 | 2025-12-31 |
| `fee_schedule_2026.pdf` | €12 | 2026-01-01 | Open-ended |

These are fictional policies and fees, not real financial regulations.

For a question about the currently applicable fee, retrieving the 2025 schedule could produce an incorrect answer.

## Design Decision

Policy applicability is determined through structured document metadata rather than semantic similarity or LLM interpretation alone.

Each policy has an identity, version, document type, and effective-date interval.

The retrieval condition is:

\[
\text{effective\_from} \leq \text{as\_of} \leq \text{effective\_to}
\]

Both boundaries are inclusive.

Metadata filtering restricts the eligible documents before similarity-ranked results are returned.

### Metadata Representation

The source manifest uses ISO-formatted dates:

```json
{
  "policy_id": "fee_schedule",
  "document_type": "fee_schedule",
  "version": "2026",
  "effective_from": "2026-01-01",
  "effective_to": null
}
```

During ingestion, additional integer fields support Chroma range filtering.

| Field | Example | Purpose |
|---|---|---|
| `effective_from` | `"2026-01-01"` | Human-readable metadata |
| `effective_from_int` | `20260101` | Lower date bound |
| `effective_to_int` | `99991231` | Upper date bound |
| `version` | `"2026"` | Policy version identification |
| `policy_id` | `"fee_schedule"` | Logical policy identity |

The integer `99991231` represents an open-ended interval for indexing. It is not an actual expiry date and does not establish legal validity.

## Implementation

### Document Manifest

**File:** `data/document_manifest.json`

The manifest provides structured metadata independently of PDF filenames.

### Metadata Validation and Enrichment

**Module:** `src/agentic_rag/ingestion/metadata.py`

- `load_document_manifest()` loads the manifest and validates required metadata and date ordering.
- `enrich_document_metadata()` attaches validated policy attributes and numeric date fields to LangChain documents without modifying the original documents.

### PDF Integration

**Module:** `src/agentic_rag/ingestion/pdf_loader.py`

**Function:** `load_and_chunk_pdfs()`

Metadata enrichment occurs after PDF extraction and before text chunking when `manifest_path` is supplied.

The text splitter propagates document metadata to the resulting chunks.

### Date-Aware Retrieval

**Module:** `src/agentic_rag/retrieval/vector_store.py`

- `policy_date_filter()` converts an `as_of` date into a Chroma-compatible metadata filter.
- `create_policy_retriever()` creates a LangChain retriever configured with the date filter.

The filter uses:

```python
{
    "$and": [
        {"effective_from_int": {"$lte": date_number}},
        {"effective_to_int": {"$gte": date_number}},
    ]
}
```

### Data Flow

```text
PDF Documents          Document Manifest
      |                       |
      v                       v
Page Extraction        Metadata Validation
      |                       |
      +-----------+-----------+
                  |
                  v
          Metadata Enrichment
                  |
                  v
             Text Chunking
                  |
                  v
       Chunks + Policy Metadata
                  |
                  v
           Chroma Indexing
                  |
                  v
User Question + as_of Date
                  |
                  v
       Policy Applicability Filter
                  |
                  v
       Dense Similarity Retrieval
                  |
                  v
          Eligible Top-k Chunks
```

## Validation

The initial demonstration retrieved the applicable 2026 fee schedule for a fee-waiver authorization question with `as_of=2026-11-01`.

Without date filtering, the three retrieved chunks included the superseded 2025 fee schedule.

After filtering, the 2025 schedule was excluded.

Automated tests verify metadata preservation and date-filter construction. End-to-end Chroma filter integration tests form a separate validation requirement.

## Alternative Approaches

| Approach | Consideration |
|---|---|
| Filename-based version detection | Simple but fragile and dependent on naming conventions |
| Static `status="current"` flag | Insufficient for historical queries |
| LLM-based self-query filtering | Supports natural-language metadata interpretation, but adds uncertainty and inference cost |
| Deterministic effective-date filtering | Predictable, auditable, and straightforward to test |

The deterministic approach is selected for the initial implementation.

## Important Limitations

- Applicability depends on the completeness and correctness of the document manifest.
- An open-ended interval does not prove that a policy remains legally applicable.
- Overlapping validity intervals are not yet detected.
- Product, jurisdiction, and customer-specific applicability are not modeled.
- Questions requiring historical interpretation need an appropriate `as_of` date supplied by the application.
- Date filtering does not enforce compliance or authorization.
- Generated answers require separate evidence and compliance validation.

## Implementation Status

**Implemented**
- Manifest-based document metadata.
- Required-field and date validation.
- Metadata enrichment before chunking.
- Numeric date attributes for Chroma.
- Deterministic date-filter construction.
- Date-filtered dense retrieval.

**Planned**
- Overlapping-version detection.
- Automatic interpretation of date constraints in natural-language questions.
- Product-specific applicability.
- Mandatory authorization filters.
- Broader retrieval integration tests and quantitative evaluation.