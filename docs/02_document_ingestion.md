# 2. Document Ingestion and Chunking

## Problem

Financial institutions maintain information across various documents, including product terms, repayment policies, fee schedules, and customer-support procedures.

A Retrieval-Augmented Generation (RAG) system must convert these documents into searchable text while preserving sufficient source information for reliable retrieval and citation.

Two challenges are particularly important:

- **Document structure:** PDF content may contain headings, paragraphs, tables, and page boundaries that influence how information should be interpreted.
- **Chunk boundaries:** Large documents must be divided into smaller passages without unnecessarily separating related information.

Poor document extraction or chunking can reduce retrieval quality and lead to incomplete or misleading answers.

## Design Decision

The ingestion pipeline uses:

| Component | Selected approach |
|---|---|
| PDF loading | LangChain `PyMuPDF4LLMLoader` |
| Extraction granularity | Page-level (`mode="page"`) |
| Text representation | Markdown-oriented text |
| Text splitting | `RecursiveCharacterTextSplitter` |
| Chunk size | 500 characters |
| Chunk overlap | 75 characters |
| Character-offset tracking | `add_start_index=True` |
| Document representation | LangChain `Document` |

These choices establish a lightweight, reproducible baseline suitable for the initial synthetic financial-policy knowledge base.

### PDF Loader Selection

`PyMuPDF4LLMLoader` converts PDF content into LangChain `Document` objects using PyMuPDF4LLM.

Its Markdown-oriented extraction helps retain recognizable structural elements such as headings and paragraphs.

Page-level loading preserves the relationship between extracted text and its original PDF page, supporting source citations.

More advanced layout analysis may be necessary for complex financial documents containing tables, multi-column layouts, or scanned pages.

### Text Splitter Selection

The pipeline uses LangChain's `RecursiveCharacterTextSplitter`.

Its default separator hierarchy is:

1. Paragraph boundaries (`\n\n`)
2. Line boundaries (`\n`)
3. Spaces (` `)
4. Individual characters when necessary

The splitter attempts to preserve larger text units before applying finer-grained splitting.

**Reasons for selecting this splitter:**

- Straightforward configuration and reproducibility.
- Suitable as an initial baseline for policy documents.
- Preserves document metadata in the resulting chunks.
- Does not require additional embedding models for segmentation.
- Supports controlled experiments with chunk size and overlap.

The initial configuration uses 500-character chunks with a requested overlap of 75 characters.

Overlap can help retain contextual continuity around chunk boundaries, although the actual overlap depends on the splitting process.

These parameter values are experimental defaults, not optimized settings.

## Implementation

### PDF Extraction and Chunking

**Module:** `src/agentic_rag/ingestion/pdf_loader.py`

**Function:** `load_and_chunk_pdfs()`

The function:

1. Validates the input directory and chunking parameters.
2. Discovers PDF files.
3. Extracts documents at page level using `PyMuPDF4LLMLoader`.
4. Preserves source metadata and adds the filename.
5. Optionally enriches documents with policy metadata from a manifest.
6. Splits pages into chunks using `RecursiveCharacterTextSplitter`.
7. Returns LangChain `Document` objects and reports ingestion statistics.

### Metadata Preservation

Each resulting chunk retains the available source metadata.

| Metadata field | Purpose |
|---|---|
| `source` | Original PDF path |
| `filename` | Source document filename |
| `page` | Zero-based page index |
| `start_index` | Character offset within the extracted page text |
| `policy_id` | Policy identity, when manifest enrichment is enabled |
| `version` | Document version, when supplied by the manifest |
| `effective_from` | Beginning of the recorded applicability interval |
| `effective_to` | End of the recorded applicability interval, when specified |

The `start_index` refers to extracted text, not to a byte position in the original PDF.

Policy metadata enrichment is performed before chunking so that the resulting chunks inherit the policy attributes.

The policy-versioning mechanism is documented separately in [2. Policy Versioning and Applicability](2_policy_versioning.md).

## Data Flow

```text
        PDF Knowledge Base
                |
                v
      PyMuPDF4LLMLoader
         (page-level)
                |
                v
       LangChain Documents
       (text + PDF metadata)
                |
                v
       Metadata Enrichment <--- Document Manifest
         (if enabled)
                |
                v
    RecursiveCharacterTextSplitter
         (500 / 75 chars)
                |
                v
        Document Chunks
     (text + source metadata)
                |
                v
       Embedding Generation
                |
                v
         Vector Database
```

## Initial Validation

The ingestion pipeline was tested using four synthetic financial-policy PDFs generated with ReportLab.

| Metric | Result |
|---|---:|
| PDF files | 4 |
| Loaded pages | 4 |
| Extracted characters | 2,019 |
| Generated chunks | 6 |
| Configured chunk size | 500 characters |
| Configured chunk overlap | 75 characters |

Automated tests cover nonempty extraction, source filename preservation, missing input directories, and invalid chunk-overlap parameters.

Additional integration tests verify that policy metadata survives the chunking process.

## Alternative Strategies

| Strategy | Potential advantage | Trade-off |
|---|---|---|
| Token-based splitting | Better alignment with model token limits | Tokenizer-dependent |
| Markdown/header-aware splitting | Better preservation of document sections | Depends on extraction quality |
| Sentence-based splitting | Reduces sentence fragmentation | May produce uneven chunk sizes |
| Semantic chunking | Groups text using semantic similarity | More computationally expensive and parameter-sensitive |
| Recursive character splitting | Simple, lightweight, reproducible | Does not explicitly model semantic boundaries |

The selected splitter provides a baseline against which alternative strategies can be evaluated using retrieval metrics.

## Important Limitations

- The current knowledge base contains only four short, synthetic PDFs.
- Extraction quality has not been evaluated on complex tables, scanned documents, or multi-column layouts.
- Character-based chunking can separate semantically related information.
- Chunk overlap does not guarantee complete contextual preservation.
- Source metadata supports traceability but does not independently verify the correctness of generated citations.
- The current chunking configuration has not been optimized through quantitative retrieval experiments.
- Document applicability depends on the accuracy and completeness of the associated policy metadata.

## Implementation Status

**Implemented**
- Page-level PDF extraction through LangChain.
- Recursive character-based text splitting.
- Source and page metadata preservation.
- Optional policy metadata enrichment before chunking.
- Ingestion reporting and automated tests.

**Not yet implemented**
- Comparative evaluation of chunking strategies.
- Advanced extraction for complex document layouts.
- OCR for scanned documents.
- Automated assessment of extraction quality.
- End-to-end citation accuracy evaluation.