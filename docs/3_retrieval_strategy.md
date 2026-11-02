## Metadata-Aware Retrieval

### Current Strategy

The retrieval layer uses a Chroma-backed LangChain `VectorStoreRetriever` with dense similarity search and deterministic metadata filtering.

**Module:** `src/agentic_rag/retrieval/vector_store.py`

**Functions:** `policy_date_filter()`, `create_policy_retriever()`

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3,
        "filter": policy_date_filter(as_of),
    },
)
```

The application supplies the `as_of` date. Chroma filters the eligible documents and returns the most similar passages among those satisfying the date constraints.

This separates policy applicability from semantic relevance.

### Alternative: Self-Query Retrieval

Self-query retrieval uses an LLM to transform natural-language questions into structured retrieval queries, potentially including metadata filters.

Example:

```text
"What was the late-payment fee in 2025?"
                  |
                  v
          LLM Query Interpretation
                  |
                  v
       Semantic Query + Date Constraint
                  |
                  v
        Validated Metadata Filter
                  |
                  v
             Vector Search
```

Self-query retrieval may improve the interpretation of natural-language constraints, including policy dates and product categories.

However, LLM-generated filters can be incomplete or incorrect and introduce additional latency and inference cost.

Mandatory authorization and compliance constraints must remain application-enforced and cannot depend solely on LLM-generated filters.

**Decision:** Retain deterministic filtering for the initial implementation. Evaluate self-query retrieval as a later extension against explicitly labeled metadata-interpretation examples.

### Evaluation Requirements

The retrieval system should be evaluated on:

- Correct applicability filtering for historical and current policies.
- Retrieval relevance using Recall@k and MRR.
- Robustness against ambiguous or missing date information.
- Latency and inference cost.
- Incorrectly included or excluded documents.