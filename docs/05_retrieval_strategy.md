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


### Configurable Retrieval Relevance Threshold

**Module:** `src/agentic_rag/retrieval/vector_store.py`  
**Function:** `create_policy_retriever()`

The retriever supports two LangChain search strategies, selected by the optional `score_threshold` parameter.

| Configuration | Search strategy | Behavior |
|---|---|---|
| `score_threshold=None` | `similarity` | Return the top-k date-applicable passages without a relevance cutoff |
| `score_threshold=0.75` | `similarity_score_threshold` | Return only passages meeting the specified minimum relevance score |

The threshold-enabled strategy uses normalized relevance scores, where higher values indicate greater similarity. These scores are not calibrated probabilities of factual correctness or answerability.

Both search strategies retain deterministic effective-date filtering.

**Design rationale**

Dense similarity search returns the nearest available documents even when the question is unrelated to the knowledge base. This can provide irrelevant context to an LLM and increase the risk of unsupported answers.

An optional relevance threshold allows the application to exclude weak retrieval matches and return no evidence when the threshold is not satisfied.

When no documents are returned, `RAGService.ask()` produces `insufficient_evidence` without invoking the LLM.

**Current limitations**

- The default remains `score_threshold=None`; threshold filtering is not enabled automatically.
- No threshold has been calibrated against labeled in-scope and out-of-scope questions.
- High semantic similarity does not guarantee that evidence is sufficient to answer a question.
- Similarity thresholds do not replace query-scope classification or factual grounding checks.
- Threshold values may require recalibration when the embedding model, vector metric, or corpus changes.

**Future evaluation**

Representative policy questions, unsupported policy questions, greetings, and unrelated questions will be used to select and evaluate a threshold. False acceptance and false rejection rates will be monitored alongside retrieval relevance.

### Evaluation Requirements

The retrieval system should be evaluated on:

- Correct applicability filtering for historical and current policies.
- Retrieval relevance using Recall@k and MRR.
- Robustness against ambiguous or missing date information.
- Latency and inference cost.
- Incorrectly included or excluded documents.