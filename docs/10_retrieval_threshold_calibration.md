# 10. Retrieval Relevance Threshold Calibration

## Objective

Dense semantic retrieval returns the nearest available document chunks even when the knowledge base does not contain the information needed to answer a question.

For financial-policy question answering, this can introduce irrelevant or incomplete evidence into the generation context and increase the risk of unsupported claims.

This experiment evaluates whether a minimum embedding-similarity threshold can distinguish answerable policy questions from unsupported and out-of-scope questions.

## Experimental Setup

**Evaluation script:** `scripts/evaluate_retrieval.py`

**Retrieval implementation:** `src/agentic_rag/retrieval/vector_store.py`

| Component | Configuration |
|---|---|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dimensions | 384 |
| Vector database | Chroma |
| Distance metric | Cosine distance |
| Embedding normalization | Enabled |
| Chunking | Recursive character splitting |
| Chunk size / overlap | 500 / 75 characters |
| Knowledge base | Four synthetic policy PDFs, six chunks |
| Metadata filtering | Effective-date applicability |
| Evaluation ranking | Top-1 similarity |

The corpus and retrieval settings match the initial RAG implementation.

## Similarity Score Definition

For query embedding $\mathbf{q}$ and document embedding $\mathbf{d}$, cosine similarity is:

$$
s(\mathbf{q},\mathbf{d})=\frac{\mathbf{q}^{\mathsf T}\mathbf{d}}{\|\mathbf{q}\|\|\mathbf{d}\|}
$$

Chroma is configured for cosine distance:

$$
d_{\mathrm{cosine}}=1-s(\mathbf{q},\mathbf{d})
$$

The evaluation retrieves the nearest document using `similarity_search_with_score()`, then computes:

$$
s=1-d_{\mathrm{cosine}}
$$

The score is therefore cosine similarity. Higher values represent greater semantic similarity.

Cosine similarity can be negative and is not a probability of answerability or factual correctness.

## Evaluation Dataset

Two manually labeled sets were used:

| Dataset | Answerable | Unanswerable | Total |
|---|---:|---:|---:|
| Development | 5 | 7 | 12 |
| Held-out test | 5 | 7 | 12 |
| Total | 10 | 14 | 24 |

Questions were assigned one of three categories:

- **Supported:** An applicable document contains the information required to answer.
- **Unsupported:** The question concerns financial services, but the applicable corpus lacks the requested information.
- **Out of scope:** The question is unrelated to the financial-policy knowledge base.

Unsupported and out-of-scope questions share the negative label for the binary calibration task.

The development set was used to select the threshold. The held-out set was not used during threshold selection.

## Threshold Selection

For the highest-scoring date-applicable chunk, retrieval acceptance is defined as:

$$
\hat y=\mathbb{1}[s_{\max}\geq\tau]
$$

The experiment evaluated candidate thresholds from 0.00 to 1.00 in increments of 0.01.

The selection criterion was balanced accuracy:

$$
\mathrm{BA}=\frac12\left(\frac{TP}{TP+FN}+\frac{TN}{TN+FP}\right)
$$

Ties were resolved by preferring fewer false positives, followed by a higher threshold.

### Development Results

The highest-scoring supported question had a score of 0.8421, while the lowest-scoring supported question had a score of 0.4746.

The highest score among unsupported and out-of-scope questions was 0.4423.

The selected development threshold was:

**$\tau=0.47$**

| Metric | Result |
|---|---:|
| True positives | 5 |
| False positives | 0 |
| True negatives | 7 |
| False negatives | 0 |
| Balanced accuracy | 1.000 |

The threshold separated all 12 development questions.

This perfect in-sample result is not sufficient evidence of generalization.

## Held-Out Evaluation

The selected threshold of 0.47 was applied without modification to the 12 held-out questions.

| Metric | Result |
|---|---:|
| True positives | 4 |
| False positives | 1 |
| True negatives | 6 |
| False negatives | 1 |
| Supported-question recall | 0.800 |
| Unsupported-question specificity | 0.857 |
| Balanced accuracy | 0.829 |

### False Negative

**Question:** Who handles cases involving disputed charges?

**Expected:** Accept — the customer-support policy specifies escalation to the appropriate specialist team.

**Top-1 score:** 0.3518

**Retrieved source:** `customer_support_policy_2026.pdf`

**Observed:** Rejected because the similarity score was below 0.47.

This illustrates that semantic paraphrases can receive relatively low similarity scores even when relevant evidence exists.

### False Positive

**Question:** How many days can a payment extension last?

**Expected:** Reject — the repayment policy discusses payment extensions but does not specify a maximum duration.

**Top-1 score:** 0.5446

**Retrieved source:** `repayment_policy_2026.pdf`

**Observed:** Accepted despite insufficient evidence.

This illustrates that a chunk can be semantically relevant without containing the specific information required to answer the question.

## Interpretation

The development set suggested a clean separation between answerable and unanswerable questions, but the held-out evaluation exposed overlapping score distributions.

The two classification errors correspond to different failure modes:

1. **False rejection:** Relevant evidence is present, but its semantic similarity is relatively low.
2. **False acceptance:** Semantically similar evidence is present, but it does not answer the question.

These errors cannot necessarily be eliminated by adjusting a single global similarity threshold.

The results also demonstrate the importance of keeping retrieval relevance and answerability as distinct evaluation objectives.


### Retrieval Threshold Evaluation: summary

An initial experiment evaluated minimum cosine-similarity thresholds for distinguishing answerable policy questions from unsupported and unrelated requests.

The selected development threshold was 0.47, achieving 100% balanced accuracy on 12 development questions but 82.9% on 12 held-out questions.

The held-out errors demonstrated that semantic relevance does not guarantee evidence sufficiency. Consequently, the threshold remains experimental and is not enforced as a strict application-wide rejection rule.

The complete methodology, evaluation dataset, results, failure analysis, and engineering decision are documented in [Retrieval Relevance Threshold Calibration].



## Engineering Decision

**The threshold of 0.47 is retained as an experimental baseline, not enabled as a strict application-wide gate.**

The initial application will continue using deterministic policy-date filtering and grounded generation.

The next reliability improvement will introduce an evidence-sufficiency assessment that evaluates whether retrieved passages actually contain the information requested by the question.

A similarity threshold may later be used as an initial coarse filter, but it should not independently determine whether an answer is supported.

## Limitations

- The dataset contains only 24 manually labeled questions.
- Both evaluation sets reference the same small synthetic corpus.
- Several questions share closely related topics and terminology.
- Each question is evaluated using only the highest-scoring chunk.
- No statistical confidence intervals or cross-corpus validation have been performed.
- The experiment measures retrieval acceptance rather than generated-answer correctness.
- Business consequences of false acceptance and false rejection have not been formally quantified.
- Threshold behavior may change with different embeddings, chunk sizes, or document collections.

## Next Steps

- Expand the labeled dataset with more diverse paraphrases and difficult unsupported questions.
- Evaluate retrieval ranking and evidence coverage using Recall@k and related metrics.
- Introduce question-evidence sufficiency assessment.
- Evaluate factual grounding and citation support independently.
- Add regression tests for identified false-positive and false-negative cases.
- Recalibrate only using designated development data and reserve fresh test data for subsequent validation.

## Reproducibility

From the repository root:

```powershell
python scripts/evaluate_retrieval.py
```

Run the associated calibration tests:

```powershell
python -m pytest tests/test_retrieval_calibration.py -q
```

The experiment uses local sentence embeddings and synthetic policies. Gemini API credentials are not required.