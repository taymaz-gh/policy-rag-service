# 9. End-to-End RAG Answer Evaluation

## Objective

This experiment evaluates whether the existing RAG application produces factually supported answers and appropriately abstains when the available financial-policy documents do not contain the requested information.

It follows the retrieval threshold-calibration experiment, which demonstrated that cosine similarity alone does not reliably determine answerability.

## System Under Evaluation

- PDF ingestion and recursive character chunking.
- MiniLM sentence embeddings and Chroma retrieval.
- Deterministic effective-date filtering.
- Grounded answer generation with Gemini.
- Citation-identifier validation.
- Basic sensitive-input screening and conversational routing.

No evidence-sufficiency assessor or similarity threshold is enabled for this baseline evaluation.

## Experimental Design

**Script:** `scripts/evaluate_rag_answers.py`

The dataset contains 11 labeled questions:

| Category | Questions |
|---|---:|
| Supported by the policy corpus | 5 |
| Financial-policy questions lacking sufficient evidence | 4 |
| Outside the knowledge-base scope | 2 |

Each question is evaluated with an explicit policy-applicability date.

The experiment uses the live Gemini model, the existing RAG service, and the synthetic FinServe documents.

Generation fallback is disabled during evaluation so that provider failures are not confused with successful abstention.

## Evaluation Criteria

### Behavior Correctness

A response is considered behaviorally correct when it:

- Answers a supported question consistently with the applicable policy.
- Acknowledges missing information for unsupported questions.
- Declines or appropriately redirects out-of-scope questions.

### Factual Grounding

Substantive claims must be supported by the retrieved evidence.

Plausible statements based on general financial knowledge are not sufficient unless supported by the provided policy passages.

### Citation Support

Citation-identifier validity and citation support are evaluated separately.

A syntactically valid citation does not guarantee that the referenced passage supports the associated claim.

### Latency

The evaluation records the elapsed time for each RAG service invocation, including retrieval and generation.

## Methodology

1. Initialize the corpus, embedding model, and Chroma collection.
2. Construct the date-aware retriever for each question.
3. Invoke the existing RAG service.
4. Record response status, generated text, source passages, and latency.
5. Manually review factual support, citation support, and behavior correctness.
6. Analyze failure cases before introducing additional safeguards.

## Results

Pending execution and manual evaluation.

Results will include response-status distributions, correct-answer and abstention rates, unsupported claims, citation-support failures, and observed latency.

## Engineering Decision

The need for a separate evidence-sufficiency assessor will be determined from the observed failure modes.

Potential interventions include retrieval improvements, grounded-prompt revisions, explicit abstention handling, evidence assessment, or post-generation verification.

Additional model calls will not be introduced without evaluating their expected reliability benefit and latency cost.

## Reproducibility

From the repository root:

```powershell
python scripts/evaluate_rag_answers.py
```

A successful run creates:
`data/evaluation/rag_answers.json`

The experiment requires a configured and available Gemini model. It is not part of the credential-free CI test suite.

The generated evaluation file contains synthetic question and evidence data and must not be used as a template for logging real customer information.