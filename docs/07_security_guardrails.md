# 7. Security Guardrails and Sensitive-Content Screening

## Problem

Financial customer-support questions may contain personal identifiers, payment information, or instructions attempting to override the application’s intended behavior.

Sending such content directly to an external LLM may create privacy, security, and compliance risks.

## Design Decision

The initial application uses deterministic input screening before retrieval and LLM invocation.

This provides a testable first layer of protection without requiring an additional moderation API.

The safeguards are designed as independently testable components rather than relying entirely on prompt instructions.

## Implementation

**Module:** `src/agentic_rag/security/guardrails.py`

**Function:** `screen_question()`

The initial screening checks for:

- Empty or oversized questions.
- Email-address patterns.
- Netherlands IBAN patterns.
- Candidate payment-card numbers passing the Luhn checksum.

**Module:** `src/agentic_rag/generation/rag_service.py`

**Method:** `RAGService.ask()`

Requests rejected by the screening layer are not forwarded to the retriever or LLM.

### Request Flow

```text
             User Question
                   |
                   v
           Input Screening
                   |
              +----+----+
              |         |
            Block      Allow
              |         |
              v         v
         Return Error  Retrieval
                        |
                        v
                 LLM Generation
                        |
                        v
                 Citation Checks
                        |
                        v
                     Response
```

## Prompt-Injection Defense

The generation prompt explicitly distinguishes trusted application instructions from retrieved document excerpts.

Retrieved passages are treated as untrusted source material.

The initial system does not provide the LLM with tools that can modify accounts, approve credit, change financial agreements, or perform transactions.

## Important Limitations

- Regular-expression screening is not comprehensive PII detection.
- Luhn validation does not establish that a payment-card number is real.
- The initial implementation does not provide multilingual or contextual sensitivity classification.
- Prompt separation and instructions cannot guarantee resistance to injection attacks.
- Citation identifier checks do not establish factual correctness.
- Business compliance and access authorization require separate controls.

## Implementation Status

**Implemented**
- Basic input validation and sensitive-identifier screening.
- Blocking before retrieval and external LLM invocation.
- Grounding instructions for retrieved evidence.
- Citation identifier validation.
- Automated tests for selected screening and response behaviors.

**Planned**
- Expanded PII detection and redaction.
- Indirect prompt-injection adversarial tests.
- Role-based document access control.
- Business-policy compliance checks.
- Secure audit logging and monitoring.
- Human escalation for consequential decisions.