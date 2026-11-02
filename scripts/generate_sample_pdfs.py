"""Generate fictional financial-policy PDFs for the RAG demo."""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "sample_documents"

POLICIES = {
    "repayment_policy_2026.pdf": {
        "title": "FinServe Repayment Policy",
        "effective": "2026-01-01",
        "sections": [
            (
                "1. Payment deadlines",
                (
                    "Monthly installments are due on the date specified "
                    "in the customer agreement."
                ),
            ),
            (
                "2. Missed payments",
                (
                    "After a missed installment, customer support must "
                    "inform the customer about the outstanding payment. "
                    "After two consecutive missed installments, the case "
                    "must be referred to the repayment assistance team."
                ),
            ),
            (
                "3. Payment extensions",
                (
                    "Customers may request an extension. Support agents "
                    "can explain the application procedure but cannot "
                    "approve an extension themselves."
                ),
            ),
        ],
    },
    "fee_schedule_2026.pdf": {
        "title": "FinServe Fee Schedule",
        "effective": "2026-01-01",
        "sections": [
            (
                "1. Late-payment fee",
                (
                    "The fictional late-payment administration fee is "
                    "EUR 12 per qualifying missed installment."
                ),
            ),
            (
                "2. Fee waivers",
                (
                    "Customer support agents may submit a fee-waiver "
                    "request, but only an authorized supervisor may "
                    "approve or reject it."
                ),
            ),
        ],
    },
    "customer_support_policy_2026.pdf": {
        "title": "FinServe Customer Support Policy",
        "effective": "2026-01-01",
        "sections": [
            (
                "1. Customer communication",
                (
                    "Agents must provide accurate information supported "
                    "by the currently effective company policies."
                ),
            ),
            (
                "2. Restricted promises",
                (
                    "Agents must not guarantee loan approval, promise "
                    "fee waivers, or modify repayment agreements without "
                    "appropriate authorization."
                ),
            ),
            (
                "3. Personal information",
                (
                    "Agents must not request customer passwords, PINs, "
                    "or full payment-card numbers through the chat."
                ),
            ),
            (
                "4. Escalation",
                (
                    "Cases involving disputed charges, repeated missed "
                    "payments, or policy exceptions must be referred "
                    "to the appropriate specialist team."
                ),
            ),
        ],
    },
    "fee_schedule_2025.pdf": {
        "title": "FinServe Fee Schedule - Superseded",
        "effective": "2025-01-01",
        "sections": [
            (
                "1. Late-payment fee",
                (
                    "The fictional late-payment administration fee "
                    "was EUR 20 per qualifying missed installment."
                ),
            ),
            (
                "2. Status",
                (
                    "This fee schedule was superseded by the "
                    "2026 fee schedule effective January 1, 2026."
                ),
            ),
        ],
    },
}


def generate_pdf(filename: str, policy: dict) -> None:
    """Create one searchable PDF from a fictional policy."""
    path = OUTPUT_DIR / filename
    styles = getSampleStyleSheet()

    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        title=policy["title"],
    )

    content = [
        Paragraph(policy["title"], styles["Title"]),
        Spacer(1, 12),
        Paragraph(
            f"Effective date: {policy['effective']}",
            styles["Normal"],
        ),
        Paragraph(
            "SYNTHETIC DOCUMENT - FOR DEMONSTRATION ONLY",
            styles["Normal"],
        ),
        Spacer(1, 20),
    ]

    for heading, body in policy["sections"]:
        content.append(Paragraph(heading, styles["Heading2"]))
        content.append(Paragraph(body, styles["BodyText"]))
        content.append(Spacer(1, 14))

    document.build(content)
    print(f"Generated: {path.name}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename, policy in POLICIES.items():
        generate_pdf(filename, policy)

    print(f"Total PDFs generated: {len(POLICIES)}")


if __name__ == "__main__":
    main()
