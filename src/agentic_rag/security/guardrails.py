"""Basic sensitive-information screening for user questions."""

import re
from dataclasses import dataclass

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

NL_IBAN_PATTERN = re.compile(
    r"\bNL\d{2}(?:[\s-]?[A-Z]{4})(?:[\s-]?\d{4}){2}" r"(?:[\s-]?\d{2})\b",
    re.IGNORECASE,
)

CARD_CANDIDATE_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")


@dataclass(frozen=True)
class ScreeningResult:
    allowed: bool
    reason: str | None = None


def _luhn_valid(number: str) -> bool:
    """Check the checksum of a candidate payment-card number."""
    digits = [int(character) for character in number]

    if not 13 <= len(digits) <= 19:
        return False

    total = 0

    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit

    return total % 10 == 0


def screen_question(question: str) -> ScreeningResult:
    """Block obvious sensitive identifiers before external processing."""
    if not question.strip():
        return ScreeningResult(False, "Question cannot be empty.")

    if len(question) > 4000:
        return ScreeningResult(False, "Question exceeds the size limit.")

    if EMAIL_PATTERN.search(question):
        return ScreeningResult(
            False, "Remove personal email addresses before continuing."
        )

    if NL_IBAN_PATTERN.search(question):
        return ScreeningResult(
            False, "Remove bank account identifiers before continuing."
        )

    for match in CARD_CANDIDATE_PATTERN.finditer(question):
        digits = re.sub(r"\D", "", match.group())
        if _luhn_valid(digits):
            return ScreeningResult(
                False, "Remove payment-card numbers before continuing."
            )

    return ScreeningResult(True)
