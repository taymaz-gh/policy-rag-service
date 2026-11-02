"""Diagnose hosted LLM latency without running the RAG pipeline."""

import argparse
import os
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure one direct Gemini generation request."
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="LangChain retry count; use zero for diagnosis.",
    )
    args = parser.parse_args()

    if args.retries < 0:
        parser.error("--retries must be nonnegative")

    load_dotenv(ROOT / ".env")

    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY is not configured.")

    model_name = os.getenv("RAG_LLM_MODEL", "gemini-3.8-flash")

    model = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        max_retries=args.retries,
    )

    print(f"Model: {model_name}")
    print(f"LangChain max_retries: {args.retries}")

    started = perf_counter()

    try:
        response = model.invoke("Reply with exactly one word: OK.")
    except Exception as exc:  # noqa: BLE001
        elapsed = perf_counter() - started
        print(f"Result: failure after {elapsed:.2f}s")
        print(f"Exception type: {type(exc).__name__}")
        # Print diagnostic details only for a synthetic request.
        message = str(exc)
        print(f"Message: {message[:500]}")
        return

    elapsed = perf_counter() - started

    print(f"Result: success after {elapsed:.2f}s")
    print(f"Response: {str(response.content)[:200]}")
    print(f"Usage metadata: {response.usage_metadata}")


if __name__ == "__main__":
    main()
