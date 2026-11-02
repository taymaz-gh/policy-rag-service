"""Streamlit interface for the FinServe financial-policy RAG service."""

import os
import re
from datetime import date
from time import perf_counter

import httpx
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
CITATION_PATTERN = re.compile(r"\[S\d+\]")

st.set_page_config(
    page_title="FinServe Copilot",
    # page_icon="  ",
    layout="wide",
)


def fetch_readiness() -> dict:
    """Read backend readiness without exposing internal errors."""
    try:
        response = httpx.get(f"{API_URL}/ready", timeout=5.0)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError):
        return {"ready": False, "generation_mode": "unavailable"}


def ask_backend(question: str, as_of: date) -> tuple[dict, float]:
    """Submit a question and return the API result and elapsed seconds."""
    started = perf_counter()

    with httpx.Client(timeout=90.0) as client:
        response = client.post(
            f"{API_URL}/ask",
            json={
                "question": question,
                "as_of": as_of.isoformat(),
            },
        )
        response.raise_for_status()
        result = response.json()

    return result, perf_counter() - started


def cited_source_ids(answer: str) -> set[str]:
    """Extract citation IDs from an answer."""
    return {match.group()[1:-1] for match in CITATION_PATTERN.finditer(answer)}


def render_assistant_message(message: dict, diagnostics: bool) -> None:
    """Display answer status, citations, and optional diagnostics."""
    result = message["result"]
    status = result.get("status", "unknown")
    answer = result.get("answer", "")
    sources = result.get("sources", [])

    if status == "retrieval_only":
        st.warning("LLM generation unavailable. Showing retrieved evidence only.")
    elif status == "blocked":
        st.warning("The request or response was blocked by screening.")
    elif status == "insufficient_evidence":
        st.info("No applicable policy evidence was retrieved.")
    elif status == "citation_validation_failed":
        st.error("The generated answer failed citation validation.")

    st.markdown(answer)

    cited = cited_source_ids(answer) if status == "answered" else set()
    cited_sources = [source for source in sources if source["id"] in cited]

    if cited_sources:
        st.markdown("**Cited sources**")
        for source in cited_sources:
            st.caption(f"[{source['id']}] {source['filename']}, page {source['page']}")

    if diagnostics:
        with st.expander("Developer diagnostics"):
            st.write(f"Response status: `{status}`")
            st.write(f"Client-observed latency: {message['latency']:.2f} s")
            st.write(f"Policy date: {message['as_of']}")
            st.write(f"Retrieved sources: {len(sources)}")
            st.write(f"Cited source IDs: {sorted(cited)}")

            if sources:
                st.dataframe(sources, hide_index=True)


def main() -> None:
    st.title("FinServe RAG Copilot")
    st.caption(
        "Financial-policy question answering using synthetic documents. "
        "Not a production financial advisory service."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Settings")
        as_of = st.date_input("Policy applicability date", value=date.today())
        diagnostics = st.toggle("Developer diagnostics", value=False)

        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.subheader("Backend")

        readiness = fetch_readiness()

        if readiness.get("ready"):
            st.success("Retrieval ready")
            st.write(
                f"Configured generation mode: `{readiness.get('generation_mode')}`"
            )
        else:
            st.error("Backend unavailable or not ready")

        st.caption(
            "Do not submit real customer identifiers, "
            "account details, or confidential records."
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_assistant_message(message, diagnostics)

    question = st.chat_input(
        "Ask about repayment rules, fees, or customer support...",
        max_chars=4000,
        disabled=not readiness.get("ready", False),
    )

    if question:
        selected_date = as_of.isoformat()

        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving applicable policy evidence..."):
                try:
                    result, latency = ask_backend(question, as_of)

                except httpx.TimeoutException:
                    st.error("The backend did not respond within the request timeout.")
                    return
                except httpx.ConnectError:
                    st.error(
                        "Cannot connect to FastAPI. Check that the backend "
                        "is running on port 8000."
                    )
                    return
                except httpx.HTTPStatusError as exc:
                    st.error(
                        f"FastAPI returned HTTP {exc.response.status_code}. "
                        "Check the backend terminal for the cause."
                    )
                    return
                except (httpx.HTTPError, ValueError):
                    st.error(
                        "The backend request failed or returned an invalid response."
                    )
                    return

            message = {
                "role": "assistant",
                "result": result,
                "latency": latency,
                "as_of": selected_date,
            }

            st.session_state.messages.append(message)
            render_assistant_message(message, diagnostics)


if __name__ == "__main__":
    main()
