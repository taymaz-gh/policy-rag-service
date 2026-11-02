"""Configure the hosted chat model for RAG generation."""

import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()


def create_chat_model() -> BaseChatModel:
    """Create a chat model using environment configuration."""
    provider = os.getenv("RAG_LLM_PROVIDER", "gemini").lower()
    model_name = os.getenv("RAG_LLM_MODEL", "gemini-3.8-flash")

    if provider == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY is required for Gemini generation.")

        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            max_retries=2,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
