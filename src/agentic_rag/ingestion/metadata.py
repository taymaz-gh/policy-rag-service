"""Validate and attach policy metadata to LangChain documents."""

import json
from datetime import date
from pathlib import Path

from langchain_core.documents import Document

REQUIRED_FIELDS = {
    "policy_id",
    "document_type",
    "version",
    "effective_from",
    "effective_to",
}


def load_document_manifest(
    manifest_path: str | Path,
) -> dict[str, dict]:
    """Load and validate the document manifest."""

    path = Path(manifest_path)

    with path.open(encoding="utf-8") as file:
        manifest = json.load(file)

    if not isinstance(manifest, dict):
        raise TypeError("Document manifest must be a JSON object")

    for filename, metadata in manifest.items():
        if not isinstance(metadata, dict):
            raise TypeError(f"Invalid metadata for {filename}")

        missing = REQUIRED_FIELDS - metadata.keys()
        if missing:
            raise ValueError(f"Missing metadata for {filename}: {sorted(missing)}")

        for field in ("policy_id", "document_type", "version"):
            if not isinstance(metadata[field], str) or not metadata[field]:
                raise ValueError(f"Invalid {field} for {filename}")

        start = date.fromisoformat(metadata["effective_from"])
        end_value = metadata["effective_to"]

        if end_value is not None:
            end = date.fromisoformat(end_value)
            if end < start:
                raise ValueError(f"Expiry precedes effective date: {filename}")

    return manifest


def enrich_document_metadata(
    pages: list[Document],
    manifest: dict[str, dict],
) -> list[Document]:
    """Return documents with their policy metadata attached."""

    enriched = []

    for page in pages:
        filename = page.metadata.get("filename")

        if filename not in manifest:
            raise ValueError(f"Document missing from manifest: {filename}")

        policy = manifest[filename]

        # Use a new metadata dictionary to avoid mutating input pages.

        metadata = {
            **page.metadata,
            "policy_id": policy["policy_id"],
            "document_type": policy["document_type"],
            "version": policy["version"],
            "effective_from": policy["effective_from"],
            "effective_from_int": int(policy["effective_from"].replace("-", "")),
            "effective_to_int": (
                int(policy["effective_to"].replace("-", ""))
                if policy["effective_to"] is not None
                else 99991231
            ),
        }

        # Keep metadata compatible with vector stores that do not
        # accept null-valued metadata fields.
        if policy["effective_to"] is not None:
            metadata["effective_to"] = policy["effective_to"]
        else:
            metadata.pop("effective_to", None)

        enriched.append(
            Document(
                page_content=page.page_content,
                metadata=metadata,
            )
        )

    return enriched
