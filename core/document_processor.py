"""
core/document_processor.py — Load and chunk mortgage documents.

PDFs are parsed via the Unstructured Platform API (high-quality layout-aware
extraction). Plain text files use LangChain's TextLoader.
"""

from pathlib import Path
from typing import List

import httpx
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import config


def _load_pdf_via_unstructured(file_path: str) -> List[Document]:
    """Call Unstructured Platform API to extract text from a PDF."""
    if not config.UNSTRUCTURED_API_KEY:
        raise ValueError(
            "UNSTRUCTURED_API_KEY is not set. Add it to your .env file."
        )

    path = Path(file_path)
    with open(file_path, "rb") as f:
        pdf_bytes = f.read()

    # Strip any path suffix the user may have copied from the platform dashboard
    # (e.g. /api/v1) — the actual partition endpoint is always /general/v0/general.
    base = config.UNSTRUCTURED_API_URL.rstrip('/')
    # Keep only the scheme + host
    from urllib.parse import urlparse
    parsed = urlparse(base)
    host_base = f"{parsed.scheme}://{parsed.netloc}"
    url = f"{host_base}/general/v0/general"
    headers = {"unstructured-api-key": config.UNSTRUCTURED_API_KEY}

    with httpx.Client(timeout=180.0) as client:
        response = client.post(
            url,
            headers=headers,
            files={"files": (path.name, pdf_bytes, "application/pdf")},
            data={"strategy": "auto", "languages": "eng"},
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"Unstructured API error {response.status_code}: {response.text[:400]}"
        )

    elements = response.json()
    if not elements:
        raise ValueError(f"Unstructured returned no content for {path.name}")

    # Group element text by page number so one Document = one page
    pages: dict[int, list[str]] = {}
    for el in elements:
        text = (el.get("text") or "").strip()
        if not text:
            continue
        meta = el.get("metadata") or {}
        page_num = int(meta.get("page_number") or 1)
        pages.setdefault(page_num, []).append(text)

    docs = []
    for page_num in sorted(pages.keys()):
        docs.append(Document(
            page_content="\n\n".join(pages[page_num]),
            metadata={"source": file_path, "page": page_num},
        ))

    return docs


def load_document(file_path: str) -> List[Document]:
    """Load a PDF (via Unstructured API) or .txt file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    if path.suffix.lower() == ".pdf":
        return _load_pdf_via_unstructured(file_path)

    return TextLoader(str(path), encoding="utf-8").load()


def split_into_chunks(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def process_document(file_path: str) -> List[Document]:
    """End-to-end: load → split → return chunks with source metadata."""
    raw_docs = load_document(file_path)
    chunks = split_into_chunks(raw_docs)
    for chunk in chunks:
        chunk.metadata["source"] = file_path
    return chunks
