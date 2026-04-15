"""
core/document_processor.py — Load and chunk mortgage documents.

WHAT THIS FILE DOES:
  1. Reads a PDF (or plain text) file from disk.
  2. Splits it into overlapping chunks so that the LLM can fit each piece
     into its context window.
  3. Returns a list of LangChain Document objects, each carrying the text
     and metadata (source file, page number).

WHY CHUNKS?
  GPT-4o has a ~128k token context window, but sending an entire 50-page
  document each time is slow and expensive. We split once, embed once, and
  only retrieve the 5 most-relevant chunks at query time.
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import config


def load_document(file_path: str) -> List[Document]:
    """
    Load a PDF or .txt file and return a list of raw Document objects
    (one per page for PDFs, one block for text files).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")

    return loader.load()


def split_into_chunks(documents: List[Document]) -> List[Document]:
    """
    Split raw documents into overlapping chunks suitable for embedding.

    RecursiveCharacterTextSplitter tries to split on paragraph breaks first,
    then sentences, then words — so chunks stay semantically coherent.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def process_document(file_path: str) -> List[Document]:
    """
    End-to-end: load → split → return chunks with source metadata.

    Each returned Document has:
      .page_content  — the text of the chunk
      .metadata      — {"source": "path/to/file.pdf", "page": 3}
    """
    raw_docs = load_document(file_path)
    chunks   = split_into_chunks(raw_docs)

    # Stamp the file path on every chunk so we can trace back the source.
    for chunk in chunks:
        chunk.metadata["source"] = file_path

    return chunks
