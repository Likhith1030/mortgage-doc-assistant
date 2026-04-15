"""
core/embeddings.py — Create and manage the FAISS vector store.

WHAT THIS FILE DOES:
  Embeddings are numerical representations of text (arrays of ~1500 numbers).
  Similar text produces similar numbers. This lets us do semantic search:
  "find chunks about income verification" even if the exact words differ.

  This file:
    1. Builds a FAISS vector store from document chunks.
    2. Saves it to disk so we don't re-embed on every run (API calls cost money).
    3. Loads an existing store when one is found.

FAISS (Facebook AI Similarity Search):
  A library that stores millions of vectors and finds the nearest ones in
  milliseconds. Think of it as a search index but for meaning, not keywords.
"""

import os
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import config


def get_embeddings_model() -> OpenAIEmbeddings:
    """Return a configured OpenAI embeddings client."""
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
    )


def build_vector_store(chunks: List[Document]) -> FAISS:
    """
    Embed all chunks and build a FAISS index in memory.

    This calls the OpenAI Embeddings API once per batch — can take a few
    seconds for large documents. The result is saved to disk automatically
    if VECTOR_STORE_PATH is set.
    """
    embeddings = get_embeddings_model()
    store = FAISS.from_documents(chunks, embeddings)

    # Persist so future runs skip re-embedding.
    os.makedirs(config.VECTOR_STORE_PATH, exist_ok=True)
    store.save_local(config.VECTOR_STORE_PATH)

    return store


def load_vector_store() -> FAISS:
    """
    Load the FAISS index from disk.
    Raises FileNotFoundError if the store hasn't been built yet.
    """
    if not os.path.exists(config.VECTOR_STORE_PATH):
        raise FileNotFoundError(
            f"No vector store found at {config.VECTOR_STORE_PATH}. "
            "Run build_vector_store() first."
        )
    embeddings = get_embeddings_model()
    return FAISS.load_local(
        config.VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,  # safe: we created this file
    )


def get_or_build_store(chunks: List[Document] | None = None) -> FAISS:
    """
    Convenience function: load from disk if available, else build from chunks.

    Usage in agents:
        store = get_or_build_store(chunks)   # first run — builds
        store = get_or_build_store()          # subsequent runs — loads
    """
    try:
        return load_vector_store()
    except FileNotFoundError:
        if chunks is None:
            raise ValueError("No vector store on disk and no chunks provided to build one.")
        return build_vector_store(chunks)
