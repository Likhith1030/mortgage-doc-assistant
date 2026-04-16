"""
core/embeddings.py — Create and manage FAISS vector stores.

Each document now gets its own store path so uploads never overwrite each other.
Pass store_path explicitly to isolate per-document indexes.
"""

import os
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import config


def get_embeddings_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
    )


def build_vector_store(chunks: List[Document], store_path: str = None) -> FAISS:
    """
    Embed chunks and build a FAISS index.

    store_path: where to persist the index.  Defaults to config.VECTOR_STORE_PATH.
                Pass a per-document path to keep indexes isolated.
    """
    embeddings = get_embeddings_model()
    store = FAISS.from_documents(chunks, embeddings)

    save_to = store_path or config.VECTOR_STORE_PATH
    os.makedirs(save_to, exist_ok=True)
    store.save_local(save_to)

    return store


def load_vector_store(store_path: str = None) -> FAISS:
    """Load a FAISS index from disk. Raises FileNotFoundError if absent."""
    load_from = store_path or config.VECTOR_STORE_PATH
    if not os.path.exists(load_from):
        raise FileNotFoundError(
            f"No vector store at {load_from}. Build one first."
        )
    embeddings = get_embeddings_model()
    return FAISS.load_local(
        load_from,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_or_build_store(
    chunks: List[Document] | None = None,
    store_path: str = None,
) -> FAISS:
    try:
        return load_vector_store(store_path)
    except FileNotFoundError:
        if chunks is None:
            raise ValueError("No vector store on disk and no chunks provided.")
        return build_vector_store(chunks, store_path)
