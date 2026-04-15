"""
core/retrieval.py — Retrieval-Augmented Generation (RAG) pipeline.

WHAT IS RAG?
  Instead of asking the LLM to memorize mortgage rules (it can't, reliably),
  we:
    1. Embed the user's question as a vector.
    2. Find the top-K document chunks most similar to that question.
    3. Paste those chunks into the prompt as context.
    4. Let the LLM answer *using that context*.

  This grounds the answer in real documents and dramatically reduces hallucination.

THIS FILE:
  Wraps LangChain's RetrievalQA chain, which handles steps 1-4 automatically.
"""

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

import config


# ── Prompt template ───────────────────────────────────────────────────────────
# {context} is filled with retrieved chunks; {question} is the user's query.
MORTGAGE_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a mortgage document analysis assistant helping loan officers
at a bank. Use only the provided context to answer the question.
If the answer is not in the context, say "I could not find this information
in the provided documents."

Context:
{context}

Question: {question}

Answer (be specific, cite page numbers or section names when available):""",
)


def build_qa_chain(vector_store: FAISS) -> RetrievalQA:
    """
    Build a RetrievalQA chain connected to the given FAISS vector store.

    The chain automatically:
      - Converts the question to an embedding
      - Fetches top-K relevant chunks from FAISS
      - Injects them into the prompt
      - Returns the LLM's answer + the source chunks used
    """
    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        temperature=0,            # 0 = deterministic/factual, not creative
        openai_api_key=config.OPENAI_API_KEY,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.RETRIEVAL_TOP_K},
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",       # "stuff" = paste all chunks into one prompt
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": MORTGAGE_QA_PROMPT},
    )


def ask(chain: RetrievalQA, question: str) -> dict:
    """
    Ask a question against the document store.

    Returns:
        {
          "answer": "...",
          "sources": ["page 3 of income_verification.pdf", ...]
        }
    """
    result = chain.invoke({"query": question})

    sources = [
        f"page {doc.metadata.get('page', '?')} of {doc.metadata.get('source', 'unknown')}"
        for doc in result.get("source_documents", [])
    ]

    return {
        "answer": result["result"],
        "sources": list(dict.fromkeys(sources)),  # deduplicate while preserving order
    }
