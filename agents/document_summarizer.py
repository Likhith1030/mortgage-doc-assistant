"""
agents/document_summarizer.py — Summarize mortgage documents for loan officers.

WORKFLOW:
  Borrower uploads a 40-page income packet → loan officer needs a 1-page brief.
  This agent reads all chunks and asks the LLM to produce a structured summary
  covering: borrower identity, income, assets, liabilities, and red flags.

TECHNIQUE: Map-Reduce summarization
  - MAP:   Summarize each chunk independently (parallel, cheap).
  - REDUCE: Combine all chunk summaries into one final summary.
  This avoids the "lost in the middle" problem where LLMs ignore content
  buried in very long prompts.
"""

from typing import List
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

import config


# ── Prompts ───────────────────────────────────────────────────────────────────

# Used on each individual chunk (the MAP step).
MAP_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are reviewing a section of a mortgage application document.
Extract and list any key facts about:
- Borrower name, SSN (last 4 only), contact info
- Employment and income figures
- Assets (bank accounts, investments, real estate)
- Liabilities (debts, monthly payments)
- Any anomalies, inconsistencies, or missing information

Section:
{text}

Key facts from this section:""",
)

# Used to combine all chunk summaries into a final brief (the REDUCE step).
REDUCE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are a senior mortgage underwriter. Below are notes extracted
from sections of a mortgage application. Combine them into a single, structured
Loan Officer Brief using these headings:

## Borrower Profile
## Income & Employment Summary
## Assets & Liabilities
## Risk Flags & Missing Items
## Recommendation for Next Steps

Notes:
{text}

Loan Officer Brief:""",
)


def summarize_document(chunks: List[Document]) -> str:
    """
    Run map-reduce summarization on a list of document chunks.

    Args:
        chunks: Output of core.document_processor.process_document()

    Returns:
        A structured Loan Officer Brief as a string.
    """
    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        temperature=0,
        openai_api_key=config.OPENAI_API_KEY,
    )

    # load_summarize_chain with "map_reduce" handles the two-pass logic.
    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=MAP_PROMPT,
        combine_prompt=REDUCE_PROMPT,
        verbose=False,
    )

    result = chain.invoke({"input_documents": chunks})
    return result["output_text"]
