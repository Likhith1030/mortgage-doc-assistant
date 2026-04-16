from typing import List
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

import config


MAP_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are reviewing a section of a mortgage application document.
Extract key facts about: borrower identity, employment, income, assets, liabilities, and anomalies.
The section header shows a page reference like [p.3] — prefix each fact you extract with that same tag.

Section:
{text}

Key facts (each starting with its [p.X] citation):""",
)

REDUCE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are a senior mortgage underwriter. Below are page-annotated facts extracted from a mortgage application.
Write a concise Loan Officer Brief using these headings. Include [p.X] citations next to specific figures and findings.

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
    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        temperature=0,
        openai_api_key=config.OPENAI_API_KEY,
    )

    # Prepend page reference to each chunk so the MAP step can cite sources.
    annotated = [
        Document(
            page_content=f"[p.{c.metadata.get('page', '?')}]\n{c.page_content}",
            metadata=c.metadata,
        )
        for c in chunks
    ]

    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=MAP_PROMPT,
        combine_prompt=REDUCE_PROMPT,
        verbose=False,
    )

    result = chain.invoke({"input_documents": annotated})
    return result["output_text"]
