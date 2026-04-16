from dataclasses import dataclass, field
from typing import List, Optional
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

import config
from core.embeddings import build_vector_store
from core.document_processor import process_document


class _RuleCheck(BaseModel):
    status: str  # "PASS", "FAIL", "NEEDS REVIEW"
    reasoning: str


@dataclass
class ComplianceResult:
    rule_name: str
    status: str
    detail: str
    sources: List[str] = field(default_factory=list)


@dataclass
class ComplianceReport:
    document_name: str
    results: List[ComplianceResult]

    @property
    def passed(self):
        return [r for r in self.results if r.status == "PASS"]

    @property
    def failed(self):
        return [r for r in self.results if r.status == "FAIL"]

    @property
    def needs_review(self):
        return [r for r in self.results if r.status == "NEEDS REVIEW"]

    def summary(self) -> str:
        lines = [
            f"Compliance Report: {self.document_name}",
            f"  PASS: {len(self.passed)}  FAIL: {len(self.failed)}  NEEDS REVIEW: {len(self.needs_review)}",
            "",
        ]
        for r in self.results:
            icon = {"PASS": "✓", "FAIL": "✗", "NEEDS REVIEW": "?"}[r.status]
            lines.append(f"  [{icon}] {r.rule_name}: {r.detail}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "document_name": self.document_name,
            "results": [
                {"rule": r.rule_name, "status": r.status, "detail": r.detail, "sources": r.sources}
                for r in self.results
            ],
            "totals": {
                "pass": len(self.passed),
                "fail": len(self.failed),
                "review": len(self.needs_review),
            },
        }


COMPLIANCE_CHECKS = [
    {
        "rule_name": "Debt-to-Income Ratio (DTI)",
        "question": "What is the borrower's total monthly debt payment and gross monthly income? Calculate the DTI ratio. Is it below 43%?",
    },
    {
        "rule_name": "Loan-to-Value Ratio (LTV)",
        "question": "What is the loan amount and the appraised property value? Calculate the LTV. Is it within acceptable limits (≤ 80% for conventional, ≤ 96.5% for FHA)?",
    },
    {
        "rule_name": "Required Document Checklist",
        "question": "Are the following documents present or referenced: W-2s (last 2 years), pay stubs (last 30 days), tax returns (last 2 years), bank statements (last 2 months), government-issued ID?",
    },
    {
        "rule_name": "Employment Verification",
        "question": "Is the borrower's employment verified? Is there at least 2 years of employment history documented?",
    },
    {
        "rule_name": "Credit Score Eligibility",
        "question": "What is the borrower's credit score? Is it at or above the minimum threshold (620 for conventional, 580 for FHA)?",
    },
    {
        "rule_name": "Fair Lending — Protected Class Flags",
        "question": "Are there any notations, denials, or conditions that appear to be based on race, color, religion, national origin, sex, marital status, or age?",
    },
    {
        "rule_name": "Qualified Mortgage (QM) Safe Harbor",
        "question": "Does the loan have: term ≤ 30 years, no negative amortization, no interest-only payments, and points/fees ≤ 3% of loan amount?",
    },
]

_SYSTEM = (
    "You are a mortgage compliance analyst. Review the document context and evaluate the compliance check. "
    "Be precise — cite specific numbers from the document (e.g., 'DTI is 38.2%, below the 43% threshold'). "
    "If the information needed to evaluate the check is absent, respond with NEEDS REVIEW and state what is missing."
)


def run_compliance_check(
    document_path: str,
    store_path: Optional[str] = None,
    report_save_path: Optional[str] = None,
) -> ComplianceReport:
    """
    Run compliance checks on a mortgage document.

    store_path: path to save/load the FAISS vector store (per-document isolation).
    report_save_path: if provided, save the JSON report to this file path.
    """
    chunks = process_document(document_path)
    store = build_vector_store(chunks, store_path=store_path)

    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        temperature=0,
        openai_api_key=config.OPENAI_API_KEY,
    ).with_structured_output(_RuleCheck)

    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": config.RETRIEVAL_TOP_K,
            "fetch_k": config.RETRIEVAL_TOP_K * 4,
            "lambda_mult": 0.7,
        },
    )

    results = []
    for check in COMPLIANCE_CHECKS:
        docs = retriever.invoke(check["question"])
        context = "\n\n".join(d.page_content for d in docs)
        sources = list(dict.fromkeys(
            f"page {d.metadata.get('page', '?')} of {d.metadata.get('source', 'unknown')}"
            for d in docs
        ))

        result = llm.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=f"Document Context:\n{context}\n\nCompliance Check: {check['question']}"),
        ])

        results.append(ComplianceResult(
            rule_name=check["rule_name"],
            status=result.status,
            detail=result.reasoning,
            sources=sources,
        ))

    report = ComplianceReport(document_name=document_path, results=results)

    if report_save_path:
        with open(report_save_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

    return report
