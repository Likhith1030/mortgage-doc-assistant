"""
agents/compliance_checker.py — Check a mortgage application against regulatory rules.

WORKFLOW:
  1. Load the compliance rules from the knowledge base (plain text file).
  2. Load the mortgage application document chunks.
  3. For each rule category, ask the LLM: "Does this application satisfy this rule?"
  4. Return a structured compliance report with PASS / FAIL / NEEDS REVIEW flags.

WHY A SEPARATE KNOWLEDGE BASE?
  Compliance rules change frequently (TRID, QM, HMDA updates). Keeping them in
  a plain text file means compliance officers can update rules without touching code.

RULES COVERED (see data/knowledge_base/compliance_rules.txt):
  - Debt-to-Income ratio thresholds
  - Loan-to-Value ratio limits
  - Required document checklist
  - Fair lending (ECOA, HMDA) flags
  - QM (Qualified Mortgage) safe harbor checks
"""

from dataclasses import dataclass, field
from typing import List

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document

import config
from core.retrieval import build_qa_chain, ask
from core.embeddings import build_vector_store, load_vector_store
from core.document_processor import process_document


@dataclass
class ComplianceResult:
    rule_name: str
    status: str          # "PASS", "FAIL", "NEEDS REVIEW"
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


# ── Compliance checks ─────────────────────────────────────────────────────────

COMPLIANCE_CHECKS = [
    {
        "rule_name": "Debt-to-Income Ratio (DTI)",
        "question": "What is the borrower's total monthly debt payment and gross monthly income? Calculate the DTI ratio. Is it below 43%?",
    },
    {
        "rule_name": "Loan-to-Value Ratio (LTV)",
        "question": "What is the loan amount and the appraised property value? Calculate the LTV. Is it within acceptable limits (typically ≤ 80% for conventional, ≤ 96.5% for FHA)?",
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


def _parse_status(answer: str) -> str:
    """Infer PASS/FAIL/NEEDS REVIEW from the LLM's free-text answer."""
    lower = answer.lower()
    if any(w in lower for w in ["pass", "yes", "compliant", "satisfied", "meets", "within"]):
        return "PASS"
    if any(w in lower for w in ["fail", "no ", "non-compliant", "does not", "missing", "below minimum", "exceed"]):
        return "FAIL"
    return "NEEDS REVIEW"


def run_compliance_check(document_path: str) -> ComplianceReport:
    """
    Run all compliance checks against a mortgage document.

    Args:
        document_path: Path to PDF or text file of the mortgage application.

    Returns:
        A ComplianceReport with per-rule results.
    """
    chunks = process_document(document_path)
    store  = build_vector_store(chunks)
    chain  = build_qa_chain(store)

    results = []
    for check in COMPLIANCE_CHECKS:
        response = ask(chain, check["question"])
        status   = _parse_status(response["answer"])
        results.append(ComplianceResult(
            rule_name=check["rule_name"],
            status=status,
            detail=response["answer"][:200],   # truncate for display
            sources=response["sources"],
        ))

    return ComplianceReport(
        document_name=document_path,
        results=results,
    )
