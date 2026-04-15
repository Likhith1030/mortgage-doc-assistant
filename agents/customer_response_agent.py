"""
agents/customer_response_agent.py — Draft borrower-facing emails.

WORKFLOW:
  Loan officer describes the situation → system drafts a compliant, empathetic
  borrower email → officer reviews, personalizes, and sends via their email client.

COMPLIANCE NOTES (built into the prompt):
  - Never reveal specific credit score thresholds to avoid gaming
  - Never promise approval or guarantee rates
  - Never use language that could be construed as discriminatory
  - Include RESPA / TILA disclosures reminder placeholders where applicable

TONE OPTIONS:
  - "standard"  — Professional, neutral, bank-branded
  - "empathetic" — Warmer, used for difficult news (denial, delays)
  - "urgent"    — Used for rate lock deadlines, missing documents
"""

from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

import config


@dataclass
class CustomerEmail:
    subject: str
    salutation: str
    body: str
    closing: str
    full_email: str


CUSTOMER_EMAIL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a mortgage loan officer at a professional bank.
Draft a borrower-facing email following these rules:
1. Be warm, clear, and professional.
2. Never guarantee approval, rates, or timelines.
3. Never mention protected class characteristics (race, gender, religion, etc.).
4. Use plain language — the borrower may not know mortgage jargon.
5. Always include a next step the borrower should take.
6. End with: "If you have questions, please contact us at [BANK_PHONE] or [BANK_EMAIL]."
Tone: {tone}"""),
    ("human", """Write a customer email for the following situation:

Situation: {situation}
Borrower first name: {borrower_first_name}
Loan officer name: {officer_name}
Specific details: {details}

Format:
SUBJECT: [subject line]
SALUTATION: [Dear ...]
BODY:
[main message — 2-4 short paragraphs]
CLOSING:
[closing paragraph with next steps and contact info]"""),
])

TONE_DESCRIPTIONS = {
    "standard":   "Professional and neutral",
    "empathetic": "Warm, understanding, and supportive (for bad news)",
    "urgent":     "Clear and action-oriented with a sense of urgency",
}


def draft_customer_email(
    situation: str,
    borrower_first_name: str,
    officer_name: str,
    tone: str = "standard",
    details: str = "",
) -> CustomerEmail:
    """
    Draft a borrower-facing email.

    Args:
        situation:           What this email is about. E.g., "Application received",
                             "Missing income documents needed", "Conditional approval".
        borrower_first_name: Used in salutation. Do not pass full name for privacy.
        officer_name:        Loan officer's name for the sign-off.
        tone:                "standard", "empathetic", or "urgent".
        details:             Specifics to include (e.g., which docs are needed, deadlines).

    Returns:
        CustomerEmail with all parts separated for easy editing.

    Example:
        email = draft_customer_email(
            situation="Conditional approval — income documents needed",
            borrower_first_name="Sarah",
            officer_name="James Patel",
            tone="standard",
            details="We need 2023 W-2 and last 2 pay stubs by June 20th."
        )
        print(email.full_email)
    """
    if tone not in TONE_DESCRIPTIONS:
        raise ValueError(f"tone must be one of: {list(TONE_DESCRIPTIONS.keys())}")

    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        temperature=0.4,
        openai_api_key=config.OPENAI_API_KEY,
    )

    chain    = CUSTOMER_EMAIL_PROMPT | llm
    response = chain.invoke({
        "tone":                tone,
        "situation":           situation,
        "borrower_first_name": borrower_first_name,
        "officer_name":        officer_name,
        "details":             details or "No additional details.",
    })

    return _parse_email(response.content.strip())


def _parse_email(raw: str) -> CustomerEmail:
    """Parse the structured output into a CustomerEmail object."""
    sections = {"SUBJECT": "", "SALUTATION": "", "BODY": "", "CLOSING": ""}
    current  = None
    buffer   = []

    for line in raw.split("\n"):
        for key in sections:
            if line.startswith(f"{key}:"):
                if current and buffer:
                    sections[current] = "\n".join(buffer).strip()
                current = key
                remainder = line[len(key) + 1:].strip()
                buffer = [remainder] if remainder else []
                break
        else:
            if current:
                buffer.append(line)

    if current and buffer:
        sections[current] = "\n".join(buffer).strip()

    full_email = (
        f"Subject: {sections['SUBJECT']}\n\n"
        f"{sections['SALUTATION']}\n\n"
        f"{sections['BODY']}\n\n"
        f"{sections['CLOSING']}"
    )

    return CustomerEmail(
        subject=sections["SUBJECT"],
        salutation=sections["SALUTATION"],
        body=sections["BODY"],
        closing=sections["CLOSING"],
        full_email=full_email,
    )
