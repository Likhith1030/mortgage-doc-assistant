from dataclasses import dataclass
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

import config


@dataclass
class InternalMessage:
    subject: str
    body: str
    message_type: str
    recipient_role: str


INTERNAL_MEMO_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a professional mortgage operations coordinator drafting
internal communications. Use formal but clear language. Never include borrower
SSNs, full account numbers, or discriminatory language. Always include:
- Clear subject line
- Purpose statement
- Specific action required and deadline
- Contact for questions"""),
    ("human", """Draft an internal {message_type} communication.

Recipient role: {recipient_role}
Situation: {situation}
Borrower reference: {borrower_ref}
Details: {details}

Format:
SUBJECT: [subject line]

[body of message]"""),
])


SITUATIONS = {
    "missing_docs": {
        "message_type": "Document Deficiency Notice",
        "description": "Borrower has not submitted required documents",
    },
    "escalate_underwriting": {
        "message_type": "Underwriting Escalation",
        "description": "File needs senior underwriter review due to complexity",
    },
    "approval_memo": {
        "message_type": "Approval Recommendation Memo",
        "description": "File has passed all checks and is recommended for approval",
    },
    "appraisal_order": {
        "message_type": "Appraisal Order Request",
        "description": "Ready to order property appraisal",
    },
    "rate_lock_advisory": {
        "message_type": "Rate Lock Advisory",
        "description": "Rate lock window approaching or expired",
    },
    "denial_notice_internal": {
        "message_type": "Internal Denial Notification",
        "description": "Application does not meet lending criteria",
    },
}


def draft_internal_message(
    situation_code: str,
    recipient_role: str,
    borrower_ref: str,
    details: str = "",
    document_context: str = "",
) -> InternalMessage:
    """
    Draft an internal communication for a given mortgage situation.

    Args:
        situation_code:    One of the keys in SITUATIONS.
        recipient_role:    e.g. "Underwriting Team", "Branch Manager".
        borrower_ref:      File or loan number (NOT SSN or full name).
        details:           Extra context the drafter wants included.
        document_context:  Retrieved chunks from the uploaded mortgage document.
                           When provided, the memo is grounded in actual file data.
    """
    if situation_code not in SITUATIONS:
        raise ValueError(
            f"Unknown situation '{situation_code}'. "
            f"Valid codes: {list(SITUATIONS.keys())}"
        )

    enriched_details = _build_details(details, document_context)

    situation_info = SITUATIONS[situation_code]
    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        temperature=0.3,
        openai_api_key=config.OPENAI_API_KEY,
    )

    chain = INTERNAL_MEMO_PROMPT | llm
    response = chain.invoke({
        "message_type":   situation_info["message_type"],
        "recipient_role": recipient_role,
        "situation":      situation_info["description"],
        "borrower_ref":   borrower_ref,
        "details":        enriched_details,
    })

    subject, body = _parse_subject_body(response.content.strip())

    return InternalMessage(
        subject=subject,
        body=body,
        message_type=situation_info["message_type"],
        recipient_role=recipient_role,
    )


def _build_details(details: str, document_context: str) -> str:
    base = details.strip() if details.strip() else "No additional details provided."
    if document_context:
        return f"{base}\n\nRelevant document context:\n{document_context}"
    return base


def _parse_subject_body(raw: str) -> tuple[str, str]:
    lines = raw.split("\n")
    subject = ""
    body_lines = []
    in_body = False

    for line in lines:
        if line.startswith("SUBJECT:") and not in_body:
            subject = line.replace("SUBJECT:", "").strip()
            in_body = True
        elif in_body:
            body_lines.append(line)

    if not subject:
        subject = "Mortgage File Update"
        body_lines = lines

    return subject, "\n".join(body_lines).strip()


def list_situations() -> None:
    print("Available situation codes:")
    for code, info in SITUATIONS.items():
        print(f"  {code:<30} — {info['description']}")
