"""
tests/test_agents.py — Unit tests that run WITHOUT calling the OpenAI API.

These tests verify the parsing and logic layers independently.
To run: python -m pytest tests/ -v
"""

import pytest
from agents.customer_response_agent import _parse_email, CustomerEmail
from agents.communication_agent import _parse_subject_body, list_situations, SITUATIONS
from agents.compliance_checker import _parse_status, ComplianceResult, ComplianceReport
from metrics.tracker import WorkflowEvent


# ── Email parser ──────────────────────────────────────────────────────────────

def test_parse_email_extracts_subject():
    raw = """SUBJECT: Your Application Update
SALUTATION: Dear Michael,
BODY:
We have reviewed your application.
Everything looks great.
CLOSING:
Please contact us if you have questions."""
    email = _parse_email(raw)
    assert email.subject == "Your Application Update"


def test_parse_email_extracts_body():
    raw = """SUBJECT: Test
SALUTATION: Dear John,
BODY:
Line one of body.
Line two of body.
CLOSING:
Goodbye."""
    email = _parse_email(raw)
    assert "Line one" in email.body
    assert "Line two" in email.body


def test_parse_email_full_email_combines_parts():
    raw = """SUBJECT: Update
SALUTATION: Dear Jane,
BODY:
Your documents were received.
CLOSING:
Thank you."""
    email = _parse_email(raw)
    assert "Subject: Update" in email.full_email
    assert "Dear Jane" in email.full_email
    assert "documents were received" in email.full_email


# ── Communication parser ──────────────────────────────────────────────────────

def test_parse_subject_body_splits_correctly():
    raw = "SUBJECT: Test Subject\n\nThis is the body line 1.\nLine 2."
    subject, body = _parse_subject_body(raw)
    assert subject == "Test Subject"
    assert "body line 1" in body


def test_parse_subject_body_fallback_when_no_subject():
    raw = "No subject marker here\nJust body text."
    subject, body = _parse_subject_body(raw)
    assert subject == "Mortgage File Update"


def test_situations_dict_has_required_keys():
    for code, info in SITUATIONS.items():
        assert "message_type" in info
        assert "description" in info


# ── Compliance status parser ──────────────────────────────────────────────────

def test_parse_status_pass():
    assert _parse_status("The DTI ratio is within limits and the borrower passes.") == "PASS"
    assert _parse_status("Yes, the credit score meets the requirement.") == "PASS"


def test_parse_status_fail():
    assert _parse_status("The income documentation is missing from the file.") == "FAIL"
    assert _parse_status("No W-2 was found in the submitted documents.") == "FAIL"


def test_parse_status_needs_review():
    assert _parse_status("The document does not clearly state the LTV calculation.") == "NEEDS REVIEW"
    assert _parse_status("Unclear from context provided.") == "NEEDS REVIEW"


# ── Metrics ───────────────────────────────────────────────────────────────────

def test_workflow_event_accuracy_score():
    accepted_no_edit = WorkflowEvent("2024-01-01", "summarization", True, False, 100)
    assert accepted_no_edit.accuracy_score == 1.0

    accepted_with_edit = WorkflowEvent("2024-01-01", "summarization", True, True, 100)
    assert accepted_with_edit.accuracy_score == 0.7

    rejected = WorkflowEvent("2024-01-01", "summarization", False, False, 100)
    assert rejected.accuracy_score == 0.0


def test_compliance_report_categorizes_results():
    results = [
        ComplianceResult("DTI", "PASS", "Within limits"),
        ComplianceResult("LTV", "FAIL", "Exceeds limit"),
        ComplianceResult("Docs", "NEEDS REVIEW", "Appraisal pending"),
    ]
    report = ComplianceReport("test.pdf", results)
    assert len(report.passed) == 1
    assert len(report.failed) == 1
    assert len(report.needs_review) == 1
