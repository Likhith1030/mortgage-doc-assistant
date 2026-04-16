"""
tests/test_agents.py — Unit tests that run WITHOUT calling the OpenAI API.

These tests verify the parsing and logic layers independently.
To run: python -m pytest tests/ -v
"""

import pytest
from pydantic import ValidationError

from agents.customer_response_agent import _parse_email, _build_details as email_build_details, CustomerEmail
from agents.communication_agent import (
    _parse_subject_body, _build_details as memo_build_details,
    list_situations, SITUATIONS,
)
from agents.compliance_checker import _RuleCheck, ComplianceResult, ComplianceReport
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


# ── Email detail enrichment ───────────────────────────────────────────────────

def test_email_build_details_no_context():
    result = email_build_details("Need W-2 by April 10.", "")
    assert result == "Need W-2 by April 10."


def test_email_build_details_with_context():
    result = email_build_details("Need W-2.", "Borrower earns $8,500/mo.")
    assert "Need W-2." in result
    assert "Borrower earns $8,500/mo." in result


def test_email_build_details_empty_base_falls_back():
    result = email_build_details("", "")
    assert result == "No additional details provided."


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


# ── Memo detail enrichment ────────────────────────────────────────────────────

def test_memo_build_details_no_context():
    result = memo_build_details("Missing: W-2.", "")
    assert result == "Missing: W-2."


def test_memo_build_details_with_context():
    result = memo_build_details("Escalation needed.", "Borrower DTI is 52%.")
    assert "Escalation needed." in result
    assert "Borrower DTI is 52%." in result


def test_memo_build_details_empty_base_falls_back():
    result = memo_build_details("", "")
    assert result == "No additional details provided."


# ── Compliance structured output model ───────────────────────────────────────

def test_rule_check_valid_pass():
    r = _RuleCheck(status="PASS", reasoning="DTI is 38%, below 43% threshold.")
    assert r.status == "PASS"
    assert "38%" in r.reasoning


def test_rule_check_valid_fail():
    r = _RuleCheck(status="FAIL", reasoning="LTV is 92%, exceeds 80% limit.")
    assert r.status == "FAIL"


def test_rule_check_valid_needs_review():
    r = _RuleCheck(status="NEEDS REVIEW", reasoning="DTI figures not found in document.")
    assert r.status == "NEEDS REVIEW"


def test_rule_check_rejects_invalid_status():
    with pytest.raises(ValidationError):
        _RuleCheck(status="UNKNOWN", reasoning="Some text.")


def test_rule_check_requires_reasoning():
    with pytest.raises(ValidationError):
        _RuleCheck(status="PASS")


# ── Compliance report categorization ─────────────────────────────────────────

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


def test_compliance_report_summary_contains_rule_names():
    results = [
        ComplianceResult("DTI Check", "PASS", "OK"),
        ComplianceResult("LTV Check", "FAIL", "Too high"),
    ]
    report = ComplianceReport("loan.pdf", results)
    summary = report.summary()
    assert "DTI Check" in summary
    assert "LTV Check" in summary
    assert "PASS: 1" in summary
    assert "FAIL: 1" in summary


# ── Metrics ───────────────────────────────────────────────────────────────────

def test_workflow_event_accuracy_score():
    accepted_no_edit = WorkflowEvent("2024-01-01", "summarization", True, False, 100)
    assert accepted_no_edit.accuracy_score == 1.0

    accepted_with_edit = WorkflowEvent("2024-01-01", "summarization", True, True, 100)
    assert accepted_with_edit.accuracy_score == 0.7

    rejected = WorkflowEvent("2024-01-01", "summarization", False, False, 100)
    assert rejected.accuracy_score == 0.0
