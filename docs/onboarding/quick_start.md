# Quick Start Card — Mortgage Document Assistant

Print this and keep it at your desk.

---

## Daily Setup (30 seconds)

```bash
cd /path/to/mortgage-doc-assistant
source venv/bin/activate
python main.py
```

---

## Menu Cheat Sheet

| Option | What It Does | When to Use |
|---|---|---|
| 1 | Summarize document | First step on every new file |
| 2 | Compliance check | After summary; before sending to UW |
| 3 | Internal memo | Deficiency notice, escalation, approval |
| 4 | Customer email | Any borrower communication |
| 5 | Metrics dashboard | End of day / manager review |
| 6 | Interactive Q&A | Specific questions about a document |

---

## Situation Codes (Option 3)

| Code | Use When |
|---|---|
| `missing_docs` | Borrower hasn't submitted required documents |
| `escalate_underwriting` | Complex file needs senior UW |
| `approval_memo` | File is ready for approval |
| `appraisal_order` | Order the appraisal |
| `rate_lock_advisory` | Rate lock window closing |
| `denial_notice_internal` | File doesn't meet criteria |

---

## Compliance Targets

| Check | Target | Action if FAIL |
|---|---|---|
| DTI | < 43% | Check math; look for compensating factors |
| LTV | < 97% conventional | Verify appraisal |
| Documents | All present | Send deficiency notice |
| Employment | 2+ years | Request LOE |
| Credit | 620+ conventional | Discuss alternatives |
| Fair Lending | No flags | STOP — call Compliance Officer |
| QM | All criteria met | Legal review before proceeding |

---

## Review Checklist (After Every AI Output)

- [ ] Every number verified against source document
- [ ] No SSN or full account numbers in any output
- [ ] FAIL compliance results escalated
- [ ] Customer email not sent without review and edits
- [ ] Feedback logged (accepted / edited)

---

## Quick Fixes

| Problem | Fix |
|---|---|
| 0 chunks indexed | Run OCR on PDF; re-upload |
| Wrong numbers in summary | Use Q&A; verify source doc manually |
| All NEEDS REVIEW | Wrong document uploaded? |
| API error | Check .env key; wait 60s |

---

## Contact

- AI System issues: IT Help Desk
- Compliance questions: Compliance Coordinator
- Borrower escalations: Your manager
