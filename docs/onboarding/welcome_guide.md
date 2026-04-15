# Welcome to the Mortgage Document Assistant

**For new employees and contractors**
**Estimated read time:** 20 minutes
**Goal:** Give you a complete picture of the system before you touch it.

---

## What This System Does

The Mortgage Document Assistant is an AI-powered tool that helps loan officers and processors analyze mortgage applications faster and more consistently. It reads documents, checks compliance rules, and drafts communications — all in under 5 minutes per file instead of 2-4 hours. You remain in control: every AI suggestion requires your review and approval before anything happens.

---

## What You'll Do With It Daily

As a **Loan Officer:**
- Drop a mortgage application into the system → get a one-page Loan Officer Brief in under a minute
- Run a 7-point compliance pre-check before sending to underwriting
- Draft and send borrower emails in 2 minutes instead of 15

As a **Loan Processor:**
- Verify document completeness using the AI checklist
- Draft internal memos for escalations, missing docs, and appraisal orders
- Monitor metrics and flag accuracy issues for your manager

As a **Compliance Coordinator:**
- Review flagged compliance results
- Update the compliance rules file when regulations change
- Run quarterly audits of AI accuracy vs. manual review

---

## The 5 Rules Everyone Must Know

**Rule 1: The AI is a drafter, not a decision-maker.**
It never approves loans, sends emails, or updates systems. You do.

**Rule 2: Always verify numbers.**
Every dollar amount in an AI output must be traceable to the source document. If you can't find it, it may be hallucinated.

**Rule 3: Log every interaction.**
After every AI output, answer the feedback prompts. This data is how we improve the system.

**Rule 4: Borrower emails go through your bank email only.**
Copy the AI draft → paste into your email client → review → send. Never copy-paste without reading.

**Rule 5: Fair lending flags = immediate escalation.**
If the compliance check flags a potential fair lending issue, stop processing and call the Compliance Officer. Do not continue.

---

## Your First Week

| Day | Activity |
|---|---|
| Day 1 | Read Module 1 (What Is This AI?) and Module 2 (Setup). Install the system. |
| Day 2 | Read Module 3 (Using the System). Complete the hands-on exercises with the sample document. |
| Day 3 | Read all four SOPs. Read Module 4 (Troubleshooting). |
| Day 4 | Shadow an experienced user for a full day. Observe how they review AI outputs. |
| Day 5 | Process your first file independently with supervisor spot-check. |

---

## Where to Find Things

| What you need | Where it is |
|---|---|
| How to run a workflow | `docs/training/module_03_using_the_system.md` |
| Step-by-step operating procedures | `docs/SOP/` folder |
| Compliance rules reference | `data/knowledge_base/compliance_rules.txt` |
| Mortgage processing guidelines | `data/knowledge_base/mortgage_guidelines.txt` |
| Practice document | `data/sample_documents/sample_mortgage_application.txt` |
| Settings and configuration | `config.py` |
| Metrics and usage data | Run `python main.py` → Option 5 |

---

## Who to Ask When You're Stuck

| Problem | Contact |
|---|---|
| System won't start | IT Help Desk |
| AI is giving wrong information | Your supervisor first, then system admin |
| Compliance question | Compliance Coordinator |
| Borrower communication question | Your manager |
| OpenAI API issues | System Administrator |

---

## Frequently Asked Questions

**Q: Do I need to know Python to use this?**
No. You only need to run `python main.py` and use the menu. The code is there for transparency, not operation.

**Q: What if the AI is wrong about something?**
Log it as "rejected," document your manual finding in Encompass, and proceed with your own assessment. The AI being wrong is expected occasionally — that's why you review everything.

**Q: Will the AI eventually replace my job?**
This AI handles mechanical document reading tasks. Judgment calls, borrower relationships, complex scenarios, and regulatory decisions remain human roles. The goal is to free up 2-3 hours per day so you can focus on higher-value work.

**Q: How much does it cost to use?**
Approximately $0.15-0.50 per file processed (all four workflows combined). The organization has a monthly API budget. Do not process documents unnecessarily just to test the system — use the sample document for practice.

**Q: Can I ask the AI anything about a borrower's file?**
Yes, using Option 6 (Q&A). The AI will only know what's in the document you uploaded — it cannot access other systems, pull credit reports, or check live data.

**Q: What about data privacy?**
Document text is sent to OpenAI's API for processing. Do not upload documents containing full SSNs without confirming your organization's data privacy agreement with OpenAI covers this. When in doubt, use the AI for documents without sensitive identifiers and verify compliance with your privacy officer.

---

## Ready to Start?

Go to `docs/training/module_01_intro_to_ai.md` and begin Module 1.

Good luck — you'll be faster than the manual process within a week.
