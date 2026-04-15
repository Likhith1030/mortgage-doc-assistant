# Mortgage Document Assistant

AI-powered mortgage document processing. Built with Python, LangChain, and OpenAI.

---

## What This Does

Automates the mechanical parts of mortgage application review:

| Task | Manual Time | With AI |
|---|---|---|
| Read and summarize 50-page application | 45 min | 30 sec |
| Run 7-point compliance pre-check | 60 min | 90 sec |
| Draft internal escalation memo | 15 min | 10 sec |
| Draft borrower update email | 20 min | 10 sec |
| **Total per file** | **~2.5 hours** | **~5 min** |

The AI drafts. You decide. Nothing happens without your review.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SYSTEM ARCHITECTURE                          │
│                                                                  │
│  Mortgage Document (PDF/TXT)                                     │
│         │                                                        │
│         ▼                                                        │
│  core/document_processor.py ──► Splits into ~1000-char chunks   │
│         │                                                        │
│         ▼                                                        │
│  core/embeddings.py ──► OpenAI API ──► Vectors ──► FAISS index  │
│         │                                  (saved to disk)       │
│         │                                                        │
│         ▼                                                        │
│  core/retrieval.py ──► RAG chain (find relevant chunks → GPT-4o)│
│                                                                  │
│  Four agents use the RAG chain:                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ agents/document_summarizer.py    → Loan Officer Brief    │   │
│  │ agents/compliance_checker.py     → 7-rule compliance report│  │
│  │ agents/communication_agent.py    → Internal memos        │   │
│  │ agents/customer_response_agent.py→ Borrower emails       │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │                                                        │
│         ▼                                                        │
│  metrics/tracker.py ──► adoption rate, accuracy, cost, time saved│
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Run

```bash
python main.py
```

Select any option from the interactive menu to get started with the sample document.

---

## Key Files

| File | What it does |
|---|---|
| `main.py` | Interactive menu — run this |
| `config.py` | All settings (model, chunk size, paths) |
| `core/document_processor.py` | Loads PDF/TXT, splits into chunks |
| `core/embeddings.py` | Builds and saves FAISS vector index |
| `core/retrieval.py` | RAG chain: question → find chunks → GPT-4o → answer |
| `agents/document_summarizer.py` | Map-reduce summarization |
| `agents/compliance_checker.py` | 7 regulatory compliance checks |
| `agents/communication_agent.py` | Internal memos and escalations |
| `agents/customer_response_agent.py` | Borrower-facing emails |
| `metrics/tracker.py` | Log events, print dashboard, export CSV |
| `data/knowledge_base/compliance_rules.txt` | Editable compliance rules |
| `data/sample_documents/` | Pre-loaded sample mortgage application |

---

## Documentation

### For Staff (Read in This Order)

1. `docs/onboarding/welcome_guide.md` — Start here. Overview in 20 minutes.
2. `docs/training/module_01_intro_to_ai.md` — What AI is and what it can't do.
3. `docs/training/module_02_system_overview.md` — Setup and architecture.
4. `docs/training/module_03_using_the_system.md` — Hands-on workflows with sample document.
5. `docs/training/module_04_troubleshooting_and_metrics.md` — Errors, quality control, cost.

### SOPs

| SOP | Topic |
|---|---|
| `docs/SOP/01_document_intake_SOP.md` | Receive, scan, upload documents |
| `docs/SOP/02_ai_workflow_SOP.md` | Run each AI workflow |
| `docs/SOP/03_compliance_check_SOP.md` | Handle PASS/FAIL/NEEDS REVIEW results |
| `docs/SOP/04_customer_communication_SOP.md` | Borrower email standards |

### Reference Card

`docs/onboarding/quick_start.md` — Print and keep at desk.

---

## Metrics

After running workflows, view the dashboard:

```
python main.py → Option 5
```

Metrics tracked:
- **Adoption rate** — % of AI outputs accepted (target: >85%)
- **Accuracy rate** — % accepted without edits (target: >80%)
- **Time saved** — Estimated minutes saved per accepted output
- **API cost** — Token usage and dollar cost

Export to CSV for reporting:
```python
from metrics.tracker import MetricsTracker
MetricsTracker().export_csv("monthly_report.csv")
```

---

## Compliance Rules

All compliance rules are in `data/knowledge_base/compliance_rules.txt`.

To update a rule (no code required):
1. Open the file in any text editor.
2. Edit the relevant section.
3. Delete `data/vector_store/` to force a rebuild.
4. Restart the system.

---

## Model Configuration

Change the model in `.env` or `config.py`:

| Model | Cost | Quality | Use for |
|---|---|---|---|
| `gpt-4o` | ~$0.15/file | Best | Production |
| `gpt-3.5-turbo` | ~$0.01/file | Good | Testing / training |
| `gpt-4o-mini` | ~$0.02/file | Very good | High-volume |

---

## Key Concepts

**RAG (Retrieval-Augmented Generation):** The AI answers questions using your actual documents, not its training data. This reduces hallucination and lets you trace every answer to a source.

**Vector Embeddings:** Text converted to numbers so that semantically similar text produces similar numbers. Enables semantic search — find "income verification" even if the document says "earnings confirmation."

**Map-Reduce Summarization:** For long documents, summarize each chunk (map) then combine summaries (reduce). Avoids losing content buried in long prompts.

**Human-in-the-Loop:** Every AI output requires human review before action. The system never sends emails, updates records, or makes decisions autonomously.

---

## Security Notes

- Your OpenAI API key is stored in `.env` — never commit this file to version control.
- Document text is sent to OpenAI's API. Confirm your organization's data privacy agreement covers PHI/PII before uploading sensitive borrower documents.
- The FAISS vector store on disk contains document embeddings but not plaintext — relatively safe, but treat it as sensitive data.
