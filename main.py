"""
main.py — Interactive demo of the Mortgage Document Assistant.

Run this file to see all four AI agents in action on the sample document:

    python main.py

MENU OPTIONS:
  1. Summarize a mortgage document
  2. Run compliance check
  3. Draft internal communication
  4. Draft customer email
  5. View metrics dashboard
  6. Interactive Q&A on a document
  0. Exit
"""

import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import print as rprint

import config
from core.document_processor import process_document
from core.embeddings import build_vector_store, get_or_build_store
from core.retrieval import build_qa_chain, ask
from agents.document_summarizer import summarize_document
from agents.compliance_checker import run_compliance_check
from agents.communication_agent import draft_internal_message, list_situations
from agents.customer_response_agent import draft_customer_email
from metrics.tracker import MetricsTracker

console = Console()

SAMPLE_DOC = "./data/sample_documents/sample_mortgage_application.txt"


def check_env():
    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == "your-openai-api-key-here":
        console.print("[bold red]ERROR:[/] OPENAI_API_KEY is not set.")
        console.print("Copy .env.example to .env and add your key.")
        raise SystemExit(1)


def get_doc_path() -> str:
    """Ask user for a document path, default to sample."""
    path = Prompt.ask(
        "Document path",
        default=SAMPLE_DOC,
    )
    if not Path(path).exists():
        console.print(f"[red]File not found:[/] {path}")
        return SAMPLE_DOC
    return path


# ── Workflow 1: Summarization ─────────────────────────────────────────────────

def run_summarization(tracker: MetricsTracker):
    console.rule("[bold blue]Workflow 1: Document Summarization[/]")
    path = get_doc_path()

    with console.status("Loading and chunking document..."):
        chunks = process_document(path)

    console.print(f"  Loaded [green]{len(chunks)} chunks[/] from {path}")

    start = time.time()
    with console.status("[yellow]AI is summarizing...[/] (this may take 15-30 seconds)"):
        summary = summarize_document(chunks)
    elapsed = time.time() - start

    console.print(Panel(summary, title="Loan Officer Brief", border_style="blue"))
    console.print(f"  Completed in {elapsed:.1f}s")

    accepted = Confirm.ask("Accept this summary?")
    edited   = Confirm.ask("Did you make any edits?") if accepted else False
    tracker.log_event("summarization", accepted, edited, tokens_used=2000, processing_time_sec=elapsed)
    console.print("[green]Logged.[/]")


# ── Workflow 2: Compliance ────────────────────────────────────────────────────

def run_compliance(tracker: MetricsTracker):
    console.rule("[bold blue]Workflow 2: Compliance Check[/]")
    path = get_doc_path()

    start = time.time()
    with console.status("[yellow]Running compliance checks...[/] (may take 60-90 seconds)"):
        report = run_compliance_check(path)
    elapsed = time.time() - start

    table = Table(title="Compliance Report", show_lines=True)
    table.add_column("Rule", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Detail")

    icons = {"PASS": "[green]✓ PASS[/]", "FAIL": "[red]✗ FAIL[/]", "NEEDS REVIEW": "[yellow]? REVIEW[/]"}
    for r in report.results:
        table.add_row(r.rule_name, icons[r.status], r.detail[:120] + "..." if len(r.detail) > 120 else r.detail)

    console.print(table)
    console.print(f"  PASS: {len(report.passed)}  FAIL: {len(report.failed)}  NEEDS REVIEW: {len(report.needs_review)}")
    console.print(f"  Completed in {elapsed:.1f}s")

    accepted = Confirm.ask("Accept this compliance report?")
    edited   = Confirm.ask("Did you make any edits?") if accepted else False
    tracker.log_event("compliance", accepted, edited, tokens_used=5000, processing_time_sec=elapsed)


# ── Workflow 3: Internal Communication ───────────────────────────────────────

def run_internal_comm(tracker: MetricsTracker):
    console.rule("[bold blue]Workflow 3: Internal Communication[/]")

    list_situations()
    situation_code = Prompt.ask("\nSituation code", default="missing_docs")
    recipient      = Prompt.ask("Recipient role", default="Loan Processor")
    borrower_ref   = Prompt.ask("Loan/file reference", default="LOAN-2024-0042")
    details        = Prompt.ask("Additional details", default="Missing: 2023 W-2 and last 2 bank statements. Deadline: April 10.")

    start = time.time()
    with console.status("[yellow]Drafting message...[/]"):
        msg = draft_internal_message(situation_code, recipient, borrower_ref, details)
    elapsed = time.time() - start

    console.print(Panel(
        f"[bold]Subject:[/] {msg.subject}\n\n{msg.body}",
        title=f"Internal {msg.message_type}",
        border_style="cyan",
    ))

    accepted = Confirm.ask("Accept this message?")
    edited   = Confirm.ask("Did you make any edits?") if accepted else False
    tracker.log_event("internal_comm", accepted, edited, tokens_used=500, processing_time_sec=elapsed)


# ── Workflow 4: Customer Email ────────────────────────────────────────────────

def run_customer_email(tracker: MetricsTracker):
    console.rule("[bold blue]Workflow 4: Customer Email[/]")

    situation  = Prompt.ask("Situation", default="Conditional approval — income documents needed")
    first_name = Prompt.ask("Borrower first name", default="Michael")
    officer    = Prompt.ask("Your name", default="James Patel")
    tone       = Prompt.ask("Tone (standard/empathetic/urgent)", default="standard")
    details    = Prompt.ask("Details to include", default="Please provide your 2023 W-2 and last 2 pay stubs by April 10.")

    start = time.time()
    with console.status("[yellow]Drafting customer email...[/]"):
        email = draft_customer_email(situation, first_name, officer, tone, details)
    elapsed = time.time() - start

    console.print(Panel(email.full_email, title="Customer Email Draft", border_style="magenta"))

    accepted = Confirm.ask("Accept this email?")
    edited   = Confirm.ask("Did you make any edits?") if accepted else False
    tracker.log_event("customer_email", accepted, edited, tokens_used=600, processing_time_sec=elapsed)


# ── Workflow 5: Interactive Q&A ───────────────────────────────────────────────

def run_qa(tracker: MetricsTracker):
    console.rule("[bold blue]Workflow 6: Interactive Document Q&A[/]")
    path = get_doc_path()

    with console.status("Building vector store..."):
        chunks = process_document(path)
        store  = build_vector_store(chunks)
        chain  = build_qa_chain(store)

    console.print("Vector store ready. Ask questions about the document (type 'done' to exit).\n")

    while True:
        question = Prompt.ask("[bold]Question[/]")
        if question.lower() in ("done", "exit", "quit"):
            break

        with console.status("Searching document and generating answer..."):
            result = ask(chain, question)

        console.print(Panel(result["answer"], title="Answer", border_style="green"))
        if result["sources"]:
            console.print(f"  Sources: {', '.join(result['sources'])}")


# ── Main menu ─────────────────────────────────────────────────────────────────

def main():
    check_env()
    tracker = MetricsTracker()

    console.print(Panel(
        "[bold]Mortgage Document Assistant[/]\n"
        "AI-powered document analysis for loan officers",
        border_style="bold blue",
    ))

    while True:
        console.print("\n[bold]Menu:[/]")
        console.print("  1. Summarize mortgage document")
        console.print("  2. Run compliance check")
        console.print("  3. Draft internal communication")
        console.print("  4. Draft customer email")
        console.print("  5. View metrics dashboard")
        console.print("  6. Interactive Q&A on document")
        console.print("  0. Exit\n")

        choice = Prompt.ask("Choose", choices=["0", "1", "2", "3", "4", "5", "6"])

        if choice == "1":
            run_summarization(tracker)
        elif choice == "2":
            run_compliance(tracker)
        elif choice == "3":
            run_internal_comm(tracker)
        elif choice == "4":
            run_customer_email(tracker)
        elif choice == "5":
            tracker.print_dashboard()
        elif choice == "6":
            run_qa(tracker)
        elif choice == "0":
            console.print("Goodbye.")
            break


if __name__ == "__main__":
    main()
