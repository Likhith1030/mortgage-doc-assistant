"""
metrics/tracker.py — Track AI workflow usage and accuracy over time.

WHAT WE TRACK:
  ┌──────────────────────┬────────────────────────────────────────────────┐
  │ Metric               │ Why it matters                                 │
  ├──────────────────────┼────────────────────────────────────────────────┤
  │ workflow_adoption    │ % of files processed using AI vs manually      │
  │ ai_accuracy          │ % of AI outputs accepted without major edits   │
  │ time_saved_minutes   │ Estimated time saved per file vs manual        │
  │ compliance_pass_rate │ % of compliance checks that pass on first try  │
  │ tokens_used          │ OpenAI API cost tracking                       │
  └──────────────────────┴────────────────────────────────────────────────┘

HOW TO USE:
  tracker = MetricsTracker()
  tracker.log_event("summarization", accepted=True, edit_made=False, tokens=1200)
  tracker.print_dashboard()
  tracker.export_csv()
"""

import csv
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import config


@dataclass
class WorkflowEvent:
    timestamp: str
    workflow: str            # "summarization", "compliance", "internal_comm", "customer_email"
    accepted: bool           # Did the officer accept the AI output?
    edit_made: bool          # Did they make edits before accepting?
    tokens_used: int         # Approximate OpenAI tokens consumed
    processing_time_sec: float = 0.0
    notes: str = ""

    @property
    def accuracy_score(self) -> float:
        """Simple proxy: accepted-with-no-edits = 1.0, edited = 0.7, rejected = 0.0"""
        if self.accepted and not self.edit_made:
            return 1.0
        if self.accepted and self.edit_made:
            return 0.7
        return 0.0


class MetricsTracker:
    def __init__(self, log_path: str = config.METRICS_LOG_PATH):
        self.log_path = Path(log_path)
        self.events: List[WorkflowEvent] = []
        self._load_existing()

    def _load_existing(self):
        """Load historical events from CSV on startup."""
        if self.log_path.exists():
            with open(self.log_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.events.append(WorkflowEvent(
                        timestamp=row["timestamp"],
                        workflow=row["workflow"],
                        accepted=row["accepted"] == "True",
                        edit_made=row["edit_made"] == "True",
                        tokens_used=int(row["tokens_used"]),
                        processing_time_sec=float(row.get("processing_time_sec", 0)),
                        notes=row.get("notes", ""),
                    ))

    def log_event(
        self,
        workflow: str,
        accepted: bool,
        edit_made: bool,
        tokens_used: int,
        processing_time_sec: float = 0.0,
        notes: str = "",
    ) -> WorkflowEvent:
        """Record a workflow event. Call this after each AI interaction."""
        event = WorkflowEvent(
            timestamp=datetime.now().isoformat(),
            workflow=workflow,
            accepted=accepted,
            edit_made=edit_made,
            tokens_used=tokens_used,
            processing_time_sec=processing_time_sec,
            notes=notes,
        )
        self.events.append(event)
        self._append_to_csv(event)
        return event

    def _append_to_csv(self, event: WorkflowEvent):
        """Append a single event row to the CSV log."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.log_path.exists()
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=asdict(event).keys())
            if write_header:
                writer.writeheader()
            writer.writerow(asdict(event))

    # ── Computed metrics ──────────────────────────────────────────────────────

    def adoption_rate(self, workflow: Optional[str] = None) -> float:
        """Fraction of events where the AI output was accepted (adopted)."""
        events = [e for e in self.events if workflow is None or e.workflow == workflow]
        if not events:
            return 0.0
        return sum(1 for e in events if e.accepted) / len(events)

    def accuracy_rate(self, workflow: Optional[str] = None) -> float:
        """Average accuracy score (0.0–1.0) across events."""
        events = [e for e in self.events if workflow is None or e.workflow == workflow]
        if not events:
            return 0.0
        return sum(e.accuracy_score for e in events) / len(events)

    def total_tokens(self) -> int:
        return sum(e.tokens_used for e in self.events)

    def estimated_cost_usd(self) -> float:
        """Rough cost estimate: $0.005 per 1K tokens (GPT-4o blended rate)."""
        return (self.total_tokens() / 1000) * 0.005

    def time_saved_estimate(self) -> float:
        """
        Estimate minutes saved. Baseline assumptions (adjust to your org):
          summarization    → saves 45 min per file
          compliance       → saves 60 min per file
          internal_comm    → saves 15 min per file
          customer_email   → saves 20 min per file
        Only counts accepted events.
        """
        savings = {
            "summarization":  45,
            "compliance":     60,
            "internal_comm":  15,
            "customer_email": 20,
        }
        total = 0.0
        for e in self.events:
            if e.accepted:
                total += savings.get(e.workflow, 10)
        return total

    def print_dashboard(self):
        """Print a human-readable metrics dashboard to the terminal."""
        print("\n" + "=" * 55)
        print("   MORTGAGE AI ASSISTANT — METRICS DASHBOARD")
        print("=" * 55)
        print(f"  Total events logged : {len(self.events)}")
        print(f"  Overall adoption    : {self.adoption_rate():.0%}")
        print(f"  Overall accuracy    : {self.accuracy_rate():.0%}")
        print(f"  Total tokens used   : {self.total_tokens():,}")
        print(f"  Estimated API cost  : ${self.estimated_cost_usd():.2f}")
        print(f"  Estimated time saved: {self.time_saved_estimate():.0f} minutes")
        print()
        print("  Per-workflow breakdown:")
        for wf in ["summarization", "compliance", "internal_comm", "customer_email"]:
            wf_events = [e for e in self.events if e.workflow == wf]
            if wf_events:
                print(f"    {wf:<20} adoption={self.adoption_rate(wf):.0%}  "
                      f"accuracy={self.accuracy_rate(wf):.0%}  "
                      f"n={len(wf_events)}")
        print("=" * 55 + "\n")

    def export_csv(self, path: Optional[str] = None) -> str:
        """Export all events to a CSV for reporting. Returns the file path."""
        out = path or str(self.log_path)
        with open(out, "w", newline="") as f:
            if self.events:
                writer = csv.DictWriter(f, fieldnames=asdict(self.events[0]).keys())
                writer.writeheader()
                writer.writerows(asdict(e) for e in self.events)
        return out
