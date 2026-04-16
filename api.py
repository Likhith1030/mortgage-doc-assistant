"""
api.py — FastAPI server for the Mortgage Document Assistant.

Run with:  uvicorn api:app --reload
Open:      http://localhost:8000
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import config
from agents.communication_agent import SITUATIONS, draft_internal_message
from agents.compliance_checker import run_compliance_check
from agents.customer_response_agent import draft_customer_email
from agents.document_summarizer import summarize_document
from core.document_processor import process_document
from core.embeddings import build_vector_store, load_vector_store
from core.retrieval import ask, build_qa_chain

app = FastAPI(title="Mortgage Document Assistant")

# ── Paths ─────────────────────────────────────────────────────────────────────

UPLOADED_DATA_DIR = Path(config.UPLOADED_DATA_PATH)
HISTORY_FILE = UPLOADED_DATA_DIR / "history.json"

UPLOADED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── In-memory QA sessions ─────────────────────────────────────────────────────
# session_id -> {"chain": ..., "store": ..., "filename": ..., "doc_id": ...}
_qa_sessions: dict = {}


# ── History helpers ───────────────────────────────────────────────────────────

def _load_history() -> list:
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_history(history: list):
    HISTORY_FILE.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def _upsert_history(entry: dict):
    history = _load_history()
    history = [h for h in history if h.get("id") != entry["id"]]
    history.insert(0, entry)
    _save_history(history)


def _get_history_entry(doc_id: str) -> dict | None:
    return next((h for h in _load_history() if h["id"] == doc_id), None)


def _update_history_flags(doc_id: str, **flags):
    history = _load_history()
    for h in history:
        if h["id"] == doc_id:
            h.update(flags)
            break
    _save_history(history)


# ── File helpers ──────────────────────────────────────────────────────────────

def _doc_dir(doc_id: str) -> Path:
    d = UPLOADED_DATA_DIR / doc_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_uploaded_file(doc_id: str, filename: str, content: bytes) -> Path:
    safe_name = Path(filename).name
    dest = _doc_dir(doc_id) / safe_name
    dest.write_bytes(content)
    return dest


def _doc_store_path(doc_id: str) -> str:
    return str(_doc_dir(doc_id) / "vector_store")


def _doc_compliance_path(doc_id: str) -> Path:
    return _doc_dir(doc_id) / "compliance_report.json"


def _doc_summary_path(doc_id: str) -> Path:
    return _doc_dir(doc_id) / "summary.md"


def _retrieve_context(session_id: str, query: str, k: int = 4) -> str:
    if not session_id or session_id not in _qa_sessions:
        return ""
    store = _qa_sessions[session_id]["store"]
    docs = store.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)


# ── Summarize ─────────────────────────────────────────────────────────────────

@app.post("/api/summarize")
async def api_summarize(file: UploadFile = File(...)):
    content = await file.read()
    doc_id = str(uuid.uuid4())
    saved_path = _save_uploaded_file(doc_id, file.filename, content)

    try:
        chunks = process_document(str(saved_path))
        summary = summarize_document(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    _doc_summary_path(doc_id).write_text(summary, encoding="utf-8")

    _upsert_history({
        "id": doc_id,
        "filename": file.filename,
        "uploaded_at": datetime.now().isoformat(),
        "file_size": len(content),
        "file_type": Path(file.filename).suffix.lower().lstrip("."),
        "chunks": len(chunks),
        "has_summary": True,
        "has_compliance": False,
        "has_vector_store": False,
    })

    return {"summary": summary, "doc_id": doc_id}


# ── Compliance ────────────────────────────────────────────────────────────────

@app.post("/api/compliance")
async def api_compliance(file: UploadFile = File(...)):
    content = await file.read()
    doc_id = str(uuid.uuid4())
    saved_path = _save_uploaded_file(doc_id, file.filename, content)
    store_path = _doc_store_path(doc_id)
    report_path = str(_doc_compliance_path(doc_id))

    try:
        report = run_compliance_check(
            str(saved_path),
            store_path=store_path,
            report_save_path=report_path,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    _upsert_history({
        "id": doc_id,
        "filename": file.filename,
        "uploaded_at": datetime.now().isoformat(),
        "file_size": len(content),
        "file_type": Path(file.filename).suffix.lower().lstrip("."),
        "chunks": 0,
        "has_summary": False,
        "has_compliance": True,
        "has_vector_store": True,
    })

    return {
        "doc_id": doc_id,
        "results": [
            {"rule": r.rule_name, "status": r.status, "detail": r.detail}
            for r in report.results
        ],
        "totals": {
            "pass": len(report.passed),
            "fail": len(report.failed),
            "review": len(report.needs_review),
        },
    }


@app.post("/api/compliance/{doc_id}")
async def api_compliance_existing(doc_id: str):
    """Re-run compliance on a previously uploaded document."""
    entry = _get_history_entry(doc_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Document not found in history.")

    doc_dir = _doc_dir(doc_id)
    candidates = [p for p in doc_dir.iterdir() if p.is_file() and p.suffix.lower() in (".pdf", ".txt")]
    if not candidates:
        raise HTTPException(status_code=404, detail="Original file not found.")

    saved_path = candidates[0]
    store_path = _doc_store_path(doc_id)
    report_path = str(_doc_compliance_path(doc_id))

    try:
        report = run_compliance_check(
            str(saved_path),
            store_path=store_path,
            report_save_path=report_path,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    _update_history_flags(doc_id, has_compliance=True, has_vector_store=True)

    return {
        "doc_id": doc_id,
        "results": [
            {"rule": r.rule_name, "status": r.status, "detail": r.detail}
            for r in report.results
        ],
        "totals": {
            "pass": len(report.passed),
            "fail": len(report.failed),
            "review": len(report.needs_review),
        },
    }


# ── Internal Communication ────────────────────────────────────────────────────

@app.post("/api/internal-comm")
async def api_internal_comm(
    situation_code: str = Form(...),
    recipient: str = Form(...),
    borrower_ref: str = Form(...),
    details: str = Form(""),
    session_id: str = Form(""),
):
    doc_context = _retrieve_context(session_id, f"{situation_code} {details}")
    msg = draft_internal_message(situation_code, recipient, borrower_ref, details, doc_context)
    return {
        "subject": msg.subject,
        "body": msg.body,
        "type": msg.message_type,
        "grounded": bool(doc_context),
    }


# ── Customer Email ─────────────────────────────────────────────────────────────

@app.post("/api/customer-email")
async def api_customer_email(
    situation: str = Form(...),
    first_name: str = Form(...),
    officer_name: str = Form(...),
    tone: str = Form("standard"),
    details: str = Form(""),
    session_id: str = Form(""),
):
    doc_context = _retrieve_context(session_id, f"{situation} {details}")
    email = draft_customer_email(situation, first_name, officer_name, tone, details, doc_context)
    return {
        "subject": email.subject,
        "full_email": email.full_email,
        "grounded": bool(doc_context),
    }


# ── Q&A Upload ────────────────────────────────────────────────────────────────

@app.post("/api/qa/upload")
async def api_qa_upload(file: UploadFile = File(...)):
    content = await file.read()
    doc_id = str(uuid.uuid4())
    saved_path = _save_uploaded_file(doc_id, file.filename, content)
    store_path = _doc_store_path(doc_id)

    try:
        chunks = process_document(str(saved_path))
        store = build_vector_store(chunks, store_path=store_path)
        chain = build_qa_chain(store)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    session_id = str(uuid.uuid4())
    _qa_sessions[session_id] = {
        "chain": chain,
        "store": store,
        "filename": file.filename,
        "doc_id": doc_id,
    }

    _upsert_history({
        "id": doc_id,
        "filename": file.filename,
        "uploaded_at": datetime.now().isoformat(),
        "file_size": len(content),
        "file_type": Path(file.filename).suffix.lower().lstrip("."),
        "chunks": len(chunks),
        "has_summary": False,
        "has_compliance": False,
        "has_vector_store": True,
    })

    return {
        "session_id": session_id,
        "doc_id": doc_id,
        "chunks": len(chunks),
        "filename": file.filename,
    }


@app.post("/api/qa/load/{doc_id}")
async def api_qa_load(doc_id: str):
    """Load a previously uploaded document into a new QA session."""
    entry = _get_history_entry(doc_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Document not found in history.")

    store_path = _doc_store_path(doc_id)

    # Try loading existing store; if missing, rebuild from the original file.
    try:
        store = load_vector_store(store_path)
    except FileNotFoundError:
        doc_dir = _doc_dir(doc_id)
        candidates = [p for p in doc_dir.iterdir() if p.is_file() and p.suffix.lower() in (".pdf", ".txt")]
        if not candidates:
            raise HTTPException(status_code=404, detail="Original file not found.")
        try:
            chunks = process_document(str(candidates[0]))
            store = build_vector_store(chunks, store_path=store_path)
            _update_history_flags(doc_id, has_vector_store=True, chunks=len(chunks))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    chain = build_qa_chain(store)
    session_id = str(uuid.uuid4())
    _qa_sessions[session_id] = {
        "chain": chain,
        "store": store,
        "filename": entry["filename"],
        "doc_id": doc_id,
    }

    return {
        "session_id": session_id,
        "doc_id": doc_id,
        "filename": entry["filename"],
        "chunks": entry.get("chunks", 0),
    }


@app.post("/api/qa/ask")
async def api_qa_ask(
    session_id: str = Form(...),
    question: str = Form(...),
):
    session_data = _qa_sessions.get(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session expired. Please re-upload the document.")
    result = ask(session_data["chain"], question)
    return {"answer": result["answer"]}


@app.post("/api/qa/save-chat")
async def api_qa_save_chat(
    session_id: str = Form(...),
    messages: str = Form(...),  # JSON string: [{"role": "user"|"assistant", "content": "..."}]
):
    session_data = _qa_sessions.get(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found.")

    doc_id = session_data.get("doc_id", "unknown")
    try:
        msgs = json.loads(messages)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid messages JSON.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chat_file = _doc_dir(doc_id) / f"chat_{timestamp}.json"
    chat_data = {
        "doc_id": doc_id,
        "filename": session_data["filename"],
        "saved_at": datetime.now().isoformat(),
        "messages": msgs,
    }
    chat_file.write_text(json.dumps(chat_data, indent=2, ensure_ascii=False), encoding="utf-8")

    return {"saved": True, "path": str(chat_file), "message_count": len(msgs)}


# ── History ───────────────────────────────────────────────────────────────────

@app.get("/api/history")
async def api_history():
    return _load_history()


@app.get("/api/history/{doc_id}")
async def api_history_detail(doc_id: str):
    entry = _get_history_entry(doc_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Not found.")

    chats = []
    doc_dir_path = UPLOADED_DATA_DIR / doc_id
    if doc_dir_path.exists():
        chats = [p.name for p in sorted(doc_dir_path.glob("chat_*.json"))]

    return {**entry, "saved_chats": chats}


@app.get("/api/history/{doc_id}/compliance")
async def api_history_compliance(doc_id: str):
    report_path = _doc_compliance_path(doc_id)
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="No compliance report for this document.")
    return json.loads(report_path.read_text(encoding="utf-8"))


@app.get("/api/history/{doc_id}/summary")
async def api_history_summary(doc_id: str):
    summary_path = _doc_summary_path(doc_id)
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="No summary for this document.")
    return {"summary": summary_path.read_text(encoding="utf-8")}


# ── Misc ──────────────────────────────────────────────────────────────────────

@app.get("/api/situations")
async def api_situations():
    return {k: v["description"] for k, v in SITUATIONS.items()}


# ── Serve frontend ────────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())
