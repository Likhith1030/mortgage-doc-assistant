"""
config.py — Central configuration for the Mortgage Document Assistant.

All environment variables and tunable constants live here.
Import this module anywhere you need a setting instead of hard-coding values.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file into os.environ

# ── LLM settings ─────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o")

EMBEDDING_MODEL = "text-embedding-3-small"

# ── Unstructured API (PDF parsing) ────────────────────────────────────────────
UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY", "")
UNSTRUCTURED_API_URL = os.getenv("UNSTRUCTURED_API_URL", "https://api.unstructuredapp.io")

# ── Vector store ──────────────────────────────────────────────────────────────
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")

RETRIEVAL_TOP_K = 12  # chunks passed to the LLM as context

# ── Document chunking ────────────────────────────────────────────────────────
CHUNK_SIZE    = 1500  # ~375 tokens per chunk; larger = more context per result
CHUNK_OVERLAP = 200   # overlap prevents losing context at chunk boundaries

# ── Uploaded documents & history ─────────────────────────────────────────────
UPLOADED_DATA_PATH = os.getenv("UPLOADED_DATA_PATH", "./data/uploaded_data")

# ── Compliance rules file ────────────────────────────────────────────────────
COMPLIANCE_RULES_PATH = "./data/knowledge_base/compliance_rules.txt"
MORTGAGE_GUIDELINES_PATH = "./data/knowledge_base/mortgage_guidelines.txt"

# ── Metrics ──────────────────────────────────────────────────────────────────
METRICS_LOG_PATH = "./data/metrics_log.csv"
