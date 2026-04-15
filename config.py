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

# Embedding model used when converting text chunks to vectors.
# text-embedding-3-small is cheap; swap for text-embedding-3-large for accuracy.
EMBEDDING_MODEL = "text-embedding-3-small"

# ── Vector store ──────────────────────────────────────────────────────────────
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")

# Number of similar chunks to retrieve when answering a question.
RETRIEVAL_TOP_K = 5

# ── Document chunking ────────────────────────────────────────────────────────
# How many characters per chunk. Larger = more context, slower + costlier.
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150  # overlap keeps context from being cut at chunk boundaries

# ── Compliance rules file ────────────────────────────────────────────────────
COMPLIANCE_RULES_PATH = "./data/knowledge_base/compliance_rules.txt"
MORTGAGE_GUIDELINES_PATH = "./data/knowledge_base/mortgage_guidelines.txt"

# ── Metrics ──────────────────────────────────────────────────────────────────
METRICS_LOG_PATH = "./data/metrics_log.csv"
