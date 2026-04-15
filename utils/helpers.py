"""
utils/helpers.py — Small utility functions shared across the project.
"""

import os
from pathlib import Path
from typing import Optional


def validate_api_key() -> bool:
    """Return True if OPENAI_API_KEY is set and looks like a real key."""
    key = os.getenv("OPENAI_API_KEY", "")
    return bool(key) and key != "your-openai-api-key-here" and key.startswith("sk-")


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist. Returns Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def truncate(text: str, max_len: int = 200, suffix: str = "...") -> str:
    """Truncate text to max_len characters, appending suffix if cut."""
    if len(text) <= max_len:
        return text
    return text[:max_len - len(suffix)] + suffix


def format_currency(amount: float) -> str:
    """Format a float as a USD currency string."""
    return f"${amount:,.2f}"


def format_percentage(ratio: float, decimals: int = 1) -> str:
    """Format a 0.0–1.0 ratio as a percentage string."""
    return f"{ratio * 100:.{decimals}f}%"
