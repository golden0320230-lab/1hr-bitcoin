"""Text normalization helpers for article ingestion and model prompts."""

from __future__ import annotations

import hashlib
import re
from html import unescape

from bs4 import BeautifulSoup

_WHITESPACE_PATTERN = re.compile(r"\s+")


def strip_html(value: str) -> str:
    """Remove HTML tags and decode entities from a text fragment."""

    soup = BeautifulSoup(value, "html.parser")
    return unescape(soup.get_text(" ", strip=True))


def normalize_whitespace(value: str) -> str:
    """Collapse repeated whitespace to single spaces."""

    return _WHITESPACE_PATTERN.sub(" ", value).strip()


def sanitize_text(value: str) -> str:
    """Remove HTML and normalize whitespace."""

    return normalize_whitespace(strip_html(value))


def truncate_text(value: str, *, max_chars: int) -> str:
    """Trim text to a maximum length while preserving whole words when practical."""

    if len(value) <= max_chars:
        return value

    clipped = value[: max_chars - 1].rstrip()
    if len(value) > max_chars and max_chars - 1 < len(value):
        next_character = value[max_chars - 1]
    else:
        next_character = " "

    if " " in clipped and next_character not in {" ", "\n", "\t", ".", ",", "!", "?"}:
        clipped = clipped.rsplit(" ", maxsplit=1)[0]
    return f"{clipped}…"


def fingerprint_article(title: str, summary: str | None = None) -> str:
    """Create a stable fingerprint from article title and summary content."""

    payload = sanitize_text(f"{title}\n{summary or ''}").lower().encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
