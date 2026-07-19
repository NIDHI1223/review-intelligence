"""Shared helpers for mapping text/config to the apps under study."""

from __future__ import annotations

import re

APP_PATTERNS = {
    "zepto": re.compile(r"\bzepto\b", re.I),
    "blinkit": re.compile(r"\bblink\s?it\b|\bgrofers\b", re.I),
    "instamart": re.compile(r"\binstamart\b|\bswiggy\s+instamart\b", re.I),
}


def detect_app(text: str) -> str | None:
    """Which app a piece of text is about; None if ambiguous or none."""
    hits = [app for app, pat in APP_PATTERNS.items() if pat.search(text)]
    return hits[0] if len(hits) == 1 else None


def mentions_any_app(text: str) -> bool:
    return any(pat.search(text) for pat in APP_PATTERNS.values())
