"""Settings loader: config/settings.yaml + .env (secrets)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "config"
# ponytail: RIP_DATA_DIR redirects the whole corpus (e.g. a v2 run) without touching data/;
# everything below derives from DATA_DIR, so one override moves it all. Reports follow suit.
DATA_DIR = Path(os.environ["RIP_DATA_DIR"]) if os.environ.get("RIP_DATA_DIR") else PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
INSIGHTS_DIR = DATA_DIR / "insights"
REPORTS_DIR = DATA_DIR / "reports" if os.environ.get("RIP_DATA_DIR") else PROJECT_ROOT / "reports"
DB_PATH = DATA_DIR / "rip.db"

load_dotenv(PROJECT_ROOT / ".env")


@lru_cache(maxsize=1)
def settings() -> dict[str, Any]:
    with open(CONFIG_DIR / "settings.yaml") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def research_questions() -> list[dict[str, Any]]:
    with open(CONFIG_DIR / "research_questions.yaml") as f:
        return yaml.safe_load(f)["research_questions"]


def env(name: str) -> str | None:
    v = os.environ.get(name, "").strip()
    return v or None


def credential_status() -> dict[str, bool]:
    return {
        "anthropic": env("ANTHROPIC_API_KEY") is not None,
        "reddit": env("REDDIT_CLIENT_ID") is not None and env("REDDIT_CLIENT_SECRET") is not None,
        "youtube": env("YOUTUBE_API_KEY") is not None,
    }


def ensure_dirs() -> None:
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, INSIGHTS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
