"""CleaningAgent — deterministic text normalization, language detection, and
spam/noise filtering. No LLM involvement; every drop carries a reason."""

from __future__ import annotations

import re
import unicodedata

from langdetect import DetectorFactory, LangDetectException, detect
from rich.console import Console

from ...core import config
from ...core.models import ProcessedDocument, RunManifest, utcnow
from ...core.storage import Store, new_run_id

DetectorFactory.seed = 42  # deterministic language detection
console = Console()

_WS = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+")
_REPEAT = re.compile(r"(.)\1{5,}")  # aaaaaaa, !!!!!!!!
_HTML_TAG = re.compile(r"<[^>]+>")

# pure-noise reviews that carry zero signal even for sentiment stats
_NOISE = re.compile(
    r"^(good|nice|ok|okay|bad|best|worst|super|great|excellent|awesome|useless|wow|no|yes|👍+|❤+|🙏+)[\s.!👍❤🙏]*$",
    re.I,
)


def clean_text(text: str) -> str:
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = _WS.sub(" ", text).strip()
    return text


def detect_language(text: str) -> str:
    # Devanagari script => hi regardless of langdetect (it mislabels short hinglish)
    if re.search(r"[ऀ-ॿ]", text):
        return "hi"
    try:
        lang = detect(text)
    except LangDetectException:
        return "unknown"
    # langdetect maps short romanized Hindi to odd languages; collapse to unknown
    return lang if lang in ("en", "hi") else "unknown"


def run_cleaning(store: Store) -> RunManifest:
    cfg = config.settings()["processing"]
    manifest = RunManifest(agent="cleaning", run_id=new_run_id("cleaning"), started_at=utcnow())
    out: list[ProcessedDocument] = []
    counts = {"total": 0, "kept": 0, "too_short": 0, "noise": 0, "language": 0}

    for raw in store.iter_raw():
        counts["total"] += 1
        text = clean_text(raw.text)
        if len(text) < cfg["min_chars"]:
            out.append(ProcessedDocument(id=raw.id, clean_text=text, language="unknown",
                                         kept=False, drop_reason="too_short"))
            counts["too_short"] += 1
            continue
        if _NOISE.match(text) or _REPEAT.search(text):
            out.append(ProcessedDocument(id=raw.id, clean_text=text, language="unknown",
                                         kept=False, drop_reason="spam"))
            counts["noise"] += 1
            continue
        lang = detect_language(text)
        if lang not in cfg["languages_kept"]:
            out.append(ProcessedDocument(id=raw.id, clean_text=text, language=lang,
                                         kept=False, drop_reason="language"))
            counts["language"] += 1
            continue
        out.append(ProcessedDocument(id=raw.id, clean_text=text, language=lang, kept=True))
        counts["kept"] += 1

    store.upsert_processed(out)
    manifest.counts = counts
    manifest.status = "ok"
    manifest.finished_at = utcnow()
    store.save_manifest(manifest)
    console.print(f"[green]✓ cleaning[/green] {counts}")
    return manifest
