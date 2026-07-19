"""DedupAgent — exact-hash dedup now, embedding near-dup pass runs inside
clustering (where embeddings already exist). Canonical copy wins; merges are
recorded via dup_of so evidence chains stay intact."""

from __future__ import annotations

import hashlib
import re

from rich.console import Console

from ...core.models import RunManifest, utcnow
from ...core.storage import Store, new_run_id

console = Console()


def _norm_key(text: str) -> str:
    """Aggressive normalization for exact-dup detection across trivial edits."""
    t = re.sub(r"[^\w]+", " ", text.lower())
    return hashlib.sha256(t.strip().encode()).hexdigest()


def run_dedup(store: Store) -> RunManifest:
    manifest = RunManifest(agent="dedup", run_id=new_run_id("dedup"), started_at=utcnow())
    seen: dict[str, str] = {}  # norm hash -> canonical doc id
    updated = []
    dups = 0
    for doc in store.iter_processed(kept_only=True):
        key = _norm_key(doc.clean_text)
        if key in seen:
            doc.kept = False
            doc.drop_reason = "exact_dup"
            doc.dup_of = seen[key]
            dups += 1
            updated.append(doc)
        else:
            seen[key] = doc.id
    store.upsert_processed(updated)
    manifest.counts = {"unique": len(seen), "exact_dups": dups}
    manifest.status = "ok"
    manifest.finished_at = utcnow()
    store.save_manifest(manifest)
    console.print(f"[green]✓ dedup[/green] unique={len(seen)} exact_dups={dups}")
    return manifest
