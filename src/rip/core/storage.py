"""SQLite-backed store for every pipeline artifact, plus JSONL raw exports.

The raw_documents table is treated as immutable: INSERT OR IGNORE only, keyed
by document id, so re-running collection never mutates or duplicates evidence.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

from . import config
from .models import (
    Cluster,
    EnrichedDocument,
    Insight,
    ProcessedDocument,
    RawDocument,
    RQCategory,
    RunManifest,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_documents (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    app TEXT,
    doc_type TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS processed_documents (
    id TEXT PRIMARY KEY,
    kept INTEGER NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS enriched_documents (
    id TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS clusters (
    cluster_id INTEGER PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS insights (
    insight_id TEXT PRIMARY KEY,
    validated INTEGER NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rq_categories (
    key TEXT PRIMARY KEY,
    rq_id TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS manifests (
    run_id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_raw_source ON raw_documents(source);
CREATE INDEX IF NOT EXISTS idx_raw_app ON raw_documents(app);
"""


class Store:
    def __init__(self, db_path: Path | None = None):
        config.ensure_dirs()
        self.db_path = db_path or config.DB_PATH
        self.conn = sqlite3.connect(self.db_path)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # ---------- raw ----------

    def add_raw(self, docs: Iterable[RawDocument]) -> int:
        """Insert raw docs; returns number actually written (new ids only)."""
        written = 0
        for d in docs:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO raw_documents (id, source, app, doc_type, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (d.id, d.source, d.app, d.doc_type, d.model_dump_json()),
            )
            written += cur.rowcount
        self.conn.commit()
        return written

    def iter_raw(self, source: str | None = None) -> Iterator[RawDocument]:
        q = "SELECT payload FROM raw_documents"
        args: tuple = ()
        if source:
            q += " WHERE source = ?"
            args = (source,)
        for (payload,) in self.conn.execute(q, args):
            yield RawDocument.model_validate_json(payload)

    def get_raw(self, doc_id: str) -> RawDocument | None:
        row = self.conn.execute(
            "SELECT payload FROM raw_documents WHERE id = ?", (doc_id,)
        ).fetchone()
        return RawDocument.model_validate_json(row[0]) if row else None

    def ids_for_apps(self, apps: set[str]) -> set[str]:
        q = ",".join("?" * len(apps))
        return {r[0] for r in self.conn.execute(
            f"SELECT id FROM raw_documents WHERE lower(app) IN ({q})", tuple(apps))}

    def raw_counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT source, COALESCE(app,'-'), COUNT(*) FROM raw_documents GROUP BY 1,2"
        ).fetchall()
        return {f"{s}/{a}": c for s, a, c in rows}

    def export_raw_jsonl(self, source: str) -> Path:
        out = config.RAW_DIR / f"{source}.jsonl"
        with open(out, "w") as f:
            for doc in self.iter_raw(source):
                f.write(doc.model_dump_json() + "\n")
        return out

    # ---------- processed ----------

    def upsert_processed(self, docs: Iterable[ProcessedDocument]) -> int:
        n = 0
        for d in docs:
            self.conn.execute(
                "INSERT OR REPLACE INTO processed_documents (id, kept, payload) VALUES (?, ?, ?)",
                (d.id, int(d.kept), d.model_dump_json()),
            )
            n += 1
        self.conn.commit()
        return n

    def iter_processed(self, kept_only: bool = True) -> Iterator[ProcessedDocument]:
        q = "SELECT payload FROM processed_documents"
        if kept_only:
            q += " WHERE kept = 1"
        for (payload,) in self.conn.execute(q):
            yield ProcessedDocument.model_validate_json(payload)

    # ---------- enriched ----------

    def upsert_enriched(self, docs: Iterable[EnrichedDocument]) -> int:
        n = 0
        for d in docs:
            self.conn.execute(
                "INSERT OR REPLACE INTO enriched_documents (id, payload) VALUES (?, ?)",
                (d.id, d.model_dump_json()),
            )
            n += 1
        self.conn.commit()
        return n

    def iter_enriched(self) -> Iterator[EnrichedDocument]:
        for (payload,) in self.conn.execute("SELECT payload FROM enriched_documents"):
            yield EnrichedDocument.model_validate_json(payload)

    def enriched_ids(self) -> set[str]:
        return {r[0] for r in self.conn.execute("SELECT id FROM enriched_documents")}

    # ---------- clusters ----------

    def replace_clusters(self, clusters: Iterable[Cluster]) -> int:
        self.conn.execute("DELETE FROM clusters")
        n = 0
        for c in clusters:
            self.conn.execute(
                "INSERT INTO clusters (cluster_id, payload) VALUES (?, ?)",
                (c.cluster_id, c.model_dump_json()),
            )
            n += 1
        self.conn.commit()
        return n

    def iter_clusters(self) -> Iterator[Cluster]:
        for (payload,) in self.conn.execute("SELECT payload FROM clusters ORDER BY cluster_id"):
            yield Cluster.model_validate_json(payload)

    # ---------- insights ----------

    def upsert_insights(self, insights: Iterable[Insight]) -> int:
        n = 0
        for i in insights:
            self.conn.execute(
                "INSERT OR REPLACE INTO insights (insight_id, validated, payload) VALUES (?, ?, ?)",
                (i.insight_id, int(i.validated), i.model_dump_json()),
            )
            n += 1
        self.conn.commit()
        return n

    def iter_insights(self, validated_only: bool = False) -> Iterator[Insight]:
        q = "SELECT payload FROM insights"
        if validated_only:
            q += " WHERE validated = 1"
        for (payload,) in self.conn.execute(q):
            yield Insight.model_validate_json(payload)

    # ---------- rq categories ----------

    def replace_rq_categories(self, cats: Iterable[RQCategory]) -> int:
        self.conn.execute("DELETE FROM rq_categories")
        n = 0
        for cat in cats:
            self.conn.execute(
                "INSERT INTO rq_categories (key, rq_id, payload) VALUES (?, ?, ?)",
                (f"{cat.rq_id}:{cat.index}", cat.rq_id, cat.model_dump_json()),
            )
            n += 1
        self.conn.commit()
        return n

    def iter_rq_categories(self) -> Iterator[RQCategory]:
        for (payload,) in self.conn.execute("SELECT payload FROM rq_categories ORDER BY key"):
            yield RQCategory.model_validate_json(payload)

    # ---------- manifests ----------

    def save_manifest(self, m: RunManifest) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO manifests (run_id, agent, payload) VALUES (?, ?, ?)",
            (m.run_id, m.agent, m.model_dump_json()),
        )
        self.conn.commit()

    def iter_manifests(self) -> Iterator[RunManifest]:
        for (payload,) in self.conn.execute("SELECT payload FROM manifests"):
            yield RunManifest.model_validate_json(payload)


def new_run_id(agent: str) -> str:
    return f"{agent}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
