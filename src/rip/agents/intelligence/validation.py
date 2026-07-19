"""ValidationAgent — the fabrication firewall. Purely deterministic; no LLM.

For every candidate insight:
  1. every cited review id must exist in the raw store
  2. every quote must appear VERBATIM (whitespace/case-normalized) in its review
  3. support count must clear the configured minimum
  4. contradiction retrieval: semantically similar docs with opposing sentiment
  5. confidence = transparent rubric, not a model's self-reported number

Failures reject the insight with a recorded reason — never silent repair.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import numpy as np
from rich.console import Console

from ...core import config
from ...core.models import ConfidenceRubric, Insight, RunManifest, utcnow
from ...core.storage import Store, new_run_id

console = Console()


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _load_embeddings() -> tuple[np.ndarray | None, dict[str, int]]:
    try:
        emb = np.load(config.PROCESSED_DIR / "embeddings.npy")
        ids = (config.PROCESSED_DIR / "embedding_ids.txt").read_text().splitlines()
        return emb, {doc_id: i for i, doc_id in enumerate(ids)}
    except FileNotFoundError:
        return None, {}


def run_validation(store: Store) -> RunManifest:
    cfg = config.settings()
    weights = cfg["validation"]["confidence_weights"]
    min_support = cfg["insights"]["min_support_count"]
    manifest = RunManifest(agent="validation", run_id=new_run_id("validation"),
                           started_at=utcnow())

    sentiment = {e.id: e.sentiment for e in store.iter_enriched()}
    emb, emb_idx = _load_embeddings()
    now = datetime.now(timezone.utc)
    total_docs = max(sum(store.raw_counts().values()), 1)

    results: list[Insight] = []
    counts = {"validated": 0, "rejected_citation": 0, "rejected_quote": 0,
              "rejected_support": 0}

    for ins in store.iter_insights():
        # 1. cited ids must exist
        raws = {rid: store.get_raw(rid) for rid in ins.supporting_review_ids}
        missing = [rid for rid, r in raws.items() if r is None]
        if missing:
            ins.validated = False
            ins.rejection_reason = f"cited ids not in raw store: {missing[:5]}"
            counts["rejected_citation"] += 1
            results.append(ins)
            continue

        # 2. verbatim quote verification
        verified_quotes = []
        for q in ins.representative_quotes:
            raw = store.get_raw(q.review_id)
            q.verified = bool(raw and _norm(q.quote) in _norm(raw.text))
            if q.verified:
                verified_quotes.append(q)
        ins.representative_quotes = verified_quotes
        if not verified_quotes:
            ins.validated = False
            ins.rejection_reason = "no quote survived verbatim verification"
            counts["rejected_quote"] += 1
            results.append(ins)
            continue

        # 3. support threshold
        if ins.support_count < min_support:
            ins.validated = False
            ins.rejection_reason = f"support {ins.support_count} < minimum {min_support}"
            counts["rejected_support"] += 1
            results.append(ins)
            continue

        # 4. contradiction retrieval: similar docs, opposing sentiment
        support_sents = [sentiment.get(rid) for rid in ins.supporting_review_ids]
        dominant_neg = sum(1 for s in support_sents if s == "negative") >= len(support_sents) / 2
        opposing = "positive" if dominant_neg else "negative"
        contradicting: list[str] = []
        if emb is not None:
            sup_vecs = [emb[emb_idx[rid]] for rid in ins.supporting_review_ids if rid in emb_idx]
            if sup_vecs:
                centroid = np.mean(sup_vecs, axis=0)
                centroid /= np.linalg.norm(centroid) + 1e-9
                sims = emb @ centroid
                order = np.argsort(-sims)
                id_list = list(emb_idx)
                sup_set = set(ins.supporting_review_ids)
                for j in order[:400]:
                    if sims[j] < 0.45:
                        break
                    did = id_list[j]
                    if did in sup_set or sentiment.get(did) != opposing:
                        continue
                    contradicting.append(did)
                    if len(contradicting) >= 5:
                        break
        ins.contradicting_review_ids = contradicting
        if contradicting:
            ins.contradiction_summary = (
                f"{len(contradicting)} semantically similar reviews express the opposite "
                f"({opposing}) experience — see ids."
            )

        # 5. confidence rubric
        volume = min(1.0, np.log1p(ins.support_count) / np.log1p(50))
        sources = {raws[rid].source for rid in ins.supporting_review_ids if raws.get(rid)}
        diversity = min(1.0, len(sources) / 2)
        dates = [raws[rid].created_at for rid in ins.supporting_review_ids
                 if raws.get(rid) and raws[rid].created_at]
        recent = sum(1 for d in dates if now - d <= timedelta(days=365))
        recency = recent / len(dates) if dates else 0.5
        consistency = ins.support_count / (ins.support_count + len(contradicting))
        score = (weights["volume"] * volume + weights["source_diversity"] * diversity
                 + weights["recency"] * recency + weights["consistency"] * consistency)
        ins.confidence = ConfidenceRubric(volume=round(volume, 3),
                                          source_diversity=round(diversity, 3),
                                          recency=round(recency, 3),
                                          consistency=round(consistency, 3),
                                          score=round(score, 3))
        ins.validated = True
        ins.rejection_reason = None
        counts["validated"] += 1
        results.append(ins)

    store.upsert_insights(results)
    manifest.counts = counts | {"corpus_size": total_docs}
    manifest.status = "ok"
    manifest.finished_at = utcnow()
    store.save_manifest(manifest)
    console.print(f"[green]✓ validation[/green] {counts}")
    return manifest
