"""InsightAgent — generates candidate insights with mandatory citations.

Two complementary passes, both via the Batches API:
  1. per-cluster (topical structure discovered by embeddings)
  2. per-research-question (behavioral pools built from enrichment signals)

Every insight must cite review ids from the exact sample shown to the model.
Nothing here is trusted: ValidationAgent verifies everything downstream.
"""

from __future__ import annotations

import random
from collections import defaultdict

from rich.console import Console

from ...core import config, llm
from ...core.models import Insight, Quote, RunManifest, utcnow
from ...core.storage import Store, new_run_id

console = Console()

PROMPT_VERSION = "insight-v1"
SAMPLE_PER_CLUSTER = 40
SAMPLE_PER_RQ = 60

SYSTEM = """You are a rigorous product researcher analyzing user feedback about Indian quick-commerce apps (Zepto, Blinkit, Swiggy Instamart).

From the reviews provided, produce:
- label: a short human-readable name for this group of reviews (<=6 words)
- insights: 0 to 3 insights. Fewer, stronger insights beat many weak ones.

Each insight must have:
- claim: one precise, falsifiable sentence about user behavior or need. Not a platitude.
- supporting_review_ids: ids from THIS input that genuinely support the claim (as many as apply)
- quotes: 2-4 objects {review_id, quote} where quote is copied VERBATIM (exact characters) from that review's text, max ~40 words. Never paraphrase inside quotes.
- research_questions: which of these it informs: RQ1 repeat-category buying, RQ2 exploration barriers, RQ3 product discovery, RQ4 habits, RQ5 missing information before trying new categories, RQ6 experimental segments, RQ7 unmet needs. [] if none.
- segments: user segments the evidence points to, [] if unclear
- apps: which apps the evidence covers

HARD RULES: cite only ids that appear in the input. Quote only text that appears character-for-character in the cited review. If the reviews don't support a real insight, return fewer or zero insights. Output only JSON."""

SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "insights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "supporting_review_ids": {"type": "array", "items": {"type": "string"}},
                    "quotes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "review_id": {"type": "string"},
                                "quote": {"type": "string"},
                            },
                            "required": ["review_id", "quote"],
                            "additionalProperties": False,
                        },
                    },
                    "research_questions": {"type": "array", "items": {"type": "string"}},
                    "segments": {"type": "array", "items": {"type": "string"}},
                    "apps": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["claim", "supporting_review_ids", "quotes",
                             "research_questions", "segments", "apps"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["label", "insights"],
    "additionalProperties": False,
}


def _render_sample(store: Store, ids: list[str], clean: dict[str, str]) -> str:
    lines = []
    for doc_id in ids:
        raw = store.get_raw(doc_id)
        if not raw:
            continue
        meta = f"app={raw.app or '?'} src={raw.source}"
        if raw.rating:
            meta += f" rating={raw.rating}"
        lines.append(f"[id={doc_id} {meta}] {clean.get(doc_id, raw.text)[:900]}")
    return "\n".join(lines)


def run_insight_generation(store: Store) -> RunManifest:
    manifest = RunManifest(agent="insight", run_id=new_run_id("insight"),
                           started_at=utcnow(), params={"prompt_version": PROMPT_VERSION})
    rng = random.Random(42)
    clean = {d.id: d.clean_text for d in store.iter_processed(kept_only=True)}
    enriched = {e.id: e for e in store.iter_enriched()}

    jobs: list[tuple[str, str, list[str]]] = []  # (custom_id, header, sampled ids)

    # Pass 1 — clusters
    for c in store.iter_clusters():
        ids = [i for i in c.member_ids if i in clean]
        if len(ids) < 10:
            continue
        sample = rng.sample(ids, min(SAMPLE_PER_CLUSTER, len(ids)))
        header = (f"Review group (cluster {c.cluster_id}, {c.size} reviews total, "
                  f"showing {len(sample)}). Frequent terms: {', '.join(c.top_terms)}.")
        jobs.append((f"cl-{c.cluster_id}", header, sample))

    # Pass 2 — research-question signal pools
    rq_pool: dict[str, list[str]] = defaultdict(list)
    signal_map = {rq["id"]: set(rq["signals"]) for rq in config.research_questions()}
    for doc_id, e in enriched.items():
        if doc_id not in clean or not e.is_informative:
            continue
        for rq_id, signals in signal_map.items():
            if signals & set(e.behavior_signals):
                rq_pool[rq_id].append(doc_id)
    for rq in config.research_questions():
        ids = rq_pool.get(rq["id"], [])
        if len(ids) < 8:
            continue
        sample = rng.sample(ids, min(SAMPLE_PER_RQ, len(ids)))
        header = (f"Reviews selected because they carry behavioral signals for "
                  f"{rq['id']}: \"{rq['question']}\" (pool of {len(ids)}, showing {len(sample)}). "
                  f"Focus insights on this question.")
        jobs.append((f"rq-{rq['id']}", header, sample))

    if not jobs:
        manifest.status = "skipped"
        manifest.notes.append("no clusters or signal pools available")
        manifest.finished_at = utcnow()
        store.save_manifest(manifest)
        console.print("[yellow]insight: nothing to analyze yet[/yellow]")
        return manifest

    c = llm.client()
    requests = [
        llm.build_request(
            custom_id=cid,
            system=SYSTEM,
            user_content=header + "\n\n" + _render_sample(store, sample, clean),
            max_tokens=4000,
            schema=SCHEMA,
        )
        for cid, header, sample in jobs
    ]
    batch_id = llm.submit_batch(c, requests)
    manifest.params["batch_id"] = batch_id
    llm.wait_for_batch(c, batch_id)

    sample_sets = {cid: set(sample) for cid, _, sample in jobs}
    insights: list[Insight] = []
    labels_by_cluster: dict[int, str] = {}
    for custom_id, parsed in llm.iter_batch_results(c, batch_id):
        if parsed is None:
            manifest.notes.append(f"{custom_id}: request errored or unparseable")
            continue
        if custom_id.startswith("cl-"):
            labels_by_cluster[int(custom_id[3:])] = parsed.get("label", "")
        for k, item in enumerate(parsed.get("insights", [])):
            cited = [i for i in item.get("supporting_review_ids", [])
                     if i in sample_sets[custom_id]]  # citations outside the sample are discarded
            insights.append(Insight(
                insight_id=f"{custom_id}-i{k}",
                claim=item.get("claim", "").strip(),
                research_questions=[r for r in item.get("research_questions", [])
                                    if r.startswith("RQ")],
                supporting_review_ids=cited,
                support_count=len(cited),
                representative_quotes=[Quote(review_id=q["review_id"], quote=q["quote"])
                                       for q in item.get("quotes", [])],
                segments=item.get("segments", []),
                apps=item.get("apps", []),
                source_cluster=int(custom_id[3:]) if custom_id.startswith("cl-") else None,
                validated=False,
            ))

    # attach labels to clusters
    clusters = list(store.iter_clusters())
    for cl in clusters:
        if cl.cluster_id in labels_by_cluster:
            cl.label = labels_by_cluster[cl.cluster_id]
    store.replace_clusters(clusters)

    n = store.upsert_insights(insights)
    manifest.counts = {"jobs": len(jobs), "candidate_insights": n}
    manifest.status = "ok"
    manifest.finished_at = utcnow()
    store.save_manifest(manifest)
    console.print(f"[green]✓ insight[/green] {n} candidate insights from {len(jobs)} jobs")
    return manifest
