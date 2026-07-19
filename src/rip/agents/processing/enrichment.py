"""EnrichmentAgent — structured tagging of each kept document via the Message
Batches API. Reviews are packed N-per-request to amortize prompt overhead;
each review echoes its id in the output schema so results can never misalign.
Never rewrites text — tags only.
"""

from __future__ import annotations

import json

from rich.console import Console

from ...core import config, llm
from ...core.models import BEHAVIOR_SIGNALS, PRODUCT_CATEGORIES, EnrichedDocument, RunManifest, utcnow
from ...core.storage import Store, new_run_id

console = Console()

PROMPT_VERSION = "enrich-v1"

SYSTEM = f"""You tag user feedback about Indian quick-commerce apps (Zepto, Blinkit, Swiggy Instamart) with structured labels for product research. Reviews may be in English, Hindi (Devanagari), or Hinglish.

For EACH review in the input, output one object with:
- id: the review id, copied exactly
- sentiment: positive | negative | mixed | neutral
- topics: 1-4 short lowercase tags for what the review is about (e.g. "delivery speed", "refund", "app crash", "price", "product quality")
- categories_mentioned: product categories explicitly mentioned, ONLY from: {json.dumps(PRODUCT_CATEGORIES)}
- behavior_signals: behavioral signals genuinely present, ONLY from: {json.dumps(BEHAVIOR_SIGNALS)}
- segment_hints: user-segment clues stated or strongly implied (e.g. "new_user", "long_time_user", "parent", "student", "bachelor", "working_professional", "tier2_city"), [] if none
- is_informative: false ONLY if the review carries no signal beyond bare sentiment ("good app", "worst")

Rules: tag only what the text actually says — never infer beyond it. Sarcasm counts by intended meaning. Output only the JSON object, nothing else."""

SCHEMA = {
    "type": "object",
    "properties": {
        "reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "sentiment": {"type": "string", "enum": ["positive", "negative", "mixed", "neutral"]},
                    "topics": {"type": "array", "items": {"type": "string"}},
                    "categories_mentioned": {"type": "array", "items": {"type": "string", "enum": PRODUCT_CATEGORIES}},
                    "behavior_signals": {"type": "array", "items": {"type": "string", "enum": BEHAVIOR_SIGNALS}},
                    "segment_hints": {"type": "array", "items": {"type": "string"}},
                    "is_informative": {"type": "boolean"},
                },
                "required": ["id", "sentiment", "topics", "categories_mentioned",
                             "behavior_signals", "segment_hints", "is_informative"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["reviews"],
    "additionalProperties": False,
}


def _pack(store: Store, per_request: int) -> list[list[tuple[str, str]]]:
    """Group (id, text) pairs of not-yet-enriched kept docs into request packs."""
    done = store.enriched_ids()
    todo = [
        (d.id, d.clean_text[:1500])  # hard cap pathological lengths
        for d in store.iter_processed(kept_only=True)
        if d.id not in done
    ]
    return [todo[i : i + per_request] for i in range(0, len(todo), per_request)]


def _user_content(pack: list[tuple[str, str]]) -> str:
    lines = ["Tag these reviews:"]
    for doc_id, text in pack:
        lines.append(f'\n[id={doc_id}] {text}')
    return "\n".join(lines)


def run_enrichment(store: Store, estimate_only: bool = False) -> RunManifest:
    cfg = config.settings()["llm"]
    manifest = RunManifest(agent="enrichment", run_id=new_run_id("enrichment"),
                           started_at=utcnow(), params={"prompt_version": PROMPT_VERSION})
    packs = _pack(store, cfg["enrichment_reviews_per_request"])
    if not packs:
        console.print("[yellow]enrichment: nothing to do (all docs already enriched)[/yellow]")
        manifest.status = "skipped"
        manifest.finished_at = utcnow()
        store.save_manifest(manifest)
        return manifest

    c = llm.client()

    # --- precise cost estimate from a sample of real packs ---
    sample = packs[: min(5, len(packs))]
    sample_tokens = [llm.count_input_tokens(c, SYSTEM, _user_content(p)) for p in sample]
    avg_in = sum(sample_tokens) / len(sample_tokens)
    est_out_per_review = 90
    total_in = int(avg_in * len(packs))
    total_out = int(est_out_per_review * sum(len(p) for p in packs))
    cost = llm.estimate_cost(total_in, total_out)
    console.print(
        f"[bold]enrichment estimate[/bold]: {sum(len(p) for p in packs)} reviews in "
        f"{len(packs)} requests · ~{total_in/1e6:.2f}M input + ~{total_out/1e6:.2f}M output tokens "
        f"· [bold]≈ ${cost:.2f}[/bold] (batched {llm.MODEL})"
    )
    if estimate_only:
        manifest.status = "skipped"
        manifest.notes.append(f"estimate_only: ${cost:.2f}")
        manifest.finished_at = utcnow()
        store.save_manifest(manifest)
        return manifest

    # --- submit + wait + persist ---
    requests = [
        llm.build_request(
            custom_id=f"enr-{i}",
            system=SYSTEM,
            user_content=_user_content(pack),
            max_tokens=min(120 * len(pack) + 500, 16000),
            schema=SCHEMA,
        )
        for i, pack in enumerate(packs)
    ]
    batch_id = llm.submit_batch(c, requests)
    manifest.params["batch_id"] = batch_id
    llm.wait_for_batch(c, batch_id, max_wait_minutes=cfg["max_batch_wait_minutes"])

    pack_by_id = {f"enr-{i}": pack for i, pack in enumerate(packs)}
    valid_ids = {doc_id for pack in packs for doc_id, _ in pack}
    enriched: list[EnrichedDocument] = []
    errored_packs = 0
    for custom_id, parsed in llm.iter_batch_results(c, batch_id):
        if parsed is None:
            errored_packs += 1
            continue
        for item in parsed.get("reviews", []):
            if item.get("id") not in valid_ids:
                continue  # hallucinated id — refuse to attach
            try:
                enriched.append(EnrichedDocument(**item, model=llm.MODEL,
                                                 prompt_version=PROMPT_VERSION))
            except Exception:
                continue
        _ = pack_by_id  # ids echo through schema; packs kept for debugging
    n = store.upsert_enriched(enriched)
    manifest.counts = {"requests": len(packs), "errored_requests": errored_packs,
                       "enriched_written": n}
    manifest.status = "ok" if errored_packs == 0 else "partial"
    manifest.finished_at = utcnow()
    store.save_manifest(manifest)
    console.print(f"[green]✓ enrichment[/green] wrote {n} enriched docs "
                  f"(errored requests: {errored_packs})")
    return manifest
