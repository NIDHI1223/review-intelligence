"""RQCategoryAgent — per research question, discover answer categories from
the data itself, then classify EVERY review in that question's signal pool
into one of them.

Two Batches-API passes:
  1. discover: a sample of the RQ pool → 5-10 question-specific categories
  2. assign:   the FULL pool, chunked, classified against those categories

Unlike insights (top-N claims that can repeat across questions), this gives
exhaustive, question-specific coverage: every pooled review lands in a
category or an explicit "none of these".
"""

from __future__ import annotations

import random
from collections import defaultdict

from rich.console import Console

from ...core import config, llm
from ...core.models import RQCategory, RunManifest, utcnow
from ...core.storage import Store, new_run_id

console = Console()

PROMPT_VERSION = "rqcat-v1"
DISCOVER_SAMPLE = 120
ASSIGN_CHUNK = 25
MIN_POOL = 20

DISCOVER_SYSTEM = """You are a product researcher analyzing user reviews of Indian quick-commerce apps (Zepto, Blinkit, Swiggy Instamart). Reviews may be in English, Hindi, or Hinglish.

You are given ONE research question and a sample of reviews selected as relevant to it. Design 5-10 mutually distinct categories that organize how these reviews answer this question. Categories must emerge from the reviews shown — not a generic taxonomy, and not a restatement of the question.

Each category:
- name: <=8 words, specific ("Melted frozen goods break trust", not "Quality issues")
- description: 1-2 sentences saying exactly which reviews belong in it

Cover the major recurring patterns. Do not invent categories the sample doesn't support. Output only JSON."""

DISCOVER_SCHEMA = {
    "type": "object",
    "properties": {
        "categories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name", "description"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["categories"],
    "additionalProperties": False,
}

ASSIGN_SYSTEM = """You classify user reviews of quick-commerce apps (Zepto, Blinkit, Swiggy Instamart) into given categories for a research question. Reviews may be in English, Hindi, or Hinglish.

For EVERY review in the input, output {id, category} where category is the number of the single best-fitting category, or 0 if none genuinely fits. Judge only from the review text — never stretch a weak fit. Output only JSON."""

ASSIGN_SCHEMA = {
    "type": "object",
    "properties": {
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "category": {"type": "integer"},
                },
                "required": ["id", "category"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["assignments"],
    "additionalProperties": False,
}


def _build_pools(store: Store) -> tuple[dict[str, str], dict[str, list[str]]]:
    """clean text by id + per-RQ pools of kept, informative doc ids (same
    signal logic as InsightAgent pass 2). RIP_APPS=zepto[,...] scopes pools
    to those apps, mirroring its meaning at collection time."""
    clean = {d.id: d.clean_text for d in store.iter_processed(kept_only=True)}
    only = config.env("RIP_APPS")
    allowed = (store.ids_for_apps({a.strip().lower() for a in only.split(",")})
               if only else None)
    pools: dict[str, list[str]] = defaultdict(list)
    signal_map = {rq["id"]: set(rq["signals"]) for rq in config.research_questions()}
    for e in store.iter_enriched():
        if e.id not in clean or not e.is_informative:
            continue
        if allowed is not None and e.id not in allowed:
            continue
        for rq_id, signals in signal_map.items():
            if signals & set(e.behavior_signals):
                pools[rq_id].append(e.id)
    return clean, pools


def _render(store: Store, ids: list[str], clean: dict[str, str], cap: int) -> str:
    lines = []
    for doc_id in ids:
        raw = store.get_raw(doc_id)
        if not raw:
            continue
        meta = f"app={raw.app or '?'}" + (f" rating={raw.rating}" if raw.rating else "")
        lines.append(f"[id={doc_id} {meta}] {clean.get(doc_id, raw.text)[:cap]}")
    return "\n".join(lines)


def _category_block(cats: list[RQCategory]) -> str:
    lines = [f"{c.index}. {c.name} — {c.description}" for c in cats]
    lines.append("0. None of these")
    return "\n".join(lines)


def _fold_assignments(cats: list[RQCategory], id_set: set[str], parsed: dict | None) -> int:
    """Apply one chunk's parsed assignments onto cats. Discards hallucinated
    ids, out-of-range or non-integer categories. Returns docs assigned."""
    n = 0
    for item in (parsed or {}).get("assignments", []):
        doc_id, k = item.get("id"), item.get("category")
        if doc_id not in id_set or not isinstance(k, int) or not 1 <= k <= len(cats):
            continue
        cats[k - 1].member_ids.append(doc_id)
        n += 1
    return n


def run_rq_categorization(store: Store, estimate_only: bool = False) -> RunManifest:
    manifest = RunManifest(agent="rq_categories", run_id=new_run_id("rqcat"),
                           started_at=utcnow(),
                           params={"prompt_version": PROMPT_VERSION,
                                   "apps": config.env("RIP_APPS") or "all"})
    rng = random.Random(42)
    clean, pools = _build_pools(store)
    rqs = [rq for rq in config.research_questions()
           if len(pools.get(rq["id"], [])) >= MIN_POOL]
    if not rqs:
        manifest.status = "skipped"
        manifest.notes.append("no RQ pool reached MIN_POOL — run enrich first")
        manifest.finished_at = utcnow()
        store.save_manifest(manifest)
        console.print("[yellow]rq-categories: nothing to categorize yet[/yellow]")
        return manifest

    c = llm.client()

    # --- cost estimate (assignment pass dominates) ---
    n_docs = sum(len(pools[rq["id"]]) for rq in rqs)
    n_chunks = sum(-(-len(pools[rq["id"]]) // ASSIGN_CHUNK) for rq in rqs)
    biggest = max((pools[rq["id"]] for rq in rqs), key=len)
    sample_user = _render(store, biggest[:ASSIGN_CHUNK], clean, 400)
    per_chunk_in = llm.count_input_tokens(c, ASSIGN_SYSTEM, sample_user) + 200  # + category block
    total_in = per_chunk_in * n_chunks + 15000 * len(rqs)  # discovery requests
    total_out = 20 * n_docs + 400 * len(rqs)
    cost = llm.estimate_cost(total_in, total_out)
    console.print(
        f"[bold]rq-categories estimate[/bold]: {n_docs} pooled reviews across {len(rqs)} questions "
        f"in {n_chunks + len(rqs)} requests · ~{total_in/1e6:.2f}M input + ~{total_out/1e6:.2f}M "
        f"output tokens · [bold]≈ ${cost:.2f}[/bold] (batched {llm.MODEL})"
    )
    if estimate_only:
        manifest.status = "skipped"
        manifest.notes.append(f"estimate_only: ${cost:.2f}")
        manifest.finished_at = utcnow()
        store.save_manifest(manifest)
        return manifest

    # --- pass 1: discover categories per question ---
    disc_reqs = []
    for rq in rqs:
        pool = pools[rq["id"]]
        sample = rng.sample(pool, min(DISCOVER_SAMPLE, len(pool)))
        header = (f'Research question: "{rq["question"]}"\n'
                  f"Sample of {len(sample)} reviews from a pool of {len(pool)} relevant ones:\n\n")
        disc_reqs.append(llm.build_request(
            custom_id=f"disc-{rq['id']}", system=DISCOVER_SYSTEM,
            user_content=header + _render(store, sample, clean, 600),
            max_tokens=2000, schema=DISCOVER_SCHEMA,
        ))
    batch_id = llm.submit_batch(c, disc_reqs)
    manifest.params["discover_batch_id"] = batch_id
    llm.wait_for_batch(c, batch_id)

    cats_by_rq: dict[str, list[RQCategory]] = {}
    for custom_id, parsed in llm.iter_batch_results(c, batch_id):
        rq_id = custom_id[5:]
        items = (parsed or {}).get("categories", [])[:12]
        if not items:
            manifest.notes.append(f"{rq_id}: category discovery errored — question skipped")
            continue
        cats_by_rq[rq_id] = [
            RQCategory(rq_id=rq_id, index=i, name=it.get("name", "").strip(),
                       description=it.get("description", "").strip(),
                       pool_size=len(pools[rq_id]))
            for i, it in enumerate(items, 1) if it.get("name", "").strip()
        ]

    # --- pass 2: classify every pooled review ---
    assign_reqs, chunk_ids = [], {}
    for rq in rqs:
        cats = cats_by_rq.get(rq["id"])
        if not cats:
            continue
        pool = pools[rq["id"]]
        header = (f'Research question: "{rq["question"]}"\n\nCategories:\n'
                  f"{_category_block(cats)}\n\nClassify every review below:\n\n")
        for j in range(0, len(pool), ASSIGN_CHUNK):
            chunk = pool[j:j + ASSIGN_CHUNK]
            cid = f"asg-{rq['id']}-{j // ASSIGN_CHUNK}"
            chunk_ids[cid] = set(chunk)
            assign_reqs.append(llm.build_request(
                custom_id=cid, system=ASSIGN_SYSTEM,
                user_content=header + _render(store, chunk, clean, 400),
                max_tokens=2000, schema=ASSIGN_SCHEMA,
            ))
    batch_id = llm.submit_batch(c, assign_reqs)
    manifest.params["assign_batch_id"] = batch_id
    llm.wait_for_batch(c, batch_id)

    assigned, errored = 0, 0
    for custom_id, parsed in llm.iter_batch_results(c, batch_id):
        if parsed is None:
            errored += 1
            continue
        rq_id = custom_id.split("-")[1]
        assigned += _fold_assignments(cats_by_rq[rq_id], chunk_ids[custom_id], parsed)

    all_cats = [cat for rq in rqs for cat in cats_by_rq.get(rq["id"], [])]
    store.replace_rq_categories(all_cats)
    manifest.counts = {"questions": len(cats_by_rq), "categories": len(all_cats),
                       "pooled_docs": n_docs, "assigned": assigned,
                       "errored_chunks": errored}
    manifest.status = "ok" if errored == 0 else "partial"
    manifest.finished_at = utcnow()
    store.save_manifest(manifest)
    console.print(f"[green]✓ rq-categories[/green] {len(all_cats)} categories across "
                  f"{len(cats_by_rq)} questions · {assigned}/{n_docs} reviews assigned "
                  f"(errored chunks: {errored})")
    return manifest
