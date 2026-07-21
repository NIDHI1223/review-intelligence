"""ResearchAgent — renders the final deliverables:

  reports/research_report.md   per-question data-derived categories with the
                               reviews that belong to each, plus a deduped
                               appendix of validated insight claims
  reports/rq_categories.json   full category membership (every review id)
  reports/limitations.md       transparent account of gaps, caps, and drops
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime

from rich.console import Console

from ...core import config
from ...core.models import Insight, RawDocument, RQCategory, RunManifest, utcnow
from ...core.storage import Store, new_run_id

console = Console()

SOURCE_NAMES = {"play_store": "Google Play", "app_store": "App Store",
                "reddit": "Reddit", "youtube": "YouTube", "web": "Web"}


def _source_tag(raw: RawDocument | None) -> str:
    if not raw:
        return "?"
    src = SOURCE_NAMES.get(raw.source, raw.source)
    app = f", {raw.app}" if raw.app else ""
    rating = f", {raw.rating}★" if raw.rating else ""
    return f"{src}{app}{rating}"


def _pick_examples(store: Store, ids: list[str], k: int = 3) -> list[RawDocument]:
    """Short, verbatim-quotable members — one per app where possible."""
    raws = [r for r in (store.get_raw(i) for i in ids) if r and len(r.text.strip()) >= 40]
    raws.sort(key=lambda r: len(r.text))
    picked: list[RawDocument] = []
    seen_apps: set = set()
    for r in raws:
        if r.app not in seen_apps:
            picked.append(r)
            seen_apps.add(r.app)
        if len(picked) == k:
            return picked
    for r in raws:
        if all(p.id != r.id for p in picked):
            picked.append(r)
            if len(picked) == k:
                break
    return picked


def _fmt_category(store: Store, cat: RQCategory, idx: int) -> str:
    n = len(cat.member_ids)
    pct = f" · {100 * n / cat.pool_size:.0f}% of pool" if cat.pool_size else ""
    lines = [
        f"#### {idx}. {cat.name} — {n} reviews{pct}",
        "",
        f"_{cat.description}_",
    ]
    apps = Counter((store.get_raw(m).app or "general") if store.get_raw(m) else "?"
                   for m in cat.member_ids)
    if apps:
        lines.append("- **Apps:** " + " · ".join(f"{a}: {c}" for a, c in apps.most_common()))
    for raw in _pick_examples(store, cat.member_ids):
        text = " ".join(raw.text.split())
        text = text[:220] + ("…" if len(text) > 220 else "")
        lines.append(f'  > "{text}" — `{raw.id}` ({_source_tag(raw)})')
    preview = ", ".join(cat.member_ids[:12])
    more = n - 12
    lines.append(f"- <sub>Members: {preview}{f' … +{more} more' if more > 0 else ''} "
                 f"(full list in rq_categories.json)</sub>")
    lines.append("")
    return "\n".join(lines)


def _fmt_insight(store: Store, ins: Insight, idx: int) -> str:
    conf = ins.confidence
    lines = [
        f"#### {idx}. {ins.claim}",
        "",
        f"- **Confidence {conf.score:.2f}** — volume {conf.volume:.2f} · "
        f"source diversity {conf.source_diversity:.2f} · recency {conf.recency:.2f} · "
        f"consistency {conf.consistency:.2f}"
        + (f" · informs {', '.join(ins.research_questions)}" if ins.research_questions else ""),
        f"- **Support:** {ins.support_count} documents"
        + (f" · apps: {', '.join(ins.apps)}" if ins.apps else "")
        + (f" · segments: {', '.join(ins.segments)}" if ins.segments else ""),
    ]
    for q in ins.representative_quotes[:3]:
        raw = store.get_raw(q.review_id)
        src = SOURCE_NAMES.get(raw.source, raw.source) if raw else "?"
        app = raw.app if raw and raw.app else ""
        rating = f", {raw.rating}★" if raw and raw.rating else ""
        lines.append(f'  > "{q.quote}" — `{q.review_id}` ({src}{", " + app if app else ""}{rating})')
    if ins.contradicting_review_ids:
        lines.append(f"- ⚖️ **Counter-evidence:** {ins.contradiction_summary} "
                     f"(`{'`, `'.join(ins.contradicting_review_ids[:5])}`)")
    supp_preview = ", ".join(ins.supporting_review_ids[:12])
    more = len(ins.supporting_review_ids) - 12
    lines.append(f"- <sub>Evidence trail: {supp_preview}{f' … +{more} more' if more > 0 else ''}</sub>")
    lines.append("")
    return "\n".join(lines)


def run_research_report(store: Store) -> RunManifest:
    manifest = RunManifest(agent="research", run_id=new_run_id("research"), started_at=utcnow())
    config.ensure_dirs()
    validated = list(store.iter_insights(validated_only=True))
    rejected = [i for i in store.iter_insights() if not i.validated]
    counts = store.raw_counts()

    # RIP_APPS scopes the report to those apps' documents and insights,
    # matching the scope the categorize stage pooled with
    only = config.env("RIP_APPS")
    scope = {a.strip().lower() for a in only.split(",")} if only else None
    if scope:
        counts = {k: v for k, v in counts.items()
                  if k.split("/", 1)[1].lower() in scope}
        validated = [i for i in validated
                     if scope & {a.lower() for a in i.apps}]
    total = sum(counts.values())

    by_rq: dict[str, list[Insight]] = defaultdict(list)  # fallback path only
    for ins in validated:
        for rq in ins.research_questions:
            by_rq[rq].append(ins)

    # ---------------- research report ----------------
    apps_label = " · ".join(a.title() for a in sorted(scope)) if scope \
        else "Zepto · Blinkit · Instamart"
    out = [
        f"# Review Intelligence Report — Quick Commerce ({apps_label})",
        "",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M} · corpus of {total:,} public documents · "
        f"every insight below survived verbatim-citation validation; counter-evidence shown "
        f"where it exists._",
        "",
        "## Corpus",
        "",
        "| Source / app | Documents |",
        "|---|---:|",
    ]
    for k in sorted(counts):
        out.append(f"| {k} | {counts[k]:,} |")
    out.append(f"| **Total** | **{total:,}** |")
    out.append("")

    cats_by_rq: dict[str, list[RQCategory]] = defaultdict(list)
    for cat in store.iter_rq_categories():
        cats_by_rq[cat.rq_id].append(cat)

    for rq in config.research_questions():
        out.append(f"## {rq['id']}: {rq['question']}")
        out.append("")
        cats = sorted((c for c in cats_by_rq.get(rq["id"], []) if c.member_ids),
                      key=lambda c: -len(c.member_ids))
        if cats:
            assigned = sum(len(c.member_ids) for c in cats)
            out.append(f"_{assigned:,} of {cats[0].pool_size:,} pooled reviews classified "
                       f"into {len(cats)} categories derived from this question's data; "
                       f"the remainder fit none of them._")
            out.append("")
            for idx, cat in enumerate(cats, 1):
                out.append(_fmt_category(store, cat, idx))
            continue
        # fallback for corpora where `rip categorize` hasn't run yet
        insights = sorted(by_rq.get(rq["id"], []), key=lambda i: -i.confidence.score)
        if not insights:
            out.append("_No categories yet (run `rip categorize`) and no validated insight "
                       "cleared the evidence bar for this question — see limitations report._\n")
            continue
        for idx, ins in enumerate(insights, 1):
            out.append(_fmt_insight(store, ins, idx))

    if validated:
        out.append("## Appendix — validated evidence-backed claims")
        out.append("")
        out.append("_Each claim listed once, with the research questions it informs._")
        out.append("")
        ranked = sorted(validated, key=lambda i: -i.confidence.score)
        for idx, ins in enumerate(ranked, 1):
            out.append(_fmt_insight(store, ins, idx))

    report_path = config.REPORTS_DIR / "research_report.md"
    report_path.write_text("\n".join(out))

    # full category membership for traceability — every review id, no preview cap
    questions = {rq["id"]: rq["question"] for rq in config.research_questions()}
    cats_dump = {
        rq_id: {
            "question": questions.get(rq_id, ""),
            "pool_size": cats[0].pool_size if cats else 0,
            "categories": [
                {"name": c.name, "description": c.description,
                 "count": len(c.member_ids), "member_ids": c.member_ids}
                for c in sorted(cats, key=lambda c: -len(c.member_ids))
            ],
        }
        for rq_id, cats in cats_by_rq.items()
    }
    if cats_dump:
        (config.REPORTS_DIR / "rq_categories.json").write_text(
            json.dumps(cats_dump, indent=1))

    # ---------------- limitations report ----------------
    lim = [
        "# Limitations & Methodology Notes",
        "",
        "This platform never fabricates data. Everything it could not access or verify is "
        "recorded here.",
        "",
        "## Sources not collected",
        "",
    ]
    skipped = [m for m in store.iter_manifests() if m.status == "skipped" and
               m.agent in ("reddit", "youtube", "web")]
    seen_agents = set()
    for m in sorted(skipped, key=lambda m: m.started_at, reverse=True):
        if m.agent in seen_agents:
            continue
        seen_agents.add(m.agent)
        lim.append(f"- **{m.agent}** — {'; '.join(m.notes) or 'skipped'}")
    lim += [
        "- **X/Twitter** — excluded by design: API is paid-only and scraping violates ToS.",
        "- **Instagram/Facebook/Quora** — excluded by design: platform ToS prohibit collection.",
        "",
        "## Structural caps",
        "",
        "- Apple App Store RSS feed caps at ~500 most recent reviews per app per country.",
        "- Google Play reviews fetched newest-first up to the configured per-app/language cap; "
        "the corpus skews recent by construction.",
        "- App-store reviews over-represent complaint/praise moments vs. everyday usage.",
        "",
        "## Processing drops (full audit in manifests table)",
        "",
    ]
    for m in store.iter_manifests():
        if m.agent in ("cleaning", "dedup", "clustering", "enrichment", "validation"):
            lim.append(f"- `{m.agent}` ({m.run_id}): {m.counts}")
    lim += [
        "",
        "## Insight rejections",
        "",
        f"{len(rejected)} candidate insights were rejected by validation "
        f"(missing citations, unverifiable quotes, or support below threshold) and are "
        "retained in the insight store with their rejection reasons for audit.",
        "",
        "## Interpretation caveats",
        "",
        "- Public reviews are self-selected feedback, not a representative user sample.",
        "- Sentiment and behavioral tags are model-generated (tag audit trail: model + "
        "prompt version stamped on every enriched record).",
        "- Per-question categories and their review membership are model-generated "
        "single-label classifications over each question's signal pool; reviews the "
        "classifier judged to fit no category are counted but not listed.",
        "- Segment hints derive only from what reviewers explicitly stated.",
    ]
    lim_path = config.REPORTS_DIR / "limitations.md"
    lim_path.write_text("\n".join(lim))

    manifest.counts = {"validated_insights": len(validated), "rejected_insights": len(rejected)}
    manifest.status = "ok"
    manifest.finished_at = utcnow()
    store.save_manifest(manifest)
    console.print(f"[green]✓ report[/green] → {report_path}")
    console.print(f"[green]✓ limitations[/green] → {lim_path}")
    return manifest
