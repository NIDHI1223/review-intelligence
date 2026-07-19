"""ResearchAgent — renders the final deliverables from VALIDATED insights only:

  reports/research_report.md   evidence-backed answers to the 7 RQs
  reports/limitations.md       transparent account of gaps, caps, and drops
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from rich.console import Console

from ...core import config
from ...core.models import Insight, RunManifest, utcnow
from ...core.storage import Store, new_run_id

console = Console()

SOURCE_NAMES = {"play_store": "Google Play", "app_store": "App Store",
                "reddit": "Reddit", "youtube": "YouTube", "web": "Web"}


def _fmt_insight(store: Store, ins: Insight, idx: int) -> str:
    conf = ins.confidence
    lines = [
        f"#### {idx}. {ins.claim}",
        "",
        f"- **Confidence {conf.score:.2f}** — volume {conf.volume:.2f} · "
        f"source diversity {conf.source_diversity:.2f} · recency {conf.recency:.2f} · "
        f"consistency {conf.consistency:.2f}",
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
    total = sum(counts.values())

    by_rq: dict[str, list[Insight]] = defaultdict(list)
    general: list[Insight] = []
    for ins in validated:
        if ins.research_questions:
            for rq in ins.research_questions:
                by_rq[rq].append(ins)
        else:
            general.append(ins)

    # ---------------- research report ----------------
    out = [
        "# Review Intelligence Report — Quick Commerce (Zepto · Blinkit · Instamart)",
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

    for rq in config.research_questions():
        insights = sorted(by_rq.get(rq["id"], []), key=lambda i: -i.confidence.score)
        out.append(f"## {rq['id']}: {rq['question']}")
        out.append("")
        if not insights:
            out.append("_No validated insight cleared the evidence bar for this question — "
                       "see limitations report._\n")
            continue
        for idx, ins in enumerate(insights, 1):
            out.append(_fmt_insight(store, ins, idx))

    if general:
        out.append("## Additional validated insights (not mapped to an RQ)")
        out.append("")
        for idx, ins in enumerate(sorted(general, key=lambda i: -i.confidence.score), 1):
            out.append(_fmt_insight(store, ins, idx))

    report_path = config.REPORTS_DIR / "research_report.md"
    report_path.write_text("\n".join(out))

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
