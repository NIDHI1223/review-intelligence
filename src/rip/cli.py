"""Review Intelligence Platform CLI.

    rip status              credential + data status
    rip collect [--source]  run collection agents (all by default)
    rip process             clean + dedup
    rip enrich              LLM enrichment via Message Batches
    rip cluster             embed + cluster
    rip insights            generate + validate insights
    rip categorize          per-question categories; classify every pooled review
    rip report              write research question + limitations reports
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from .core import config
from .core.storage import Store

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


def _collection_params() -> dict:
    s = config.settings()
    apps = dict(s["apps"])
    youtube = s["collection"]["youtube"]
    # ponytail: RIP_APPS=zepto[,blinkit...] scopes a run to a subset. Filters the app-keyed
    # collectors (play/app store) and narrows YouTube queries to ones naming a kept app.
    only = config.env("RIP_APPS")
    if only:
        keys = {k.strip().lower() for k in only.split(",")}
        apps = {k: v for k, v in apps.items() if k.lower() in keys}
        names = {n.lower() for k, v in apps.items() for n in (k, v["display_name"])}
        youtube = dict(youtube)
        youtube["queries"] = [q for q in youtube["queries"]
                              if any(n in q.lower() for n in names)]
    return {
        "apps": apps,
        "country": s["collection"]["country"],
        "play": s["collection"]["play_store"],
        "app_store": s["collection"]["app_store"],
        "reddit": s["collection"]["reddit"],
        "youtube": youtube,
    }


@app.command()
def status() -> None:
    """Show credentials, raw corpus counts, and stage progress."""
    creds = config.credential_status()
    t = Table(title="Credentials")
    t.add_column("Service"); t.add_column("Status")
    for k, v in creds.items():
        t.add_row(k, "[green]configured[/green]" if v else "[yellow]missing[/yellow]")
    console.print(t)

    store = Store()
    counts = store.raw_counts()
    t2 = Table(title="Raw corpus")
    t2.add_column("source/app"); t2.add_column("docs", justify="right")
    for k in sorted(counts):
        t2.add_row(k, str(counts[k]))
    t2.add_row("[bold]total[/bold]", f"[bold]{sum(counts.values())}[/bold]")
    console.print(t2)

    n_proc = sum(1 for _ in store.iter_processed(kept_only=True))
    n_enr = len(store.enriched_ids())
    n_clu = sum(1 for _ in store.iter_clusters())
    n_ins = sum(1 for _ in store.iter_insights())
    n_val = sum(1 for _ in store.iter_insights(validated_only=True))
    console.print(
        f"processed(kept)={n_proc}  enriched={n_enr}  clusters={n_clu}  "
        f"insights={n_ins} (validated={n_val})"
    )


@app.command()
def collect(
    source: str = typer.Option("all", help="all | play_store | app_store | reddit | youtube"),
) -> None:
    """Run collection agents. Missing credentials skip the agent gracefully."""
    from .agents.collection.app_store import AppStoreAgent
    from .agents.collection.play_store import PlayStoreAgent
    from .agents.collection.reddit import RedditAgent
    from .agents.collection.youtube import YouTubeAgent

    registry = {
        "play_store": PlayStoreAgent,
        "app_store": AppStoreAgent,
        "reddit": RedditAgent,
        "youtube": YouTubeAgent,
    }
    targets = list(registry) if source == "all" else [source]
    store = Store()
    params = _collection_params()
    for name in targets:
        console.rule(f"[bold]{name}[/bold]")
        registry[name](store, params).run()
        store.export_raw_jsonl(name)
    console.print("\n[bold]Corpus now:[/bold]", store.raw_counts())


@app.command()
def process() -> None:
    """Clean, language-detect, and dedup the raw corpus."""
    from .agents.processing.cleaning import run_cleaning
    from .agents.processing.dedup import run_dedup

    store = Store()
    run_cleaning(store)
    run_dedup(store)


@app.command()
def enrich(
    estimate_only: bool = typer.Option(False, help="Print cost estimate and exit"),
) -> None:
    """LLM enrichment of kept documents via the Message Batches API."""
    from .agents.processing.enrichment import run_enrichment

    run_enrichment(Store(), estimate_only=estimate_only)


@app.command()
def cluster() -> None:
    """Embed kept+informative documents and cluster them."""
    from .agents.processing.clustering import run_clustering

    run_clustering(Store())


@app.command()
def insights() -> None:
    """Generate candidate insights per cluster, then validate them."""
    from .agents.intelligence.insight import run_insight_generation
    from .agents.intelligence.validation import run_validation

    store = Store()
    run_insight_generation(store)
    run_validation(store)


@app.command()
def categorize(
    estimate_only: bool = typer.Option(False, help="Print cost estimate and exit"),
) -> None:
    """Discover per-question answer categories and classify every pooled review."""
    from .agents.intelligence.rq_categories import run_rq_categorization

    run_rq_categorization(Store(), estimate_only=estimate_only)


@app.command()
def report() -> None:
    """Write the research-question report and the limitations report."""
    from .agents.intelligence.research import run_research_report

    run_research_report(Store())


if __name__ == "__main__":
    app()
