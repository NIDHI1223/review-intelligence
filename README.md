# Review Intelligence Platform

An AI-powered pipeline that turns thousands of public user reviews of Indian
quick-commerce apps (**Zepto, Blinkit, Swiggy Instamart**) into traceable,
evidence-backed product insights — built for a Product Management case study.

**Core principle: no fabrication.** Every insight must cite real reviews,
every quote is verified verbatim against the raw corpus, counter-evidence is
surfaced rather than hidden, and anything that couldn't be collected or
verified is disclosed in a limitations report.

> 📊 **Pitch deliverables (NL Zepto — Smart Missions):** the live deck and the
> interactive MVP prototype are listed in **[LINKS.md](LINKS.md)**.

## Architecture

```
INGESTION      PlayStoreAgent · AppStoreAgent · RedditAgent · YouTubeAgent
               (fetch raw text + metadata ONLY — schemas forbid interpretation)
      ↓
RAW STORE      immutable SQLite + JSONL, keyed by document id
      ↓
PROCESSING     CleaningAgent → DedupAgent → EnrichmentAgent (Claude, batched)
               → EmbeddingAgent (multilingual) → ClusteringAgent (UMAP+HDBSCAN)
      ↓
INTELLIGENCE   InsightAgent (cited claims) → ValidationAgent (verbatim-quote
               verification, contradiction retrieval, confidence rubric)
               → RQCategoryAgent (per-question categories discovered from the
               data; every pooled review classified into one)
      ↓
OUTPUTS        reports/research_report.md · reports/limitations.md
```

Agents communicate only through persisted, Pydantic-validated artifacts; every
run emits an audit manifest (counts, drops, reasons, model + prompt versions).

## Research questions answered

1. Why do users repeatedly buy from the same categories?
2. What prevents exploration of new categories?
3. How are new products discovered today?
4. What role do habits play?
5. What information is missing before users try a new category?
6. Which user segments are more experimental?
7. What unmet needs consistently appear?

## Compliance stance

Official APIs and public endpoints only: Google Play public review endpoints,
Apple's official RSS feed, Reddit's OAuth API (PRAW), YouTube Data API v3.
X/Twitter, Instagram, Facebook and Quora are excluded because their terms
prohibit collection — recorded transparently in the limitations report, never
worked around.

## Quickstart

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt --python .venv/bin/python
cp .env.example .env   # fill in keys; missing keys degrade gracefully

PYTHONPATH=src .venv/bin/python -m rip.cli status
PYTHONPATH=src .venv/bin/python -m rip.cli collect
PYTHONPATH=src .venv/bin/python -m rip.cli process
PYTHONPATH=src .venv/bin/python -m rip.cli enrich --estimate-only  # cost preview
PYTHONPATH=src .venv/bin/python -m rip.cli enrich
PYTHONPATH=src .venv/bin/python -m rip.cli cluster
PYTHONPATH=src .venv/bin/python -m rip.cli insights
PYTHONPATH=src .venv/bin/python -m rip.cli categorize   # per-question categories, full-pool classification
PYTHONPATH=src .venv/bin/python -m rip.cli report
```

LLM stages use the Anthropic **Message Batches API** (50% pricing) with
`claude-opus-4-8`; a token-counted cost estimate is printed before any batch
is submitted.

## Repository layout

```
config/            settings.yaml, research_questions.yaml
src/rip/core/      schemas, storage, config, LLM helpers
src/rip/agents/    collection/ · processing/ · intelligence/
src/rip/cli.py     pipeline entrypoints
data/              (gitignored) raw + processed corpus, insight store
reports/           generated research + limitations reports
```
