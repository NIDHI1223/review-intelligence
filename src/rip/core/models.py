"""Schemas for every artifact that flows between agents.

Agents communicate ONLY through these persisted, validated records. Collection
agents can only produce RawDocument — the schema has no fields for sentiment,
summaries, or interpretation, which structurally enforces the "fetch only,
never infer" contract.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Source = Literal["play_store", "app_store", "reddit", "youtube", "web"]
DocType = Literal["review", "post", "comment"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def author_hash(name: str | None) -> Optional[str]:
    """Store a one-way hash of author names, never the name itself."""
    if not name:
        return None
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]


class RawDocument(BaseModel):
    """One piece of raw user feedback, exactly as fetched. Immutable."""

    id: str  # e.g. gp_<reviewId>, as_<id>, rd_<fullname>, yt_<commentId>
    source: Source
    app: Optional[str] = None  # zepto | blinkit | instamart | None (general discussion)
    doc_type: DocType
    text: str
    title: Optional[str] = None
    rating: Optional[int] = None  # 1-5 where the source has ratings
    author: Optional[str] = None  # sha256 prefix, never raw name
    created_at: Optional[datetime] = None
    fetched_at: datetime = Field(default_factory=utcnow)
    url: Optional[str] = None
    lang_hint: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunManifest(BaseModel):
    """Audit record emitted by every agent run."""

    agent: str
    run_id: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    status: Literal["running", "ok", "partial", "failed", "skipped"] = "running"
    params: dict[str, Any] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ProcessedDocument(BaseModel):
    """Cleaning + dedup verdict for one raw document."""

    id: str
    clean_text: str
    language: str  # ISO 639-1 or "unknown"
    kept: bool
    drop_reason: Optional[str] = None  # too_short | spam | language | exact_dup | near_dup
    dup_of: Optional[str] = None


BEHAVIOR_SIGNALS = [
    "habit", "routine", "reorder", "repeat_purchase", "category_loyalty",
    "convenience", "discovery", "recommendation", "search_behavior",
    "promotion_response", "social_influence", "trust_barrier", "price_concern",
    "quality_concern", "freshness_concern", "awareness_gap", "information_gap",
    "sizing_uncertainty", "authenticity_concern", "review_seeking",
    "experimentation", "early_adopter", "deal_seeker", "unmet_need",
    "feature_request", "assortment_gap", "service_gap",
]

PRODUCT_CATEGORIES = [
    "groceries_staples", "fruits_vegetables", "dairy_bread_eggs", "snacks_beverages",
    "meat_fish", "beauty_personal_care", "household_cleaning", "baby_care",
    "pharma_wellness", "electronics_accessories", "pet_care", "stationery_toys",
    "apparel", "flowers_gifting", "paan_tobacco", "other",
]


class EnrichedDocument(BaseModel):
    """LLM-added structured tags. Text is never altered here."""

    id: str
    sentiment: Literal["positive", "negative", "mixed", "neutral"]
    topics: list[str] = Field(default_factory=list)  # free-form short topic tags
    categories_mentioned: list[str] = Field(default_factory=list)  # from PRODUCT_CATEGORIES
    behavior_signals: list[str] = Field(default_factory=list)  # from BEHAVIOR_SIGNALS
    segment_hints: list[str] = Field(default_factory=list)  # e.g. new_user, parent, bachelor, metro
    is_informative: bool = True  # False for pure "good app" noise
    model: str = ""
    prompt_version: str = ""


class Cluster(BaseModel):
    cluster_id: int
    label: str = ""  # LLM-assigned, human-readable
    size: int
    member_ids: list[str]
    top_terms: list[str] = Field(default_factory=list)


class RQCategory(BaseModel):
    """One data-derived answer category for a research question, holding every
    pooled review the classifier assigned to it. Discovered fresh per question
    by RQCategoryAgent — not a shared taxonomy."""

    rq_id: str  # RQ1..RQ7
    index: int  # 1-based position in the category list shown to the classifier
    name: str
    description: str = ""
    pool_size: int = 0  # size of the RQ pool the members were classified from
    member_ids: list[str] = Field(default_factory=list)


class Quote(BaseModel):
    review_id: str
    quote: str  # must appear VERBATIM in the raw document text
    verified: bool = False


class ConfidenceRubric(BaseModel):
    volume: float
    source_diversity: float
    recency: float
    consistency: float
    score: float


class Insight(BaseModel):
    """The evidence contract. ValidationAgent is the only writer of
    `validated=True`; InsightAgent output always has validated=False."""

    insight_id: str
    claim: str
    research_questions: list[str] = Field(default_factory=list)  # RQ1..RQ7
    supporting_review_ids: list[str]
    support_count: int = 0
    representative_quotes: list[Quote] = Field(default_factory=list)
    contradicting_review_ids: list[str] = Field(default_factory=list)
    contradiction_summary: Optional[str] = None
    segments: list[str] = Field(default_factory=list)
    apps: list[str] = Field(default_factory=list)
    confidence: Optional[ConfidenceRubric] = None
    validated: bool = False
    rejection_reason: Optional[str] = None
    source_cluster: Optional[int] = None
