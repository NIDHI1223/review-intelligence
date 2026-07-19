"""PlayStoreAgent — Google Play reviews via google-play-scraper.

Public, unauthenticated endpoints. We resolve/verify each app id through the
library's search() before fetching, paginate with continuation tokens in
polite chunks, and record exactly what was fetched in the manifest.
"""

from __future__ import annotations

from typing import Iterable

from google_play_scraper import Sort, app as app_details, reviews, search

from ...core.models import RawDocument, author_hash
from .base import BaseCollector, console

CHUNK = 200


class PlayStoreAgent(BaseCollector):
    name = "play_store"

    def available(self) -> tuple[bool, str]:
        return True, ""

    def _resolve_app_id(self, key: str, cfg: dict) -> str | None:
        seed = cfg.get("play_store_id")
        country = self.params["country"]
        if seed:
            try:  # verify the seed id actually exists
                details = app_details(seed, lang="en", country=country)
                self.manifest.notes.append(f"{key}: verified seed id {seed} ({details['title']})")
                return seed
            except Exception as e:
                self.manifest.notes.append(f"{key}: seed id {seed} failed verification ({e})")
        query = cfg.get("play_store_query") or cfg["display_name"]
        try:
            results = search(query, lang="en", country=country, n_hits=5)
        except Exception as e:
            self.manifest.notes.append(f"{key}: search failed ({e}); using seed id {seed}")
            return seed
        blocked = ("partner", "rider", "warehouse", "seller")
        for r in results:  # search sometimes returns appId=None for the top hit
            if r.get("appId") and not any(b in r["title"].lower() for b in blocked):
                self.manifest.notes.append(
                    f"{key}: resolved app id via search -> {r['appId']} ({r['title']})"
                )
                return r["appId"]
        return seed

    def fetch(self) -> Iterable[RawDocument]:
        cfg = self.params
        for app_key, app_cfg in cfg["apps"].items():
            app_id = self._resolve_app_id(app_key, app_cfg)
            if not app_id:
                self.manifest.notes.append(f"{app_key}: no Play Store id resolvable, skipped")
                continue
            for lang in cfg["play"]["languages"]:
                target = cfg["play"]["reviews_per_app_per_lang"]
                token = None
                got = 0
                while got < target:
                    n = min(CHUNK, target - got)
                    try:
                        batch, token = reviews(
                            app_id,
                            lang=lang,
                            country=cfg["country"],
                            sort=Sort.NEWEST,
                            count=n,
                            continuation_token=token,
                        )
                    except Exception as e:
                        self.manifest.notes.append(
                            f"{app_key}/{lang}: stopped at {got} ({type(e).__name__}: {e})"
                        )
                        break
                    if not batch:
                        break
                    for r in batch:
                        text = (r.get("content") or "").strip()
                        if not text:
                            continue
                        yield RawDocument(
                            id=f"gp_{r['reviewId']}",
                            source="play_store",
                            app=app_key,
                            doc_type="review",
                            text=text,
                            rating=r.get("score"),
                            author=author_hash(r.get("userName")),
                            created_at=r.get("at"),
                            url=f"https://play.google.com/store/apps/details?id={app_id}",
                            lang_hint=lang,
                            metadata={
                                "thumbs_up": r.get("thumbsUpCount"),
                                "app_version": r.get("appVersion"),
                                "reply": bool(r.get("replyContent")),
                                "play_app_id": app_id,
                            },
                        )
                    got += len(batch)
                    if token is None:
                        break
                    self.sleep(cfg["play"]["request_delay_seconds"])
                console.print(f"  play_store {app_key}/{lang}: {got} reviews")
