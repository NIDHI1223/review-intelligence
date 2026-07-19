"""AppStoreAgent — Apple App Store reviews via the official iTunes RSS feed.

Fully official surface (itunes.apple.com/{cc}/rss/customerreviews). Capped by
Apple at roughly 500 most recent reviews per country; the cap is recorded in
the manifest and surfaces in the limitations report.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

import httpx

from ...core.models import RawDocument, author_hash
from .base import BaseCollector, console

SEARCH_URL = "https://itunes.apple.com/search"
RSS_URL = "https://itunes.apple.com/{cc}/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/json"
HEADERS = {"User-Agent": "review-intelligence/0.1 (research; contact via GitHub)"}


class AppStoreAgent(BaseCollector):
    name = "app_store"

    def available(self) -> tuple[bool, str]:
        return True, ""

    def _resolve_app_id(self, client: httpx.Client, key: str, cfg: dict, country: str) -> int | None:
        r = client.get(
            SEARCH_URL,
            params={"term": cfg["app_store_query"], "country": country, "entity": "software", "limit": 5},
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return None
        top = results[0]
        self.manifest.notes.append(
            f"{key}: resolved App Store id -> {top['trackId']} ({top['trackName']})"
        )
        return top["trackId"]

    def fetch(self) -> Iterable[RawDocument]:
        cfg = self.params
        with httpx.Client(headers=HEADERS, timeout=30, follow_redirects=True) as client:
            for country in cfg["app_store"]["countries"]:
                for app_key, app_cfg in cfg["apps"].items():
                    try:
                        app_id = self._resolve_app_id(client, app_key, app_cfg, country)
                    except Exception as e:
                        self.manifest.notes.append(f"{app_key}/{country}: id resolution failed ({e})")
                        continue
                    if not app_id:
                        self.manifest.notes.append(f"{app_key}/{country}: not found in App Store")
                        continue
                    got = 0
                    for page in range(1, cfg["app_store"]["max_rss_pages"] + 1):
                        try:
                            r = client.get(RSS_URL.format(cc=country, page=page, app_id=app_id))
                            if r.status_code != 200:
                                break
                            entries = r.json().get("feed", {}).get("entry", [])
                        except Exception as e:
                            self.manifest.notes.append(
                                f"{app_key}/{country} page {page}: {type(e).__name__}: {e}"
                            )
                            break
                        # first entry on page 1 is app metadata, not a review
                        reviews = [e for e in entries if "im:rating" in e]
                        if not reviews:
                            break
                        for e in reviews:
                            text = (e.get("content", {}).get("label") or "").strip()
                            if not text:
                                continue
                            created = None
                            if "updated" in e:
                                try:
                                    created = datetime.fromisoformat(e["updated"]["label"])
                                except ValueError:
                                    pass
                            yield RawDocument(
                                id=f"as_{e['id']['label']}",
                                source="app_store",
                                app=app_key,
                                doc_type="review",
                                text=text,
                                title=e.get("title", {}).get("label"),
                                rating=int(e["im:rating"]["label"]),
                                author=author_hash(e.get("author", {}).get("name", {}).get("label")),
                                created_at=created,
                                url=f"https://apps.apple.com/{country}/app/id{app_id}",
                                metadata={
                                    "app_version": e.get("im:version", {}).get("label"),
                                    "vote_sum": e.get("im:voteSum", {}).get("label"),
                                    "itunes_app_id": app_id,
                                    "country": country,
                                },
                            )
                            got += 1
                        self.sleep(cfg["app_store"]["request_delay_seconds"])
                    console.print(f"  app_store {app_key}/{country}: {got} reviews")
