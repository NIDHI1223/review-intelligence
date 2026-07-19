"""YouTubeAgent — comments on quick-commerce review videos via the official
YouTube Data API v3. Key-gated; skips itself cleanly when YOUTUBE_API_KEY is
absent."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

from ...core import config
from ...core.models import RawDocument, author_hash
from .apps import detect_app
from .base import BaseCollector, console


class YouTubeAgent(BaseCollector):
    name = "youtube"

    def available(self) -> tuple[bool, str]:
        if not config.env("YOUTUBE_API_KEY"):
            return False, "YOUTUBE_API_KEY not set in .env"
        return True, ""

    def fetch(self) -> Iterable[RawDocument]:
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError

        cfg = self.params["youtube"]
        yt = build("youtube", "v3", developerKey=config.env("YOUTUBE_API_KEY"))
        seen_videos: set[str] = set()

        for query in cfg["queries"]:
            try:
                search = (
                    yt.search()
                    .list(q=query, part="id,snippet", type="video",
                          maxResults=cfg["videos_per_query"], regionCode="IN",
                          relevanceLanguage="en")
                    .execute()
                )
            except HttpError as e:
                self.manifest.notes.append(f"search '{query}': {e.status_code}")
                continue
            for item in search.get("items", []):
                video_id = item["id"]["videoId"]
                if video_id in seen_videos:
                    continue
                seen_videos.add(video_id)
                video_title = item["snippet"]["title"]
                video_app = detect_app(video_title + " " + item["snippet"].get("description", ""))
                fetched = 0
                page_token = None
                while fetched < cfg["comments_per_video"]:
                    try:
                        resp = (
                            yt.commentThreads()
                            .list(part="snippet", videoId=video_id, maxResults=100,
                                  pageToken=page_token, textFormat="plainText", order="relevance")
                            .execute()
                        )
                    except HttpError as e:
                        # comments disabled on many videos — normal, not an error
                        if e.status_code == 403:
                            break
                        self.manifest.notes.append(f"comments {video_id}: {e.status_code}")
                        break
                    for th in resp.get("items", []):
                        s = th["snippet"]["topLevelComment"]["snippet"]
                        text = (s.get("textDisplay") or "").strip()
                        if len(text) < 10:
                            continue
                        created = None
                        try:
                            created = datetime.fromisoformat(s["publishedAt"].replace("Z", "+00:00"))
                        except (KeyError, ValueError):
                            pass
                        yield RawDocument(
                            id=f"yt_{th['id']}",
                            source="youtube",
                            app=detect_app(text) or video_app,
                            doc_type="comment",
                            text=text,
                            author=author_hash(s.get("authorDisplayName")),
                            created_at=created,
                            url=f"https://www.youtube.com/watch?v={video_id}",
                            metadata={"video_id": video_id, "video_title": video_title,
                                      "likes": s.get("likeCount"), "query": query},
                        )
                        fetched += 1
                    page_token = resp.get("nextPageToken")
                    if not page_token:
                        break
                    self.sleep(cfg["request_delay_seconds"])
                console.print(f"  youtube '{video_title[:50]}': {fetched} comments")
