"""RedditAgent — posts + comments via the official Reddit API (PRAW).

Credential-gated: without REDDIT_CLIENT_ID/SECRET the agent skips itself and
the gap is recorded for the limitations report. Author names are hashed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from ...core import config
from ...core.models import RawDocument, author_hash
from .apps import detect_app, mentions_any_app
from .base import BaseCollector, console


class RedditAgent(BaseCollector):
    name = "reddit"

    def available(self) -> tuple[bool, str]:
        if not (config.env("REDDIT_CLIENT_ID") and config.env("REDDIT_CLIENT_SECRET")):
            return False, "REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set in .env"
        return True, ""

    def fetch(self) -> Iterable[RawDocument]:
        import praw  # imported lazily so missing creds never touch the lib

        cfg = self.params["reddit"]
        reddit = praw.Reddit(
            client_id=config.env("REDDIT_CLIENT_ID"),
            client_secret=config.env("REDDIT_CLIENT_SECRET"),
            user_agent=config.env("REDDIT_USER_AGENT") or "macos:review-intelligence:v0.1",
        )
        reddit.read_only = True
        seen_posts: set[str] = set()

        for sub_name in cfg["subreddits"]:
            subreddit = reddit.subreddit(sub_name)
            for query in cfg["queries"]:
                try:
                    submissions = list(
                        subreddit.search(query, sort="relevance", time_filter="all",
                                         limit=cfg["posts_per_query"])
                    )
                except Exception as e:
                    self.manifest.notes.append(f"r/{sub_name} '{query}': {type(e).__name__}: {e}")
                    continue
                for post in submissions:
                    if post.id in seen_posts:
                        continue
                    seen_posts.add(post.id)
                    full_text = f"{post.title}\n{post.selftext or ''}"
                    if not mentions_any_app(full_text) and "commerce" not in query:
                        continue
                    created = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
                    yield RawDocument(
                        id=f"rd_{post.id}",
                        source="reddit",
                        app=detect_app(full_text),
                        doc_type="post",
                        text=full_text.strip(),
                        title=post.title,
                        author=author_hash(str(post.author) if post.author else None),
                        created_at=created,
                        url=f"https://reddit.com{post.permalink}",
                        metadata={"subreddit": sub_name, "score": post.score,
                                  "num_comments": post.num_comments, "query": query},
                    )
                    # top-level comments, best first
                    try:
                        post.comment_sort = "top"
                        post.comments.replace_more(limit=0)
                        comments = post.comments.list()[: cfg["max_comments_per_post"]]
                    except Exception:
                        comments = []
                    for c in comments:
                        body = (getattr(c, "body", "") or "").strip()
                        if len(body) < 15 or body in ("[deleted]", "[removed]"):
                            continue
                        yield RawDocument(
                            id=f"rd_{c.id}",
                            source="reddit",
                            app=detect_app(body) or detect_app(full_text),
                            doc_type="comment",
                            text=body,
                            author=author_hash(str(c.author) if c.author else None),
                            created_at=datetime.fromtimestamp(c.created_utc, tz=timezone.utc),
                            url=f"https://reddit.com{post.permalink}{c.id}",
                            metadata={"subreddit": sub_name, "score": c.score,
                                      "parent_post": post.id},
                        )
                    self.sleep(cfg["request_delay_seconds"])
                console.print(f"  reddit r/{sub_name} '{query}': {len(seen_posts)} posts total so far")
