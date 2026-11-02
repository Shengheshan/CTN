import json
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field

import capturethenarrative as ctn

@dataclass
class RawSnapshot:
    started_at: str
    finished_at: str
    trending_posts: list[dict]
    latest_posts: list[dict]
    news_articles: list[dict]
    home_posts: list[dict] = field(default_factory=list)
    tagged_posts: list[dict] = field(default_factory=list)
    notification_posts: list[dict] = field(default_factory=list)


class EnvironmentObserver:

    def __init__(self):
        self.social = ctn.legitsocial.LegitSocial()
        self.news = ctn.newssites.NewsSites()

    def collect(self) -> RawSnapshot:

        started_at = datetime.now(timezone.utc).isoformat()

        trending = list(
            self.social.feed_trending(limit=15)
        )

        latest = list(
            self.social.feed_latest(limit=30)
        )

        articles = list(
            self.news.latest(20)
        )

        finished_at = datetime.now(timezone.utc).isoformat()

        return RawSnapshot(
            started_at=started_at,
            finished_at=finished_at,
            trending_posts=trending,
            latest_posts=latest,
            news_articles=articles
        )


def save_raw_snapshot(snapshot: RawSnapshot) -> Path:

    directory = Path("data/raw")
    directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    filepath = directory / f"snapshot_{timestamp}.json"

    filepath.write_text(
        json.dumps(
            asdict(snapshot),
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    return filepath
