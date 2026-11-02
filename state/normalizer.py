
from dataclasses import dataclass

@dataclass
class PostRecord:
    post_id: int
    author_id: int | None
    body: str | None
    created_at: str | None

    kind: str | None
    reply_to_post_id: int | None
    root_post_id: int | None

    hot_score: float | None
    rank: int | None

    feed_sources: list[str]


def normalize_posts(raw_snapshot: dict) -> list[PostRecord]:

    posts = {}

    feeds = [
        ("trending", raw_snapshot["trending_posts"]),
        ("latest", raw_snapshot["latest_posts"]),
        ("home", raw_snapshot.get("home_posts", [])),
        ("tagged", raw_snapshot.get("tagged_posts", [])),
        ("notification", raw_snapshot.get("notification_posts", []))
    ]

    for source, feed in feeds:

        for raw in feed:
            if not isinstance(raw, dict) or not isinstance(raw.get("post_id"), int):
                continue

            post_id = raw["post_id"]

            if post_id in posts:
                # 重复帖子：只补充来源
                post = posts[post_id]

                if source not in post.feed_sources:
                    post.feed_sources.append(source)

                continue

            # 创建新的标准化 PostRecord
            posts[post_id] = PostRecord(
                post_id=post_id,
                author_id=raw.get("author_id"),
                body=raw.get("body"),
                created_at=raw.get("created_at"),

                kind=raw.get("kind"),
                reply_to_post_id=raw.get("reply_to_post_id"),
                root_post_id=raw.get("root_post_id"),

                hot_score=raw.get("global_hot_score"),
                rank=raw.get("rank_position"),

                feed_sources=[source]
            )

    return list(posts.values())


@dataclass
class NewsRecord:
    article_id: str
    story_id: str | None
    site_name: str
    title: str
    summary: str | None
    published_at: str | None
    url: str | None


@dataclass
class EventRecord:
    event_id: str
    article_ids: list[str]


def normalize_news(raw_snapshot: dict):

    articles = {}
    events = {}

    for raw in raw_snapshot["news_articles"]:

        site = raw["site"]
        site_id = site["sub_url"]
        slug = raw["slug"]

        # 同一网站上的文章唯一标识
        article_id = f"{site_id}:{slug}"

        # 同一文章不重复存储
        if article_id in articles:
            continue

        story_id = raw.get("story_id")

        # 优先以 story_id 聚合事件
        # 没有 story_id 则暂时视为独立事件
        if story_id:
            event_id = f"story:{story_id}"
        else:
            event_id = f"article:{article_id}"

        articles[article_id] = NewsRecord(
            article_id=article_id,
            story_id=story_id,
            site_name=site["name"],
            title=raw["title"],
            summary=raw.get("summary"),
            published_at=raw.get("published_at"),
            url=raw.get("url")
        )

        if event_id not in events:
            events[event_id] = EventRecord(
                event_id=event_id,
                article_ids=[]
            )

        events[event_id].article_ids.append(article_id)

    return list(articles.values()), list(events.values())

