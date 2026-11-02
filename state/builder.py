
# state/builder.py

from dataclasses import dataclass

from state.store import StateStore
from state.features import extract_post_features
from state.topic_classification.registry import TopicRegistry
from state.topic_classification.classifier import PROMPT_VERSION


@dataclass
class PostState:
    post_id: int
    author_id: int | None
    body: str | None
    hot_score: float | None
    rank: int | None
    feed_sources: list[str]
    features: dict
    topic_ids: list[str]
    created_at: str | None = None


@dataclass
class EventState:
    event_id: str
    article_ids: list[str]
    topic_ids: list[str]


@dataclass
class TopicState:
    topic_id: str
    post_ids: list[int]
    event_ids: list[str]
    observed_post_count: int
    observed_event_count: int


@dataclass
class GlobalState:
    observed_post_count: int
    trending_post_count: int
    latest_post_count: int
    article_count: int
    event_count: int
    active_topic_count: int | None


@dataclass
class WorldState:
    metadata: dict
    global_state: GlobalState
    posts: list[PostState]
    events: list[EventState]
    topics: list[TopicState]


class StateBuilder:

    def __init__(self, store: StateStore):
        self.store = store
        self.registry = TopicRegistry(store.db_path)

    def build(self, snapshot_id: int) -> WorldState:

        current = self.store.load_snapshot_record(snapshot_id)

        history = self.store.load_previous_post_observations(
            snapshot_id
        )

        assignments = self.registry.load_snapshot_assignments(
            snapshot_id,
            PROMPT_VERSION
        )

        # 1. PostState
        post_states = []

        for post in current["posts"]:
            previous = history.get(post["post_id"])

            features = extract_post_features(
                current=post,
                previous=previous["post"] if previous else None,
                current_time=current["finished_at"],
                previous_time=(
                    previous["finished_at"] if previous else None
                )
            )

            post_states.append(PostState(
                post_id=post["post_id"],
                author_id=post["author_id"],
                body=post["body"],
                hot_score=post["hot_score"],
                rank=post["rank"],
                feed_sources=post["feed_sources"],
                features=features,
                topic_ids=assignments.get(
                    f"post:{post['post_id']}", []
                ),
                created_at=post.get("created_at")
            ))

        # 2. EventState
        event_states = []

        for event in current["events"]:
            event_states.append(EventState(
                event_id=event["event_id"],
                article_ids=event["article_ids"],
                topic_ids=assignments.get(
                    f"event:{event['event_id']}", []
                )
            ))

        # 3. TopicState 聚合
        groups = {}

        for post in post_states:
            for topic_id in post.topic_ids:
                group = groups.setdefault(
                    topic_id, {"posts": set(), "events": set()}
                )
                group["posts"].add(post.post_id)

        for event in event_states:
            for topic_id in event.topic_ids:
                group = groups.setdefault(
                    topic_id, {"posts": set(), "events": set()}
                )
                group["events"].add(event.event_id)

        topic_states = []

        for topic_id, group in sorted(groups.items()):
            post_ids = sorted(group["posts"])
            event_ids = sorted(group["events"])

            topic_states.append(TopicState(
                topic_id=topic_id,
                post_ids=post_ids,
                event_ids=event_ids,
                observed_post_count=len(post_ids),
                observed_event_count=len(event_ids)
            ))

        # 4. 分类覆盖率
        eligible_posts = [
            p for p in post_states if p.body and p.body.strip()
        ]

        classified_posts = sum(
            f"post:{p.post_id}" in assignments
            for p in eligible_posts
        )

        classified_events = sum(
            f"event:{e.event_id}" in assignments
            for e in event_states
        )

        total = len(eligible_posts) + len(event_states)
        completed = classified_posts + classified_events

        if completed == 0:
            topic_status = "not_classified"
        elif completed == total:
            topic_status = "complete"
        else:
            topic_status = "partial"

        # 5. GlobalState
        global_state = GlobalState(
            observed_post_count=len(post_states),
            trending_post_count=sum(
                "trending" in p.feed_sources
                for p in post_states
            ),
            latest_post_count=sum(
                "latest" in p.feed_sources
                for p in post_states
            ),
            article_count=len(current["articles"]),
            event_count=len(event_states),
            active_topic_count=(
                len(topic_states) if completed else None
            )
        )

        # 6. WorldState
        return WorldState(
            metadata={
                "snapshot_id": snapshot_id,
                "started_at": current["started_at"],
                "finished_at": current["finished_at"],
                "topic_status": topic_status,
                "topic_classifier_version": PROMPT_VERSION,
                "classification": {
                    "eligible_posts": len(eligible_posts),
                    "classified_posts": classified_posts,
                    "eligible_events": len(event_states),
                    "classified_events": classified_events
                }
            },
            global_state=global_state,
            posts=post_states,
            events=event_states,
            topics=topic_states
        )
