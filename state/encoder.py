
# state/encoder.py

from dataclasses import dataclass
from math import isfinite
from statistics import mean

from state.builder import WorldState


ENCODER_VERSION = "state-encoder-v1"


GLOBAL_FEATURE_NAMES = (
    "post_volume",
    "trending_fraction",
    "latest_fraction",
    "article_volume",
    "event_volume",
    "mean_hot_score",
    "max_hot_score",
    "hot_score_coverage",
    "mean_hot_velocity",
    "velocity_coverage",
    "post_topic_coverage",
    "event_topic_coverage",
)

POST_FEATURE_NAMES = (
    "hot_score",
    "has_hot_score",
    "is_trending",
    "is_latest",
    "rank_score",
    "has_rank",
    "hot_velocity",
    "has_hot_velocity",
)

EVENT_FEATURE_NAMES = (
    "article_count",
)

TOPIC_FEATURE_NAMES = (
    "post_share",
    "event_share",
)


@dataclass(frozen=True)
class EncodedWorldState:
    version: str
    snapshot_id: int

    global_vector: tuple[float, ...]
    post_vectors: dict[int, tuple[float, ...]]
    event_vectors: dict[str, tuple[float, ...]]
    topic_vectors: dict[str, tuple[float, ...]]


def _number(value):
    """Return a finite number, or None when unavailable."""
    if value is None or isinstance(value, bool):
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    return result if isfinite(result) else None


def _positive_scale(value, scale):
    """
    Map nonnegative values into [0, 1).

    x / (x + scale)
    """
    x = _number(value)

    if x is None:
        return 0.0

    x = max(0.0, x)
    return x / (x + scale)


def _signed_scale(value, scale):
    """
    Preserve direction, mapping numbers into (-1, 1).
    """
    x = _number(value)

    if x is None:
        return 0.0

    return x / (abs(x) + scale)


def _ratio(numerator, denominator):
    if not denominator or denominator <= 0:
        return 0.0

    return max(
        0.0,
        min(1.0, numerator / denominator)
    )


class StateEncoder:

    def __init__(
        self,
        hot_scale=100.0,
        velocity_scale=2.0,
    ):
        if hot_scale <= 0 or velocity_scale <= 0:
            raise ValueError("Scales must be positive")

        self.hot_scale = hot_scale
        self.velocity_scale = velocity_scale

    def encode(self, world: WorldState) -> EncodedWorldState:

        posts = world.posts
        events = world.events
        topics = world.topics

        post_vectors = {}
        hot_values = []
        velocity_values = []

        # 1. Per-post encoding
        for post in posts:

            hot = _number(post.hot_score)

            features = (
                post.features
                if isinstance(post.features, dict)
                else {}
            )

            velocity = _number(
                features.get("hot_velocity")
            )

            rank = _number(post.rank)

            sources = set(post.feed_sources or [])

            if hot is not None:
                hot_values.append(
                    _positive_scale(hot, self.hot_scale)
                )

            if velocity is not None:
                velocity_values.append(
                    _signed_scale(
                        velocity,
                        self.velocity_scale
                    )
                )

            valid_rank = rank is not None and rank > 0

            vector = (
                _positive_scale(hot, self.hot_scale),
                float(hot is not None),
                float("trending" in sources),
                float("latest" in sources),
                1.0 / rank if valid_rank else 0.0,
                float(valid_rank),
                _signed_scale(
                    velocity, self.velocity_scale
                ),
                float(velocity is not None),
            )

            post_vectors[post.post_id] = vector

        # 2. Per-event encoding
        event_vectors = {}

        for event in events:
            event_vectors[event.event_id] = (
                _positive_scale(
                    len(event.article_ids),
                    3.0
                ),
            )

        # 3. Per-topic encoding
        # Only descriptive structural features.
        # Do not use as trusted policy signals yet.
        topic_vectors = {}

        for topic in topics:
            topic_vectors[topic.topic_id] = (
                _ratio(
                    topic.observed_post_count,
                    len(posts)
                ),
                _ratio(
                    topic.observed_event_count,
                    len(events)
                ),
            )

        # 4. Classification coverage
        classification = world.metadata.get(
            "classification", {}
        )

        post_topic_coverage = _ratio(
            classification.get("classified_posts", 0),
            classification.get("eligible_posts", 0),
        )

        event_topic_coverage = _ratio(
            classification.get("classified_events", 0),
            classification.get("eligible_events", 0),
        )

        # 5. Global encoding
        global_state = world.global_state
        post_count = len(posts)

        global_vector = (
            _positive_scale(post_count, 50.0),

            _ratio(
                global_state.trending_post_count,
                post_count
            ),

            _ratio(
                global_state.latest_post_count,
                post_count
            ),

            _positive_scale(
                global_state.article_count, 30.0
            ),

            _positive_scale(
                len(events), 20.0
            ),

            mean(hot_values) if hot_values else 0.0,

            max(hot_values) if hot_values else 0.0,

            _ratio(len(hot_values), post_count),

            (
                mean(velocity_values)
                if velocity_values else 0.0
            ),

            _ratio(len(velocity_values), post_count),

            post_topic_coverage,
            event_topic_coverage,
        )

        return EncodedWorldState(
            version=ENCODER_VERSION,
            snapshot_id=world.metadata["snapshot_id"],
            global_vector=global_vector,
            post_vectors=post_vectors,
            event_vectors=event_vectors,
            topic_vectors=topic_vectors,
        )
