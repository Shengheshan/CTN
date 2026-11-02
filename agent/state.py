"""Agent-owned state, separate from external WorldState."""

from dataclasses import dataclass, field


@dataclass
class AgentState:
    recent_actions: list[dict] = field(default_factory=list)
    completed_keys: set[str] = field(default_factory=set)
    content_hashes: set[str] = field(default_factory=set)
    liked_post_ids: set[int] = field(default_factory=set)
    reposted_post_ids: set[int] = field(default_factory=set)
    followed_account_ids: set[int] = field(default_factory=set)
    mentioned_account_ids: set[int] = field(default_factory=set)
    mentioned_post_ids: set[int] = field(default_factory=set)
    own_published_posts: list[dict] = field(default_factory=list)
    published_news_sources: list[str] = field(default_factory=list)
    action_count_this_round: int = 0


@dataclass
class PlatformState:
    feed_locked: bool = False
    account_id: int | None = None
    liked_post_ids: set[int] = field(default_factory=set)
    reposted_post_ids: set[int] = field(default_factory=set)
    followed_account_ids: set[int] = field(default_factory=set)
    api_remaining: int | None = None
    articles_by_id: dict[str, dict] = field(default_factory=dict)
    max_actions_per_round: int = 1
    alignment_by_post: dict[int, float] = field(default_factory=dict)
    relevance_by_target: dict[str, float] = field(default_factory=dict)
    popular_tags: list[dict] = field(default_factory=list)
    reply_to_post_ids: dict[int, int | None] = field(default_factory=dict)
    root_post_ids: dict[int, int | None] = field(default_factory=dict)
    priority_reply_post_ids: set[int] = field(default_factory=set)
    notification_ids_by_post: dict[int, int] = field(default_factory=dict)
