"""Generate concrete action candidates from observed state and agent history."""

import logging
import math
import re
from agent.state import AgentState, PlatformState
from datetime import datetime, timezone
from urllib.parse import urljoin, urlsplit, urlunsplit
from policy.models import CandidateAction
from policy.relevance import (campaign_post_relevant, campaign_relevant,
                              timely)
from policy.settings import PolicySettings
from state.builder import WorldState
from state.encoder import EncodedWorldState


_LOGGER = logging.getLogger(__name__)


_HOT_NEWS_STOPWORDS = {
    "about", "after", "again", "amid", "among", "been", "before", "being",
    "bill", "could", "from", "have", "into", "more", "most", "news", "over",
    "says", "said", "that", "their", "there", "these", "they", "this", "those",
    "through", "under", "with", "without", "would", "your", "kingstonia",
    "referendum", "h2d2", "vaccine", "vaccines", "vaccination", "mandate",
    "vote", "voting", "yes", "no", "public", "health", "just", "what",
    "which", "does", "every", "people", "once", "campaign", "community",
    "citizens", "healthcare", "hospitals", "hospital", "outbreak", "policy",
    "report", "reported", "reports", "latest", "today", "safety",
}


def _news_terms(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z]{4,}", text.casefold())
            if token not in _HOT_NEWS_STOPWORDS}


def _hot_score(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = float(value)
    return max(0.0, score) if math.isfinite(score) else None


_URL_RE = re.compile(
    r"(?:https?://[^\s<>\])}]+|/[a-z0-9_-]+/articles/[^\s<>\])}]+)",
    re.IGNORECASE)


def _canonical_news_url(value: str | None) -> str | None:
    """Treat SDK relative news paths and published absolute links as one URL."""
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip().rstrip(".,;:!?)]}")
    parsed = urlsplit(urljoin("https://sites.legitreal.com/", raw))
    if (parsed.scheme.lower() not in ("http", "https") or not parsed.netloc
            or "/articles/" not in parsed.path.casefold()):
        return None
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(),
                       parsed.path.rstrip("/").casefold(), "", ""))


def _source_url(article: dict) -> str | None:
    return _canonical_news_url(article.get("url"))


def _already_published_article(article: dict, published_sources: list[str]) -> bool:
    """Use the cited article URL, not broad topic overlap, to suppress repeats."""
    target = _source_url(article)
    if target is None:
        return False
    return any(
        target == _canonical_news_url(observed)
        for previous in published_sources if isinstance(previous, str)
        for observed in _URL_RE.findall(previous)
    )


def _news_match(post_terms: set[str], article: dict,
                linked_news_urls: set[str]):
    """Prefer exact cited news; otherwise require two concrete shared terms."""
    if not isinstance(article, dict) or article.get("source_type") == "trending_post":
        return None
    if not all(isinstance(article.get(key), str) and article[key].strip()
               for key in ("title", "site_name")) or _source_url(article) is None:
        return None
    article_text = f"{article['title']} {article.get('summary') or ''}"
    if not campaign_post_relevant(article_text):
        return None
    title_terms = _news_terms(article["title"])
    article_terms = title_terms | _news_terms(str(article.get("summary") or ""))
    shared = post_terms & article_terms
    title_shared = post_terms & title_terms
    if linked_news_urls:
        if _source_url(article) not in linked_news_urls:
            return None
    elif len(shared) < 2 or not title_shared:
        return None
    return shared, title_shared


class CandidateGenerator:
    def __init__(self, settings: PolicySettings | None = None, state_store=None):
        self.settings = settings or PolicySettings()
        self.state_store = state_store
        self._ever_trending_posts: dict[int, dict] = {}
        self._last_history_snapshot_id = 0

    def _trending_history(self, world: WorldState) -> dict[int, dict]:
        snapshot_id = int(world.metadata["snapshot_id"])
        if snapshot_id < self._last_history_snapshot_id:
            self._ever_trending_posts.clear()
            self._last_history_snapshot_id = 0
        if self.state_store is not None and snapshot_id > self._last_history_snapshot_id:
            try:
                self._ever_trending_posts.update(
                    self.state_store.load_historically_trending_posts(
                        snapshot_id, self._last_history_snapshot_id))
            except Exception as exc:
                _LOGGER.warning("Historical trend lookup failed: %s", type(exc).__name__)
            else:
                self._last_history_snapshot_id = snapshot_id
        for post in world.posts:
            if "trending" in (post.feed_sources or []):
                self._ever_trending_posts[post.post_id] = {
                    **vars(post), "observed_snapshot_id": snapshot_id}
        return dict(self._ever_trending_posts)

    def generate(self, world: WorldState, encoded: EncodedWorldState,
                 agent: AgentState, platform: PlatformState) -> tuple[CandidateAction, ...]:
        if world.metadata["snapshot_id"] != encoded.snapshot_id:
            raise ValueError("snapshot mismatch")
        wait = CandidateAction("WAIT", rationale="No eligible action or action budget")
        if (platform.feed_locked or platform.api_remaining == 0 or
                agent.action_count_this_round >= platform.max_actions_per_round):
            return (wait,)

        candidates: list[CandidateAction] = []
        observed_at = world.metadata.get("finished_at")
        own_post_ids = {item.get("post_id") for item in agent.own_published_posts
                        if isinstance(item.get("post_id"), int)}
        for event in world.events:
            # Keep a valid post opportunity in the policy pool during its
            # cooldown. The scorer lowers its probability instead of making
            # CREATE_POST impossible for Softmax to explore.
            event_cooldown_active = self._event_cooling_down(
                event.event_id, agent, observed_at)
            articles = [platform.articles_by_id[aid] for aid in event.article_ids
                        if aid in platform.articles_by_id
                        and platform.articles_by_id[aid].get("title")
                        and campaign_post_relevant(" ".join(
                            str(platform.articles_by_id[aid].get(field) or "")
                            for field in ("title", "summary")))
                        and timely(platform.articles_by_id[aid].get("published_at"),
                                   observed_at, 72)]
            if not articles:
                continue
            candidate = CandidateAction(
                "CREATE_POST", source_event_id=event.event_id,
                context={"articles": articles[:3], "event_id": event.event_id,
                         "post_generation_snapshot_id": world.metadata["snapshot_id"],
                         "post_intent": "normal_post",
                         "event_cooldown_active": event_cooldown_active,
                         "content_features": {"source_article_count": len(articles[:3])},
                         "target_features": {"event_id": event.event_id,
                                             "topic_ids": list(event.topic_ids)},
                         "timing_features": {"observed_at": observed_at,
                                             "source_published_at": [a.get("published_at") for a in articles[:3]]},
                         "environment_features": {"snapshot_id": world.metadata.get("snapshot_id"),
                                                   "source": "news_articles"}},
                rationale="Report a sourced news event", estimated_cost=self.settings.costs["CREATE_POST"],
                required_parameters=("body",),
            )
            if candidate.key not in agent.completed_keys:
                candidates.append(candidate)

        for post in world.posts:
            if not post.body or not post.body.strip() or post.post_id not in encoded.post_vectors:
                continue
            reply_to = platform.reply_to_post_ids.get(post.post_id)
            root_post = platform.root_post_ids.get(post.post_id)
            is_reply_to_own_post = (reply_to in own_post_ids or root_post in own_post_ids
                                    or post.post_id in platform.priority_reply_post_ids)
            if not campaign_relevant(post.body) and not is_reply_to_own_post:
                continue
            if not timely(post.created_at, observed_at, 36):
                continue
            if platform.account_id is not None and post.author_id == platform.account_id:
                continue
            context = {"post_body": post.body, "author_id": post.author_id,
                       "source_post_id": post.post_id,
                       "feed_sources": list(post.feed_sources),
                       "created_at": post.created_at,
                       "reply_to_own_post": is_reply_to_own_post,
                       "notification_id": platform.notification_ids_by_post.get(post.post_id),
                       "own_post_id_replied_to": (reply_to if reply_to in own_post_ids else
                                                  root_post if root_post in own_post_ids else None),
                       "content_features": {"target_text": post.body},
                       "target_features": {
                           "parent_post_id": post.post_id,
                           "topic_ids": list(post.topic_ids),
                           "pre_action_interaction_baseline": {
                               "hot_score": post.hot_score,
                               "rank": post.rank,
                               "hot_velocity": post.features.get("hot_velocity"),
                           },
                       },
                       "timing_features": {"observed_at": observed_at,
                                           "parent_created_at": post.created_at,
                                           "parent_age_seconds": post.features.get("post_age_seconds")},
                       "supporting_articles": [article for article in
                           platform.articles_by_id.values()
                           if campaign_relevant(" ".join(str(article.get(key) or "")
                               for key in ("title", "summary")))
                           and timely(article.get("published_at"), observed_at, 72)][:2]}
            reply = CandidateAction(
                "REPLY", target_id=post.post_id, context=context,
                rationale=("Respond to a direct reply to our post" if is_reply_to_own_post
                           else "Respond to an observed post"), estimated_cost=self.settings.costs["REPLY"],
                required_parameters=("body",),
            )
            if reply.key not in agent.completed_keys:
                candidates.append(reply)

            # Endorsement and relationship actions need an explicit positive
            # alignment signal; heat alone is insufficient.
            if platform.alignment_by_post.get(post.post_id, 0) <= 0:
                continue
            if post.post_id not in agent.liked_post_ids | platform.liked_post_ids:
                candidate = CandidateAction(
                    "LIKE", target_id=post.post_id, context=context,
                    rationale="Positive alignment confirmed", estimated_cost=self.settings.costs["LIKE"])
                if candidate.key not in agent.completed_keys:
                    candidates.append(candidate)
            if post.post_id not in agent.reposted_post_ids | platform.reposted_post_ids:
                candidate = CandidateAction(
                    "REPOST", target_id=post.post_id, context=context,
                    rationale="Positive alignment confirmed", estimated_cost=self.settings.costs["REPOST"])
                if candidate.key not in agent.completed_keys:
                    candidates.append(candidate)
            if (post.author_id and post.author_id not in
                    agent.followed_account_ids | platform.followed_account_ids):
                candidate = CandidateAction(
                    "FOLLOW", target_id=post.author_id, context=context,
                    rationale="Positive alignment confirmed", estimated_cost=self.settings.costs["FOLLOW"])
                if candidate.key not in agent.completed_keys:
                    candidates.append(candidate)
            if (post.author_id and post.author_id not in agent.mentioned_account_ids
                    and post.author_id != platform.account_id):
                own_post = next((item for item in agent.own_published_posts
                                 if item["post_id"] not in agent.mentioned_post_ids
                                 and campaign_relevant(item["source_text"])), None)
                if own_post is not None:
                    candidate = CandidateAction(
                        "MENTION", target_id=post.author_id,
                        context={**context, "post_id": post.post_id,
                                 "mention_on_post_id": own_post["post_id"]},
                        rationale="Mention an explicitly aligned author on a relevant own post",
                        estimated_cost=self.settings.costs["MENTION"])
                    if candidate.key not in agent.completed_keys:
                        candidates.append(candidate)
        candidates = self._append_hot_trend_candidates(
            candidates, world, agent, platform, self._trending_history(world))
        candidates.append(wait)
        # Several posts from one author may propose the same FOLLOW action.
        return tuple({candidate.key: candidate for candidate in candidates}.values())

    def _append_hot_trend_candidates(self, candidates: list[CandidateAction],
                                     world: WorldState, agent: AgentState,
                                     platform: PlatformState,
                                     trending_sources: dict[int, dict]) -> list[CandidateAction]:
        """Use sufficiently hot Trending history to find citable current news."""
        result = list(candidates)
        current_posts = {post.post_id: post for post in world.posts}
        for post_id, post in sorted(trending_sources.items()):
            body = post.get("body")
            author_id = post.get("author_id")
            if (not isinstance(body, str) or not body.strip()
                    or not campaign_relevant(body)
                    or (author_id is not None and author_id == platform.account_id)):
                continue
            prior_key = f"CREATE_POST:hot-trending:{post_id}"
            if any(action.get("action_type") == "CREATE_POST"
                   and action.get("status") == "SUCCEEDED"
                   and str(action.get("action_key") or "").startswith(prior_key)
                   for action in agent.recent_actions):
                continue

            linked_news_urls = {
                canonical for observed_url in _URL_RE.findall(body)
                if (canonical := _canonical_news_url(observed_url)) is not None}
            historical_hot = _hot_score(post.get("hot_score"))
            current = current_posts.get(post_id)
            current_hot = _hot_score(current.hot_score) if current is not None else None
            observed_heat = max(historical_hot or 0.0, current_hot or 0.0)
            if ((historical_hot is None and current_hot is None)
                    or observed_heat < self.settings.hot_news_score_threshold):
                continue
            current_trending = bool(current is not None
                                    and "trending" in (current.feed_sources or []))
            if current_trending:
                cooling_fraction = 0.0
            elif current_hot is None:
                cooling_fraction = 1.0
            else:
                reference = max(historical_hot or current_hot, 1.0)
                decline = max(0.0, (reference - current_hot) / reference)
                cooling_fraction = max(0.6, min(1.0, decline))
            heat_strength = (min(1.0, current_hot /
                             max(1.0, 1.5 * self.settings.hot_news_score_threshold))
                             if current_hot is not None else 0.0)

            post_terms = _news_terms(_URL_RE.sub(" ", body))
            matches = []
            for article_id, article in platform.articles_by_id.items():
                match = _news_match(post_terms, article, linked_news_urls)
                if match is None or _already_published_article(
                        article, agent.published_news_sources):
                    continue
                shared, title_shared = match
                matches.append((article_id, article, shared, title_shared))
            if not matches:
                continue
            article_id, news, shared, title_shared = max(
                matches, key=lambda item: (
                    len(item[2]), len(item[3]),
                    str(item[1].get("published_at") or ""), str(item[0])))

            hot_source_post = {
                "site_name": "Legit Social",
                "title": f"Trending post {post_id}",
                "summary": body,
                "published_at": post.get("created_at"),
                "url": f"https://social.legitreal.com/twoot/{post_id}",
                "source_type": "trending_post",
                "post_id": post_id,
                "author_id": author_id,
            }
            candidate = CandidateAction(
                "CREATE_POST", source_event_id=f"hot-trending:{post_id}",
                context={
                    "articles": [dict(news)],
                    "hot_source_post": hot_source_post,
                    "post_intent": "trend_hitchhike",
                    "post_generation_snapshot_id": world.metadata["snapshot_id"],
                    "hot_news_evidence": {
                        "reason": "historical_trending_post_matched_current_news",
                        "related_post_id": post_id,
                        "related_post_hot_score": post.get("hot_score"),
                        "historical_hot_score": historical_hot,
                        "current_hot_score": current_hot,
                        "current_trending": current_trending,
                        "heat_threshold": self.settings.hot_news_score_threshold,
                        "heat_strength": heat_strength,
                        "cooling_fraction": cooling_fraction,
                        "observed_snapshot_id": post.get("observed_snapshot_id"),
                        "matched_article_id": str(article_id),
                        "matched_article_url": news["url"],
                        "match_method": ("linked_article_url" if linked_news_urls
                                         else "specific_term_overlap"),
                        "shared_terms": sorted(shared),
                        "title_shared_terms": sorted(title_shared),
                    },
                    "content_features": {
                        "source_article_count": 1,
                        "source_type": "news_article",
                        "matched_article_id": str(article_id),
                    },
                    "target_features": {"source_post_id": post_id,
                                         "topic_ids": list(post.get("topic_ids") or [])},
                    "timing_features": {"observed_at": world.metadata.get("finished_at"),
                                         "source_created_at": post.get("created_at"),
                                         "news_published_at": news.get("published_at")},
                    "environment_features": {
                        "snapshot_id": world.metadata.get("snapshot_id"),
                        "source": "current_news_matched_to_trending_post",
                    },
                },
                rationale="Create an original post about news related to a Trending discussion",
                estimated_cost=self.settings.costs["CREATE_POST"],
                required_parameters=("body",),
            )
            if candidate.key not in agent.completed_keys:
                result.append(candidate)
        return result

    def _event_cooling_down(self, event_id: str, agent: AgentState,
                            observed_at: str | None) -> bool:
        cooldown = self.settings.post_event_cooldown_hours
        if cooldown <= 0 or not observed_at:
            return False
        try:
            observed = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
            if observed.tzinfo is None:
                observed = observed.replace(tzinfo=timezone.utc)
        except (AttributeError, TypeError, ValueError):
            return False
        prefix = f"CREATE_POST:{event_id}"
        for action in agent.recent_actions:
            key = str(action.get("action_key") or "")
            if (action.get("action_type") != "CREATE_POST" or
                    action.get("status") != "SUCCEEDED" or
                    not (key == prefix or key.startswith(prefix + ":"))):
                continue
            try:
                created = datetime.fromisoformat(str(action.get("created_at")).replace("Z", "+00:00"))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age = (observed - created).total_seconds() / 3600
                return age < cooldown
            except (TypeError, ValueError):
                return True
        return False
