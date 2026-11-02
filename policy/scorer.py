"""Explainable heuristic for Action + Target + Context, not reward prediction."""

from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp, isfinite, log

from agent.state import AgentState, PlatformState
from policy.models import CandidateAction
from policy.settings import PolicySettings
from state.builder import WorldState
from state.encoder import EncodedWorldState, POST_FEATURE_NAMES


_P = {name: i for i, name in enumerate(POST_FEATURE_NAMES)}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _freshness(created: str | None, observed: str | None) -> float:
    if not created or not observed:
        return 0.5
    try:
        def parse(value):
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        hours = (parse(observed) - parse(created)).total_seconds() / 3600
    except (ValueError, TypeError):
        return 0.5
    if hours < -1:
        return 0.5
    return exp(-log(2) * max(0.0, hours) / 6)


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: CandidateAction
    total_score: float
    score_components: dict[str, float]


class PolicyScorer:
    def __init__(self, settings: PolicySettings | None = None):
        self.settings = settings or PolicySettings()

    def score(self, candidate: CandidateAction, world: WorldState,
              encoded: EncodedWorldState, agent: AgentState,
              platform: PlatformState) -> ScoredCandidate:
        w = self.settings
        if candidate.action_type == "WAIT":
            return ScoredCandidate(candidate, w.wait_score, {"wait_baseline": w.wait_score})
        target_key = str(candidate.target_id or candidate.source_event_id or "global")
        if candidate.action_type in ("FOLLOW", "MENTION"):
            target_key = str(candidate.context.get("source_post_id", candidate.target_id))
        relevance = _clamp(platform.relevance_by_target.get(target_key, 0.5))
        freshness = 0.5
        heat = trend = evidence = 0.0
        completeness = 0.0
        if candidate.action_type == "CREATE_POST":
            articles = candidate.context.get("articles", [])
            evidence = _clamp(len(articles) / 2)
            completeness = _clamp(sum(bool(a.get("title") and a.get("summary"))
                                      for a in articles) / max(1, len(articles)))
            freshness = max((_freshness(a.get("published_at"), world.metadata.get("finished_at"))
                             for a in articles), default=0.5)
        else:
            post_id = candidate.context.get("post_id", candidate.target_id)
            if candidate.action_type == "FOLLOW":
                post_id = candidate.context.get("source_post_id")
            vector = encoded.post_vectors.get(post_id)
            if vector is not None:
                heat = vector[_P["hot_score"]] if vector[_P["has_hot_score"]] else 0.0
                trend = max(0.0, vector[_P["hot_velocity"]]) if vector[_P["has_hot_velocity"]] else 0.0
                completeness = (vector[_P["has_hot_score"]] + vector[_P["has_rank"]]
                                + vector[_P["has_hot_velocity"]]) / 3
            freshness = _freshness(candidate.context.get("created_at"),
                                   world.metadata.get("finished_at"))
            evidence = 0.25 if candidate.context.get("post_body") else 0.0

        quota = 0.5 if platform.api_remaining is None else float(platform.api_remaining > 0)
        sources = set(candidate.context.get("feed_sources", []))
        source_bonus = (0.10 if "notification" in sources else
                        0.03 if "home" in sources else
                        0.02 if "tagged" in sources else 0.0)
        repeated = float(any(a.get("action_type") == candidate.action_type
                             for a in agent.recent_actions[:5]))
        hot_evidence = (candidate.context.get("hot_news_evidence") or {}
                        if candidate.context.get("post_intent") == "trend_hitchhike"
                        else {})
        hot_heat_bonus = (w.hot_news_heat_weight *
                          _clamp(float(hot_evidence.get("heat_strength") or 0.0)))
        hot_cooling_penalty = (-w.hot_news_cooldown_penalty *
                               _clamp(float(hot_evidence.get("cooling_fraction") or 0.0)))
        components = {
            "action_priority": (w.create_post_priority if candidate.action_type == "CREATE_POST"
                                else w.own_reply_priority if candidate.action_type == "REPLY"
                                and candidate.context.get("reply_to_own_post")
                                else w.reply_priority if candidate.action_type == "REPLY" else 0.0),
            "relevance": w.relevance * relevance,
            "freshness": w.freshness * freshness,
            "heat": w.heat * heat,
            "trend": w.trend * trend,
            "evidence": w.evidence * evidence,
            "completeness": w.completeness * completeness,
            "quota": w.quota * quota,
            "source_bonus": source_bonus,
            "repeat_penalty": -w.repeat_penalty * repeated,
            "cost_penalty": -w.cost_penalty * candidate.estimated_cost,
            "hot_news_heat_bonus": hot_heat_bonus,
            "hot_news_cooldown_penalty": hot_cooling_penalty,
        }
        total = _clamp(sum(components.values()))
        if not isfinite(total):
            raise ValueError("non-finite policy score")
        return ScoredCandidate(candidate, total, components)
