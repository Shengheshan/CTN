"""Evaluate candidate context before policy selection."""

from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp, log, sqrt

from agent.state import PlatformState
from policy.models import CandidateAction
from state.builder import WorldState
from state.encoder import EncodedWorldState, POST_FEATURE_NAMES


_P = {name: i for i, name in enumerate(POST_FEATURE_NAMES)}


def _freshness(created: str | None, observed: str | None) -> float | None:
    if not created or not observed:
        return None
    try:
        def parse(value):
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        hours = (parse(observed) - parse(created)).total_seconds() / 3600
    except (ValueError, TypeError, AttributeError):
        return None
    if hours < -1:
        return None
    return exp(-log(2) * max(0.0, hours) / 6)


@dataclass(frozen=True)
class OpportunityAssessment:
    score: float
    evidence_coverage: float
    components: dict[str, float]
    missing_signals: tuple[str, ...]


class OpportunityEvaluator:
    def evaluate(self, candidate: CandidateAction, world: WorldState,
                 encoded: EncodedWorldState, platform: PlatformState) -> OpportunityAssessment:
        if candidate.action_type == "WAIT":
            return OpportunityAssessment(0.0, 1.0, {}, ())
        observed_at = world.metadata.get("finished_at")
        influence = freshness = reach = relevance = None
        if candidate.action_type == "CREATE_POST":
            event = encoded.event_vectors.get(candidate.source_event_id)
            influence = event[0] if event else None
            articles = candidate.context.get("articles", [])
            freshness_values = [_freshness(a.get("published_at"), observed_at)
                                for a in articles]
            freshness = max((v for v in freshness_values if v is not None), default=None)
            reach = min(1.0, len({a.get("site_name") for a in articles if a.get("site_name")}) / 2)
        else:
            post_id = candidate.context.get("source_post_id", candidate.target_id)
            vector = encoded.post_vectors.get(post_id)
            if vector:
                influence = vector[_P["hot_score"]] if vector[_P["has_hot_score"]] else None
                reach = sqrt(vector[_P["rank_score"]]) if vector[_P["has_rank"]] else None
                velocity = max(0.0, vector[_P["hot_velocity"]]) if vector[_P["has_hot_velocity"]] else None
                if influence is not None and velocity is not None:
                    influence = 0.7 * influence + 0.3 * velocity
                elif influence is None:
                    influence = velocity
            freshness = _freshness(candidate.context.get("created_at"), observed_at)
        key = str(candidate.target_id or candidate.source_event_id or "global")
        if candidate.action_type in ("FOLLOW", "MENTION"):
            key = str(candidate.context.get("source_post_id", candidate.target_id))
        if key in platform.relevance_by_target:
            relevance = max(0.0, min(1.0, platform.relevance_by_target[key]))
        components = {}
        missing = []
        covered = 0.0
        for name, value, weight in (("influence", influence, 0.30),
                                    ("freshness", freshness, 0.20),
                                    ("reach", reach, 0.15),
                                    ("relevance", relevance, 0.20)):
            if value is None:
                missing.append(name)
            else:
                components[name] = weight * value
                covered += weight
        missing.extend(("strategic_fit", "competition"))
        return OpportunityAssessment(
            score=max(0.0, min(1.0, sum(components.values()))),
            evidence_coverage=covered,
            components=components,
            missing_signals=tuple(missing),
        )
