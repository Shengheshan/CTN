"""Versioned action-specific reward using owned-object feedback only."""
from dataclasses import dataclass

DEFAULT_WEIGHTS = {"execution_component": 0.30, "factual_component": 0.35,
                   "novelty_component": 0.20, "compliance_component": 0.15}


@dataclass(frozen=True)
class RewardResult:
    components: dict
    total: float | None
    status: str
    calculation: dict


class RewardEvaluator:
    VERSION = "reward-v2-action-feedback"

    def __init__(self, weights: dict | None = None, version: str = VERSION):
        self.version = version
        self.weights = dict(DEFAULT_WEIGHTS)
        self.weights.update(weights or {})
        if set(self.weights) != set(DEFAULT_WEIGHTS):
            raise ValueError("reward weights must name all and only the four quality components")
        if any(value < 0 for value in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError("reward weights must be nonnegative and nonzero")

    def evaluate(self, action_type: str, execution_success: bool | None,
                 assessment: dict, platform_accepted: bool | None,
                 feedback_complete: bool, feedback_window: str | None,
                 feedback_statuses: list[dict] | None = None,
                 observed_metrics: dict | None = None) -> RewardResult:
        observed_metrics = observed_metrics or {}
        observed_points = 0.0
        observed_any = False
        for key, award in (("views", 1), ("likes", 5), ("reposts", 10), ("replies", 15)):
            value = observed_metrics.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                observed_points += float(value) * award
                observed_any = True
        observed_feedback = 1.0 - __import__("math").exp(-observed_points / 50.0) if observed_any else None
        components = {
            "execution_component": float(execution_success) if execution_success is not None else None,
            "factual_component": assessment.get("source_alignment_score"),
            "novelty_component": assessment.get("novelty_score"),
            "compliance_component": float(platform_accepted) if platform_accepted is not None else None,
            "context_relevance_component": assessment.get("context_relevance_score"),
            "observed_feedback_component": observed_feedback,
        }
        applicable = ["execution_component", "compliance_component"]
        if action_type in ("CREATE_POST", "REPLY"):
            if assessment.get("source_validation", {}).get("source_count", 0) > 0:
                applicable.append("factual_component")
            if action_type == "REPLY" and components["context_relevance_component"] is not None:
                applicable.append("context_relevance_component")
            if observed_any:
                applicable.append("observed_feedback_component")
        weights = {**self.weights, "context_relevance_component": 0.15,
                   "observed_feedback_component": 0.15}
        known = [key for key in applicable if components.get(key) is not None]
        denom = sum(weights[key] for key in known)
        total = (sum(weights[key] * components[key] for key in known) / denom
                 if denom else None)
        missing = [key for key in applicable if components.get(key) is None]
        status = "FINAL" if feedback_complete else "PROVISIONAL"
        if total is None:
            status = "INVALID"
        calculation = {
            "reward_version": self.version, "weights": weights,
            "components": components, "applicable_components": applicable,
            "analysis_only_components": ["novelty_component"],
            "used_components": known, "missing_components": missing,
            "feedback_complete": feedback_complete, "feedback_window": feedback_window,
            "feedback_statuses": feedback_statuses or [],
            "interpretation": "owned-object interaction proxy plus content quality; descriptive and non-causal",
            "engagement_counters_used": observed_any,
            "parent_post_counters_used": False,
            "action_type_evaluation": ("original_post_quality_and_owned_feedback" if action_type == "CREATE_POST"
                                       else "reply_quality_context_timing_and_reply_own_feedback"
                                       if action_type == "REPLY" else "execution_and_compliance"),
        }
        return RewardResult(components, round(total, 6) if total is not None else None,
                            status, calculation)
