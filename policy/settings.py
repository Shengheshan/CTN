"""Single place for heuristic weights, Action Priors and approximate costs."""
import json
import math
import os
from dataclasses import dataclass, field

_ACTIONS = ("CREATE_POST", "REPLY", "LIKE", "REPOST", "FOLLOW", "MENTION", "WAIT")


@dataclass(frozen=True)
class PolicySettings:
    relevance: float = 0.18
    freshness: float = 0.14
    heat: float = 0.17
    trend: float = 0.09
    evidence: float = 0.16
    completeness: float = 0.12
    quota: float = 0.06
    repeat_penalty: float = 0.20
    cost_penalty: float = 0.08
    wait_score: float = 0.22
    create_post_priority: float = 0.35
    reply_priority: float = -0.12
    own_reply_priority: float = 0.45
    minimum_post_coverage: float = 0.55
    softmax_temperature: float = 0.35
    post_coverage_window: int = 20
    post_event_cooldown_hours: float = 2.0
    text_actions_per_engagement: int = 3
    hot_news_score_threshold: float = 30.0
    hot_news_heat_weight: float = 0.14
    hot_news_cooldown_penalty: float = 0.28
    hot_news_gate_cooling_factor: float = 0.80
    new_news_window_hours: float = 2.0
    prior_heuristic: float = 0.75
    prior_opportunity: float = 0.20
    prior_urgency: float = 0.05
    bandit_opportunity_prior: float = 0.10
    bandit_risk_weight: float = 1.0
    costs: dict[str, float] = field(default_factory=lambda: {
        "CREATE_POST": 0.9, "REPLY": 0.7, "LIKE": 0.2,
        "REPOST": 0.6, "FOLLOW": 0.3, "MENTION": 0.4, "WAIT": 0.0,
    })
    action_priors: dict[str, float] = field(default_factory=lambda: {
        action: 1.0 for action in _ACTIONS})

    @classmethod
    def from_env(cls) -> "PolicySettings":
        raw = os.getenv("CTN_ACTION_PRIORS", "{}")
        try:
            configured = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("CTN_ACTION_PRIORS must be a JSON object") from exc
        if not isinstance(configured, dict):
            raise ValueError("CTN_ACTION_PRIORS must be a JSON object")
        priors = {action: 1.0 for action in _ACTIONS}
        for action, value in configured.items():
            if action not in priors or isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"invalid Action Prior: {action}")
            if not 0 <= float(value) <= 2:
                raise ValueError("Action Prior values must be in [0,2]")
            priors[action] = float(value)
        coverage = float(os.getenv("CTN_MINIMUM_POST_COVERAGE", "0.55"))
        coverage_window = int(os.getenv("CTN_POST_COVERAGE_WINDOW", "20"))
        event_cooldown = float(os.getenv("CTN_POST_EVENT_COOLDOWN_HOURS", "2"))
        text_per_engagement = int(os.getenv("CTN_TEXT_ACTIONS_PER_ENGAGEMENT", "3"))
        hot_news_threshold = float(os.getenv("CTN_HOT_NEWS_SCORE_THRESHOLD", "30"))
        new_news_window = float(os.getenv("CTN_NEW_NEWS_WINDOW_HOURS", "2"))
        create_priority = float(os.getenv("CTN_CREATE_POST_PRIORITY", "0.35"))
        reply_priority = float(os.getenv("CTN_REPLY_PRIORITY", "-0.12"))
        softmax_temperature = float(os.getenv("CTN_SOFTMAX_TEMPERATURE", "0.35"))
        own_reply_priority = float(os.getenv("CTN_OWN_REPLY_PRIORITY", "0.45"))
        if (not math.isfinite(coverage) or not 0 <= coverage <= 1 or coverage_window < 1
                or not math.isfinite(event_cooldown) or event_cooldown < 0
                or text_per_engagement < 1
                or not math.isfinite(hot_news_threshold) or hot_news_threshold < 0
                or not math.isfinite(new_news_window) or new_news_window <= 0
                or not math.isfinite(create_priority) or not math.isfinite(reply_priority)
                or not math.isfinite(own_reply_priority)
                or not math.isfinite(softmax_temperature) or not 0.05 <= softmax_temperature <= 2.0
                or not -1 <= create_priority <= 1 or not -1 <= reply_priority <= 1
                or not 0 <= own_reply_priority <= 1):
            raise ValueError("post coverage must be in [0,1] and window must be positive")
        return cls(action_priors=priors, minimum_post_coverage=coverage,
                   post_coverage_window=coverage_window,
                   post_event_cooldown_hours=event_cooldown,
                   text_actions_per_engagement=text_per_engagement,
                   hot_news_score_threshold=hot_news_threshold,
                   new_news_window_hours=new_news_window,
                   create_post_priority=create_priority, reply_priority=reply_priority,
                   own_reply_priority=own_reply_priority,
                   softmax_temperature=softmax_temperature)
