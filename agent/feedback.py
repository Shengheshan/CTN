"""Record observable execution and engagement without causal reward claims."""

import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from math import expm1, isfinite

from policy.bandit import FEATURE_VERSION, MIN_TRAINING_SAMPLES, LinUCB

from agent.executor import ExecutionResult
from agent.history import ActionHistory
from policy.models import ActionPlan


class FeedbackCollector:
    def __init__(self, social, history: ActionHistory):
        self.social = social
        self.history = history

    def collect(self, plan: ActionPlan, result: ExecutionResult) -> dict:
        metrics = {}
        if result.status == "SUCCEEDED" and result.remote_id:
            try:
                metrics = self.social.counters(int(result.remote_id)) or {}
            except Exception as exc:
                metrics = {"read_error": type(exc).__name__}
        self.history.save_feedback(
            plan.action_id,
            result.status == "SUCCEEDED" if result.status not in ("UNKNOWN", "DRY_RUN") else None,
            result.status == "SUCCEEDED" if result.status not in ("UNKNOWN", "DRY_RUN") else None,
            metrics,
        )
        return metrics

    def collect_delayed(self, horizon_minutes: float = 30.0) -> int:
        """Create a direct-engagement proxy only for mature real posts."""
        cutoff = (datetime.now(timezone.utc) -
                  timedelta(minutes=horizon_minutes)).isoformat()
        count = 0
        for item in self.history.pending_delayed_feedback(cutoff):
            try:
                metrics = self.social.counters(int(item["remote_id"])) or {}
            except Exception:
                continue
            reward = engagement_proxy(metrics)
            if reward is None:
                continue
            self.history.save_feedback(item["action_id"], True, True, metrics)
            if self.history.save_reward(item["action_id"], reward,
                                        "official_base_engagement_proxy_v1"):
                count += 1
        return count


def engagement_proxy(metrics: dict) -> float | None:
    """Bounded proxy using official base points for observable post counters.

    Official citizen scores award 1/view, 5/like, 10/repost and 15/reply.
    The public counter API does not identify NPCs, apply the team-wide view
    cap/multiplier, or expose persuasion/follow awards, so this is not an
    official score or a causal estimate.
    """
    fields = (
        ("view_count", 1.0),
        ("like_count", 5.0),
        ("repost_count", 10.0),
        ("reply_count", 15.0),
    )
    points = 0.0
    observed = False
    for key, award in fields:
        value = metrics.get(key)
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and isfinite(value) and value >= 0):
            points += award * value
            observed = True
    if not observed:
        return None
    # Keep rewards in the LinUCB [0, 1] range while preserving the official
    # relative point values at low engagement.
    return -expm1(-points / 50.0)

class V03FeedbackSystem:
    """Windowed feedback collector with opt-in contextual policy learning."""
    ACTION_TYPES = ("CREATE_POST", "REPLY", "LIKE", "REPOST", "FOLLOW", "MENTION", "WAIT")
    LEARNABLE_ACTION_TYPES = frozenset(("CREATE_POST", "REPLY"))
    ONLINE_POLICY_VERSION = "ctn-contextual-linucb-v1"

    def __init__(self, social, store, config, base_priors=None):
        from agent.v03_store import V03Store
        from policy.feature_analyzer import ContentFeatureAnalyzer
        from policy.learning import ActionPriorLearningPolicy, HeuristicPolicy
        from policy.reward import RewardEvaluator
        self.social = social
        self.store = store if isinstance(store, V03Store) else V03Store(store)
        self.config = config
        self.bandit = LinUCB()
        saved_bandit = self.store.load_policy_state(FEATURE_VERSION)
        if saved_bandit:
            self.bandit = LinUCB.from_dict(saved_bandit)
        self.analyzer = ContentFeatureAnalyzer(
            config.novelty_history_size, config.novelty_duplicate_threshold)
        self.reward_evaluator = RewardEvaluator(
            config.reward_weights, version=config.reward_version)
        self.learning_enabled = bool(config.enable_learning)
        self.learning_start_key = f"{self.ONLINE_POLICY_VERSION}-start"
        learning_start = self.store.load_policy_state(self.learning_start_key)
        if self.learning_enabled:
            if not isinstance(learning_start, dict) or not learning_start.get("started_at"):
                self.learning_started_at = datetime.now(timezone.utc).isoformat()
                self.store.save_policy_state(
                    self.learning_start_key, {"started_at": self.learning_started_at})
            else:
                self.learning_started_at = learning_start["started_at"]
        else:
            self.learning_started_at = None
        self.action_priors = dict(base_priors or {})
        if self.learning_enabled:
            self.learning_policy = ActionPriorLearningPolicy(self.ACTION_TYPES)
            state = self.store.load_policy_state(self.learning_policy.version)
            if state:
                self.learning_policy.set_parameters(state)
            else:
                for action, prior in self.action_priors.items():
                    if action in self.learning_policy.means:
                        self.learning_policy.means[action] = max(0.0, min(1.0, float(prior) - 0.5))
        else:
            self.learning_policy = HeuristicPolicy(self.action_priors)

    def current_action_priors(self) -> dict:
        if self.learning_enabled:
            return self.learning_policy.get_parameters()["prior_multipliers"]
        return dict(self.action_priors)

    def record_plan(self, plan):
        if plan.action_type == "WAIT":
            plan.parameters["wait_reason"] = plan.reason or "policy_wait"
        return self.store.record_decision(plan)

    def record_content(self, plan):
        if plan.action_type not in ("CREATE_POST", "REPLY") or not plan.generated_content:
            return None
        past = self.store.recent_contents(self.config.novelty_history_size)
        features = self.analyzer.analyze(plan, past)
        if plan.action_type == "REPLY" and plan.target_id:
            context = plan.parameters.get("context", {})
            target_features = (plan.parameters.get("target_features") or
                               context.get("target_features", {}))
            try:
                baseline = self.social.counters(int(plan.target_id)) or {}
                allowed = {"like_count", "reply_count", "repost_count", "view_count", "follow_count"}
                baseline = {internal: baseline[source] for internal, source in {
                    "likes": "like_count", "replies": "reply_count",
                    "reposts": "repost_count", "views": "view_count",
                    "follows": "follow_count"}.items()
                    if source in allowed and isinstance(baseline.get(source), int)
                    and not isinstance(baseline.get(source), bool) and baseline[source] >= 0}
                features["target_features"] = {
                    **target_features,
                    "parent_post_id": plan.target_id,
                    "pre_action_parent_metrics": baseline,
                    "pre_action_baseline_status": "COLLECTED" if baseline else "UNAVAILABLE",
                }
                plan.parameters["target_features"] = features["target_features"]
            except Exception as exc:
                features["target_features"] = {
                    **target_features,
                    "parent_post_id": plan.target_id,
                    "pre_action_parent_metrics": {},
                    "pre_action_baseline_status": "UNAVAILABLE",
                    "baseline_error": type(exc).__name__,
                }
        plan.parameters["content_assessment"] = features
        plan.parameters["content_features"] = {
            "content": plan.generated_content,
            "target_text": plan.parameters.get("context", {}).get("post_body"),
            "relevance_score": features.get("context_relevance_score"),
            "originality": features.get("novelty_score"),
            "factual_basis": features.get("source_validation"),
        }
        self.store.save_content_assessment(plan.action_id, plan.generated_content, features)
        return features

    def record_result(self, plan, result):
        if self.config.enable_feedback:
            self.store.schedule_feedback(plan, result, self.config.feedback_windows)
        reward_excluded = bool(plan.parameters.get("exclude_from_reward"))
        if (result.status in ("SUCCEEDED", "WAIT") and self.config.enable_feedback
                and not reward_excluded):
            last_window = (f"t+{max(self.config.feedback_windows)}s"
                           if result.status == "SUCCEEDED" else "next_observation")
            self.store.save_reward(plan.action_id, self.config.reward_version, last_window,
                {}, None, "PENDING", {"reason": "waiting_for_scheduled_feedback",
                                      "feedback_windows_seconds": list(self.config.feedback_windows)})
        saved_status = self.store.action_status(plan.action_id)
        if (result.status in ("FAILED", "BLOCKED") and saved_status == result.status
                and not reward_excluded):
            evaluated = self.reward_evaluator.evaluate(
                plan.action_type, False, plan.parameters.get("content_assessment", {}),
                None, True, "execution")
            calculation = dict(evaluated.calculation)
            calculation["failure_reason"] = result.error
            reward_id = self.store.save_reward(
                plan.action_id, self.config.reward_version, "execution",
                evaluated.components, evaluated.total, "FINAL", calculation)
            self._record_learning(plan.action_id, plan.action_type, reward_id,
                                  evaluated.total, final=True)
        elif (result.status == "UNKNOWN" and saved_status == "UNKNOWN"
              and not reward_excluded):
            self.store.save_reward(plan.action_id, self.config.reward_version,
                "execution", {}, None, "INVALID",
                {"reason": "execution_outcome_unknown", "error": result.error,
                 "reward_total": None})

    def process_due(self, limit: int = 20) -> dict:
        collected = 0
        errors = 0
        for item in self.store.due_feedback(limit):
            action_type = item["action_type"]
            if item["window_name"] == "next_observation":
                try:
                    raw = json.loads(item.get("raw_metrics_json") or "{}")
                    parameters = json.loads(item.get("action_parameters") or "{}")
                except (TypeError, json.JSONDecodeError):
                    raw, parameters = {}, {}
                raw["wait_reason"] = parameters.get("wait_reason") or "policy_wait"
                from datetime import datetime, timezone
                scheduled = datetime.fromisoformat(item["scheduled_at"].replace("Z", "+00:00"))
                raw["wait_seconds"] = max(0.0, (datetime.now(timezone.utc)-scheduled).total_seconds())
                self.store.complete_feedback(item, raw, "COLLECTED", error=None)
                collected += 1
                continue
            if action_type not in ("CREATE_POST", "REPLY"):
                self.store.complete_feedback(
                    item, {}, "UNAVAILABLE",
                    "No action-specific public counter endpoint for this action type")
                errors += 1
                continue
            try:
                response = self.social.counters(int(item["remote_id"])) or {}
                allowed = {"like_count", "reply_count", "repost_count", "view_count", "follow_count"}
                metrics = {key: value for key, value in response.items()
                           if key in allowed and isinstance(value, int)
                           and not isinstance(value, bool) and value >= 0}
                parent_metrics = {}
                parent_id = item.get("parent_remote_id")
                if action_type == "REPLY" and parent_id:
                    try:
                        parent_response = self.social.counters(int(parent_id)) or {}
                        parent_metrics = {key: value for key, value in parent_response.items()
                                          if key in allowed and isinstance(value, int)
                                          and not isinstance(value, bool) and value >= 0}
                    except Exception as parent_exc:
                        parent_metrics = {}
                        item["parent_feedback_error"] = type(parent_exc).__name__
                expected = {"like_count", "reply_count", "repost_count", "view_count"}
                if not metrics:
                    status = "UNAVAILABLE"
                    feedback_error = "Counter API returned no supported metrics"
                elif expected.issubset(metrics):
                    status = "COLLECTED"
                    feedback_error = None
                else:
                    status = "PARTIAL"
                    feedback_error = "One or more counter metrics were absent"
                self.store.complete_feedback(item, metrics, status, feedback_error,
                                             parent_metrics=parent_metrics)
                collected += 1
            except Exception as exc:
                http_status = getattr(exc, "status", None)
                if http_status == 429 and int(item.get("attempt_count", 0)) < 2:
                    retry_after = getattr(exc, "retry_after", 60)
                    delay = max(1, float(retry_after or 60))
                    from datetime import datetime, timedelta, timezone
                    retry_at = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()
                    self.store.complete_feedback(item, {}, "PENDING",
                        f"HTTP 429; retry scheduled ({type(exc).__name__})", retry_at)
                elif http_status == 404:
                    self.store.complete_feedback(item, {}, "UNAVAILABLE", "Remote object not found")
                    errors += 1
                else:
                    self.store.complete_feedback(item, {}, "FAILED", type(exc).__name__)
                    errors += 1

        rewards = 0
        updates = 0
        for action in self.store.pending_rewards():
            if action.get("parameters", {}).get("exclude_from_reward"):
                # Counter snapshots remain available for reporting, but this
                # independent action layer never contributes reward or learning.
                continue
            snapshots = action["snapshots"]
            if not snapshots:
                continue
            pending = any(item["feedback_status"] == "PENDING" for item in snapshots)
            complete = not pending
            latest = max(snapshots, key=lambda row: row["scheduled_at"])
            try:
                observed_metrics = json.loads(latest.get("delta_metrics_json") or "{}")
            except (TypeError, json.JSONDecodeError):
                observed_metrics = {}
            if not observed_metrics:
                observed_metrics = {key: latest.get(key) for key in
                                    ("views", "likes", "reposts", "replies")}
            assessment = action["assessment"]
            evaluation = self.reward_evaluator.evaluate(
                action["action_type"], True, assessment,
                (None if action["action_type"] == "WAIT" else True), complete,
                latest["window_name"],
                [{"window": snap["window_name"], "status": snap["feedback_status"],
                  "observed_at": snap["observed_at"],
                  "own_delta": snap.get("delta_metrics_json"),
                  "parent_delta": snap.get("parent_delta_metrics_json"),
                  "parent_delta_attribution": "NOT_ATTRIBUTED_TO_REPLY"}
                 for snap in snapshots], observed_metrics)
            reward_id = self.store.save_reward(
                action["action_id"], self.config.reward_version, latest["window_name"],
                evaluation.components, evaluation.total, evaluation.status,
                evaluation.calculation)
            rewards += 1
            if complete and evaluation.total is not None:
                if self._record_learning(action["action_id"], action["action_type"],
                                         reward_id, evaluation.total, final=True):
                    updates += 1
        return {"feedback_collected": collected, "feedback_errors": errors,
                "rewards_recomputed": rewards, "policy_updates": updates}

    def _record_learning(self, action_id, action_type, reward_id, reward, final):
        if not final or reward is None:
            return False
        before = {
            "contextual_bandit": deepcopy(self.bandit.to_dict()),
            "action_priors": deepcopy(self.learning_policy.get_parameters()),
        }
        if action_type == "WAIT":
            if not self.store.has_policy_update(reward_id):
                self.store.add_policy_update(self.ONLINE_POLICY_VERSION, action_id, reward_id,
                    before, before, "wait_baseline_not_learned", None, False)
            return False
        if action_type not in self.LEARNABLE_ACTION_TYPES:
            if not self.store.has_policy_update(reward_id):
                self.store.add_policy_update(self.ONLINE_POLICY_VERSION, action_id, reward_id,
                    before, before, "no_action_specific_observed_outcome", None, False)
            return False
        if not self.learning_enabled:
            if not self.store.has_policy_update_for_action(action_id):
                self.store.add_policy_update(self.ONLINE_POLICY_VERSION, action_id, reward_id,
                    before, before, "learning_disabled_fixed_baseline", None, False)
            return False
        created_at = self.store.action_created_at(action_id)
        try:
            pre_learning_action = (
                created_at is None or self.learning_started_at is None or
                datetime.fromisoformat(created_at) <
                datetime.fromisoformat(self.learning_started_at))
        except (TypeError, ValueError):
            pre_learning_action = True
        if pre_learning_action:
            if not self.store.has_policy_update_for_action(action_id):
                self.store.add_policy_update(
                    self.ONLINE_POLICY_VERSION, action_id, reward_id, before, before,
                    "pre_learning_start_action_excluded", None, False)
            return False
        if self.store.has_policy_update_for_action(action_id):
            return False
        old_count = self.bandit.count
        features = self.store.selected_policy_features(action_id)
        bandit_applied = False
        if features is not None:
            try:
                self.bandit.update(features, float(reward))
            except ValueError:
                features = None
            else:
                self.store.save_policy_state(FEATURE_VERSION, self.bandit.to_dict())
                bandit_applied = True
        prior_result = {}
        prior_applied = False
        if old_count < MIN_TRAINING_SAMPLES:
            self.learning_policy.observe_feedback(action_type, float(reward))
            prior_result = self.learning_policy.update(action_type)
            prior_applied = bool(prior_result.get("applied"))
            if prior_applied:
                self.learning_policy.save_state(self.store)
        after = {
            "contextual_bandit": deepcopy(self.bandit.to_dict()),
            "action_priors": deepcopy(self.learning_policy.get_parameters()),
        }
        applied = bandit_applied or prior_applied
        if bandit_applied and prior_applied:
            reason = "contextual_linucb_and_cold_start_prior_update"
        elif bandit_applied:
            reason = "contextual_linucb_update"
        elif prior_applied:
            reason = "cold_start_action_prior_update"
        elif features is None:
            reason = "missing_or_invalid_context_features"
        else:
            reason = "no_parameter_change"
        self.store.add_policy_update(
            self.ONLINE_POLICY_VERSION, action_id, reward_id, before, after,
            reason, prior_result.get("learning_rate"), applied)
        return applied
