"""Baseline and optional conservative action-prior learning interface."""
from copy import deepcopy
import math
import os


class BaseLearningPolicy:
    version = "base-learning-v1"

    def select_action(self, candidates, scores):
        raise NotImplementedError

    def observe_feedback(self, action_type: str, reward: float):
        raise NotImplementedError

    def update(self, action_type: str):
        raise NotImplementedError

    def get_parameters(self) -> dict:
        raise NotImplementedError

    def save_state(self, store):
        store.save_policy_state(self.version, self.get_parameters())

    def load_state(self, store):
        state = store.load_policy_state(self.version)
        if state:
            self.set_parameters(state)


class HeuristicPolicy(BaseLearningPolicy):
    version = "heuristic-baseline-v1"

    def __init__(self, priors=None):
        self.priors = dict(priors or {})

    def select_action(self, candidates, scores):
        if not candidates:
            return None
        index = max(range(len(candidates)), key=lambda i: scores[i])
        return candidates[index]

    def observe_feedback(self, action_type, reward):
        return False

    def update(self, action_type):
        return {"applied": False, "reason": "fixed_heuristic_baseline"}

    def get_parameters(self):
        return {"action_priors": self.priors}

    def set_parameters(self, parameters):
        self.priors = dict(parameters.get("action_priors", self.priors))


class ActionPriorLearningPolicy(BaseLearningPolicy):
    version = "quality-action-priors-v1"

    def __init__(self, action_types, learning_rate: float = 0.05,
                 max_update_step: float | None = None,
                 baseline_decay: float | None = None,
                 max_prior_deviation: float | None = None):
        max_update_step = (float(os.getenv("CTN_POLICY_MAX_UPDATE_STEP", "0.01"))
                           if max_update_step is None else max_update_step)
        baseline_decay = (float(os.getenv("CTN_POLICY_BASELINE_DECAY", "0.02"))
                          if baseline_decay is None else baseline_decay)
        max_prior_deviation = (float(os.getenv("CTN_POLICY_MAX_PRIOR_DEVIATION", "0.20"))
                               if max_prior_deviation is None else max_prior_deviation)
        values = (learning_rate, max_update_step, baseline_decay, max_prior_deviation)
        if (any(not math.isfinite(value) for value in values)
                or not 0 < learning_rate <= 1 or not 0 < max_update_step <= 0.5
                or not 0 <= baseline_decay < 1 or not 0 < max_prior_deviation <= 0.5):
            raise ValueError("invalid clipped action-prior learning configuration")
        self.learning_rate = learning_rate
        self.max_update_step = max_update_step
        self.baseline_decay = baseline_decay
        self.max_prior_deviation = max_prior_deviation
        self.means = {name: 0.5 for name in action_types}
        self.counts = {name: 0 for name in action_types}
        self._pending = {}

    def select_action(self, candidates, scores):
        if not candidates:
            return None
        adjusted = [scores[i] * self.prior_for(c.action_type)
                    for i, c in enumerate(candidates)]
        return candidates[max(range(len(candidates)), key=lambda i: adjusted[i])]

    def observe_feedback(self, action_type: str, reward: float):
        if action_type not in self.means or not 0 <= reward <= 1:
            raise ValueError("invalid learning observation")
        self._pending[action_type] = float(reward)

    def update(self, action_type: str):
        if action_type not in self._pending:
            return {"applied": False, "reason": "no_pending_reward"}
        before = self.get_parameters()
        reward = self._pending.pop(action_type)
        mean_before = self.means[action_type]
        decayed_mean = 0.5 + (mean_before - 0.5) * (1.0 - self.baseline_decay)
        requested_delta = self.learning_rate * (reward - decayed_mean)
        applied_delta = max(-self.max_update_step,
                            min(self.max_update_step, requested_delta))
        proposed = decayed_mean + applied_delta
        bounded = max(0.5 - self.max_prior_deviation,
                      min(0.5 + self.max_prior_deviation, proposed))
        self.means[action_type] = max(0.0, min(1.0, bounded))
        self.counts[action_type] += 1
        after = self.get_parameters()
        return {"applied": before != after, "before": before, "after": after,
                "learning_rate": self.learning_rate, "reward": reward,
                "mean_before": mean_before, "mean_after": self.means[action_type],
                "requested_delta": requested_delta, "applied_delta": self.means[action_type] - mean_before,
                "update_clipped": abs(requested_delta) > self.max_update_step,
                "baseline_decay": self.baseline_decay,
                "max_update_step": self.max_update_step,
                "max_prior_deviation": self.max_prior_deviation}

    def prior_for(self, action_type: str) -> float:
        return max(0.0, min(2.0, 0.5 + self.means.get(action_type, 0.5)))

    def get_parameters(self) -> dict:
        return {"means": dict(self.means), "counts": dict(self.counts),
                "prior_multipliers": {key: self.prior_for(key) for key in self.means},
                "update_config": {"learning_rate": self.learning_rate,
                                  "max_update_step": self.max_update_step,
                                  "baseline_decay": self.baseline_decay,
                                  "max_prior_deviation": self.max_prior_deviation}}

    def set_parameters(self, parameters):
        means = parameters.get("means", {})
        counts = parameters.get("counts", {})
        for key in self.means:
            if key in means:
                value = max(0.0, min(1.0, float(means[key])))
                self.means[key] = max(0.5 - self.max_prior_deviation,
                                      min(0.5 + self.max_prior_deviation, value))
            if key in counts:
                self.counts[key] = int(counts[key])
