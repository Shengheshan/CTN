"""Pure-Python contextual LinUCB with stable Softmax sampling."""

import random
from math import exp, isfinite, sqrt

from policy.models import CandidateAction
from policy.opportunity import OpportunityAssessment
from policy.strategic import StrategicAssessment
from state.encoder import EncodedWorldState, GLOBAL_FEATURE_NAMES, POST_FEATURE_NAMES


# Keep the v1 feature schema stable. MENTION has no direct training reward.
ACTION_ORDER = ("CREATE_POST", "FOLLOW", "LIKE", "REPLY", "REPOST", "WAIT")
FEATURE_VERSION = "ctn-bandit-features-v1"
MIN_TRAINING_SAMPLES = 5
FEATURE_DIM = 1 + len(GLOBAL_FEATURE_NAMES) + len(POST_FEATURE_NAMES) + 1 + len(ACTION_ORDER) + 4


class ContextFeatureEncoder:
    def encode(self, candidate: CandidateAction, encoded: EncodedWorldState,
               strategy: StrategicAssessment, opportunity: OpportunityAssessment,
               risk: float) -> tuple[float, ...]:
        post_id = candidate.context.get("source_post_id", candidate.target_id)
        post = encoded.post_vectors.get(post_id, (0.0,) * len(POST_FEATURE_NAMES))
        event = encoded.event_vectors.get(candidate.source_event_id, (0.0,))
        action = tuple(float(candidate.action_type == name) for name in ACTION_ORDER)
        vector = (1.0,) + encoded.global_vector + tuple(post) + tuple(event) + action + (
            opportunity.score, opportunity.evidence_coverage, strategy.urgency, risk)
        if len(vector) != FEATURE_DIM or any(not isfinite(value) for value in vector):
            raise ValueError("invalid contextual feature vector")
        return vector


class LinUCB:
    def __init__(self, dimension: int = FEATURE_DIM, ridge: float = 1.0,
                 alpha: float = 0.15):
        if dimension < 1 or ridge <= 0 or alpha < 0:
            raise ValueError("invalid bandit parameters")
        self.dimension = dimension
        self.ridge = ridge
        self.alpha = alpha
        self.inverse = [[(1.0 / ridge if i == j else 0.0)
                         for j in range(dimension)] for i in range(dimension)]
        self.b = [0.0] * dimension
        self.count = 0

    def predict(self, features: tuple[float, ...]) -> tuple[float, float]:
        self._validate(features)
        ax = [sum(row[j] * features[j] for j in range(self.dimension))
              for row in self.inverse]
        theta = [sum(row[j] * self.b[j] for j in range(self.dimension))
                 for row in self.inverse]
        expected = sum(theta[i] * features[i] for i in range(self.dimension))
        uncertainty = sqrt(max(0.0, sum(features[i] * ax[i]
                                         for i in range(self.dimension))))
        return expected, uncertainty

    def update(self, features: tuple[float, ...], reward: float) -> None:
        self._validate(features)
        if not isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError("reward must be finite and in [0,1]")
        ax = [sum(row[j] * features[j] for j in range(self.dimension))
              for row in self.inverse]
        denominator = 1.0 + sum(features[i] * ax[i] for i in range(self.dimension))
        for i in range(self.dimension):
            for j in range(self.dimension):
                self.inverse[i][j] -= ax[i] * ax[j] / denominator
            self.b[i] += reward * features[i]
        self.count += 1

    def _validate(self, features):
        if len(features) != self.dimension or any(not isfinite(x) for x in features):
            raise ValueError("feature dimension or value mismatch")

    def to_dict(self) -> dict:
        return {"version": FEATURE_VERSION, "dimension": self.dimension,
                "ridge": self.ridge, "alpha": self.alpha, "inverse": self.inverse,
                "b": self.b, "count": self.count}

    @classmethod
    def from_dict(cls, data: dict) -> "LinUCB":
        if data.get("version") != FEATURE_VERSION or data.get("dimension") != FEATURE_DIM:
            raise ValueError("bandit feature schema changed")
        model = cls(data["dimension"], data["ridge"], data["alpha"])
        if (len(data["inverse"]) != model.dimension or
                any(len(row) != model.dimension for row in data["inverse"]) or
                len(data["b"]) != model.dimension):
            raise ValueError("invalid persisted bandit model")
        model.inverse = data["inverse"]
        model.b = data["b"]
        model.count = data["count"]
        return model


def softmax_sample(scores: list[float], temperature: float,
                   rng: random.Random) -> tuple[int, tuple[float, ...]]:
    if not scores or temperature <= 0 or any(not isfinite(s) for s in scores):
        raise ValueError("invalid softmax inputs")
    peak = max(scores)
    weights = [exp((score - peak) / temperature) for score in scores]
    total = sum(weights)
    probabilities = tuple(weight / total for weight in weights)
    draw = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if draw < cumulative:
            return index, probabilities
    return len(scores) - 1, probabilities
