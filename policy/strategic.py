"""Strategic state estimates, with unavailable signals represented explicitly."""

from dataclasses import dataclass

from state.encoder import EncodedWorldState, GLOBAL_FEATURE_NAMES


_G = {name: i for i, name in enumerate(GLOBAL_FEATURE_NAMES)}


@dataclass(frozen=True)
class StrategicAssessment:
    favorability: float | None
    urgency: float
    evidence_coverage: float
    missing_signals: tuple[str, ...]


class StrategicEvaluator:
    def evaluate(self, encoded: EncodedWorldState) -> StrategicAssessment:
        g = encoded.global_vector
        if len(g) != len(GLOBAL_FEATURE_NAMES):
            raise ValueError("global feature schema mismatch")
        velocity_coverage = g[_G["velocity_coverage"]]
        urgency = min(1.0, max(0.0,
            0.25 * g[_G["trending_fraction"]]
            + 0.75 * max(0.0, g[_G["mean_hot_velocity"]]) * velocity_coverage))
        # Poll, side and official score data are not observed in WorldState.
        return StrategicAssessment(
            favorability=None, urgency=urgency,
            evidence_coverage=velocity_coverage,
            missing_signals=("poll_position", "side_alignment", "official_score"),
        )
