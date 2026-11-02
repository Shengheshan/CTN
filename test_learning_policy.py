import random
import tempfile
import unittest
from pathlib import Path

from agent.feedback import FeedbackCollector, engagement_proxy
from agent.history import ActionHistory
from agent.state import AgentState, PlatformState
from policy.bandit import (FEATURE_DIM, ContextFeatureEncoder, LinUCB,
                           softmax_sample)
from policy.engine import DecisionEngine
from policy.models import ActionPlan, CandidateAction
from policy.opportunity import OpportunityEvaluator
from policy.settings import PolicySettings
from policy.strategic import StrategicEvaluator
from state.builder import GlobalState, PostState, WorldState
from state.encoder import StateEncoder


class Social:
    def counters(self, remote_id):
        return {"like_count": 10, "reply_count": 2, "repost_count": 1}


def fixture():
    world = WorldState(
        metadata={"snapshot_id": 1, "finished_at": "2026-11-01T12:00:00Z"},
        global_state=GlobalState(1, 1, 0, 0, 0, None),
        posts=[PostState(11, 22, "H2D2 referendum update", 80.0, 1,
                         ["trending"], {"hot_velocity": 0.5}, [],
                         "2026-11-01T11:50:00Z")],
        events=[], topics=[],
    )
    encoded = StateEncoder().encode(world)
    candidate = CandidateAction("REPLY", target_id=11,
                                context={"post_body": "Observed text",
                                         "created_at": "2026-11-01T11:50:00Z",
                                         "source_post_id": 11},
                                required_parameters=("body",))
    platform = PlatformState()
    strategy = StrategicEvaluator().evaluate(encoded)
    opportunity = OpportunityEvaluator().evaluate(candidate, world, encoded, platform)
    features = ContextFeatureEncoder().encode(candidate, encoded, strategy,
                                              opportunity, 0.1)
    return world, encoded, candidate, platform, strategy, opportunity, features


class LearningPolicyTests(unittest.TestCase):
    def test_explicit_missing_strategy_and_feature_schema(self):
        _, _, _, _, strategy, opportunity, features = fixture()
        self.assertIsNone(strategy.favorability)
        self.assertIn("poll_position", strategy.missing_signals)
        self.assertIn("relevance", opportunity.missing_signals)
        self.assertEqual(len(features), FEATURE_DIM)
        self.assertGreater(opportunity.score, 0)

    def test_linucb_updates_prediction_and_reduces_uncertainty(self):
        features = fixture()[-1]
        model = LinUCB()
        before_prediction, before_uncertainty = model.predict(features)
        model.update(features, 0.8)
        after_prediction, after_uncertainty = model.predict(features)
        self.assertGreater(after_prediction, before_prediction)
        self.assertLess(after_uncertainty, before_uncertainty)
        self.assertEqual(LinUCB.from_dict(model.to_dict()).count, 1)

    def test_softmax_probabilities_and_engine_switch_after_training(self):
        index, probabilities = softmax_sample([0.1, 0.2, 0.3], 0.1,
                                               random.Random(2))
        self.assertIn(index, range(3))
        self.assertAlmostEqual(sum(probabilities), 1.0)
        world, encoded, _, platform, _, _, features = fixture()
        model = LinUCB()
        normal_only = PolicySettings(hot_news_score_threshold=100.0)
        cold = DecisionEngine(bandit=model, settings=normal_only,
                              rng=random.Random(1)).decide(
            world, encoded, AgentState(), platform)
        self.assertEqual(cold.parameters["selection_method"], "prior_softmax")
        self.assertIn("selection_probability", cold.score_components)
        for _ in range(5):
            model.update(features, 0.7)
        warm = DecisionEngine(bandit=model, settings=normal_only,
                              rng=random.Random(1)).decide(
            world, encoded, AgentState(), platform)
        self.assertEqual(warm.parameters["selection_method"], "linucb_softmax")
        self.assertIn("selection_probability", warm.score_components)

    def test_only_real_success_can_train_and_model_persists(self):
        with tempfile.TemporaryDirectory() as temp:
            history = ActionHistory(Path(temp) / "state.db")
            world, encoded, candidate, platform, _, _, features = fixture()
            plan = ActionPlan.from_candidate(1, candidate, 0.5, {})
            plan.generated_content = "A grounded reply"
            plan.parameters["policy_features"] = list(features)
            history.save_result(plan, "DRY_RUN")
            with self.assertRaises(ValueError):
                history.save_reward(plan.action_id, 0.7, "test")
            history.save_result(plan, "SUCCEEDED", remote_id="999")
            self.assertEqual(FeedbackCollector(Social(), history).collect_delayed(0), 1)
            repost = ActionPlan.from_candidate(
                1, CandidateAction("REPOST", target_id=11), 0.5, {})
            history.save_result(repost, "SUCCEEDED", remote_id="11")
            self.assertEqual(FeedbackCollector(Social(), history).collect_delayed(0), 0)
            updates = history.load_untrained_rewards()
            self.assertEqual(len(updates), 1)
            model = LinUCB()
            model.update(updates[0][1], updates[0][2])
            history.commit_policy_model("linucb_v1", model.to_dict(), [plan.action_id])
            self.assertEqual(history.load_untrained_rewards(), [])
            self.assertEqual(LinUCB.from_dict(history.load_policy_model("linucb_v1")).count, 1)
            # 10 likes, 2 replies and 1 repost are 90 official base points.
            from math import exp
            expected = 1 - exp(-90 / 50)
            self.assertAlmostEqual(engagement_proxy(Social().counters(999)), expected)
            self.assertAlmostEqual(engagement_proxy({"view_count": 500}),
                                   1 - exp(-500 / 50))
            self.assertIsNone(engagement_proxy({"like_count": float("inf")}))


if __name__ == "__main__":
    unittest.main()
