import unittest

from policy.models import ActionPlan, CandidateAction


class ActionContractTests(unittest.TestCase):
    def test_required_fields_and_text_validation(self):
        candidate = CandidateAction(
            "REPLY", target_id=10, context={"post_body": "hello"},
            rationale="respond", estimated_cost=0.5,
            required_parameters=("body",),
        )
        plan = ActionPlan.from_candidate(1, candidate, 0.4, {"relevance": 0.4})
        self.assertEqual(plan.target_id, 10)
        with self.assertRaises(ValueError):
            plan.validate()
        plan.generated_content = "A grounded reply"
        plan.validate()
        self.assertEqual(plan.action_key, "REPLY:10")

    def test_invalid_candidate_is_rejected(self):
        for kwargs in ({"action_type": "LIKE"},
                       {"action_type": "WAIT", "target_id": 1},
                       {"action_type": "FOLLOW", "target_id": 0}):
            with self.assertRaises(ValueError):
                CandidateAction(**kwargs)


if __name__ == "__main__":
    unittest.main()
