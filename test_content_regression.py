"""Offline checks for source links and bounded content failures."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openai import RateLimitError

from agent.content_generator import ContentGenerationError, ContentGenerator, LLMRateLimitError
from agent.history import ActionHistory
from agent.runtime import AgentRuntime
from policy.models import ActionPlan, CandidateAction
from policy.source_validator import validate_sources
from state.store import StateStore
from test_v02 import FakeObserver, FakeSocial


class RecordingLLM:
    def __init__(self, draft, fail_review=False):
        self.draft = draft
        self.fail_review = fail_review
        self.requests = []

    def ask(self, prompt, system, max_tokens, temperature=None):
        request = json.loads(prompt)
        self.requests.append(request)
        if request.get("task") == "Review a fictional social-media draft before publication.":
            if self.fail_review:
                raise FakeRateLimitError()
            return json.dumps({"decision": "approve", "reason": ""})
        return self.draft


class FakeRateLimitError(RateLimitError):
    def __init__(self):
        self.response = SimpleNamespace(headers={"retry-after": "20"})


def make_plan(action_type, context, target_id=None):
    candidate = CandidateAction(action_type, target_id=target_id,
                                source_event_id="news:1" if action_type == "CREATE_POST" else None,
                                context=context)
    return ActionPlan.from_candidate(1, candidate, 0.5, {})


class ContentRegressionTests(unittest.TestCase):
    def test_news_relative_url_becomes_original_absolute_url_and_matches_source(self):
        article = {
            "site_name": "Kingstonia Herald",
            "title": "The Trap of the Perpetual Emergency",
            "summary": "An opinion piece questions the bill's missing sunset clause.",
            "url": "/kingstonia-herald/articles/the-trap-of-the-perpetual-emergency",
        }
        plan = make_plan("CREATE_POST", {"articles": [article]})
        plan.parameters["content_brief"] = {"popular_tags": ["h2d2"]}
        llm = RecordingLLM(
            "The Herald raises a fair question about the sunset clause. I still back the mandate because preparation matters, while clear public review matters too.")
        output = ContentGenerator(llm).generate(plan)
        expected = ("https://sites.legitreal.com/kingstonia-herald/"
                    "articles/the-trap-of-the-perpetual-emergency")
        self.assertIn(expected, output)
        self.assertEqual(plan.parameters["context"]["articles"][0]["url"], expected)
        self.assertEqual(llm.requests[0]["sources"][0]["url"], expected)
        self.assertLessEqual(len(output), 500)
        self.assertEqual(validate_sources(output, plan.parameters["context"]["articles"])
                         ["matched_source_count"], 1)

    def test_trending_post_uses_internal_source_but_publishes_no_social_link(self):
        article = {
            "site_name": "Legit Social", "title": "Trending post 101",
            "summary": "Someone questions oversight of the H2D2 mandate.",
            "source_type": "trending_post", "post_id": 101,
            "url": "https://social.legitreal.com/twoot/101",
        }
        plan = make_plan("CREATE_POST", {"articles": [article]})
        llm = RecordingLLM(
            "Oversight is a fair question. I still support the mandate, and I'd want its safeguards explained clearly.")
        output = ContentGenerator(llm).generate(plan)
        self.assertNotIn("social.legitreal.com/twoot", output)
        self.assertNotIn("Source: Legit Social", output)
        self.assertEqual(llm.requests[0]["sources"][0]["source_type"], "trending_post")

    def test_hot_branch_rejects_social_only_source_before_llm(self):
        social = {
            "site_name": "Legit Social", "title": "Trending post 101",
            "summary": "A popular H2D2 debate.",
            "source_type": "trending_post",
            "url": "https://social.legitreal.com/twoot/101",
        }
        plan = make_plan("CREATE_POST", {
            "articles": [social], "post_intent": "trend_hitchhike",
        })
        llm = RecordingLLM("This should never be generated.")
        with self.assertRaisesRegex(ContentGenerationError,
                                    "No attributed news source"):
            ContentGenerator(llm).generate(plan)
        self.assertEqual(llm.requests, [])

    def test_hot_news_plan_cites_matched_news_not_the_trending_post(self):
        article = {
            "site_name": "Kingstonia Herald",
            "title": "Hospital staffing before H2D2",
            "summary": "The report covers vacant nursing posts and preparation.",
            "url": "/kingstonia-herald/articles/hospital-staffing-before-h2d2",
        }
        hot_post = {
            "summary": "People are discussing hospital staffing ahead of H2D2.",
            "url": "https://social.legitreal.com/twoot/101",
            "post_id": 101,
        }
        plan = make_plan("CREATE_POST", {
            "articles": [article], "hot_source_post": hot_post,
            "post_intent": "trend_hitchhike",
        })
        llm = RecordingLLM(
            "The staffing report matters to me: I support the mandate, and I want preparation to include the nurses who will carry it out.")
        output = ContentGenerator(llm).generate(plan)
        self.assertIn("https://sites.legitreal.com/kingstonia-herald/"
                      "articles/hospital-staffing-before-h2d2", output)
        self.assertNotIn("social.legitreal.com/twoot", output)
        self.assertIn("discussion signal", llm.requests[0]["reply_focus"])
        self.assertEqual(llm.requests[0]["sources"][0]["site_name"],
                         "Kingstonia Herald")

    def test_model_hashtag_is_removed_before_review_without_extra_llm_call(self):
        article = {
            "site_name": "Kingstonia Herald", "title": "H2D2 update",
            "summary": "The article discusses the H2D2 mandate.",
            "url": "/kingstonia-herald/articles/h2d2-update",
        }
        plan = make_plan("CREATE_POST", {"articles": [article]})
        plan.parameters["content_brief"] = {"popular_tags": ["h2d2"]}
        llm = RecordingLLM(
            "I still support the mandate after reading this update, though the details deserve scrutiny. #ProtectOurCommunity")
        output = ContentGenerator(llm).generate(plan)
        self.assertTrue(plan.parameters["model_hashtags_removed"])
        self.assertIn("#ProtectOurCommunity", plan.parameters["llm_draft_raw"])
        self.assertNotIn("#ProtectOurCommunity", output)
        self.assertIn("#h2d2", output)
        self.assertEqual(len(llm.requests), 2)

    def test_review_429_is_typed_for_runtime_and_does_not_repair(self):
        plan = make_plan("REPLY", {"post_body": "The H2D2 mandate needs clear oversight."},
                         target_id=101)
        llm = RecordingLLM("I support early action, and I agree the oversight should be clear.",
                           fail_review=True)
        with self.assertRaises(LLMRateLimitError):
            ContentGenerator(llm).generate(plan)
        self.assertEqual(len(llm.requests), 2)

    def test_runtime_does_not_repeat_full_generation_after_429(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "state.db"
            runtime = AgentRuntime(
                FakeObserver(), StateStore(db), ActionHistory(db), FakeSocial(),
                llm_client=object(), mode="dry-run", policy_mode="heuristic",
                save_raw=lambda raw: Path(temp) / "raw.json",
            )
            def decide(world, encoded, agent, platform, excluded_action_types):
                if "REPLY" in excluded_action_types:
                    return make_plan("WAIT", {})
                return make_plan(
                    "REPLY", {"post_body": "The H2D2 referendum needs careful debate."},
                    target_id=101)
            runtime.engine.decide = decide
            with patch("agent.runtime.ContentGenerator.generate",
                       side_effect=LLMRateLimitError("LLM unavailable: RateLimitError")) as generate:
                result = runtime.run_round()
            self.assertEqual(generate.call_count, 1)
            self.assertEqual(result["actions"][0]["status"], "FAILED")
            self.assertIn("RateLimitError", result["actions"][0]["error"])


if __name__ == "__main__":
    unittest.main()
