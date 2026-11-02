import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from agent.content_generator import ContentGenerationError, ContentGenerator
from agent.executor import ActionExecutor
from agent.history import ActionHistory
from agent.runtime import AgentRuntime
from agent.state import AgentState, PlatformState
from policy.bandit import FEATURE_DIM
from policy import CandidateGenerator, DecisionEngine, PolicyScorer
from policy.models import ActionPlan, CandidateAction
from policy.relevance import explicit_yes_alignment, select_popular_tags
from state.builder import StateBuilder
from state.encoder import StateEncoder
from state.normalizer import normalize_news, normalize_posts
from state.observer import RawSnapshot
from state.store import StateStore


class FakeObserver:
    def __init__(self): self.calls = 0
    def collect(self):
        self.calls += 1
        return RawSnapshot(
            started_at=f"2026-11-01T12:{self.calls:02d}:00+00:00",
            finished_at=f"2026-11-01T12:{self.calls:02d}:05+00:00",
            trending_posts=[{
                "post_id": 101, "author_id": 9, "body": "A source discusses the H2D2 referendum.",
                "created_at": "2026-11-01T11:30:00+00:00", "kind": "post",
                "global_hot_score": 60.0, "rank_position": 1,
            }], latest_posts=[], news_articles=[{
                "site": {"sub_url": "news", "name": "News Site"},
                "slug": "h2d2", "story_id": "1", "title": "H2D2 referendum update",
                "summary": "The referendum campaign changed today.",
                "published_at": "2026-11-01T11:00:00+00:00",
                "url": "https://example.test/transport",
            }],
        )


class FakeLLM:
    def __init__(self, answer=None):
        self.answer = answer
        self.calls = 0
        self.requests = []
    def ask(self, prompt, system, max_tokens):
        self.calls += 1
        request = json.loads(prompt)
        self.requests.append(request)
        if request.get("task") == "Review a fictional social-media draft before publication.":
            return json.dumps({"decision": "approve", "reason": ""})
        if self.answer is not None: return self.answer
        return "The H2D2 referendum deserves a clear discussion about public health and choice."


class FakeSocial:
    def __init__(self, locked=False): self.locked = locked; self.lock_calls = 0
    def feed_lock(self): self.lock_calls += 1; return {"locked": self.locked}
    def counters(self, post_id): return {"like_count": 2}
    def popular_tags(self):
        return [{"name": "voteno", "post_count": 500},
                {"name": "h2d2", "tag_id": 1, "post_count": 400},
                {"name": "referendum", "post_count": 200},
                {"name": "voteyes", "tag_id": 2, "post_count": 70}]
    def tag_posts(self, tag_id, limit=15):
        return [{"post_id": 102, "author_id": 10, "body": "H2D2 referendum update",
                 "created_at": "2026-11-01T11:45:00+00:00", "kind": "post"}]
    def get_post(self, post_id):
        return {"post_id": post_id, "author_id": 11,
                "body": "H2D2 referendum discussion", "root_post_id": 100,
                "created_at": "2026-11-01T11:50:00+00:00"}
    def replies(self, post_id, limit=3): return []
    def following(self, account_id, limit=100): return []


class FakeAPIError(Exception):
    def __init__(self, status, retry_after=None):
        self.status = status; self.retry_after = retry_after


class FakeAccount:
    account_id = 500
    def __init__(self, error=None): self.calls = []; self.error = error
    def _write(self, name, *args):
        self.calls.append((name, args))
        if self.error:
            error = self.error
            self.error = None
            raise error
        return {"post_id": 999} if name in ("post", "reply", "repost") else None
    def post(self, body): return self._write("post", body)
    def reply(self, post_id, body, root_post_id=None):
        return self._write("reply", post_id, body, root_post_id)
    def like(self, post_id): return self._write("like", post_id)
    def repost(self, post_id): return self._write("repost", post_id)
    def follow(self, account_id): return self._write("follow", account_id)
    def mention(self, post_id, account_id): return self._write("mention", post_id, account_id)
    def my_likes(self, ids): return set()
    def my_reposts(self, ids): return {}
    def feed_home(self, limit=30):
        return [{"post_id": 103, "author_id": 12,
                 "body": "H2D2 referendum home post",
                 "created_at": "2026-11-01T11:50:00+00:00", "kind": "post"}]
    def notifications(self, unresolved=True, limit=10):
        return [{"post_id": 104}]


def plan(snapshot_id, action_type, target=None, event=None, text=None):
    candidate = CandidateAction(action_type, target_id=target, source_event_id=event)
    action = ActionPlan.from_candidate(snapshot_id, candidate, 0.8, {"test": 0.8})
    action.generated_content = text
    return action


class V02Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "state.db"
        self.store = StateStore(self.db)
        self.history = ActionHistory(self.db)

    def world(self):
        raw = FakeObserver().collect()
        posts = normalize_posts(asdict(raw))
        articles, events = normalize_news(asdict(raw))
        sid = self.store.save_snapshot(Path(self.tmp.name) / "raw1.json", asdict(raw),
                                       posts, articles, events)
        return StateBuilder(self.store).build(sid), articles

    def test_snapshot_state_encoder_candidate_score_and_plan(self):
        world, articles = self.world()
        encoded = StateEncoder().encode(world)
        self.assertEqual(len(encoded.global_vector), 12)
        platform = PlatformState(articles_by_id={a.article_id: asdict(a) for a in articles})
        candidates = CandidateGenerator().generate(world, encoded, AgentState(), platform)
        self.assertIn("CREATE_POST", {c.action_type for c in candidates})
        self.assertIn("REPLY", {c.action_type for c in candidates})
        self.assertIn("WAIT", {c.action_type for c in candidates})
        self.assertTrue(all(0 <= PolicyScorer().score(c, world, encoded, AgentState(), platform).total_score <= 1
                            for c in candidates))
        selected = DecisionEngine().decide(world, encoded, AgentState(), platform)
        self.assertIn(selected.action_type, {"CREATE_POST", "REPLY", "WAIT"})
        self.assertEqual(selected.snapshot_id, world.metadata["snapshot_id"])

    def test_llm_content_uses_source_and_rejects_unsupported_number(self):
        p = plan(1, "CREATE_POST", event="story:1")
        p.parameters["context"] = {"articles": [{"title": "H2D2 referendum update",
                              "summary": "The referendum is approaching.",
                              "site_name": "News Site", "url": "https://example.test"}]}
        p.parameters["content_brief"] = {"popular_tags": ["h2d2", "voteyes"]}
        llm = FakeLLM()
        generated = ContentGenerator(llm).generate(p)
        self.assertIn("clear discussion", generated)
        self.assertIn("https://example.test", generated)
        self.assertIn("#h2d2 #voteyes Source:", generated)
        self.assertTrue(generated.endswith("https://example.test"))
        self.assertEqual(p.parameters["selected_tags"], ["h2d2", "voteyes"])
        self.assertEqual(p.parameters["generated_content"], generated)
        # CREATE_POST has its own perspective and source-writing instructions.
        self.assertEqual(llm.requests[0]["task"], "Write one original social media post")
        post_constraints=" ".join(llm.requests[0]["constraints"])
        self.assertIn("one meaningful issue",post_constraints)
        self.assertIn("clear and distinctive perspective",post_constraints)
        self.assertIn("outspoken, independent-minded person",post_constraints)
        self.assertIn("Never invent numbers, quotes",post_constraints)
        self.assertNotIn("PERSONA:", post_constraints)
        self.assertNotIn("follow_hint_allowed", llm.requests[0])
        self.assertFalse(any("hint that readers can follow" in constraint
                             for constraint in llm.requests[0]["constraints"]))
        self.assertIn("Do not follow a fixed opening",post_constraints)
        self.assertNotIn("central argument", " ".join(llm.requests[0]["constraints"]))
        with self.assertRaises(ContentGenerationError):
            ContentGenerator(FakeLLM("H2D2 has killed 999 people.")).generate(p)

    def test_hot_trend_post_generation_attributes_social_claim_as_social_source(self):
        p = plan(1, "CREATE_POST", event="hot-trending:101")
        p.parameters["context"] = {"articles": [{
            "title": "Trending post 101",
            "summary": "The 30 Oct draft applies after the vote.",
            "site_name": "Legit Social",
            "url": "https://social.legitreal.com/twoot/101",
            "source_type": "trending_post", "post_id": 101,
        }]}
        llm = FakeLLM("The draft's terms matter to ordinary people, and I want clear limits before rules affect everyday access.")
        generated = ContentGenerator(llm).generate(p)
        self.assertNotIn("Source: Legit Social", generated)
        self.assertNotIn("https://social.legitreal.com/twoot/101", generated)
        self.assertEqual(llm.requests[0]["sources"][0]["source_type"], "trending_post")
        self.assertIn("not proof that its factual claims are true", llm.requests[0]["reply_focus"])
        self.assertIn("account's Vote Yes position",
                      " ".join(llm.requests[0]["constraints"]))

    def test_create_post_repairs_format_failure_once_and_keeps_diagnostics(self):
        p = plan(1, "CREATE_POST", event="story:1")
        p.parameters["context"] = {"articles": [{
            "title": "Harbour test corrected after lab contamination",
            "summary": "Repeat sampling found no H2D2; the earlier signal came from lab contamination.",
            "site_name": "News Site", "url": "https://example.test/story"}]}
        p.parameters["content_brief"] = {"popular_tags": []}

        class RepairLLM(FakeLLM):
            def ask(self, prompt, system, max_tokens):
                request = json.loads(prompt)
                self.requests.append(request)
                if request.get("task") == "Review a fictional social-media draft before publication.":
                    return json.dumps({"decision": "approve", "reason": ""})
                if request.get("task", "").startswith("Repair a social-media post"):
                    return "That Harbour result was a false alarm traced to lab contamination, not proof of a wider failure. Vote Yes means taking clear public-health checks seriously and keeping people informed."
                return "x" * 351

        llm = RepairLLM()
        generated = ContentGenerator(llm).generate(p)
        self.assertIn("lab contamination", generated)
        self.assertTrue(p.parameters["format_validation"]["repaired"])
        self.assertIn("over_350_characters",
                      p.parameters["format_validation"]["initial_issues"])
        self.assertTrue(any(request.get("task", "").startswith("Repair a social-media post")
                            for request in llm.requests))
        repair_request = next(request for request in llm.requests
                              if request.get("task", "").startswith("Repair a social-media post"))
        repair_requirements = " ".join(repair_request["requirements"])
        self.assertIn("supports Vote Yes", repair_requirements)
        self.assertIn("Do not include @mentions or hashtags", repair_requirements)

    def test_reply_repairs_format_failure_once_and_keeps_diagnostics(self):
        p = plan(1, "REPLY", target=101)
        p.parameters["context"] = {
            "post_body": "I oppose the H2D2 mandate because oversight is unclear. What safeguard would change your mind?"}
        p.parameters["content_brief"] = {"tone": "plainspoken"}

        class RepairLLM(FakeLLM):
            def ask(self, prompt, system, max_tokens):
                request = json.loads(prompt)
                self.requests.append(request)
                if request.get("task") == "Review a fictional social-media draft before publication.":
                    return json.dumps({"decision": "approve", "reason": ""})
                if request.get("task", "").startswith("Revise this reply"):
                    return "For me, regular public reviews would address the oversight concern; with that safeguard, I could support the mandate."
                return "x" * 351

        llm = RepairLLM()
        generated = ContentGenerator(llm).generate(p)
        self.assertIn("oversight concern", generated)
        self.assertTrue(p.parameters["format_validation"]["repaired"])
        self.assertIn("over_350_characters",
                      p.parameters["format_validation"]["initial_issues"])
        self.assertTrue(p.parameters["reply_repair_format_validation"]["passed"])
        repair_request = next(request for request in llm.requests
                              if request.get("task", "").startswith("Revise this reply"))
        repair_requirements = " ".join(repair_request["requirements"])
        self.assertIn("supports Vote Yes", repair_requirements)
        self.assertIn("Do not include @mentions or hashtags", repair_requirements)

    def test_llm_reply_uses_its_own_prompt_and_answers_the_post(self):
        p = plan(1, "REPLY", target=101)
        p.parameters["context"] = {
            "post_body": "I oppose the H2D2 mandate because oversight is unclear. What safeguard would change your mind?"
        }
        p.parameters["content_brief"] = {"tone": "plainspoken",
                                          "follow_hint_allowed": True}
        llm = FakeLLM(
            "For me, regular public reviews would address the oversight concern; with that safeguard, I could support the mandate."
        )
        generated = ContentGenerator(llm).generate(p)
        self.assertIn("oversight concern", generated)
        self.assertEqual(llm.requests[0]["task"], "Write one original social media reply")
        self.assertEqual(llm.requests[0]["tone"], "plainspoken")
        self.assertIn("central argument", " ".join(llm.requests[0]["constraints"]))
        stance_constraints = " ".join(llm.requests[0]["constraints"])
        self.assertIn("account's Vote Yes position", stance_constraints)
        self.assertIn("do not force a voting declaration", stance_constraints)
        self.assertNotIn("PERSONA:", stance_constraints)
        self.assertTrue(any("hint that readers can follow" in constraint
                            for constraint in llm.requests[0]["constraints"]))
        self.assertIn("sources", llm.requests[0])
        self.assertEqual(llm.requests[0]["sources"], [])

    def test_semantic_review_rate_limit_uses_runtime_retry_instead_of_repair(self):
        p = plan(1, "REPLY", target=101)
        p.parameters["context"] = {
            "post_body": "There are forty vacant nursing posts, and H2D2 is two islands away."
        }

        class ReviewUnavailableLLM(FakeLLM):
            def ask(self, prompt, system, max_tokens):
                request = json.loads(prompt)
                self.requests.append(request)
                if request.get("task") == "Review a fictional social-media draft before publication.":
                    raise RuntimeError("rate limited")
                return "Those vacancies matter; I support the mandate because preparation has to start before H2D2 reaches the hospital door."

        llm = ReviewUnavailableLLM()
        with self.assertRaisesRegex(ContentGenerationError,
                                    "Semantic review unavailable: RuntimeError"):
            ContentGenerator(llm).generate(p)
        self.assertEqual(len(llm.requests), 2)
        self.assertFalse(any(request.get("task", "").startswith("Revise this reply")
                             for request in llm.requests))

    def test_popular_tags_exclude_opposition_and_unrelated_tags(self):
        tags = FakeSocial().popular_tags() + [{"name": "turtle", "post_count": 1000}]
        self.assertEqual(select_popular_tags(tags), ("h2d2", "voteyes"))
        self.assertEqual(explicit_yes_alignment("H2D2 referendum #VoteYes"), 1.0)
        self.assertEqual(explicit_yes_alignment("H2D2 referendum #VoteNo"), 0.0)

    def test_personal_and_tag_feeds_join_snapshot_with_source_labels(self):
        raw = FakeObserver().collect()
        runtime = AgentRuntime(FakeObserver(), self.store, self.history,
                               FakeSocial(), account=FakeAccount())
        runtime._enrich_observation(raw)
        posts = {post.post_id: post for post in normalize_posts(asdict(raw))}
        self.assertIn("tagged", posts[102].feed_sources)
        self.assertIn("home", posts[103].feed_sources)
        self.assertIn("notification", posts[104].feed_sources)

    def test_live_reply_refreshes_target_and_preserves_thread_root(self):
        observer, social, account = FakeObserver(), FakeSocial(), FakeAccount()
        def save_raw(raw):
            path = Path(self.tmp.name) / "reply.json"
            path.write_text(json.dumps(asdict(raw)))
            return path
        runtime = AgentRuntime(observer, self.store, self.history, social,
                               llm_client=FakeLLM(), account=account,
                               mode="live", policy_mode="heuristic", save_raw=save_raw)
        class ReplyOnlyCandidates:
            def generate(self, world, encoded, agent, platform):
                return tuple(candidate for candidate in CandidateGenerator().generate(
                    world, encoded, agent, platform) if candidate.action_type == "REPLY")
        runtime.engine.generator = ReplyOnlyCandidates()
        result = runtime.run_round()
        self.assertEqual(result["actions"][0]["status"], "SUCCEEDED")
        self.assertEqual(account.calls[0][0], "reply")
        self.assertIsInstance(account.calls[0][1][0], int)
        self.assertEqual(account.calls[0][1][-1], 100)

    def test_unresolved_reply_notification_is_resolved_after_selected_reply(self):
        class ReplyNotificationAccount(FakeAccount):
            def notifications(self, unresolved=True, limit=10):
                self.requested_notification_limit = limit
                return [{"notification_id": 204, "kind": "reply",
                         "post": {"post_id": 104}}]
            def resolve_notification(self, notification_id):
                self.resolved_notifications = getattr(self, "resolved_notifications", [])
                self.resolved_notifications.append(notification_id)

        observer, social, account = FakeObserver(), FakeSocial(), ReplyNotificationAccount()
        def save_raw(raw):
            path = Path(self.tmp.name) / "priority_reply.json"
            path.write_text(json.dumps(asdict(raw)))
            return path
        runtime = AgentRuntime(observer, self.store, self.history, social,
                               llm_client=FakeLLM(), account=account,
                               mode="live", policy_mode="heuristic", save_raw=save_raw)
        class NotificationReplyCandidates:
            def generate(self, world, encoded, agent, platform):
                return tuple(candidate for candidate in CandidateGenerator().generate(
                    world, encoded, agent, platform)
                    if candidate.action_type == "REPLY"
                    and candidate.context.get("notification_id") == 204)
        runtime.engine.generator = NotificationReplyCandidates()
        result = runtime.run_round()
        self.assertEqual(result["actions"][0]["action"], "REPLY")
        self.assertEqual(account.calls[0][0], "reply")
        self.assertEqual(account.calls[0][1][0], 104)
        self.assertEqual(account.requested_notification_limit, 50)
        self.assertEqual(account.resolved_notifications, [204])

    def test_dry_run_does_not_write_or_check_feed_lock(self):
        social, account = FakeSocial(), FakeAccount()
        result = ActionExecutor(social, account, self.history).execute(
            plan(1, "REPLY", target=101, text="Grounded reply"), "dry-run")
        self.assertEqual(result.status, "DRY_RUN")
        self.assertEqual(account.calls, [])
        self.assertEqual(social.lock_calls, 0)
        self.assertIn("REPLY:101", self.history.load_state(include_dry_runs=True).completed_keys)
        self.assertNotIn("REPLY:101", self.history.load_state().completed_keys)

    def test_recent_failed_action_has_ten_minute_cooldown(self):
        from datetime import datetime, timedelta, timezone

        recent = plan(1, "REPLY", target=777, text="Grounded reply")
        self.history.save_result(recent, "FAILED", error="generation failed")
        self.assertIn("REPLY:777", self.history.load_state().completed_keys)

        old = plan(1, "REPLY", target=778, text="Grounded reply")
        old.created_at = (datetime.now(timezone.utc) - timedelta(minutes=11)).isoformat()
        self.history.save_result(old, "FAILED", error="generation failed")
        self.assertNotIn("REPLY:778", self.history.load_state().completed_keys)

    def test_mock_sdk_actions_and_restart_dedup(self):
        social, account = FakeSocial(), FakeAccount()
        executor = ActionExecutor(social, account, self.history)
        for kind, target, event, body in [
            ("CREATE_POST", None, "story:1", "News quote"),
            ("REPLY", 101, None, "Reply quote"),
            ("LIKE", 101, None, None),
            ("REPOST", 101, None, None),
            ("FOLLOW", 9, None, None),
        ]:
            self.assertEqual(executor.execute(plan(1, kind, target, event, body), "live").status,
                             "SUCCEEDED")
        self.assertEqual([name for name, _ in account.calls],
                         ["post", "reply", "like", "repost", "follow"])
        restarted = ActionHistory(self.db)
        self.assertIn("LIKE:101", restarted.load_state().completed_keys)
        duplicate = ActionExecutor(social, account, restarted).execute(
            plan(2, "LIKE", 101), "live")
        self.assertEqual(duplicate.status, "BLOCKED")
        self.assertEqual(len(account.calls), 5)

        mention = plan(5, "MENTION", target=9)
        mention.parameters["context"] = {"mention_on_post_id": 999}
        self.assertEqual(executor.execute(mention, "live").status, "SUCCEEDED")
        self.assertEqual(account.calls[-1][0], "mention")
        mention_state = self.history.load_state()
        self.assertIn(9, mention_state.mentioned_account_ids)
        self.assertIn(999, mention_state.mentioned_post_ids)

    def test_feed_lock_rate_limit_and_unknown_timeout(self):
        account = FakeAccount()
        locked = ActionExecutor(FakeSocial(True), account, self.history).execute(
            plan(1, "LIKE", 101), "live")
        self.assertEqual(locked.status, "BLOCKED")
        self.assertEqual(account.calls, [])
        waits = []
        account.error = FakeAPIError(429, 2)
        limited = ActionExecutor(FakeSocial(), account, self.history, sleep=waits.append).execute(
            plan(2, "LIKE", 102), "live")
        self.assertEqual(limited.status, "SUCCEEDED")
        self.assertEqual(waits, [2])
        account.error = TimeoutError()
        unknown = ActionExecutor(FakeSocial(), account, self.history).execute(
            plan(3, "REPLY", 103, text="A reply"), "live")
        self.assertEqual(unknown.status, "UNKNOWN")
        self.assertEqual(ActionExecutor(FakeSocial(), account, self.history).execute(
            plan(4, "REPLY", 103, text="A reply"), "live").status, "BLOCKED")

    def test_authorization_conflict_and_duplicate_content(self):
        account = FakeAccount()
        executor = ActionExecutor(FakeSocial(), account, self.history)
        for index, status in enumerate((401, 403, 409), start=1):
            account.error = FakeAPIError(status)
            result = executor.execute(plan(index, "LIKE", 200 + index), "live")
            self.assertEqual(result.status, "BLOCKED")
        first = plan(10, "CREATE_POST", event="story:a", text="A source quote")
        second = plan(11, "CREATE_POST", event="story:b", text="A source quote")
        self.assertEqual(executor.execute(first, "live").status, "SUCCEEDED")
        self.assertEqual(executor.execute(second, "live").status, "BLOCKED")
        self.assertEqual([name for name, _ in account.calls].count("post"), 1)

    def test_invalid_llm_output_skips_runtime_write(self):
        observer = FakeObserver()
        def save_raw(raw):
            path = Path(self.tmp.name) / "bad_llm.json"
            path.write_text(json.dumps(asdict(raw)))
            return path
        account = FakeAccount()
        runtime = AgentRuntime(observer, self.store, self.history, FakeSocial(),
                               llm_client=FakeLLM("Invented fact"), account=account,
                               mode="dry-run", save_raw=save_raw)
        class TextOnlyCandidates:
            def generate(self, world, encoded, agent, platform):
                return tuple(candidate for candidate in CandidateGenerator().generate(
                    world, encoded, agent, platform)
                    if candidate.action_type in ("CREATE_POST", "REPLY"))
        runtime.engine.generator = TextOnlyCandidates()
        result = runtime.run_round()
        self.assertEqual(result["actions"][0]["status"], "FAILED")
        self.assertEqual(account.calls, [])

    def test_runtime_multiple_dry_run_rounds(self):
        observer = FakeObserver()
        counter = iter(range(100))
        def save_raw(raw):
            path = Path(self.tmp.name) / f"raw{next(counter)}.json"
            path.write_text(json.dumps(asdict(raw)))
            return path
        social = FakeSocial()
        runtime = AgentRuntime(observer, self.store, self.history, social,
                               llm_client=FakeLLM(), save_raw=save_raw)
        result = runtime.run(max_rounds=2, interval=0)
        self.assertEqual(len(result), 2)
        self.assertEqual(observer.calls, 2)
        self.assertTrue(all(r["actions"] for r in result))
        with self.history._connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0], 2)
            self.assertGreater(db.execute("SELECT COUNT(*) FROM agent_actions").fetchone()[0], 0)
            keys = [row[0] for row in db.execute(
                "SELECT action_key FROM agent_actions WHERE status='DRY_RUN'")]
            self.assertEqual(len(keys), len(set(keys)))

    def test_runtime_does_not_train_legacy_engagement_model(self):
        previous = plan(1, "REPLY", target=999, text="Earlier sourced reply")
        previous.parameters["policy_features"] = [1.0] + [0.0] * (FEATURE_DIM - 1)
        self.history.save_result(previous, "SUCCEEDED", remote_id="888")
        observer = FakeObserver()
        def save_raw(raw):
            path = Path(self.tmp.name) / "round.json"
            path.write_text(json.dumps(asdict(raw)))
            return path
        runtime = AgentRuntime(observer, self.store, self.history, FakeSocial(),
                               llm_client=FakeLLM(), save_raw=save_raw,
                               reward_horizon_minutes=0)
        result = runtime.run_round()
        self.assertEqual(result["policy_samples"], 0)
        self.assertIsNone(self.history.load_policy_model("linucb_official_proxy_v1"))
        self.assertEqual(self.history.load_untrained_rewards(), [])


if __name__ == "__main__":
    unittest.main()
