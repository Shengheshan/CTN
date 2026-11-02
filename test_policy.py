import unittest
import random

from agent.state import AgentState, PlatformState
from policy import CandidateGenerator, DecisionEngine, PolicyScorer
from policy.models import ActionPlan, CandidateAction
from state.builder import EventState, GlobalState, PostState, WorldState
from state.encoder import StateEncoder


class PolicyV02Tests(unittest.TestCase):
    def setUp(self):
        self.world = WorldState(
            metadata={"snapshot_id": 1, "finished_at": "2026-11-01T12:00:00Z"},
            global_state=GlobalState(1, 1, 0, 0, 0, None),
            posts=[PostState(5, 9, "I will vote YES in the H2D2 referendum", 80.0, 1,
                             ["trending"], {"hot_velocity": 1.0}, [],
                             "2026-11-01T11:30:00Z")],
            events=[], topics=[],
        )
        self.encoded = StateEncoder().encode(self.world)

    def test_action_target_context_and_score(self):
        candidates = CandidateGenerator().generate(
            self.world, self.encoded, AgentState(), PlatformState())
        reply = next(c for c in candidates if c.action_type == "REPLY")
        self.assertEqual(reply.target_id, 5)
        self.assertEqual(reply.context["post_body"], "I will vote YES in the H2D2 referendum")
        score = PolicyScorer().score(reply, self.world, self.encoded,
                                     AgentState(), PlatformState())
        self.assertTrue(0 <= score.total_score <= 1)
        self.assertIn("cost_penalty", score.score_components)
        self.assertLess(score.score_components["action_priority"], 0)
        plan = DecisionEngine().decide(self.world, self.encoded,
                                       AgentState(), PlatformState())
        self.assertEqual(plan.snapshot_id, 1)
        self.assertTrue(plan.action_id)
        trace = plan.parameters["decision_trace"]
        selected = next(item for item in trace["candidates"] if item["selected"])
        self.assertAlmostEqual(trace["content_confidence"],
                               selected["opportunity"]["evidence_coverage"] / 0.85
                               if plan.action_type in ("CREATE_POST", "REPLY") else 0.0)
        self.assertEqual(trace["follow_hint_allowed"],
                         plan.action_type in ("CREATE_POST", "REPLY")
                         and trace["content_confidence"] >= 0.80)

    def test_endorsement_requires_alignment_and_lock_forces_wait(self):
        plain = CandidateGenerator().generate(self.world, self.encoded,
                                              AgentState(), PlatformState())
        self.assertNotIn("LIKE", {c.action_type for c in plain})
        aligned = CandidateGenerator().generate(
            self.world, self.encoded, AgentState(),
            PlatformState(alignment_by_post={5: 1.0}))
        self.assertTrue({"LIKE", "REPOST", "FOLLOW"}.issubset(
            {c.action_type for c in aligned}))
        locked = CandidateGenerator().generate(
            self.world, self.encoded, AgentState(), PlatformState(feed_locked=True))
        self.assertEqual([c.action_type for c in locked], ["WAIT"])

    def test_irrelevant_post_is_excluded_and_mention_needs_own_post(self):
        turtle = WorldState(
            metadata=self.world.metadata, global_state=self.world.global_state,
            posts=[PostState(6, 10, "Green turtle returns to the sea", 40.0, 2,
                             ["latest"], {}, [], "2026-11-01T11:30:00Z")],
            events=[], topics=[])
        candidates = CandidateGenerator().generate(
            turtle, StateEncoder().encode(turtle), AgentState(), PlatformState())
        self.assertEqual({candidate.action_type for candidate in candidates}, {"WAIT"})

        agent = AgentState(own_published_posts=[
            {"post_id": 777, "source_text": "H2D2 referendum news"}])
        platform = PlatformState(alignment_by_post={5: 1.0})
        aligned = CandidateGenerator().generate(self.world, self.encoded, agent, platform)
        self.assertIn("MENTION", {candidate.action_type for candidate in aligned})
        agent.mentioned_post_ids.add(777)
        after = CandidateGenerator().generate(self.world, self.encoded, agent, platform)
        self.assertNotIn("MENTION", {candidate.action_type for candidate in after})

    def test_bandit_cold_start_samples_more_than_one_action_type(self):
        engine = DecisionEngine(policy_mode="bandit", rng=random.Random(2))
        platform = PlatformState(alignment_by_post={5: 1.0})
        chosen = {engine.decide(self.world, self.encoded, AgentState(),
                                platform).action_type for _ in range(30)}
        self.assertGreater(len(chosen), 1)

    def test_heuristic_softmax_prefers_posts_but_keeps_randomness(self):
        class Generator:
            def generate(self, world, encoded, agent, platform):
                return (CandidateAction("CREATE_POST", source_event_id="event-1"),
                        CandidateAction("REPLY", target_id=5,
                                        context={"post_body": "H2D2 discussion"}))
        engine = DecisionEngine(generator=Generator(), policy_mode="heuristic",
                                rng=random.Random(19))
        selected = [engine.decide(self.world, self.encoded, AgentState(),
                                  PlatformState()).action_type for _ in range(40)]
        self.assertGreater(selected.count("CREATE_POST"), selected.count("REPLY"))
        self.assertGreater(selected.count("REPLY"), 0)
        self.assertEqual(engine.temperature, 0.35)
        self.assertEqual(engine.settings.minimum_post_coverage, 0.55)
        self.assertEqual(engine.settings.create_post_priority, 0.35)

    def test_stale_referendum_post_is_not_an_action_target(self):
        old_world = WorldState(
            metadata=self.world.metadata, global_state=self.world.global_state,
            posts=[PostState(8, 12, "I will vote YES in the H2D2 referendum",
                             90.0, 1, ["trending"], {}, [],
                             "2026-10-25T11:30:00Z")], events=[], topics=[])
        candidates = CandidateGenerator().generate(
            old_world, StateEncoder().encode(old_world),
            AgentState(), PlatformState(alignment_by_post={8: 1.0}))
        self.assertEqual({candidate.action_type for candidate in candidates}, {"WAIT"})

    def test_post_candidate_skips_procedural_news_but_keeps_policy_news(self):
        world=WorldState(metadata={**self.world.metadata,"snapshot_id":3},
            global_state=self.world.global_state,posts=[],events=[
                EventState("story:ID",["admin"],[]),
                EventState("story:POLICY",["policy"],[])],topics=[])
        encoded=StateEncoder().encode(world)
        platform=PlatformState(articles_by_id={
            "admin":{"title":"National ID card not required to vote in H2D2 referendum",
                     "summary":"Electoral Commission says staff must not record vaccination status at polls",
                     "published_at":"2026-11-01T11:30:00Z"},
            "policy":{"title":"Hospitals prepare for Dolphin Flu outbreak",
                       "summary":"Vaccination mandate and public-space access are debated",
                       "published_at":"2026-11-01T11:30:00Z"}})
        candidates=CandidateGenerator().generate(world,encoded,AgentState(),platform)
        posts=[c for c in candidates if c.action_type=="CREATE_POST"]
        self.assertEqual([c.source_event_id for c in posts],["story:POLICY"])

    def test_post_coverage_scheduler_forces_eligible_post_below_target(self):
        class Generator:
            def generate(self, world, encoded, agent, platform):
                return (CandidateAction("CREATE_POST", context={"articles": []}),
                        CandidateAction("REPLY", target_id=5,
                                        context={"source_post_id": 5, "post_body": "H2D2"}),
                        CandidateAction("WAIT"))
        agent = AgentState(recent_actions=[{"action_type":"REPLY","status":"SUCCEEDED"}])
        engine = DecisionEngine(generator=Generator(), policy_mode="heuristic")
        plan = engine.decide(self.world, self.encoded, agent, PlatformState())
        self.assertEqual(plan.action_type, "CREATE_POST")
        self.assertTrue(plan.parameters["decision_trace"]["post_coverage_forced"])

    def test_reply_to_own_post_is_generated_for_non_campaign_text(self):
        world = WorldState(
            metadata=self.world.metadata, global_state=self.world.global_state,
            posts=[PostState(6, 10, "The ferry was delayed again this morning.",
                             12.0, 3, ["latest"], {}, [],
                             "2026-11-01T11:45:00Z")], events=[], topics=[])
        encoded = StateEncoder().encode(world)
        agent = AgentState(own_published_posts=[{"post_id": 777}])
        platform = PlatformState(reply_to_post_ids={6: 777})
        candidates = CandidateGenerator().generate(world, encoded, agent, platform)
        reply = next(c for c in candidates if c.action_type == "REPLY")
        self.assertTrue(reply.context["reply_to_own_post"])
        self.assertEqual(reply.context["own_post_id_replied_to"], 777)

    def test_reply_notification_marks_candidate_priority_without_parent_in_feed(self):
        world = WorldState(
            metadata=self.world.metadata, global_state=self.world.global_state,
            posts=[PostState(7, 10, "I still have a question about that.",
                             12.0, 3, ["notification"], {}, [],
                             "2026-11-01T11:45:00Z")], events=[], topics=[])
        encoded = StateEncoder().encode(world)
        platform = PlatformState(priority_reply_post_ids={7},
                                 notification_ids_by_post={7: 204})
        candidates = CandidateGenerator().generate(
            world, encoded, AgentState(), platform)
        reply = next(c for c in candidates if c.action_type == "REPLY")
        self.assertTrue(reply.context["reply_to_own_post"])
        self.assertEqual(reply.context["notification_id"], 204)

    def test_reply_to_own_post_is_prioritized_without_hard_override(self):
        class Generator:
            def generate(self, world, encoded, agent, platform):
                return (CandidateAction("CREATE_POST"),
                        CandidateAction("REPLY", target_id=6,
                                        context={"reply_to_own_post": True}),
                        CandidateAction("REPLY", target_id=5),
                        CandidateAction("WAIT"))
        engine = DecisionEngine(generator=Generator(), policy_mode="heuristic")
        plan = engine.decide(self.world, self.encoded, AgentState(), PlatformState())
        trace = plan.parameters["decision_trace"]
        self.assertTrue(trace["priority_reply_available"])
        self.assertFalse(trace["priority_reply_override"])
        self.assertFalse(trace["post_coverage_forced"])
        self.assertTrue(any(
            item["key"] == "REPLY:6" and item["selection_probability"] > 0
            for item in trace["candidates"]))

    def test_hot_related_news_is_eligible_and_uses_real_news_source(self):
        from policy.settings import PolicySettings
        observed = "2026-11-01T12:00:00Z"
        news_url = "/daily-kingstonia/articles/crown-harbour-signal"
        world = WorldState(
            metadata={**self.world.metadata, "finished_at": observed},
            global_state=self.world.global_state,
            posts=[PostState(80, 21,
                "The H2D2 Crown Harbour wastewater signal was a false alarm after retesting.",
                55.0, 1, ["trending"], {}, [], "2026-11-01T11:50:00Z")],
            events=[EventState("story:WATER", ["article:WATER"], [])], topics=[])
        encoded = StateEncoder().encode(world)
        platform = PlatformState(articles_by_id={"article:WATER": {
            "article_id": "article:WATER",
            "site_name": "Daily Kingstonia",
            "title": "Crown Harbour wastewater H2D2 signal ruled false alarm",
            "summary": "Repeat sampling at Crown Harbour found no H2D2 after lab contamination.",
            "published_at": "2026-11-01T11:30:00Z", "url": news_url}})
        settings = PolicySettings(hot_news_score_threshold=50)
        candidates = CandidateGenerator(settings).generate(
            world, encoded, AgentState(), platform)
        trend = next(c for c in candidates if c.source_event_id == "hot-trending:80")
        self.assertEqual(trend.context["post_intent"], "trend_hitchhike")
        self.assertEqual(trend.context["articles"][0]["url"], news_url)
        self.assertNotEqual(trend.context["articles"][0].get("source_type"), "trending_post")
        self.assertEqual(trend.context["hot_source_post"]["source_type"], "trending_post")
        self.assertEqual(trend.context["hot_news_evidence"]["matched_article_id"],
                         "article:WATER")
        self.assertGreaterEqual(len(trend.context["hot_news_evidence"]["shared_terms"]), 2)
        self.assertTrue(trend.context["hot_news_evidence"]["title_shared_terms"])

    def test_hot_trend_below_heat_threshold_has_no_hot_candidate(self):
        world = WorldState(
            metadata=self.world.metadata, global_state=self.world.global_state,
            posts=[PostState(87, 10,
                "H2D2 Crown Harbour wastewater signal needs retesting.",
                29.0, 15, ["trending"], {}, [],
                "2026-11-01T11:50:00Z")], events=[], topics=[])
        platform = PlatformState(articles_by_id={"news:water": {
            "site_name": "Daily Kingstonia",
            "title": "Crown Harbour wastewater H2D2 signal retested",
            "summary": "Crown Harbour workers repeated H2D2 sampling after the signal.",
            "url": "/daily-kingstonia/articles/retesting"}})
        candidates = CandidateGenerator().generate(
            world, StateEncoder().encode(world), AgentState(), platform)
        self.assertFalse(any(c.context.get("post_intent") == "trend_hitchhike"
                             for c in candidates))

    def test_hot_trend_without_matching_news_has_no_hot_candidate(self):
        world = WorldState(
            metadata=self.world.metadata, global_state=self.world.global_state,
            posts=[PostState(81, 10,
                "The H2D2 Crown Harbour wastewater signal was a false alarm.",
                55.0, 1, ["trending"], {}, [],
                "2026-11-01T11:50:00Z")], events=[], topics=[])
        encoded = StateEncoder().encode(world)
        no_news = CandidateGenerator().generate(
            world, encoded, AgentState(), PlatformState())
        self.assertFalse(any(c.context.get("post_intent") == "trend_hitchhike"
                             for c in no_news))

        unrelated = PlatformState(articles_by_id={"news:turtle": {
            "article_id": "news:turtle", "site_name": "Daily Kingstonia",
            "title": "Green turtle returns to New Haven waters",
            "summary": "Residents cheered the rehabilitation of the turtle.",
            "url": "https://sites.legitreal.com/daily/articles/green-turtle"}})
        candidates = CandidateGenerator().generate(
            world, encoded, AgentState(), unrelated)
        self.assertFalse(any(c.context.get("post_intent") == "trend_hitchhike"
                             for c in candidates))

        same_terms_without_policy = PlatformState(articles_by_id={"news:water": {
            "site_name": "Daily Kingstonia",
            "title": "Crown Harbour wastewater signal update",
            "summary": "Crown Harbour wastewater monitoring continues.",
            "url": "/daily-kingstonia/articles/wastewater-monitoring"}})
        candidates = CandidateGenerator().generate(
            world, encoded, AgentState(), same_terms_without_policy)
        self.assertFalse(any(c.context.get("post_intent") == "trend_hitchhike"
                             for c in candidates))

    def test_hot_news_match_rejects_only_generic_or_summary_terms(self):
        world = WorldState(
            metadata=self.world.metadata, global_state=self.world.global_state,
            posts=[PostState(82, 10,
                "H2D2 Crown Harbour wastewater signal was discussed after retesting.",
                55.0, 1, ["trending"], {}, [],
                "2026-11-01T11:50:00Z")], events=[], topics=[])
        encoded = StateEncoder().encode(world)
        articles = {
            "generic": {
                "site_name": "Daily Kingstonia",
                "title": "H2D2 vaccine mandate referendum update",
                "summary": "The health policy campaign continues.",
                "url": "https://sites.legitreal.com/daily/articles/generic"},
            "summary_only": {
                "site_name": "Daily Kingstonia",
                "title": "Public safety briefing",
                "summary": "H2D2 Crown Harbour wastewater retesting was discussed.",
                "url": "https://sites.legitreal.com/daily/articles/summary-only"},
            "one_term": {
                "site_name": "Daily Kingstonia",
                "title": "Crown H2D2 briefing",
                "summary": "No further finding was announced.",
                "url": "https://sites.legitreal.com/daily/articles/one-term"},
        }
        candidates = CandidateGenerator().generate(
            world, encoded, AgentState(), PlatformState(articles_by_id=articles))
        self.assertFalse(any(c.context.get("post_intent") == "trend_hitchhike"
                             for c in candidates))

    def test_historical_trending_post_can_match_current_news_after_cooling(self):
        class HistoryStore:
            def load_historically_trending_posts(self, snapshot_id, after_snapshot_id):
                return {83: {
                    "post_id": 83, "author_id": 10,
                    "body": "H2D2 Crown Harbour wastewater signal needs retesting.",
                    "hot_score": 60.0, "created_at": "2026-10-29T12:00:00Z",
                    "feed_sources": ["trending"], "observed_snapshot_id": 1}}
        world = WorldState(
            metadata={**self.world.metadata, "snapshot_id": 2},
            global_state=self.world.global_state,
            posts=[PostState(83, 10,
                "H2D2 Crown Harbour wastewater signal needs retesting.",
                2.0, 40, ["latest"], {}, [], "2026-10-29T12:00:00Z")],
            events=[], topics=[])
        platform = PlatformState(articles_by_id={"news:current": {
            "site_name": "Daily Kingstonia",
            "title": "Crown Harbour wastewater H2D2 signal retested",
            "summary": "Crown Harbour workers repeated H2D2 sampling after the signal.",
            "url": "https://sites.legitreal.com/daily/articles/retesting"}})
        candidates = CandidateGenerator(state_store=HistoryStore()).generate(
            world, StateEncoder().encode(world), AgentState(), platform)
        cooled = next(c for c in candidates
                      if c.source_event_id == "hot-trending:83")
        self.assertGreaterEqual(
            cooled.context["hot_news_evidence"]["cooling_fraction"], 0.6)
        self.assertFalse(
            cooled.context["hot_news_evidence"]["current_trending"])

    def test_hot_post_link_matches_observed_news_without_shared_terms(self):
        world = WorldState(
            metadata=self.world.metadata, global_state=self.world.global_state,
            posts=[PostState(85, 10,
                "H2D2: this article matters https://sites.legitreal.com/"
                "daily-kingstonia/articles/crown-harbour-signal",
                44.0, 8, ["trending"], {}, [],
                "2026-11-01T11:50:00Z")], events=[], topics=[])
        article_url = "/daily-kingstonia/articles/crown-harbour-signal"
        platform = PlatformState(articles_by_id={"news:linked": {
            "site_name": "Daily Kingstonia",
            "title": "Crown Harbour wastewater H2D2 signal ruled false alarm",
            "summary": "Repeat sampling found no H2D2 after lab contamination.",
            "url": article_url}})
        candidates = CandidateGenerator().generate(
            world, StateEncoder().encode(world), AgentState(), platform)
        trend = next(c for c in candidates if c.source_event_id == "hot-trending:85")
        self.assertEqual(trend.context["articles"][0]["url"], article_url)
        self.assertEqual(trend.context["hot_news_evidence"]["match_method"],
                         "linked_article_url")

    def test_unavailable_link_does_not_fall_back_to_loose_term_match(self):
        world = WorldState(
            metadata=self.world.metadata, global_state=self.world.global_state,
            posts=[PostState(86, 10,
                "H2D2 Crown Harbour wastewater signal "
                "https://sites.legitreal.com/daily-kingstonia/articles/another-story",
                55.0, 1, ["trending"], {}, [],
                "2026-11-01T11:50:00Z")], events=[], topics=[])
        platform = PlatformState(articles_by_id={"news:other": {
            "site_name": "Daily Kingstonia",
            "title": "Crown Harbour wastewater H2D2 signal update",
            "summary": "Crown Harbour wastewater readings were reviewed.",
            "url": "/daily-kingstonia/articles/current-story"}})
        candidates = CandidateGenerator().generate(
            world, StateEncoder().encode(world), AgentState(), platform)
        self.assertFalse(any(c.context.get("post_intent") == "trend_hitchhike"
                             for c in candidates))

    def test_hot_news_layer_skips_a_previously_published_article(self):
        world = WorldState(
            metadata=self.world.metadata, global_state=self.world.global_state,
            posts=[PostState(84, 10,
                "H2D2 Crown Harbour wastewater signal needs retesting.",
                55.0, 1, ["trending"], {}, [],
                "2026-11-01T11:50:00Z")],
            events=[EventState("story:WATER", ["news:current"], [])], topics=[])
        url = "/daily-kingstonia/articles/retesting"
        platform = PlatformState(articles_by_id={"news:current": {
            "site_name": "Daily Kingstonia",
            "title": "Crown Harbour wastewater H2D2 signal retested",
            "summary": "Crown Harbour workers repeated H2D2 sampling after the signal.",
            "published_at": "2026-11-01T11:30:00Z",
            "url": url}})
        agent = AgentState(published_news_sources=[
            "Earlier post Source: Daily Kingstonia\n"
            "https://sites.legitreal.com/daily-kingstonia/articles/retesting"])
        candidates = CandidateGenerator().generate(
            world, StateEncoder().encode(world), agent, platform)
        self.assertFalse(any(c.context.get("post_intent") == "trend_hitchhike"
                             for c in candidates))
        self.assertTrue(any(c.source_event_id == "story:WATER"
                            for c in candidates))

    def test_hot_news_layer_uses_half_probability_and_excludes_its_reward(self):
        class Generator:
            def generate(self, world, encoded, agent, platform):
                return (CandidateAction("CREATE_POST", source_event_id="hot",
                            context={"post_intent": "trend_hitchhike",
                                     "hot_news_evidence": {"related_post_id": 80}}),
                        CandidateAction("REPLY", target_id=5),
                        CandidateAction("WAIT"))
        engine = DecisionEngine(generator=Generator(), policy_mode="heuristic",
                                 rng=random.Random(7))
        selected_hot_layer = 0
        for _ in range(100):
            plan = engine.decide(self.world, self.encoded, AgentState(), PlatformState())
            trace = plan.parameters["decision_trace"]
            layer = trace["hot_news_layer"]
            self.assertAlmostEqual(
                sum(item["selection_probability"] for item in trace["candidates"]),
                0.5, places=6)
            selected_hot_layer += int(layer["selected"])
            if layer["selected"]:
                self.assertEqual(plan.action_type, "CREATE_POST")
                self.assertEqual(plan.parameters["selection_method"], "hot_news_layer_softmax")
                self.assertTrue(plan.parameters["exclude_from_reward"])
                self.assertEqual(plan.parameters["reward_exclusion_reason"],
                                 "independent_hot_news_layer")
            else:
                self.assertNotEqual(plan.action_type, "CREATE_POST")
                self.assertNotIn("exclude_from_reward", plan.parameters)
        self.assertGreater(selected_hot_layer, 35)
        self.assertLess(selected_hot_layer, 65)

    def test_cooling_penalty_lowers_hot_score_and_layer_probability(self):
        fresh = CandidateAction(
            "CREATE_POST", source_event_id="fresh",
            context={"post_intent": "trend_hitchhike",
                     "hot_news_evidence": {
                         "heat_strength": 1.0, "cooling_fraction": 0.0}})
        cooled = CandidateAction(
            "CREATE_POST", source_event_id="cooled",
            context={"post_intent": "trend_hitchhike",
                     "hot_news_evidence": {
                         "heat_strength": 0.0, "cooling_fraction": 1.0}})
        scorer = PolicyScorer()
        fresh_score = scorer.score(
            fresh, self.world, self.encoded, AgentState(), PlatformState())
        cooled_score = scorer.score(
            cooled, self.world, self.encoded, AgentState(), PlatformState())
        self.assertGreater(fresh_score.total_score, cooled_score.total_score)
        self.assertGreater(fresh_score.score_components["hot_news_heat_bonus"], 0)
        self.assertLess(cooled_score.score_components["hot_news_cooldown_penalty"], 0)

        class BothGenerator:
            def generate(self, world, encoded, agent, platform):
                return (fresh, cooled, CandidateAction("WAIT"))
        both = DecisionEngine(
            generator=BothGenerator(), policy_mode="heuristic",
            rng=random.Random(1)).decide(
                self.world, self.encoded, AgentState(), PlatformState())
        trace = both.parameters["decision_trace"]
        self.assertTrue(trace["hot_news_layer"]["selected"])
        scores = {item["source_event_id"]: item["selection_probability"]
                  for item in trace["candidates"]}
        self.assertGreater(scores["fresh"], scores["cooled"])
        self.assertAlmostEqual(sum(scores.values()), 0.5)

        class CooledOnlyGenerator:
            def generate(self, world, encoded, agent, platform):
                return (cooled, CandidateAction("WAIT"))
        cooled_plan = DecisionEngine(
            generator=CooledOnlyGenerator(), policy_mode="heuristic",
            rng=random.Random(2)).decide(
                self.world, self.encoded, AgentState(), PlatformState())
        self.assertAlmostEqual(
            cooled_plan.parameters["decision_trace"]["hot_news_layer"]["probability"],
            0.1)

    def test_own_post_reply_priority_precedes_hot_layer_coin_flip(self):
        class ReplyGenerator:
            def generate(self, world, encoded, agent, platform):
                return (CandidateAction("CREATE_POST", source_event_id="hot",
                            context={"post_intent": "trend_hitchhike",
                                     "hot_news_evidence": {"related_post_id": 80}}),
                        CandidateAction("REPLY", target_id=5,
                            context={"reply_to_own_post": True}))
        reply = DecisionEngine(generator=ReplyGenerator(), policy_mode="heuristic",
                               rng=random.Random(12)).decide(
            self.world, self.encoded, AgentState(), PlatformState())
        self.assertEqual(reply.action_type, "REPLY")
        layer = reply.parameters["decision_trace"]["hot_news_layer"]
        self.assertIsNone(layer["roll"])
        self.assertFalse(layer["selected"])

    def test_create_post_event_remains_explorable_during_cooldown(self):
        from policy.settings import PolicySettings
        event_world=WorldState(metadata={**self.world.metadata,"snapshot_id":2},
            global_state=self.world.global_state,posts=self.world.posts,
            events=[EventState("story:H2D2",["article-1"],[])],topics=[])
        encoded=StateEncoder().encode(event_world)
        platform=PlatformState(articles_by_id={"article-1":{
            "title":"H2D2 referendum update","summary":"Public health update",
            "published_at":"2026-11-01T11:30:00Z"}})
        settings=PolicySettings(post_event_cooldown_hours=2.0)
        generator=CandidateGenerator(settings)
        recent={"action_type":"CREATE_POST","status":"SUCCEEDED",
                "action_key":"CREATE_POST:story:H2D2:1",
                "created_at":"2026-11-01T11:00:00Z"}
        cooling=generator.generate(event_world,encoded,
            AgentState(recent_actions=[recent]),platform)
        cooling_post=next(c for c in cooling
                          if c.action_type=="CREATE_POST"
                          and c.source_event_id=="story:H2D2")
        self.assertTrue(cooling_post.context["event_cooldown_active"])
        recent["created_at"]="2026-11-01T09:00:00Z"
        eligible=generator.generate(event_world,encoded,
            AgentState(recent_actions=[recent],completed_keys={"CREATE_POST:story:H2D2"}),platform)
        post=next(c for c in eligible if c.action_type=="CREATE_POST")
        self.assertFalse(post.context["event_cooldown_active"])
        self.assertEqual(post.key,"CREATE_POST:story:H2D2:2")
        plan=ActionPlan.from_candidate(2,post,0.5,{})
        self.assertEqual(plan.action_key,post.key)

    def test_post_coverage_is_not_reward(self):
        from policy.reward import RewardEvaluator
        result=RewardEvaluator().evaluate("CREATE_POST",True,{},True,True,"execution")
        self.assertNotIn("action_priority", result.components)


if __name__ == "__main__":
    unittest.main()
