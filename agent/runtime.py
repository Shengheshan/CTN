"""Bounded observe → decide → execute → feedback runtime."""

import logging
import time
from dataclasses import asdict, replace

from agent.content_generator import ContentGenerationError, ContentGenerator, LLMRateLimitError
from agent.executor import ActionExecutor, ExecutionResult
from agent.feedback import V03FeedbackSystem
from agent.history import ActionHistory
from agent.state import PlatformState
from policy.engine import DecisionEngine
from policy.settings import PolicySettings
from config import RuntimeConfig
from policy.models import utc_now
from policy.relevance import (campaign_relevant, explicit_yes_alignment,
                              select_popular_tags)
from state.builder import StateBuilder
from state.encoder import StateEncoder
from state.normalizer import normalize_news, normalize_posts
from state.observer import save_raw_snapshot


log = logging.getLogger(__name__)


class AgentRuntime:
    def __init__(self, observer, store, history: ActionHistory, social,
                 llm_client=None, account=None, mode="dry-run",
                 max_actions_per_round=1, enable_topics=False,
                 policy_mode="bandit", reward_horizon_minutes=30.0,
                 runtime_config: RuntimeConfig | None = None,
                 save_raw=save_raw_snapshot):
        if mode not in ("dry-run", "live"):
            raise ValueError("invalid mode")
        if max_actions_per_round < 1:
            raise ValueError("action budget must be positive")
        self.observer = observer
        self.store = store
        self.history = history
        self.social = social
        self.llm_client = llm_client
        self.account = account
        self.mode = mode
        self.max_actions_per_round = max_actions_per_round
        self.enable_topics = enable_topics
        self.reward_horizon_minutes = reward_horizon_minutes
        self.save_raw = save_raw
        self.runtime_config = runtime_config or RuntimeConfig.from_env()
        self.policy_settings = PolicySettings.from_env()
        self.feedback_system = V03FeedbackSystem(
            self.social, self.history.db_path, self.runtime_config,
            self.policy_settings.action_priors)
        self.bandit = self.feedback_system.bandit
        self.engine = DecisionEngine(policy_mode=policy_mode, bandit=self.bandit,
                                     settings=self.policy_settings,
                                     state_store=self.store)
        self.encoder = StateEncoder()

    @staticmethod
    def _successful_text_actions_since_engagement(
            recent_actions, include_dry_runs: bool = False) -> int | None:
        """Count eligible text actions after the latest completed engagement.

        ActionHistory supplies newest-first rows. None means there is no prior
        eligible engagement, so the first interaction is allowed. Dry-run rows
        count only when evaluating another dry-run round, never in live mode.
        """
        text_actions = {"CREATE_POST", "REPLY"}
        engagement_actions = {"LIKE", "REPOST", "FOLLOW", "MENTION"}
        successful_statuses = {"SUCCEEDED"}
        if include_dry_runs:
            successful_statuses.add("DRY_RUN")
        count = 0
        for action in recent_actions:
            if action.get("status") not in successful_statuses:
                continue
            action_type = action.get("action_type")
            if action_type in engagement_actions:
                return count
            if action_type in text_actions:
                count += 1
        return None

    def _enrich_observation(self, raw) -> list[dict]:
        """Include personal and tag feeds in the same durable snapshot."""
        tags = []
        try:
            tags = list(self.social.popular_tags())
        except Exception as exc:
            log.warning("Popular tags unavailable: %s", type(exc).__name__)
        if self.account is not None:
            try:
                raw.home_posts = list(self.account.feed_home(limit=30))
            except Exception as exc:
                log.warning("Home feed unavailable: %s", type(exc).__name__)
            try:
                # Notifications are newest-first. Read the available unresolved
                # page and hydrate each trimmed post so reply/root IDs survive
                # normalization and can be targeted immediately.
                notifications = list(self.account.notifications(unresolved=True, limit=50))
                for item in notifications:
                    if not isinstance(item, dict):
                        continue
                    summary = item.get("post")
                    post_id = (summary.get("post_id") if isinstance(summary, dict)
                               else item.get("post_id"))
                    if not isinstance(post_id, int):
                        continue
                    try:
                        post = self.social.get_post(post_id)
                    except Exception:
                        post = summary if isinstance(summary, dict) else item
                    if isinstance(post, dict) and isinstance(post.get("post_id"), int):
                        post = dict(post)
                        post["_notification_kind"] = item.get("kind")
                        post["_notification_id"] = item.get("notification_id")
                        raw.notification_posts.append(post)
            except Exception as exc:
                log.warning("Notifications unavailable: %s", type(exc).__name__)
        for name in select_popular_tags(tags):
            match = next((item for item in tags if isinstance(item, dict)
                          and item.get("name", "").lower() == name), None)
            tag_id = match.get("tag_id") if match else None
            if not isinstance(tag_id, int):
                continue
            try:
                raw.tagged_posts.extend(self.social.tag_posts(tag_id, limit=15))
            except Exception as exc:
                log.warning("Tag feed unavailable for %s: %s", name, type(exc).__name__)
        return tags

    def run_round(self) -> dict:
        if self.runtime_config.enable_feedback:
            try:
                feedback_summary = self.feedback_system.process_due()
                self.policy_settings = replace(
                    self.policy_settings,
                    action_priors=self.feedback_system.current_action_priors())
                self.engine.settings = self.policy_settings
                self.engine.scorer.settings = self.policy_settings
                if any(feedback_summary.values()):
                    log.info("v0.3 feedback: %s", feedback_summary)
            except Exception as exc:
                # A feedback outage must never block observation or action planning.
                log.warning("Feedback processing skipped: %s", type(exc).__name__)
        raw = self.observer.collect()
        popular_tags = self._enrich_observation(raw)
        raw_dict = asdict(raw)
        raw_path = self.save_raw(raw)
        posts = normalize_posts(raw_dict)
        articles, events = normalize_news(raw_dict)
        snapshot_id = self.store.save_snapshot(raw_path, raw_dict, posts, articles, events)
        log.info("Snapshot %s: %s posts, %s articles, %s events",
                 snapshot_id, len(posts), len(articles), len(events))
        if self.enable_topics and self.llm_client is not None:
            try:
                from state.topic_classification.pipeline import TopicClassificationPipeline
                TopicClassificationPipeline(self.store, self.llm_client).run(snapshot_id)
            except Exception as exc:
                log.warning("Topic classification skipped: %s", type(exc).__name__)

        world = StateBuilder(self.store).build(snapshot_id)
        encoded = self.encoder.encode(world)
        platform = PlatformState(
            account_id=getattr(self.account, "account_id", None),
            articles_by_id={a.article_id: asdict(a) for a in articles},
            max_actions_per_round=max(2, self.max_actions_per_round),
            reply_to_post_ids={p.post_id: p.reply_to_post_id for p in posts
                               if p.reply_to_post_id is not None},
            root_post_ids={p.post_id: p.root_post_id for p in posts
                           if p.root_post_id is not None},
            priority_reply_post_ids={
                item["post_id"] for item in raw_dict.get("notification_posts", [])
                if item.get("_notification_kind") == "reply"
                and isinstance(item.get("post_id"), int)},
            notification_ids_by_post={
                item["post_id"]: item["_notification_id"]
                for item in raw_dict.get("notification_posts", [])
                if item.get("_notification_kind") == "reply"
                and isinstance(item.get("post_id"), int)
                and isinstance(item.get("_notification_id"), int)},
        )
        platform.popular_tags = popular_tags
        for post in world.posts:
            if campaign_relevant(post.body):
                platform.relevance_by_target[str(post.post_id)] = 1.0
                alignment = explicit_yes_alignment(post.body)
                if alignment > 0:
                    platform.alignment_by_post[post.post_id] = alignment
        for event in world.events:
            if any(campaign_relevant(" ".join(
                    str(platform.articles_by_id[article_id].get(key) or "")
                    for key in ("title", "summary")))
                    for article_id in event.article_ids
                    if article_id in platform.articles_by_id):
                platform.relevance_by_target[event.event_id] = 1.0
        if self.mode == "live":
            try:
                platform.feed_locked = bool(self.social.feed_lock().get("locked"))
            except Exception as exc:
                log.warning("Feed lock unavailable; no writes this round: %s", type(exc).__name__)
                platform.feed_locked = True
            if self.account is not None and not platform.feed_locked:
                ids = [p.post_id for p in world.posts]
                try:
                    platform.liked_post_ids = set(self.account.my_likes(ids))
                    platform.reposted_post_ids = set(self.account.my_reposts(ids))
                    platform.followed_account_ids = {
                        item["account_id"] for item in self.social.following(
                            platform.account_id, limit=100)
                        if isinstance(item, dict) and isinstance(item.get("account_id"), int)
                    }
                except Exception as exc:
                    log.warning("Account relation check failed: %s", type(exc).__name__)
                    platform.feed_locked = True

        executor = ActionExecutor(self.social, self.account, self.history)
        generator = ContentGenerator(self.llm_client) if self.llm_client is not None else None
        results = []
        skipped = set()
        # One text action plus one non-text interaction can share a round.
        text_actions = {"CREATE_POST", "REPLY"}
        engagement_actions = {"LIKE", "REPOST", "FOLLOW", "MENTION"}
        max_round_actions = max(2, self.max_actions_per_round)
        for action_number in range(max_round_actions):
            agent = self.history.load_state(include_dry_runs=self.mode == "dry-run")
            agent.completed_keys.update(skipped)
            agent.action_count_this_round = action_number
            if action_number > 0:
                text_since_engagement = self._successful_text_actions_since_engagement(
                    agent.recent_actions, include_dry_runs=self.mode == "dry-run")
                required_text_actions = self.policy_settings.text_actions_per_engagement
                if (text_since_engagement is not None
                        and text_since_engagement < required_text_actions):
                    log.debug(
                        "Skipping non-LLM interaction: %s successful post/reply actions "
                        "since last engagement; requires %s",
                        text_since_engagement, required_text_actions)
                    break
            # Keep the content slot independent from non-LLM interactions.
            # Slot 0 chooses content or WAIT; slot 1 chooses interaction or WAIT.
            excluded = engagement_actions if action_number == 0 else text_actions
            if action_number == 0:
                consecutive_content_waits = 0
                for prior in agent.recent_actions:
                    if prior.get("action_type") == "WAIT":
                        consecutive_content_waits += 1
                    elif prior.get("action_type") in text_actions:
                        break
                if consecutive_content_waits >= 2:
                    # Avoid a long silent streak while still waiting when no
                    # eligible recent content candidate exists.
                    excluded = engagement_actions | {"WAIT"}
            plan = self.engine.decide(world, encoded, agent, platform,
                                      excluded_action_types=set(excluded))
            try:
                self.feedback_system.record_plan(plan)
            except Exception as exc:
                log.warning("Decision trace storage skipped: %s", type(exc).__name__)
            if plan.action_type == "WAIT":
                result = executor.execute(plan, self.mode)
                try:
                    self.feedback_system.record_result(plan, result)
                except Exception as exc:
                    log.warning("Action feedback scheduling skipped: %s", type(exc).__name__)
                results.append({"action": "WAIT", "status": result.status})
                if action_number == 0:
                    continue
                break
            if plan.action_type in ("CREATE_POST", "REPLY"):
                if generator is None:
                    error = "LLM unavailable"
                    self.history.save_result(plan, "FAILED", error=error)
                    try:
                        self.feedback_system.record_result(
                            plan, ExecutionResult("FAILED", error=error))
                    except Exception as exc:
                        log.warning("Failure reward storage skipped: %s", type(exc).__name__)
                    results.append({"action": plan.action_type, "status": "FAILED", "error": error})
                    skipped.add(plan.action_key)
                    continue
                if plan.action_type == "REPLY" and self.mode == "live":
                    try:
                        context = plan.parameters.setdefault("context", {})
                        current = self.social.get_post(plan.target_id)
                        is_reply_to_own = bool(context.get("reply_to_own_post"))
                        if (not isinstance(current, dict) or
                                (not is_reply_to_own and not campaign_relevant(current.get("body"))) or
                                current.get("author_id") == platform.account_id):
                            raise ValueError("Reply target is unavailable or no longer relevant")
                        context["post_body"] = current["body"]
                        context["root_post_id"] = current.get("root_post_id")
                        try:
                            thread_context = []
                            for item in self.social.replies(plan.target_id, limit=3):
                                if not isinstance(item, dict) or not isinstance(item.get("body"), str):
                                    continue
                                author = (item.get("author_username") or item.get("username")
                                          or item.get("handle"))
                                author_data = item.get("author") or item.get("user")
                                if isinstance(author_data, dict):
                                    author = (author or author_data.get("username")
                                              or author_data.get("handle"))
                                elif isinstance(author_data, str):
                                    author = author or author_data
                                reply_context = {"body": item["body"][:200]}
                                if isinstance(author, str) and author.strip():
                                    reply_context["author"] = author.strip()[:48]
                                thread_context.append(reply_context)
                            context["thread_replies"] = thread_context
                        except Exception as exc:
                            log.warning("Thread context unavailable: %s", type(exc).__name__)
                    except Exception as exc:
                        error = f"Reply preflight failed: {type(exc).__name__}"
                        self.history.save_result(plan, "FAILED", error=error)
                        try:
                            self.feedback_system.record_result(
                                plan, ExecutionResult("FAILED", error=error))
                        except Exception as store_exc:
                            log.warning("Failure reward storage skipped: %s", type(store_exc).__name__)
                        results.append({"action": "REPLY", "status": "FAILED", "error": error})
                        skipped.add(plan.action_key)
                        continue
                generation_attempts = []
                generation_error = None
                generated = False
                for attempt in range(2):
                    if attempt:
                        plan.parameters["generation_retry_feedback"] = (
                            f"Previous draft failed: {generation_error}")
                    try:
                        generator.generate(plan)
                    except ContentGenerationError as exc:
                        generation_error = exc
                        generation_attempts.append({
                            "attempt": attempt + 1, "status": "FAILED",
                            "error": str(exc)[:240]})
                        permanent = any(message in str(exc).casefold() for message in (
                            "invalid tone", "no attributed news source",
                            "reply requires a real post with text"))
                        if attempt == 0 and not permanent and not isinstance(exc, LLMRateLimitError):
                            log.warning("%s content generation failed; retrying once: %s",
                                        plan.action_type, type(exc).__name__)
                            continue
                        plan.parameters["generation_attempts"] = generation_attempts
                        self.history.save_result(plan, "FAILED", error=str(exc))
                        try:
                            self.feedback_system.record_result(
                                plan, ExecutionResult("FAILED", error=str(exc)))
                        except Exception as store_exc:
                            log.warning("Failure reward storage skipped: %s", type(store_exc).__name__)
                        results.append({"action": plan.action_type, "status": "FAILED", "error": str(exc)})
                        skipped.add(plan.action_key)
                        break
                    else:
                        generation_attempts.append({
                            "attempt": attempt + 1, "status": "SUCCEEDED"})
                        plan.parameters["generation_attempts"] = generation_attempts
                        generated = True
                        break
                if not generated:
                    continue
            try:
                self.feedback_system.record_content(plan)
            except Exception as exc:
                log.warning("Content assessment skipped: %s", type(exc).__name__)
            result = executor.execute(plan, self.mode)
            if (plan.action_type in text_actions and result.status == "FAILED"
                    and result.error in {"HTTP 400", "HTTP 413", "HTTP 422"}):
                first_error = result.error
                attempts = plan.parameters.setdefault("execution_attempts", [])
                attempts.append({"attempt": 1, "status": "FAILED", "error": first_error})
                attempts.append({"attempt": 2, "status": "REGENERATING"})
                self.history.save_result(plan, "FAILED", error=first_error)
                plan.parameters["generation_retry_feedback"] = (
                    f"The platform rejected the previous submission ({first_error}). "
                    "Write a fresh, concise version that fits the platform limits.")
                try:
                    generator.generate(plan)
                except ContentGenerationError as exc:
                    attempts[-1] = {"attempt": 2, "status": "GENERATION_FAILED",
                                    "error": str(exc)[:240]}
                    result = ExecutionResult(
                        "FAILED", error=f"{first_error}; retry generation failed: {exc}")
                    self.history.save_result(plan, result.status, error=result.error)
                else:
                    attempts[-1] = {"attempt": 2, "status": "SUBMITTING"}
                    generation_attempts = plan.parameters.setdefault("generation_attempts", [])
                    generation_attempts.append({
                        "attempt": len(generation_attempts) + 1,
                        "status": "SUCCEEDED", "trigger": "platform_rejection"})
                    try:
                        self.feedback_system.record_content(plan)
                    except Exception as exc:
                        log.warning("Retry content assessment skipped: %s", type(exc).__name__)
                    retry_result = executor.execute(plan, self.mode)
                    attempts[-1] = {"attempt": 2, "status": retry_result.status,
                                    "error": retry_result.error}
                    result = retry_result
                    self.history.save_result(
                        plan, result.status, result.remote_id, result.error, result.response)
                    log.info("%s platform rejection %s; one regenerated retry returned %s",
                             plan.action_type, first_error, result.status)
            try:
                self.feedback_system.record_result(plan, result)
            except Exception as exc:
                log.warning("Action feedback scheduling skipped: %s", type(exc).__name__)
            results.append({"action": plan.action_type, "status": result.status,
                            "remote_id": result.remote_id})
            if (plan.action_type == "REPLY" and result.status == "SUCCEEDED"
                    and self.account is not None):
                context = plan.parameters.get("context", {})
                notification_id = context.get("notification_id")
                resolver = getattr(self.account, "resolve_notification", None)
                if isinstance(notification_id, int) and callable(resolver):
                    try:
                        resolver(notification_id)
                    except Exception as exc:
                        log.warning("Reply notification resolution failed: %s",
                                    type(exc).__name__)
            skipped.add(plan.action_key)
            if result.status in ("BLOCKED", "UNKNOWN"):
                break
        return {"snapshot_id": snapshot_id, "encoded_version": encoded.version,
                "actions": results, "policy_samples": self.bandit.count,
                "completed_at": utc_now()}

    def run(self, max_rounds: int = 1, interval: float = 60.0) -> list[dict]:
        if max_rounds < 1 or interval < 0:
            raise ValueError("max_rounds must be positive and interval nonnegative")
        completed = []
        for index in range(max_rounds):
            try:
                result = self.run_round()
                completed.append(result)
                log.info("Round %s/%s: %s", index + 1, max_rounds, result["actions"])
            except KeyboardInterrupt:
                log.info("Stopped by user")
                break
            except Exception:
                log.exception("Round %s failed; continuing within bounded run", index + 1)
            if index + 1 < max_rounds:
                try:
                    delay = (interval * self.engine.rng.uniform(0.7, 1.3)
                             if self.mode == "live" and interval > 0 else interval)
                    time.sleep(delay)
                except KeyboardInterrupt:
                    log.info("Stopped by user")
                    break
        return completed
