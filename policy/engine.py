"""Cold-start heuristic or contextual UCB/Softmax ActionPlan selection."""

import logging
import random
from math import exp

from agent.state import AgentState, PlatformState
from policy.bandit import (FEATURE_VERSION, MIN_TRAINING_SAMPLES,
                           ContextFeatureEncoder, LinUCB, softmax_sample)
from policy.candidates import CandidateGenerator
from policy.generation_profiles import select_generation_profile
from policy.models import ActionPlan, CandidateAction
from policy.opportunity import OpportunityEvaluator
from policy.relevance import select_popular_tags, select_post_tags
from policy.scorer import PolicyScorer, ScoredCandidate
from policy.settings import PolicySettings
from policy.strategic import StrategicEvaluator
from state.builder import WorldState
from state.encoder import EncodedWorldState


_LOGGER = logging.getLogger(__name__)


class DecisionEngine:
    def __init__(self, generator: CandidateGenerator | None = None,
                 scorer: PolicyScorer | None = None,
                 settings: PolicySettings | None = None,
                 policy_mode: str = "bandit", bandit: LinUCB | None = None,
                 min_samples: int = MIN_TRAINING_SAMPLES, temperature: float | None = None,
                 rng: random.Random | None = None,
                 content_rng: random.Random | None = None,
                 state_store=None):
        self.settings = settings or PolicySettings()
        temperature = (self.settings.softmax_temperature if temperature is None
                       else temperature)
        if policy_mode not in ("heuristic", "bandit") or min_samples < 1 or temperature <= 0:
            raise ValueError("invalid policy configuration")
        self.generator = generator or CandidateGenerator(
            self.settings, state_store=state_store)
        self.scorer = scorer or PolicyScorer(self.settings)
        self.policy_mode = policy_mode
        self.bandit = bandit or LinUCB()
        self.min_samples = min_samples
        self.temperature = temperature
        self.rng = rng or random.Random()
        # Keep content-style exploration from consuming action-policy randomness.
        self.content_rng = content_rng or random.Random()
        self.strategic = StrategicEvaluator()
        self.opportunity = OpportunityEvaluator()
        self.features = ContextFeatureEncoder()

    @staticmethod
    def _softmax_probabilities(scores: list[float], temperature: float) -> tuple[float, ...]:
        peak = max(scores)
        weights = [exp((score - peak) / temperature) for score in scores]
        total = sum(weights)
        return tuple(weight / total for weight in weights)

    def _sample(self, scored: list[ScoredCandidate], scores: list[float]
                ) -> tuple[int, float, tuple[float, ...]]:
        """Sample action type first so numerous REPLY targets do not dominate."""
        groups: dict[str, list[int]] = {}
        for index, item in enumerate(scored):
            groups.setdefault(item.candidate.action_type, []).append(index)
        names = sorted(groups)
        type_scores = [max(scores[index] for index in groups[name]) for name in names]
        type_index, type_probabilities = softmax_sample(type_scores, self.temperature, self.rng)
        probability_by_index = [0.0] * len(scored)
        for group_index, name in enumerate(names):
            indices = groups[name]
            local_probabilities = self._softmax_probabilities(
                [scores[index] for index in indices], self.temperature)
            for index, local_probability in zip(indices, local_probabilities):
                probability_by_index[index] = (
                    type_probabilities[group_index] * local_probability)
        indices = groups[names[type_index]]
        candidate_index, candidate_probabilities = softmax_sample(
            [scores[index] for index in indices], self.temperature, self.rng)
        chosen_index = indices[candidate_index]
        return chosen_index, (type_probabilities[type_index] *
                              candidate_probabilities[candidate_index]), tuple(probability_by_index)

    def decide(self, world: WorldState, encoded: EncodedWorldState,
               agent: AgentState, platform: PlatformState,
               excluded_action_types: set[str] | None = None) -> ActionPlan:
        strategic = self.strategic.evaluate(encoded)
        generated_candidates = tuple(self.generator.generate(
            world, encoded, agent, platform))
        excluded_types = excluded_action_types or set()
        excluded_candidates = tuple(candidate for candidate in generated_candidates
                                     if candidate.action_type in excluded_types)
        candidates = tuple(candidate for candidate in generated_candidates
            if candidate.action_type not in (excluded_action_types or set()))
        post_coverage = self._post_coverage(agent)
        coverage_forced = False
        priority_reply_override = False
        priority_reply_available = False
        hot_news_layer_selected = False
        hot_news_layer_roll = None
        hot_news_layer_probability = 0.0
        hot_news_gate_cooling = 0.0
        scheduler_exclusion_reason = "minimum_original_post_coverage"
        coverage_excluded_candidates = ()
        eligible_posts = tuple(c for c in candidates if c.action_type == "CREATE_POST")
        priority_replies = tuple(c for c in candidates
            if c.action_type == "REPLY" and c.context.get("reply_to_own_post"))
        hot_news_posts = tuple(c for c in eligible_posts
            if c.context.get("post_intent") == "trend_hitchhike")
        priority_reply_available = bool(priority_replies)
        if priority_reply_available:
            # An own-post reply keeps its existing score boost, but remains a
            # weighted choice so it cannot suppress every CREATE_POST option.
            coverage_excluded_candidates = hot_news_posts
            candidates = tuple(c for c in candidates if c not in hot_news_posts)
            scheduler_exclusion_reason = "hot_news_layer_deferred_for_priority_reply"
        else:
            if hot_news_posts:
                hot_news_gate_cooling = min(
                    max(0.0, min(1.0, float(
                        c.context.get("hot_news_evidence", {}).get(
                            "cooling_fraction") or 0.0)))
                    for c in hot_news_posts)
                hot_news_layer_probability = 0.5 * (
                    1.0 - self.settings.hot_news_gate_cooling_factor
                    * hot_news_gate_cooling)
                hot_news_layer_roll = self.rng.random()
                hot_news_layer_selected = (
                    hot_news_layer_roll < hot_news_layer_probability)
                if hot_news_layer_selected:
                    coverage_excluded_candidates = tuple(c for c in candidates
                        if c not in hot_news_posts)
                    candidates = hot_news_posts
                    scheduler_exclusion_reason = "hot_news_layer_selected"
                else:
                    coverage_excluded_candidates = hot_news_posts
                    candidates = tuple(c for c in candidates if c not in hot_news_posts)
                    scheduler_exclusion_reason = "hot_news_layer_not_selected"
            eligible_posts = tuple(c for c in candidates if c.action_type == "CREATE_POST")
            coverage_posts = tuple(c for c in eligible_posts
                                   if not c.context.get("event_cooldown_active"))
            if (not hot_news_layer_selected and coverage_posts and post_coverage is not None
                and post_coverage < self.settings.minimum_post_coverage
                and platform.api_remaining != 0 and not platform.feed_locked):
                # Coverage is a scheduling constraint; it is not part of reward.
                coverage_excluded_candidates += tuple(c for c in candidates
                    if c not in coverage_posts)
                candidates = coverage_posts
                coverage_forced = True
                scheduler_exclusion_reason = "minimum_original_post_coverage"
        fallback_wait = not candidates
        if not candidates:
            candidates = (CandidateAction("WAIT", rationale="No eligible action"),)
        opportunities = {
            candidate.key: self.opportunity.evaluate(
                candidate, world, encoded, platform) for candidate in candidates
        }
        base_scored = [self.scorer.score(c, world, encoded, agent, platform)
                       for c in candidates]
        scored: list[ScoredCandidate] = []
        for item in base_scored:
            action_prior = float(self.settings.action_priors.get(
                item.candidate.action_type, 1.0))
            adjusted = max(0.0, min(1.0, item.total_score * action_prior))
            score_components = dict(item.score_components)
            score_components["base_score"] = item.total_score
            score_components["action_prior"] = action_prior
            score_components["prior_adjustment"] = adjusted - item.total_score
            scored.append(ScoredCandidate(item.candidate, adjusted, score_components))
        features = {}
        for item in scored:
            risk = -sum(value for name, value in item.score_components.items()
                        if name.endswith("penalty"))
            features[item.candidate.key] = self.features.encode(
                item.candidate, encoded, strategic,
                opportunities[item.candidate.key], risk)

        trained = self.policy_mode == "bandit" and self.bandit.count >= self.min_samples
        selection_scores: dict[str, float] = {}
        selection_probabilities: dict[str, float] = {}
        exploration_bonuses: dict[str, float] = {}
        predicted_values: dict[str, float | None] = {}
        if hot_news_layer_selected:
            # Keep the hot-news layer separate from learned reward, while
            # applying heat and cooling scores to its own random choice.
            scored.sort(key=lambda item: item.candidate.key)
            scores = [item.total_score for item in scored]
            index, conditional_probabilities = softmax_sample(
                scores, self.temperature, self.rng)
            chosen = scored[index]
            conditional_probability = conditional_probabilities[index]
            probability = hot_news_layer_probability * conditional_probability
            selection_scores = {item.candidate.key: item.total_score for item in scored}
            selection_probabilities = {
                item.candidate.key: hot_news_layer_probability * local_probability
                for item, local_probability in zip(scored, conditional_probabilities)}
            selected_score = max(0.0, min(1.0, chosen.total_score))
            components = dict(chosen.score_components)
            components["hot_news_layer_probability"] = hot_news_layer_probability
            components["hot_news_candidate_probability"] = conditional_probability
            components["selection_probability"] = probability
            method = "hot_news_layer_softmax"
        elif trained:
            scored.sort(key=lambda item: item.candidate.key)
            raw_scores = []
            for item in scored:
                if item.candidate.action_type == "WAIT":
                    raw_scores.append(self.settings.wait_score)
                    exploration_bonuses[item.candidate.key] = 0.0
                    predicted_values[item.candidate.key] = self.settings.wait_score
                elif item.candidate.action_type in ("LIKE", "REPOST", "FOLLOW", "MENTION"):
                    # These actions do not directly earn the official team
                    # awards. Keep their exposure heuristic separate from the
                    # bandit trained on counters for our own published content.
                    raw_scores.append(item.total_score)
                    exploration_bonuses[item.candidate.key] = 0.0
                    predicted_values[item.candidate.key] = item.total_score
                else:
                    predicted, uncertainty = self.bandit.predict(features[item.candidate.key])
                    exploration_bonuses[item.candidate.key] = self.bandit.alpha * uncertainty
                    predicted_values[item.candidate.key] = predicted
                    risk = -sum(value for name, value in item.score_components.items()
                                if name.endswith("penalty"))
                    raw_scores.append(predicted + exploration_bonuses[item.candidate.key]
                                      + item.score_components.get("action_priority", 0.0)
                                      + item.score_components.get("prior_adjustment", 0.0)
                                      + self.settings.bandit_opportunity_prior
                                      * opportunities[item.candidate.key].score
                                      - self.settings.bandit_risk_weight * risk)
            index, probability, candidate_probabilities = self._sample(scored, raw_scores)
            selection_scores = {item.candidate.key: score
                                for item, score in zip(scored, raw_scores)}
            selection_probabilities = {
                item.candidate.key: candidate_probability
                for item, candidate_probability in zip(scored, candidate_probabilities)}
            chosen = scored[index]
            selected_score = max(0.0, min(1.0, raw_scores[index]))
            components = dict(chosen.score_components)
            components["ucb_score"] = raw_scores[index]
            components["selection_probability"] = probability
            method = "linucb_softmax"
        else:
            prior_scores = {
                item.candidate.key: (
                    self.settings.wait_score if item.candidate.action_type == "WAIT"
                    else max(0.0, min(1.0,
                        self.settings.prior_heuristic * item.total_score
                        + self.settings.prior_opportunity
                        * opportunities[item.candidate.key].score
                        + self.settings.prior_urgency * strategic.urgency)))
                for item in scored
            }
            scored.sort(key=lambda item: item.candidate.key)
            index, probability, candidate_probabilities = self._sample(
                scored, [prior_scores[item.candidate.key] for item in scored])
            selection_probabilities = {
                item.candidate.key: candidate_probability
                for item, candidate_probability in zip(scored, candidate_probabilities)}
            chosen = scored[index]
            selection_scores = dict(prior_scores)
            selected_score = prior_scores[chosen.candidate.key]
            components = dict(chosen.score_components)
            components["cold_start_prior"] = selected_score
            components["selection_probability"] = probability
            method = "prior_softmax" if self.policy_mode == "bandit" else "heuristic_softmax"
        if (hot_news_layer_roll is not None and not hot_news_layer_selected
                and not priority_reply_available):
            normal_layer_probability = 1.0 - hot_news_layer_probability
            probability *= normal_layer_probability
            selection_probabilities = {
                key: value * normal_layer_probability
                for key, value in selection_probabilities.items()}
            components["selection_probability"] = probability
            components["hot_news_layer_probability"] = hot_news_layer_probability
        opportunity = opportunities[chosen.candidate.key]
        # Opportunity evidence weights sum to 0.85; the remaining strategic
        # and competition signals are explicitly unavailable. Normalize the
        # observed coverage only, never the action-selection score.
        content_confidence = (
            max(0.0, min(1.0, opportunity.evidence_coverage / 0.85))
            if chosen.candidate.action_type in ("CREATE_POST", "REPLY") else 0.0)
        follow_hint_allowed = (chosen.candidate.action_type in ("CREATE_POST", "REPLY")
                               and content_confidence >= 0.80)
        components["opportunity_score"] = opportunity.score
        components["urgency"] = strategic.urgency
        trace_candidates = []
        for item in scored:
            candidate = item.candidate
            candidate_opportunity = opportunities[candidate.key]
            trace_candidates.append({
                "key": candidate.key,
                "action_type": candidate.action_type,
                "target_id": candidate.target_id,
                "source_event_id": candidate.source_event_id,
                "rationale": candidate.rationale,
                "post_intent": candidate.context.get("post_intent"),
                "base_score": item.score_components.get("base_score", item.total_score),
                "action_prior": item.score_components.get("action_prior", 1.0),
                "heuristic_score": item.total_score,
                "predicted_value": predicted_values.get(candidate.key),
                "exploration_bonus": exploration_bonuses.get(candidate.key, 0.0),
                "score_components": dict(item.score_components),
                "selection_score": selection_scores.get(candidate.key),
                "selection_probability": selection_probabilities.get(candidate.key, 0.0),
                "opportunity": {
                    "score": candidate_opportunity.score,
                    "evidence_coverage": candidate_opportunity.evidence_coverage,
                    "components": candidate_opportunity.components,
                    "missing_signals": list(candidate_opportunity.missing_signals),
                },
                "policy_features": list(features[candidate.key]),
                "content_features": candidate.context.get("content_features", {}),
                "target_features": candidate.context.get("target_features", {}),
                "timing_features": candidate.context.get("timing_features", {}),
                "hot_news_evidence": candidate.context.get("hot_news_evidence", {}),
                "environment_features": {
                    **candidate.context.get("environment_features", {}),
                    "snapshot_id": world.metadata.get("snapshot_id"),
                    "feed_sources": candidate.context.get("feed_sources", []),
                    "data_completeness": world.metadata.get("classification"),
                },
                "selected": candidate.key == chosen.candidate.key,
            })
        decision_trace = {
            "version": "decision-trace-v1",
            "decision_id": __import__("uuid").uuid4().hex,
            "snapshot_id": world.metadata["snapshot_id"],
            "encoder_version": encoded.version,
            "policy_version": FEATURE_VERSION,
            "action_priors": dict(self.settings.action_priors),
            "score_configuration": {
                "prior_heuristic": self.settings.prior_heuristic,
                "prior_opportunity": self.settings.prior_opportunity,
                "prior_urgency": self.settings.prior_urgency,
                "bandit_opportunity_prior": self.settings.bandit_opportunity_prior,
                "bandit_risk_weight": self.settings.bandit_risk_weight,
                "wait_score": self.settings.wait_score,
                "create_post_priority": self.settings.create_post_priority,
                "reply_priority": self.settings.reply_priority,
                "own_reply_priority": self.settings.own_reply_priority,
                "minimum_post_coverage": self.settings.minimum_post_coverage,
                "post_coverage_window": self.settings.post_coverage_window,
                "post_event_cooldown_hours": self.settings.post_event_cooldown_hours,
                "text_actions_per_engagement": self.settings.text_actions_per_engagement,
                "hot_news_eligibility": "historically_trending_above_heat_threshold",
                "hot_news_score_threshold": self.settings.hot_news_score_threshold,
                "hot_news_heat_weight": self.settings.hot_news_heat_weight,
                "hot_news_cooldown_penalty": self.settings.hot_news_cooldown_penalty,
                "hot_news_layer_probability_base": 0.5,
                "hot_news_gate_cooling_factor": self.settings.hot_news_gate_cooling_factor,
            },
            "observed_at": world.metadata.get("finished_at"),
            "policy_mode": self.policy_mode,
            "selection_method": method,
            "bandit_sample_count": self.bandit.count,
            "min_samples": self.min_samples,
            "temperature": self.temperature,
            "excluded_action_types": sorted(excluded_types),
            "generated_candidate_count": len(generated_candidates),
            "post_coverage": post_coverage,
            "post_coverage_forced": coverage_forced,
            "priority_reply_override": priority_reply_override,
            "priority_reply_available": priority_reply_available,
            "hot_news_layer": {
                "available": bool(hot_news_posts),
                "probability": hot_news_layer_probability,
                "gate_cooling_fraction": hot_news_gate_cooling,
                "roll": hot_news_layer_roll,
                "selected": hot_news_layer_selected,
                "candidate_count": len(hot_news_posts),
                "reward_excluded": hot_news_layer_selected,
            },
            "content_confidence": content_confidence,
            "follow_hint_allowed": follow_hint_allowed,
            "excluded_candidates": [{
                "key": candidate.key,
                "action_type": candidate.action_type,
                "target_id": candidate.target_id,
                "source_event_id": candidate.source_event_id,
                "rationale": candidate.rationale,
                "excluded_reason": "action_type_excluded_for_round",
            } for candidate in excluded_candidates],
            "coverage_excluded_candidates": [{
                "key": candidate.key, "action_type": candidate.action_type,
                "target_id": candidate.target_id,
                "excluded_reason": (
                    "priority_reply_to_own_post" if priority_reply_override else
                    "hot_news_layer_not_selected" if (
                        candidate in hot_news_posts and not hot_news_layer_selected
                        and hot_news_layer_roll is not None) else
                    scheduler_exclusion_reason),
            } for candidate in coverage_excluded_candidates],
            "evaluated_candidate_count": len(trace_candidates),
            "fallback_wait": fallback_wait,
            "chosen_candidate_key": chosen.candidate.key,
            "strategic": {
                "urgency": strategic.urgency,
                "favorability": strategic.favorability,
                "evidence_coverage": strategic.evidence_coverage,
                "missing_signals": list(strategic.missing_signals),
            },
            "candidates": trace_candidates,
        }
        plan = ActionPlan.from_candidate(world.metadata["snapshot_id"],
                                         chosen.candidate, selected_score,
                                         components)
        plan.parameters.update({
            "policy_version": FEATURE_VERSION,
            "selection_method": method,
            "score_components": components,
            "policy_features": list(features[chosen.candidate.key]),
            "opportunity_missing": list(opportunity.missing_signals),
            "strategic_missing": list(strategic.missing_signals),
            "decision_trace": decision_trace,
        })
        if plan.action_type == "CREATE_POST":
            plan.parameters["post_intent"] = chosen.candidate.context.get(
                "post_intent", "normal_post")
        if hot_news_layer_selected:
            plan.parameters["exclude_from_reward"] = True
            plan.parameters["reward_exclusion_reason"] = "independent_hot_news_layer"
        if plan.action_type in ("CREATE_POST", "REPLY"):
            recent_profiles = [action.get("generation_profile")
                               for action in agent.recent_actions
                               if isinstance(action.get("generation_profile"), dict)]
            profile = select_generation_profile(
                plan.action_type, recent_profiles, self.content_rng)
            _LOGGER.info("Selected writing profile action=%s config=%r",
                         plan.action_type, profile)
            plan.parameters["generation_profile"] = profile
            decision_trace["generation_profile"] = profile
            plan.parameters["content_brief"] = {
                "tone": profile["tone"],
                "generation_profile": profile,
                "purpose": "share_source" if plan.action_type == "CREATE_POST" else "respond_to_post",
                "publish_decision": "now",
                "observed_at": world.metadata.get("finished_at"),
                "source_age_limit_hours": 72 if plan.action_type == "CREATE_POST" else 36,
                "content_confidence": content_confidence,
                "follow_hint_allowed": follow_hint_allowed,
                # Keep this field for compatibility, but don't give peer copy to the LLM.
                "trend_examples": [],
            }
            if plan.action_type == "CREATE_POST":
                articles = chosen.candidate.context.get("articles") or ()
                # A context-specific tag must fit whichever article the
                # generator eventually selects from this candidate.
                source_texts = [
                    " ".join(str(article.get(field) or "")
                             for field in ("title", "summary"))
                    for article in articles if isinstance(article, dict)]
                source_text = source_texts[0] if len(source_texts) == 1 else ""
                plan.parameters["content_brief"]["popular_tags"] = list(
                    select_post_tags(platform.popular_tags, source_text,
                                     agent.recent_actions, self.content_rng))
        return plan

    def _post_coverage(self, agent: AgentState) -> float:
        window = max(1, self.settings.post_coverage_window)
        completed = [action for action in agent.recent_actions
                     if action.get("status") == "SUCCEEDED"
                     and action.get("action_type") in ("CREATE_POST", "REPLY")][:window]
        if not completed:
            return None
        return sum(action.get("action_type") == "CREATE_POST" for action in completed) / len(completed)
