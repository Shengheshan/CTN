"""Independent, recency-balanced writing profiles for generated text actions."""

from __future__ import annotations

import random
from typing import Iterable


_PROFILES = {
    "CREATE_POST": (
        {"profile_id": "post_fact_reaction", "tone": "plainspoken",
         "angle": "one concrete source fact and the account's own judgment",
         "structure": "fact, personal interpretation, one practical implication",
         "voice": "direct and pragmatic", "intensity": "firm"},
        {"profile_id": "post_everyday_consequence", "tone": "warm",
         "angle": "how the sourced issue bears on ordinary community life",
         "structure": "human consequence first, source detail, candid reaction",
         "voice": "concerned and grounded", "intensity": "measured"},
        {"profile_id": "post_fairness_tradeoff", "tone": "thoughtful",
         "angle": "the fairness or policy trade-off supported by the source",
         "structure": "set out the tension, choose a side, give one reason",
         "voice": "considered and clear", "intensity": "measured"},
        {"profile_id": "post_accountability", "tone": "plainspoken",
         "angle": "who is accountable and what the source says remains uncertain",
         "structure": "blunt observation followed by a specific accountability point",
         "voice": "skeptical but fair", "intensity": "strong"},
        {"profile_id": "post_public_service", "tone": "warm",
         "angle": "the practical effect on public services or shared responsibilities",
         "structure": "source detail, community consequence, decisive closing thought",
         "voice": "civic-minded and vivid", "intensity": "firm"},
        {"profile_id": "post_uncertainty", "tone": "thoughtful",
         "angle": "what the source establishes and what it leaves unresolved",
         "structure": "separate known from unknown, then state the account's view",
         "voice": "measured and candid", "intensity": "low"},
        {"profile_id": "post_practical_safeguard", "tone": "plainspoken",
         "angle": "a concrete safeguard or consequence clearly framed as a proposal",
         "structure": "name the concern, propose a practical safeguard, state the view",
         "voice": "solutions-focused and direct", "intensity": "firm"},
        {"profile_id": "post_sharp_observation", "tone": "thoughtful",
         "angle": "a source-grounded contradiction or overlooked implication",
         "structure": "short observation, explain why it matters, stop without a slogan",
         "voice": "wry and incisive", "intensity": "strong"},
    ),
    "REPLY": (
        {"profile_id": "reply_direct_answer", "tone": "plainspoken",
         "angle": "answer the author's central question directly",
         "structure": "answer first, then one source-grounded reason",
         "voice": "clear and decisive", "intensity": "firm"},
        {"profile_id": "reply_concede_counterpoint", "tone": "thoughtful",
         "angle": "recognize one valid concern, then challenge its implication",
         "structure": "brief concession, specific counterpoint, clear conclusion",
         "voice": "fair but confident", "intensity": "measured"},
        {"profile_id": "reply_practical_effect", "tone": "warm",
         "angle": "the practical consequence behind the author's argument",
         "structure": "name the practical concern, connect it to the policy, respond",
         "voice": "grounded and empathetic", "intensity": "measured"},
        {"profile_id": "reply_challenge_assumption", "tone": "plainspoken",
         "angle": "challenge one assumption or unsupported leap in the post",
         "structure": "identify the leap, give a concrete counterpoint, finish cleanly",
         "voice": "sharp and composed", "intensity": "strong"},
        {"profile_id": "reply_evidence_distinction", "tone": "thoughtful",
         "angle": "separate a reported fact from an interpretation or uncertainty",
         "structure": "clarify what is known, address the interpretation, state the view",
         "voice": "careful and assured", "intensity": "low"},
        {"profile_id": "reply_safeguard", "tone": "warm",
         "angle": "respond to a concern with a clearly labeled proposal or safeguard",
         "structure": "answer the concern, offer a proposal, explain its relevance",
         "voice": "constructive and direct", "intensity": "firm"},
        {"profile_id": "reply_short_reaction", "tone": "plainspoken",
         "angle": "one concise, original reaction to the strongest point",
         "structure": "one pointed observation and one reason; no forced question",
         "voice": "natural and unscripted", "intensity": "strong"},
        {"profile_id": "reply_genuine_question", "tone": "thoughtful",
         "angle": "clarify one unresolved point that matters to the argument",
         "structure": "state the account's view, then ask one specific non-generic question",
         "voice": "curious but confident", "intensity": "measured"},
    ),
}


def select_generation_profile(action_type: str, recent_profiles: Iterable[dict],
                              rng: random.Random) -> dict:
    """Randomly choose a profile while avoiding recent exact and structural repeats."""
    if action_type not in _PROFILES:
        raise ValueError(f"no generation profiles for {action_type}")
    recent = [item for item in recent_profiles if isinstance(item, dict)]
    recent = recent[:5]
    recent_exact = {item.get("profile_id") for item in recent if item.get("profile_id")}
    recent_angles = {item.get("angle") for item in recent[:2] if item.get("angle")}
    recent_structures = {item.get("structure") for item in recent[:2]
                         if item.get("structure")}
    recent_voices = {item.get("voice") for item in recent[:2] if item.get("voice")}

    candidates = [item for item in _PROFILES[action_type]
                  if item["profile_id"] not in recent_exact
                  and item["angle"] not in recent_angles
                  and item["structure"] not in recent_structures
                  and item["voice"] not in recent_voices]
    if not candidates:
        candidates = [item for item in _PROFILES[action_type]
                      if item["profile_id"] not in recent_exact]
    if not candidates:
        candidates = list(_PROFILES[action_type])
    return dict(rng.choice(candidates))
