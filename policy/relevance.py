"""Conservative eligibility checks for the explicit H2D2 referendum task."""

import random
import re
from datetime import datetime, timezone
from math import log1p


_CAMPAIGN = re.compile(
    r"\b(?:h2d2|dolphin flu|referendum|vaccin\w*|vax\w*|mandate|"
    r"health pass|public[- ]space access|30 oct draft|voteyes|voteno)\b", re.IGNORECASE)
_VOTE_CONTEXT = re.compile(r"\b(?:vote|voting|ballot|draft|bill)\b", re.IGNORECASE)
_SIDE = re.compile(r"\b(?:yes|no)\b", re.IGNORECASE)
_SUPPORT = re.compile(
    r"\b(?:vote|voting|support|back|choose)\s+yes\b|"
    r"\ba yes (?:vote|win)\b|\bmandate now\b|#voteyes\b|#protectkingstonia\b",
    re.IGNORECASE)
_OPPOSITION = re.compile(
    r"\b(?:vote|voting|support|back|choose)\s+no\b|"
    r"\b(?:don't|do not|won't|never)\s+(?:vote|support|back)\s+yes\b|#voteno\b",
    re.IGNORECASE)
_GENERAL_TAGS = ("h2d2", "referendum")
_YES_TAGS = ("voteyes", "protectkingstonia", "proofnotlockdown", "ournurses")
_ADMINISTRATIVE = re.compile(
    r"\b(?:national id|id card|electoral commission|campaign silence|polling place|"
    r"voter registration|ballot access|voting procedure)\b", re.IGNORECASE)
_DIRECT_POLICY = re.compile(
    r"\b(?:mandate bill|compulsory vaccination|vaccine mandate|"
    r"proof of vaccination for (?:public )?(?:spaces|access)|"
    r"public[- ]space access|public access requirement)\b", re.IGNORECASE)


def campaign_relevant(text: str | None) -> bool:
    if not isinstance(text, str):
        return False
    return bool(_CAMPAIGN.search(text) or
                (_VOTE_CONTEXT.search(text) and _SIDE.search(text)))


def campaign_post_relevant(text: str | None) -> bool:
    """Require substantive policy relevance for original sourced posts.

    General campaign relevance remains useful for replies, but procedural
    referendum administration alone is not a suitable CREATE_POST source.
    """
    if not campaign_relevant(text):
        return False
    if _ADMINISTRATIVE.search(text) and not _DIRECT_POLICY.search(text):
        return False
    return True


def explicit_yes_alignment(text: str | None) -> float:
    if not campaign_relevant(text) or not isinstance(text, str):
        return 0.0
    if _OPPOSITION.search(text):
        return 0.0
    return 1.0 if _SUPPORT.search(text) else 0.0


def timely(created_at: str | None, observed_at: str | None,
           max_age_hours: float) -> bool:
    if not created_at or not observed_at:
        return False
    try:
        def parse(value):
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        age = (parse(observed_at) - parse(created_at)).total_seconds() / 3600
    except (ValueError, TypeError, AttributeError):
        return False
    return -1 <= age <= max_age_hours


def select_popular_tags(tags: list[dict]) -> tuple[str, ...]:
    """Pick current popular tags from one general and one side-relevant group."""
    ranked = sorted((item for item in tags if isinstance(item, dict)
                     and isinstance(item.get("name"), str)
                     and isinstance(item.get("post_count"), int)),
                    key=lambda item: -item["post_count"])
    selected = []
    for allowed in (_GENERAL_TAGS, _YES_TAGS):
        match = next((item["name"].lower() for item in ranked
                      if item["name"].lower() in allowed), None)
        if match is not None:
            selected.append(match)
    return tuple(selected)


_POST_TAG_CONTEXT = {
    "proofnotlockdown": re.compile(
        r"\b(?:proof|mandate|lockdown|venue|public[- ]space)\b", re.IGNORECASE),
    "proofkeepsusopen": re.compile(
        r"\b(?:proof|venue|open|business|mandate)\b", re.IGNORECASE),
    "ournurses": re.compile(
        r"\b(?:nurse|hospital|healthcare|staffing)\b", re.IGNORECASE),
    "smallbusiness": re.compile(
        r"\b(?:small business|business|employer|market)\b", re.IGNORECASE),
    "newhaven": re.compile(r"\bnew haven\b", re.IGNORECASE),
    "crownharbour": re.compile(r"\bcrown harbour\b", re.IGNORECASE),
}
_POST_TAG_ALWAYS = frozenset({
    "h2d2", "referendum", "kingstonia", "voteyes", "protectkingstonia",
})
_POST_TAG_GENERAL = frozenset({"h2d2", "referendum", "kingstonia"})


def select_post_tags(tags: list[dict], context_text: str = "",
                     recent_actions: list[dict] | None = None,
                     rng: random.Random | None = None,
                     max_tags: int = 3) -> tuple[str, ...]:
    """Sample current relevant tags, reducing repeats from recent own posts."""
    if max_tags < 1:
        return ()
    rng = rng or random.Random()
    text = context_text if isinstance(context_text, str) else ""
    counts: dict[str, int] = {}
    for item in tags:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        name = item["name"].strip().lower().lstrip("#")
        count = item.get("post_count")
        if (not re.fullmatch(r"[a-z0-9_]{2,32}", name)
                or type(count) is not int or count < 0):
            continue
        if name not in _POST_TAG_ALWAYS and not (
                name in _POST_TAG_CONTEXT and
                _POST_TAG_CONTEXT[name].search(text)):
            continue
        counts[name] = max(counts.get(name, 0), count)
    if not counts:
        return ()

    recent_posts = [
        action for action in (recent_actions or [])
        if action.get("action_type") == "CREATE_POST"
        and action.get("status") == "SUCCEEDED"
    ][:3]
    most_recent_use: dict[str, int] = {}
    for age, action in enumerate(recent_posts):
        for name in action.get("selected_tags") or ():
            if isinstance(name, str):
                most_recent_use.setdefault(name.lower(), age)

    def draw(pool: list[str]) -> str:
        weights = []
        for name in pool:
            recency_factor = (0.10, 0.30, 0.55)[most_recent_use[name]] if (
                name in most_recent_use) else 1.0
            weights.append((1.0 + log1p(counts[name])) * recency_factor)
        return rng.choices(pool, weights=weights, k=1)[0]

    pool = sorted(counts)
    selected = []
    general = [name for name in pool if name in _POST_TAG_GENERAL]
    if general:
        first = draw(general)
        selected.append(first)
        pool.remove(first)
    while pool and len(selected) < max_tags:
        name = draw(pool)
        selected.append(name)
        pool.remove(name)
    return tuple(selected)
