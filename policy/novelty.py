"""Deterministic near-duplicate scoring for authored content."""
import re
import unicodedata

ALGORITHM_VERSION = "word-shingle-jaccard-v1"


def normalize_content(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").casefold()
    text = re.sub(r"https?://\S+", " ", text)
    return " ".join(re.findall(r"[\w’'-]+", text, flags=re.UNICODE))


def exact_hash(text: str) -> str:
    import hashlib
    return hashlib.sha256(normalize_content(text).encode("utf-8")).hexdigest()


def _shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = normalize_content(text).split()
    if len(words) < n:
        return set()
    return {tuple(words[index:index + n]) for index in range(len(words) - n + 1)}


def similarity(left: str, right: str) -> float | None:
    a, b = _shingles(left), _shingles(right)
    if not a or not b:
        return None
    return len(a & b) / len(a | b)


def novelty(text: str, history: list[str], history_size: int = 100,
            duplicate_threshold: float = 0.90) -> dict:
    normalized = normalize_content(text)
    words = normalized.split()
    prior = [item for item in history[-history_size:] if isinstance(item, str)]
    if len(normalized) < 20 or len(words) < 5:
        return {"novelty_score": None, "similarity_max": None,
                "duplicate": None, "status": "UNKNOWN",
                "history_compared": 0, "history_size": history_size,
                "algorithm_version": ALGORITHM_VERSION,
                "reason": "content_too_short_for_stable_similarity"}
    current_hash = exact_hash(text)
    if any(exact_hash(item) == current_hash for item in prior):
        return {"novelty_score": 0.0, "similarity_max": 1.0,
                "duplicate": True, "status": "ASSESSED",
                "history_compared": len(prior), "history_size": history_size,
                "algorithm_version": ALGORITHM_VERSION,
                "reason": "exact_normalized_duplicate"}
    similarities = [score for item in prior
                    if (score := similarity(text, item)) is not None]
    if not similarities:
        return {"novelty_score": None, "similarity_max": None,
                "duplicate": None, "status": "UNKNOWN",
                "history_compared": 0, "history_size": history_size,
                "algorithm_version": ALGORITHM_VERSION,
                "reason": "no_comparable_history"}
    maximum = max(similarities)
    return {"novelty_score": round(1.0 - maximum, 6),
            "similarity_max": round(maximum, 6),
            "duplicate": maximum >= duplicate_threshold,
            "status": "ASSESSED", "history_compared": len(similarities),
            "history_size": history_size, "algorithm_version": ALGORITHM_VERSION,
            "reason": "maximum_word_shingle_similarity"}
