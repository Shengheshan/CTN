"""Deterministic structural features for published text; no semantic claims."""

import re
import unicodedata


_WORD = re.compile(r"\b[\w’'-]+\b", re.UNICODE)
_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_FIRST_PERSON = re.compile(r"\b(?:i|we|my|our|me|us)\b", re.IGNORECASE)
_HASHTAG = re.compile(r"(?<!\w)#[\w]+", re.UNICODE)


def extract_content_features(text: str) -> dict:
    """Return reproducible surface features useful for offline stratification."""
    if not isinstance(text, str):
        text = ""
    without_urls = _URL.sub(" ", text)
    words = _WORD.findall(without_urls)
    normalized_words = [word.casefold() for word in words]
    unique_ratio = (len(set(normalized_words)) / len(normalized_words)
                    if normalized_words else 0.0)
    emoji_count = sum(
        unicodedata.category(char) in ("So", "Sk")
        or "REGIONAL INDICATOR" in unicodedata.name(char, "")
        for char in text
    )
    question_count = text.count("?")
    exclamation_count = text.count("!")
    sentence_count = len(re.findall(r"[.!?]+", without_urls))
    url_count = len(_URL.findall(text))
    hashtag_count = len(_HASHTAG.findall(text))
    first_person_count = len(_FIRST_PERSON.findall(without_urls))

    if question_count:
        primary_structure = "question"
    elif url_count:
        primary_structure = "link_bearing"
    elif first_person_count:
        primary_structure = "first_person"
    elif emoji_count:
        primary_structure = "emoji_bearing"
    elif exclamation_count:
        primary_structure = "exclamatory"
    elif sentence_count > 1:
        primary_structure = "multi_sentence"
    else:
        primary_structure = "plain_statement"

    return {
        "char_count": len(text),
        "word_count": len(words),
        "sentence_count": sentence_count,
        "question_count": question_count,
        "exclamation_count": exclamation_count,
        "emoji_count_approx": emoji_count,
        "hashtag_count": hashtag_count,
        "url_count": url_count,
        "first_person_count": first_person_count,
        "unique_word_ratio": round(unique_ratio, 4),
        "line_count": text.count("\n") + 1,
        "primary_structure": primary_structure,
    }
