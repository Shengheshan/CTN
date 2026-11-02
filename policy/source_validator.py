"""Conservative local URL/source matching; it does not claim to fact-check prose."""
import re
from urllib.parse import urlsplit, urlunsplit

_URL = re.compile(r"https?://[^\s<>\])}]+", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9][a-z0-9'-]*", re.IGNORECASE)
_NUMBER = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b")


def _normal_url(url: str) -> str:
    try:
        parts = urlsplit(url.rstrip(".,;:!?)]}"))
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(),
                           parts.path.rstrip("/"), parts.query, ""))
    except ValueError:
        return url.rstrip("/")


def validate_sources(content: str, articles: list[dict] | None) -> dict:
    articles = [item for item in (articles or []) if isinstance(item, dict)]
    urls = list(dict.fromkeys(_normal_url(item) for item in _URL.findall(content or "")))
    by_url = {_normal_url(str(item.get("url"))): item for item in articles
              if item.get("url")}
    matched = [by_url[url] for url in urls if url in by_url]
    unmatched = [url for url in urls if url not in by_url]
    supplied = [item for item in articles if item.get("title") or item.get("summary")]

    # Exclude the mechanically appended attribution so it cannot inflate alignment.
    prose = re.sub(r"\bSource\s*:\s*.+$", " ", content or "", flags=re.I)
    prose = _URL.sub(" ", prose)
    content_terms = {word.casefold() for word in _WORD.findall(prose)
                     if len(word) > 2}
    matched_text = " ".join(" ".join(str(item.get(key) or "")
                                    for key in ("title", "summary"))
                            for item in matched)
    source_terms = {word.casefold() for word in _WORD.findall(matched_text)
                    if len(word) > 2}
    overlap = len(content_terms & source_terms)
    alignment = (overlap / len(content_terms)) if content_terms and source_terms else None
    content_numbers = {item.replace(",", "") for item in _NUMBER.findall(prose)}
    source_numbers = {item.replace(",", "") for item in _NUMBER.findall(matched_text)}
    unsupported_values = content_numbers - source_numbers
    unsupported = len(unsupported_values) if matched else None
    contradicted = False
    if matched:
        source_sentences = re.split(r"[.!?;]+", matched_text)
        prose_sentences = re.split(r"[.!?;]+", prose)
        for sentence in prose_sentences:
            sentence_numbers = {item.replace(",", "") for item in _NUMBER.findall(sentence)}
            if not sentence_numbers:
                continue
            sentence_terms = {word.casefold() for word in _WORD.findall(sentence)
                              if len(word) > 3}
            for source_sentence in source_sentences:
                source_sentence_numbers = {item.replace(",", "")
                                           for item in _NUMBER.findall(source_sentence)}
                source_terms = {word.casefold() for word in _WORD.findall(source_sentence)
                                if len(word) > 3}
                if (sentence_terms & source_terms and source_sentence_numbers
                        and sentence_numbers.isdisjoint(source_sentence_numbers)):
                    contradicted = True
                    break
            if contradicted:
                break

    if matched and alignment is not None:
        validation = "CONTRADICTED" if contradicted else "PARTIALLY_VERIFIED"
        # Lexical relevance is deliberately capped: it cannot establish truth.
        score = 0.0 if contradicted else min(0.5, alignment * 0.5)
        if unsupported and not contradicted:
            score *= max(0.0, 1.0 - unsupported / max(1, len(content_numbers)))
    else:
        validation = "UNVERIFIED"
        score = 0.0 if urls or supplied else None
    if not urls and not supplied:
        validation = "UNVERIFIED"
        score = None

    return {
        "source_count": len(urls),
        "matched_source_count": len(matched),
        "unverified_source_count": len(unmatched) + max(0, len(supplied) - len(matched)),
        "source_alignment_score": round(score, 6) if score is not None else None,
        "unsupported_claim_count": unsupported,
        "validation_status": validation,
        "fact_check_status": "CONTRADICTED" if contradicted else "UNKNOWN",
        "fact_check_evidence": {
            "matched_urls": [url for url in urls if url in by_url],
            "unmatched_urls": unmatched,
            "lexical_overlap_terms": overlap,
            "numeric_claims_without_source_match": sorted(unsupported_values) if matched else None,
            "numeric_context_conflict_detected": contradicted,
            "note": "URL matching and lexical overlap do not establish factual truth; no claim-level contradiction checker ran.",
        },
        "validator_version": "local-source-match-v1",
    }
