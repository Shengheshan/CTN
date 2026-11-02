"""Versioned surface/content assessment for action plans."""
import re
from policy.novelty import novelty
from policy.source_validator import validate_sources

_NUMBER = re.compile(r"\b\d+(?:\.\d+)?%?\b")
_QUESTION = re.compile(r"\?")
_PERSONAL_STORY = re.compile(r"\b(?:i|we)\b.{0,100}\b(?:saw|felt|went|worked|lived|remember|experienced|visited|helped)\b", re.I)


class ContentFeatureAnalyzer:
    VERSION = "content-analyzer-v1"

    def __init__(self, history_size: int = 100, duplicate_threshold: float = 0.90):
        if history_size < 1 or not 0 <= duplicate_threshold <= 1:
            raise ValueError("invalid novelty configuration")
        self.history_size = history_size
        self.duplicate_threshold = duplicate_threshold

    def analyze(self, plan, past_contents: list[str]) -> dict:
        content = plan.generated_content or ""
        context = plan.parameters.get("context", {})
        articles = context.get("articles", []) if isinstance(context, dict) else []
        novelty_result = novelty(content, past_contents, self.history_size,
                                 self.duplicate_threshold)
        source_result = validate_sources(content, articles)
        words = re.findall(r"\b[\w’'-]+\b", content)
        target_text = context.get("post_body", "") if isinstance(context, dict) else ""
        tokens = lambda value: [word.lower() for word in re.findall(r"\b[\w’'-]+\b", value or "")
                                if len(word) > 2]
        content_terms, target_terms = set(tokens(content)), set(tokens(target_text))
        relevance = (len(content_terms & target_terms) / len(content_terms | target_terms)
                     if content_terms and target_terms else None)
        return {
            "has_news_reference": source_result["source_count"] > 0,
            "reference_count": source_result["source_count"],
            "has_numeric_evidence": bool(_NUMBER.search(content)),
            "has_question": bool(_QUESTION.search(content)),
            "has_personal_story": bool(_PERSONAL_STORY.search(content)),
            "content_length": len(content),
            "word_count": len(words),
            "context_relevance_score": relevance,
            "content_features": {"content": content,
                                 "context_relevance": relevance,
                                 "originality": novelty_result["novelty_score"],
                                 "factual_basis": source_result},
            "target_features": plan.parameters.get("target_features", {}),
            "timing_features": {"created_at": plan.created_at,
                                **(context.get("timing_features", {}) if isinstance(context, dict) else {})},
            "environment_features": {"snapshot_id": plan.snapshot_id,
                                      "feed_sources": context.get("feed_sources", []) if isinstance(context, dict) else []},
            "novelty_score": novelty_result["novelty_score"],
            "duplicate": novelty_result["duplicate"],
            "novelty_status": novelty_result["status"],
            "novelty_details": novelty_result,
            "source_alignment_score": source_result["source_alignment_score"],
            "fact_check_status": source_result["fact_check_status"],
            "source_validation": source_result,
            "analyzer_version": self.VERSION,
        }
