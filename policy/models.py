"""Shared candidate and action contracts for CTN Agent v0.2."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import NAMESPACE_URL, uuid5


ACTION_TYPES = frozenset({"CREATE_POST", "REPLY", "LIKE", "REPOST", "FOLLOW", "MENTION", "WAIT"})
TEXT_ACTIONS = frozenset({"CREATE_POST", "REPLY"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CandidateAction:
    action_type: str
    target_id: int | None = None
    source_event_id: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    estimated_cost: float = 0.0
    required_parameters: tuple[str, ...] = ()

    def __post_init__(self):
        if self.action_type not in ACTION_TYPES:
            raise ValueError("unknown action type")
        if self.action_type in {"REPLY", "LIKE", "REPOST", "FOLLOW", "MENTION"}:
            if isinstance(self.target_id, bool) or not isinstance(self.target_id, int) or self.target_id <= 0:
                raise ValueError("targeted action requires a positive integer target_id")
        elif self.target_id is not None:
            raise ValueError("this action cannot have a target_id")
        if self.action_type == "WAIT" and (self.source_event_id or self.required_parameters):
            raise ValueError("WAIT cannot require context or parameters")
        if not 0 <= self.estimated_cost <= 1:
            raise ValueError("estimated_cost must be in [0,1]")

    @property
    def key(self) -> str:
        if self.action_type == "CREATE_POST":
            generation = self.context.get("post_generation_snapshot_id")
            if generation is not None:
                return f"CREATE_POST:{self.source_event_id or 'global'}:{generation}"
        return f"{self.action_type}:{self.target_id or self.source_event_id or 'global'}"


@dataclass
class ActionPlan:
    action_id: str
    snapshot_id: int
    action_type: str
    target_id: int | None
    source_event_id: str | None
    score: float
    score_components: dict[str, float]
    reason: str
    parameters: dict[str, Any]
    generated_content: str | None
    status: str
    created_at: str

    @property
    def action_key(self) -> str:
        if self.action_type == "CREATE_POST":
            context = self.parameters.get("context", {})
            generation = context.get("post_generation_snapshot_id")
            if generation is not None:
                return f"CREATE_POST:{self.source_event_id or 'global'}:{generation}"
        return f"{self.action_type}:{self.target_id or self.source_event_id or 'global'}"

    @classmethod
    def from_candidate(cls, snapshot_id: int, candidate: CandidateAction,
                       score: float, components: dict[str, float]) -> "ActionPlan":
        identity = f"{snapshot_id}:{candidate.key}"
        return cls(
            action_id=str(uuid5(NAMESPACE_URL, "ctn-agent-v0.2:" + identity)),
            snapshot_id=snapshot_id,
            action_type=candidate.action_type,
            target_id=candidate.target_id,
            source_event_id=candidate.source_event_id,
            score=score,
            score_components=components,
            reason=candidate.rationale,
            parameters={"context": candidate.context, "required_parameters": list(candidate.required_parameters)},
            generated_content=None,
            status="PLANNED",
            created_at=utc_now(),
        )

    def validate(self, max_chars: int = 500) -> None:
        if self.action_type not in ACTION_TYPES or not 0 <= self.score <= 1:
            raise ValueError("invalid action plan")
        if self.action_type in TEXT_ACTIONS:
            if not self.generated_content or not self.generated_content.strip():
                raise ValueError("text action requires generated content")
            if len(self.generated_content) > max_chars:
                raise ValueError("generated content exceeds character limit")
        if self.action_type == "WAIT" and (self.target_id is not None or self.generated_content):
            raise ValueError("WAIT cannot target or publish")

