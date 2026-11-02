# Central environment configuration; secrets remain in .env and are never logged.
import json
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
CTN_TEAM_CODE = os.getenv("CTN_TEAM_CODE")


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _env_windows() -> tuple[int, ...]:
    raw = os.getenv("CTN_FEEDBACK_WINDOWS", "300,1800,7200,21600")
    try:
        values = tuple(sorted({int(item.strip()) for item in raw.split(",") if item.strip()}))
    except ValueError as exc:
        raise ValueError("CTN_FEEDBACK_WINDOWS must be comma-separated seconds") from exc
    if not values or any(value <= 0 for value in values):
        raise ValueError("feedback windows must be positive seconds")
    return values


@dataclass(frozen=True)
class RuntimeConfig:
    team_code: str | None
    username: str | None
    password: str | None
    token: str | None
    observe_interval: float
    max_actions_per_round: int
    enable_topics: bool
    enable_feedback: bool
    enable_policy_analytics: bool
    enable_learning: bool
    feedback_windows: tuple[int, ...]
    reward_version: str
    reward_weights: dict[str, float]
    novelty_history_size: int
    novelty_duplicate_threshold: float

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        threshold = float(os.getenv("CTN_NOVELTY_DUPLICATE_THRESHOLD", "0.90"))
        if not 0 <= threshold <= 1:
            raise ValueError("CTN_NOVELTY_DUPLICATE_THRESHOLD must be in [0,1]")
        history_size = int(os.getenv("CTN_NOVELTY_HISTORY_SIZE", "100"))
        try:
            reward_weights = json.loads(os.getenv("CTN_REWARD_WEIGHTS", "{}"))
        except json.JSONDecodeError as exc:
            raise ValueError("CTN_REWARD_WEIGHTS must be a JSON object") from exc
        if not isinstance(reward_weights, dict) or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
                for value in reward_weights.values()):
            raise ValueError("CTN_REWARD_WEIGHTS must map component names to nonnegative numbers")
        if history_size < 1:
            raise ValueError("CTN_NOVELTY_HISTORY_SIZE must be positive")
        return cls(
            team_code=os.getenv("CTN_TEAM_CODE"), username=os.getenv("CTN_USERNAME"),
            password=os.getenv("CTN_PASSWORD"), token=os.getenv("CTN_TOKEN"),
            observe_interval=float(os.getenv("CTN_OBSERVE_INTERVAL", "60")),
            max_actions_per_round=int(os.getenv("CTN_MAX_ACTIONS_PER_ROUND", "1")),
            enable_topics=_env_bool("CTN_ENABLE_TOPICS"),
            enable_feedback=_env_bool("CTN_ENABLE_FEEDBACK", True),
            enable_policy_analytics=_env_bool("CTN_ENABLE_POLICY_ANALYTICS", True),
            enable_learning=_env_bool("CTN_ENABLE_LEARNING", False),
            feedback_windows=_env_windows(),
            reward_version=os.getenv("CTN_REWARD_VERSION", "reward-v2-action-feedback"),
            reward_weights={key: float(value) for key, value in reward_weights.items()},
            novelty_history_size=history_size,
            novelty_duplicate_threshold=threshold,
        )
