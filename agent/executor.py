"""Dry-run or bounded live SDK execution with durable pre-write reservation."""

import time
import urllib.error
from dataclasses import dataclass

from agent.history import ActionHistory
from policy.models import ActionPlan


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    remote_id: str | None = None
    error: str | None = None
    response: dict | None = None


class ActionExecutor:
    def __init__(self, social, account=None, history: ActionHistory | None = None,
                 sleep=time.sleep, max_retry_wait: float = 30.0):
        self.social = social
        self.account = account
        self.history = history
        self.sleep = sleep
        self.max_retry_wait = max_retry_wait

    def execute(self, plan: ActionPlan, mode: str = "dry-run") -> ExecutionResult:
        if mode not in ("dry-run", "live"):
            raise ValueError("mode must be dry-run or live")
        plan.validate()
        if plan.action_type == "WAIT":
            result = ExecutionResult("WAIT")
            self._save(plan, result)
            return result
        if mode == "dry-run":
            result = ExecutionResult("DRY_RUN")
            self._save(plan, result)
            return result
        if self.account is None or self.history is None:
            raise RuntimeError("Live execution requires account and history")
        try:
            lock = self.social.feed_lock()
            if lock.get("locked", False):
                result = ExecutionResult("BLOCKED", error="feed locked")
                self._save(plan, result)
                return result
        except Exception as exc:
            result = ExecutionResult("BLOCKED", error=f"feed lock unavailable: {type(exc).__name__}")
            self._save(plan, result)
            return result
        if not self.history.reserve(plan):
            return ExecutionResult("BLOCKED", error="duplicate or uncertain action")

        for attempt in range(2):
            try:
                payload = self._call_sdk(plan)
                remote_id = str(payload.get("post_id")) if isinstance(payload, dict) and payload.get("post_id") else None
                result = ExecutionResult("SUCCEEDED", remote_id=remote_id,
                                         response=payload if isinstance(payload, dict) else {})
                self._save(plan, result)
                return result
            except Exception as exc:
                status = getattr(exc, "status", None)
                if status == 429:
                    wait = getattr(exc, "retry_after", None)
                    if attempt == 0 and isinstance(wait, (int, float)) and 0 <= wait <= self.max_retry_wait:
                        self.sleep(wait)
                        continue
                    result = ExecutionResult("BLOCKED", error="rate limited")
                elif status in (401, 403, 409):
                    result = ExecutionResult("BLOCKED", error=f"HTTP {status}")
                elif isinstance(exc, (TimeoutError, ConnectionError, urllib.error.URLError)):
                    # A write may have arrived even when its response was lost.
                    result = ExecutionResult("UNKNOWN", error=type(exc).__name__)
                elif (plan.action_type in ("CREATE_POST", "REPLY")
                      and isinstance(status, int) and status >= 500):
                    # A server error can arrive after the write was committed.
                    # Do not retry a non-idempotent post/reply blindly.
                    result = ExecutionResult("UNKNOWN", error=f"HTTP {status}; write outcome uncertain")
                elif status is not None:
                    result = ExecutionResult("FAILED", error=f"HTTP {status}")
                else:
                    result = ExecutionResult("UNKNOWN", error=type(exc).__name__)
                self._save(plan, result)
                return result
        raise AssertionError("unreachable")

    def _call_sdk(self, plan: ActionPlan):
        account = self.account
        if plan.action_type == "CREATE_POST":
            return account.post(plan.generated_content)
        if plan.action_type == "REPLY":
            root_id = plan.parameters.get("context", {}).get("root_post_id")
            return account.reply(plan.target_id, plan.generated_content,
                                 root_post_id=root_id if isinstance(root_id, int) else None)
        if plan.action_type == "LIKE":
            if plan.target_id in account.my_likes([plan.target_id]):
                return {"already_present": True}
            return account.like(plan.target_id)
        if plan.action_type == "REPOST":
            if plan.target_id in account.my_reposts([plan.target_id]):
                return {"already_present": True}
            return account.repost(plan.target_id)
        if plan.action_type == "FOLLOW":
            return account.follow(plan.target_id)
        if plan.action_type == "MENTION":
            post_id = plan.parameters.get("context", {}).get("mention_on_post_id")
            if not isinstance(post_id, int) or post_id <= 0:
                raise ValueError("mention requires an own post id")
            return account.mention(post_id, plan.target_id)
        raise ValueError("unsupported action")

    def _save(self, plan: ActionPlan, result: ExecutionResult):
        plan.status = result.status
        if self.history is not None:
            self.history.save_result(plan, result.status, result.remote_id,
                                     result.error, result.response)
