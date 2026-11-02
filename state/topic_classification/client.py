
# state/classification/client.py

import hashlib
import os
import random
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import capturethenarrative as ctn

from openai import (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    AuthenticationError,
    PermissionDeniedError,
    BadRequestError
)


_LOCAL_LOCKS = {}
_LOCAL_LOCKS_GUARD = threading.Lock()


def _local_lock(team_key: str):
    with _LOCAL_LOCKS_GUARD:
        return _LOCAL_LOCKS.setdefault(team_key, threading.Lock())


@contextmanager
def _file_lock(handle):
    """Serialize requests across local processes sharing the same team code."""
    if os.name == "nt":
        import msvcrt
        while True:
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                time.sleep(0.1)
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class LLMClient:

    MIN_REQUEST_INTERVAL_SECONDS = 21.0

    def __init__(self, team_code: str):
        team_key = hashlib.sha256(team_code.encode("utf-8")).hexdigest()[:16]
        self._local_lock = _local_lock(team_key)
        self._rate_limit_path = (
            Path(__file__).resolve().parents[2] / "data"
            / f".llm_rate_limit_{team_key}.lock")
        self._min_request_interval = self.MIN_REQUEST_INTERVAL_SECONDS
        self._sleep = time.sleep
        self._clock = time.time
        self.client = ctn.competitorllm.CompetitorLLM(
            team_code=team_code,
            max_retries=0
        )

    @staticmethod
    def _read_deadline(handle) -> float:
        handle.seek(0)
        try:
            return float(handle.readline().decode("ascii").strip() or 0.0)
        except (ValueError, UnicodeDecodeError):
            return 0.0

    @staticmethod
    def _write_deadline(handle, deadline: float):
        handle.seek(0)
        handle.truncate()
        handle.write(f"{deadline:.6f}\n".encode("ascii"))
        handle.flush()
        os.fsync(handle.fileno())

    @contextmanager
    def _request_slot(self):
        self._rate_limit_path.parent.mkdir(parents=True, exist_ok=True)
        with self._local_lock:
            with self._rate_limit_path.open("a+b") as handle:
                # Keep a lockable byte at offset zero on Windows. A concurrent
                # first start may write twice; the first line remains valid.
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"0\n")
                    handle.flush()
                with _file_lock(handle):
                    wait = max(0.0, self._read_deadline(handle) - self._clock())
                    if wait:
                        self._sleep(wait)
                    request_started = self._clock()
                    self._write_deadline(
                        handle, request_started + self._min_request_interval)

                    def defer(seconds: float):
                        next_allowed = max(
                            self._read_deadline(handle),
                            self._clock() + max(0.0, seconds))
                        self._write_deadline(handle, next_allowed)

                    yield defer

    def ask(
        self,
        prompt: str,
        system: str,
        max_tokens: int = 500,
        temperature: float | None = None,
    ) -> str:

        max_attempts = 3

        for attempt in range(max_attempts):
            retry_error = None
            retry_delay = 0.0
            try:
                with self._request_slot() as defer:
                    options = {"system": system, "max_tokens": max_tokens}
                    if temperature is not None:
                        options["temperature"] = temperature
                    try:
                        return self.client.ask(prompt, **options)
                    except RateLimitError as exc:
                        retry_error = exc
                        headers = getattr(
                            getattr(exc, "response", None), "headers", {})
                        try:
                            retry_after = float(headers.get("retry-after", 20))
                        except (ValueError, TypeError):
                            retry_after = 20.0
                        retry_delay = max(1.0, retry_after) + random.uniform(0, 0.5)
                        defer(retry_delay)
                    except (APIConnectionError, APITimeoutError,
                            InternalServerError) as exc:
                        retry_error = exc
                        retry_delay = min(2 ** attempt, 8) + random.uniform(0, 0.5)
                        defer(retry_delay)
            except (AuthenticationError, PermissionDeniedError, BadRequestError):
                raise

            if retry_error is None:
                raise RuntimeError("LLM request ended without a response or retryable error")
            if attempt == max_attempts - 1:
                raise retry_error
            if isinstance(retry_error, RateLimitError):
                print(f"Rate limited. Retry in {retry_delay:.1f}s")
            else:
                print(f"Temporary error: {type(retry_error).__name__}. Retrying...")

        raise RuntimeError("LLM request failed")
