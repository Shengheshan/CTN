"""Read-only platform counter sampling and local action audit."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


COUNTER_FIELDS = ("view_count", "like_count", "reply_count", "repost_count")
FEEDBACK_WINDOWS = (
    ("1m", 60), ("5m", 300), ("15m", 900), ("30m", 1800),
    ("1h", 3600), ("3h", 10800), ("6h", 21600), ("24h", 86400),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ObservationMonitor:
    def __init__(self, db_path: str | Path, social):
        self.db_path = Path(db_path)
        self.social = social
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS observed_post_counters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_id TEXT NOT NULL,
                remote_id TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                delta_json TEXT NOT NULL,
                error_type TEXT,
                FOREIGN KEY(action_id) REFERENCES agent_actions(action_id))""")
            db.execute("""CREATE INDEX IF NOT EXISTS idx_counter_action
                ON observed_post_counters(action_id, id)""")
            db.execute("""CREATE TABLE IF NOT EXISTS feedback_window_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_id TEXT NOT NULL,
                remote_id TEXT NOT NULL,
                window_label TEXT NOT NULL,
                target_age_seconds INTEGER NOT NULL,
                observed_at TEXT NOT NULL,
                actual_age_seconds REAL NOT NULL,
                status TEXT NOT NULL,
                metrics_json TEXT NOT NULL DEFAULT '{}',
                delta_json TEXT NOT NULL DEFAULT '{}',
                interval_start_at TEXT,
                interval_start_age_seconds REAL,
                crosses_age_window INTEGER,
                error_type TEXT,
                UNIQUE(action_id, window_label))""")
            db.execute("""CREATE INDEX IF NOT EXISTS idx_feedback_window_action
                ON feedback_window_observations(action_id, target_age_seconds)""")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.db_path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def recent_actions(self, limit: int = 20) -> list[dict]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._connect() as db:
            rows = db.execute("""SELECT action_id,action_type,target_id,status,
                    created_at,executed_at,remote_id,parameters_json,error_message
                FROM agent_actions ORDER BY created_at DESC LIMIT ?""", (limit,)).fetchall()
        actions = []
        for row in rows:
            parameters = json.loads(row[7])
            actions.append({
                "action_id": row[0], "action_type": row[1], "target_id": row[2],
                "status": row[3], "created_at": row[4], "executed_at": row[5],
                "remote_id": row[6],
                "selection_method": parameters.get("selection_method"),
                "error_type": row[8],
            })
        return actions

    def observe_once(self, limit: int = 20) -> list[dict]:
        """Read counters for recent successful own posts and store raw counts."""
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._connect() as db:
            rows = db.execute("""SELECT action_id,action_type,remote_id
                FROM agent_actions
                WHERE status='SUCCEEDED'
                  AND action_type IN ('CREATE_POST','REPLY')
                  AND remote_id IS NOT NULL
                ORDER BY executed_at DESC LIMIT ?""", (limit,)).fetchall()
        results = []
        for action_id, action_type, remote_id in rows:
            observed_at = utc_now()
            error_type = None
            try:
                response = self.social.counters(int(remote_id)) or {}
                metrics = {
                    key: value for key in COUNTER_FIELDS
                    if isinstance((value := response.get(key)), int)
                    and not isinstance(value, bool) and value >= 0
                }
            except Exception as exc:
                metrics = {}
                error_type = type(exc).__name__
            with self._connect() as db:
                previous = db.execute("""SELECT metrics_json
                    FROM observed_post_counters WHERE action_id=? AND error_type IS NULL
                    ORDER BY id DESC LIMIT 1""", (action_id,)).fetchone()
                previous_metrics = json.loads(previous[0]) if previous else {}
                delta = {key: value - previous_metrics[key] for key, value in metrics.items()
                         if key in previous_metrics}
                db.execute("""INSERT INTO observed_post_counters
                    (action_id,remote_id,observed_at,metrics_json,delta_json,error_type)
                    VALUES (?,?,?,?,?,?)""",
                    (action_id, remote_id, observed_at, json.dumps(metrics),
                     json.dumps(delta), error_type))
            results.append({
                "action_id": action_id, "action_type": action_type,
                "remote_id": remote_id, "observed_at": observed_at,
                "metrics": metrics, "delta": delta, "error_type": error_type,
            })
        return results

    def observe_due_windows(self, limit: int = 500,
                            poll_interval_seconds: float = 60.0) -> list[dict]:
        """Sample published text at fixed ages after execution, read-only.

        A late/missed window is recorded as MISSED rather than pretending that
        a later cumulative counter is a measurement at the requested age.
        """
        if limit < 1 or poll_interval_seconds < 1:
            raise ValueError("limit and polling interval must be positive")
        cycle_time = datetime.now(timezone.utc)
        grace_seconds = max(2 * poll_interval_seconds, 120.0)
        with self._connect() as db:
            actions = db.execute("""SELECT action_id,action_type,remote_id,executed_at
                FROM agent_actions
                WHERE status='SUCCEEDED'
                  AND action_type IN ('CREATE_POST','REPLY')
                  AND remote_id IS NOT NULL AND executed_at IS NOT NULL
                ORDER BY executed_at DESC LIMIT ?""", (limit,)).fetchall()

        observations = []
        for action_id, action_type, remote_id, executed_at in actions:
            try:
                created = datetime.fromisoformat(executed_at.replace("Z", "+00:00"))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError, AttributeError):
                continue
            age_now = max(0.0, (cycle_time - created.astimezone(timezone.utc)).total_seconds())
            for label, target_age in FEEDBACK_WINDOWS:
                if age_now < target_age:
                    continue
                with self._connect() as db:
                    existing = db.execute("""SELECT status
                        FROM feedback_window_observations
                        WHERE action_id=? AND window_label=?""",
                        (action_id, label)).fetchone()
                    if existing and existing[0] != "ERROR":
                        continue
                    previous = db.execute("""SELECT observed_at,actual_age_seconds,metrics_json
                        FROM feedback_window_observations
                        WHERE action_id=? AND status='OBSERVED'
                        ORDER BY observed_at DESC LIMIT 1""", (action_id,)).fetchone()

                if age_now > target_age + grace_seconds:
                    status, metrics, delta, error_type = "MISSED", {}, {}, None
                    sampled_at = datetime.now(timezone.utc)
                else:
                    try:
                        response = self.social.counters(int(remote_id)) or {}
                        metrics = {
                            key: value for key in COUNTER_FIELDS
                            if isinstance((value := response.get(key)), int)
                            and not isinstance(value, bool) and value >= 0
                        }
                        status, error_type = "OBSERVED", None
                    except Exception as exc:
                        metrics = {}
                        status, error_type = "ERROR", type(exc).__name__
                    sampled_at = datetime.now(timezone.utc)
                    previous_metrics = json.loads(previous[2]) if previous else {}
                    delta = {key: value - previous_metrics[key]
                             for key, value in metrics.items()
                             if key in previous_metrics}

                age = max(0.0, (sampled_at - created.astimezone(timezone.utc)).total_seconds())
                interval_start_at = previous[0] if previous else None
                interval_start_age = float(previous[1]) if previous else None
                crossed = None
                if interval_start_age is not None:
                    crossed = int(_crosses_age_window(interval_start_age, age))
                observed_at = sampled_at.isoformat()
                with self._connect() as db:
                    db.execute("""INSERT INTO feedback_window_observations
                        (action_id,remote_id,window_label,target_age_seconds,observed_at,
                         actual_age_seconds,status,metrics_json,delta_json,interval_start_at,
                         interval_start_age_seconds,crosses_age_window,error_type)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(action_id,window_label) DO UPDATE SET
                         observed_at=excluded.observed_at,
                         actual_age_seconds=excluded.actual_age_seconds,
                         status=excluded.status,metrics_json=excluded.metrics_json,
                         delta_json=excluded.delta_json,interval_start_at=excluded.interval_start_at,
                         interval_start_age_seconds=excluded.interval_start_age_seconds,
                         crosses_age_window=excluded.crosses_age_window,
                         error_type=excluded.error_type""",
                        (action_id, remote_id, label, target_age, observed_at, age,
                         status, json.dumps(metrics), json.dumps(delta), interval_start_at,
                         interval_start_age, crossed, error_type))
                observations.append({
                    "action_id": action_id, "action_type": action_type,
                    "remote_id": remote_id, "window": label,
                    "target_age_seconds": target_age,
                    "actual_age_seconds": round(age, 1), "status": status,
                    "observed_at": observed_at, "metrics": metrics,
                    "delta": delta, "interval_start_at": interval_start_at,
                    "interval_start_age_seconds": interval_start_age,
                    "crosses_age_window": crossed, "error_type": error_type,
                })
        return observations


def _age_window(age_seconds: float) -> str:
    """Bucket an observation age for interval-boundary diagnostics."""
    bounds = ((300, "0-5m"), (900, "5-15m"), (1800, "15-30m"),
              (3600, "30-60m"), (10800, "1-3h"), (21600, "3-6h"),
              (86400, "6-24h"))
    for upper, label in bounds:
        if age_seconds <= upper:
            return label
    return ">24h"


def _crosses_age_window(start_age: float, end_age: float) -> bool:
    """Whether an observed counter delta straddles an internal bin boundary."""
    lower_bounds = (300, 900, 1800, 3600, 10800, 21600, 86400)
    start_window = _age_window(start_age)
    end_window = _age_window(end_age)
    if start_window == end_window:
        return False
    end_lower = next((bound for bound in lower_bounds
                      if _age_window(bound + 0.001) == end_window), None)
    return end_lower is None or start_age < end_lower
