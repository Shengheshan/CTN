"""Additive v0.3 SQLite tables. Existing v0.2 tables and rows are untouched."""
import hashlib
import json
import sqlite3
from math import isfinite
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

_METRICS = {"likes": "like_count", "replies": "reply_count",
            "reposts": "repost_count", "views": "view_count",
            "follows": "follow_count"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _parse_time(value):
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class V03Store:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS policy_decisions (
              decision_id TEXT PRIMARY KEY, snapshot_id INTEGER NOT NULL,
              policy_version TEXT, encoder_version TEXT, selected_action_id TEXT,
              selection_method TEXT, candidate_count INTEGER NOT NULL,
              created_at TEXT NOT NULL, decision_context_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS candidate_scores (
              decision_id TEXT NOT NULL, candidate_id TEXT NOT NULL,
              action_type TEXT NOT NULL, target_id TEXT, base_score REAL,
              action_prior REAL, novelty_score REAL, source_quality_score REAL,
              cost_penalty REAL, risk_penalty REAL, exploration_bonus REAL,
              final_score REAL, selected INTEGER NOT NULL,
              components_json TEXT NOT NULL,
              PRIMARY KEY(decision_id,candidate_id),
              FOREIGN KEY(decision_id) REFERENCES policy_decisions(decision_id));
            CREATE TABLE IF NOT EXISTS content_assessments (
              action_id TEXT PRIMARY KEY, content_hash TEXT, analyzer_version TEXT NOT NULL,
              features_json TEXT NOT NULL, assessed_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS action_feedback_snapshots (
              feedback_id TEXT PRIMARY KEY, action_id TEXT NOT NULL, remote_id TEXT,
              action_type TEXT NOT NULL, window_name TEXT NOT NULL,
              scheduled_at TEXT NOT NULL, observed_at TEXT,
              likes INTEGER, replies INTEGER, reposts INTEGER, views INTEGER, follows INTEGER,
              previous_metrics_json TEXT, delta_metrics_json TEXT, interval_seconds REAL,
              raw_metrics_json TEXT NOT NULL DEFAULT '{}',
              feedback_status TEXT NOT NULL, error_message TEXT, attempt_count INTEGER NOT NULL DEFAULT 0,
              UNIQUE(action_id,window_name),
              FOREIGN KEY(action_id) REFERENCES agent_actions(action_id));
            CREATE INDEX IF NOT EXISTS idx_feedback_due
              ON action_feedback_snapshots(feedback_status,scheduled_at);
            CREATE TABLE IF NOT EXISTS action_rewards (
              reward_id TEXT PRIMARY KEY, action_id TEXT NOT NULL,
              reward_version TEXT NOT NULL, feedback_window TEXT,
              execution_component REAL, factual_component REAL, novelty_component REAL,
              compliance_component REAL, context_relevance_component REAL,
              observed_feedback_component REAL, reward_total REAL, reward_status TEXT NOT NULL,
              computed_at TEXT NOT NULL, calculation_json TEXT NOT NULL,
              calculation_hash TEXT NOT NULL,
              UNIQUE(action_id,reward_version,feedback_window,calculation_hash),
              FOREIGN KEY(action_id) REFERENCES agent_actions(action_id));
            CREATE INDEX IF NOT EXISTS idx_rewards_action
              ON action_rewards(action_id,reward_version,computed_at);
            CREATE TABLE IF NOT EXISTS policy_updates (
              update_id TEXT PRIMARY KEY, policy_version TEXT NOT NULL,
              action_id TEXT NOT NULL, reward_id TEXT,
              parameters_before_json TEXT NOT NULL, parameters_after_json TEXT NOT NULL,
              update_reason TEXT NOT NULL, learning_rate REAL,
              created_at TEXT NOT NULL, update_applied INTEGER NOT NULL,
              FOREIGN KEY(action_id) REFERENCES agent_actions(action_id));
            CREATE TABLE IF NOT EXISTS policy_states (
              policy_version TEXT PRIMARY KEY, parameters_json TEXT NOT NULL,
              updated_at TEXT NOT NULL);
            """)
            # Additive in-place migration; keep all historical action rows.
            columns = {row[1] for row in db.execute("PRAGMA table_info(action_feedback_snapshots)")}
            for name in ("parent_remote_id", "parent_metrics_json", "parent_previous_metrics_json",
                         "parent_delta_metrics_json", "parent_baseline_metrics_json"):
                if name not in columns:
                    db.execute(f"ALTER TABLE action_feedback_snapshots ADD COLUMN {name} TEXT")
            reward_columns = {row[1] for row in db.execute("PRAGMA table_info(action_rewards)")}
            for name in ("context_relevance_component", "observed_feedback_component"):
                if name not in reward_columns:
                    db.execute(f"ALTER TABLE action_rewards ADD COLUMN {name} REAL")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db_path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=15000")
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def record_decision(self, plan):
        trace = plan.parameters.get("decision_trace")
        if not isinstance(trace, dict):
            return None
        decision_id = str(trace.get("decision_id") or uuid4().hex)
        trace["decision_id"] = decision_id
        candidates = trace.get("candidates", [])
        with self.connect() as db:
            db.execute("""INSERT OR IGNORE INTO policy_decisions
              (decision_id,snapshot_id,policy_version,encoder_version,selected_action_id,
               selection_method,candidate_count,created_at,decision_context_json)
              VALUES (?,?,?,?,?,?,?,?,?)""",
              (decision_id, trace.get("snapshot_id", plan.snapshot_id),
               trace.get("policy_version", plan.parameters.get("policy_version")),
               trace.get("encoder_version"), plan.action_id,
               trace.get("selection_method"), len(candidates), plan.created_at, _json(trace)))
            for item in candidates:
                components = item.get("score_components") or {}
                penalties = [abs(float(value)) for name, value in components.items()
                             if name.endswith("penalty") and name != "cost_penalty"]
                db.execute("""INSERT OR IGNORE INTO candidate_scores
                  (decision_id,candidate_id,action_type,target_id,base_score,action_prior,
                   novelty_score,source_quality_score,cost_penalty,risk_penalty,
                   exploration_bonus,final_score,selected,components_json)
                  VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                  (decision_id, str(item.get("key")), item.get("action_type"),
                   str(item.get("target_id")) if item.get("target_id") is not None else None,
                   item.get("base_score", item.get("heuristic_score")),
                   item.get("action_prior", 1.0), None, None,
                   abs(float(components.get("cost_penalty", 0.0))), sum(penalties),
                   max(0.0, float(item.get("exploration_bonus", components.get("exploration_bonus", 0.0)))),
                   item.get("selection_score"), int(bool(item.get("selected"))), _json({
                       "score_components": components,
                       "policy_features": item.get("policy_features"),
                       "selection_probability": item.get("selection_probability"),
                       "opportunity": item.get("opportunity"),
                       "rationale": item.get("rationale"),
                   })))
        return decision_id

    def selected_policy_features(self, action_id: str) -> tuple[float, ...] | None:
        """Load the exact selected-candidate context saved with its decision."""
        with self.connect() as db:
            row = db.execute("""SELECT cs.components_json
              FROM policy_decisions pd JOIN candidate_scores cs USING(decision_id)
              WHERE pd.selected_action_id=? AND cs.selected=1
              ORDER BY pd.created_at DESC LIMIT 1""", (action_id,)).fetchone()
        if row is None:
            return None
        try:
            values = json.loads(row["components_json"] or "{}").get("policy_features")
        except (TypeError, json.JSONDecodeError):
            return None
        if (not isinstance(values, list) or not values
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not isfinite(value) for value in values)):
            return None
        return tuple(float(value) for value in values)

    def action_status(self, action_id: str):
        with self.connect() as db:
            row=db.execute("SELECT status FROM agent_actions WHERE action_id=?",
                           (action_id,)).fetchone()
        return row[0] if row else None

    def recent_contents(self, limit: int = 100) -> list[str]:
        with self.connect() as db:
            rows = db.execute("""SELECT a.parameters_json,a.result_json
              FROM agent_actions a WHERE a.action_type IN ('CREATE_POST','REPLY')
                AND a.status='SUCCEEDED' ORDER BY a.executed_at DESC LIMIT ?""",
              (limit,)).fetchall()
        contents = []
        for row in rows:
            params = json.loads(row["parameters_json"] or "{}")
            result = json.loads(row["result_json"] or "{}")
            value = params.get("generated_content") or params.get("llm_draft") or result.get("body")
            if isinstance(value, str) and value.strip():
                contents.append(value)
        return contents

    def save_content_assessment(self, action_id: str, content: str, features: dict):
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        with self.connect() as db:
            db.execute("""INSERT INTO content_assessments
              (action_id,content_hash,analyzer_version,features_json,assessed_at)
              VALUES (?,?,?,?,?) ON CONFLICT(action_id) DO UPDATE SET
              content_hash=excluded.content_hash,analyzer_version=excluded.analyzer_version,
              features_json=excluded.features_json,assessed_at=excluded.assessed_at""",
              (action_id, digest, features.get("analyzer_version", "unknown"),
               _json(features), _now()))
            decision = db.execute("SELECT decision_id,decision_context_json FROM policy_decisions WHERE selected_action_id=?",
                                  (action_id,)).fetchone()
            if decision:
                trace=json.loads(decision["decision_context_json"] or "{}")
                trace["post_decision_content_assessment"] = features
                db.execute("UPDATE policy_decisions SET decision_context_json=? WHERE decision_id=?",
                           (_json(trace),decision["decision_id"]))
                candidate=db.execute("""SELECT candidate_id,components_json FROM candidate_scores
                  WHERE decision_id=? AND selected=1 LIMIT 1""",(decision["decision_id"],)).fetchone()
                if candidate:
                    components=json.loads(candidate["components_json"] or "{}")
                    components["post_decision_content_assessment"]={
                        "novelty":features.get("novelty_details"),
                        "source_validation":features.get("source_validation")}
                    db.execute("""UPDATE candidate_scores SET novelty_score=?,source_quality_score=?,
                      components_json=? WHERE decision_id=? AND candidate_id=?""",
                      (features.get("novelty_score"),features.get("source_alignment_score"),
                       _json(components),decision["decision_id"],candidate["candidate_id"]))

    def schedule_feedback(self, plan, result, windows: tuple[int, ...]):
        if result.status == "DRY_RUN":
            return 0
        if plan.action_type == "WAIT" or result.status == "WAIT":
            rows = [("next_observation", _now())]
        elif result.status == "SUCCEEDED":
            with self.connect() as db:
                saved = db.execute("SELECT executed_at FROM agent_actions WHERE action_id=?",
                                   (plan.action_id,)).fetchone()
            try:
                executed = _parse_time(saved[0]) if saved and saved[0] else _parse_time(plan.created_at)
            except (ValueError, TypeError):
                executed = datetime.now(timezone.utc)
            rows = [(f"t+{seconds}s", (executed + timedelta(seconds=seconds)).isoformat())
                    for seconds in windows]
        else:
            return 0
        with self.connect() as db:
            parent_remote_id = (plan.parameters.get("context", {}).get("source_post_id")
                                if plan.action_type == "REPLY" else None)
            assessment = db.execute("SELECT features_json FROM content_assessments WHERE action_id=?",
                                    (plan.action_id,)).fetchone()
            parent_baseline = {}
            if assessment:
                try:
                    parent_baseline = json.loads(assessment[0]).get("target_features", {}).get(
                        "pre_action_parent_metrics", {})
                except (TypeError, json.JSONDecodeError):
                    parent_baseline = {}
            for name, scheduled in rows:
                db.execute("""INSERT OR IGNORE INTO action_feedback_snapshots
                  (feedback_id,action_id,remote_id,action_type,window_name,scheduled_at,
                   raw_metrics_json,feedback_status,parent_remote_id,parent_baseline_metrics_json)
                  VALUES (?,?,?,?,?,?,?,?,?,?)""",
                  (uuid4().hex, plan.action_id, result.remote_id, plan.action_type,
                   name, scheduled, _json({"execution_status": result.status}), "PENDING",
                   str(parent_remote_id) if parent_remote_id is not None else None,
                   _json(parent_baseline) if parent_baseline else None))
        return len(rows)

    def due_feedback(self, limit: int = 100) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("""SELECT f.*,a.status AS action_status,a.created_at AS action_created_at,
              a.error_message AS action_error,a.parameters_json AS action_parameters FROM action_feedback_snapshots f
              JOIN agent_actions a USING(action_id)
              WHERE f.feedback_status='PENDING' AND f.scheduled_at<=?
              ORDER BY f.scheduled_at LIMIT ?""", (_now(), limit)).fetchall()
        return [dict(row) for row in rows]

    def complete_feedback(self, item: dict, metrics: dict | None, status: str,
                          error: str | None = None, retry_at: str | None = None,
                          parent_metrics: dict | None = None):
        metrics = metrics or {}
        now = _now()
        with self.connect() as db:
            previous_row = db.execute("""SELECT observed_at,likes,replies,reposts,views,follows
              FROM action_feedback_snapshots WHERE action_id=? AND observed_at IS NOT NULL
                AND feedback_status IN ('COLLECTED','PARTIAL')
              ORDER BY observed_at DESC LIMIT 1""", (item["action_id"],)).fetchone()
            previous_parent_row = db.execute("""SELECT parent_metrics_json FROM action_feedback_snapshots
              WHERE action_id=? AND parent_metrics_json IS NOT NULL
                AND feedback_status IN ('COLLECTED','PARTIAL')
              ORDER BY observed_at DESC LIMIT 1""", (item["action_id"],)).fetchone()
            current = {name: metrics.get(source) if isinstance(metrics.get(source), int)
                       and not isinstance(metrics.get(source), bool) else None
                       for name, source in _METRICS.items()}
            previous = ({name: previous_row[name] for name in _METRICS}
                        if previous_row else {name: None for name in _METRICS})
            delta = {name: current[name] - previous[name]
                     for name in current if current[name] is not None and previous[name] is not None}
            parent_metrics = parent_metrics or {}
            try:
                previous_parent = (json.loads(previous_parent_row[0]) if previous_parent_row
                                   else json.loads(item.get("parent_baseline_metrics_json") or "{}"))
            except (TypeError, json.JSONDecodeError):
                previous_parent = {}
            parent_current = {name: parent_metrics.get(source) if isinstance(parent_metrics.get(source), int)
                              and not isinstance(parent_metrics.get(source), bool) else None
                              for name, source in _METRICS.items()}
            parent_delta = {name: parent_current[name] - previous_parent.get(name)
                            for name in parent_current if parent_current[name] is not None
                            and isinstance(previous_parent.get(name), int)}
            interval = None
            if previous_row:
                interval = max(0.0, (_parse_time(now) - _parse_time(previous_row["observed_at"])).total_seconds())
            attempt = int(item.get("attempt_count", 0)) + (1 if status in ("FAILED", "UNAVAILABLE", "PENDING") else 0)
            if retry_at:
                db.execute("""UPDATE action_feedback_snapshots SET scheduled_at=?,attempt_count=?,
                  error_message=?,raw_metrics_json=? WHERE feedback_id=?""",
                  (retry_at, attempt, error, _json({"last_error": error}), item["feedback_id"]))
                return
            db.execute("""UPDATE action_feedback_snapshots SET observed_at=?,likes=?,replies=?,
              reposts=?,views=?,follows=?,previous_metrics_json=?,delta_metrics_json=?,
              interval_seconds=?,raw_metrics_json=?,feedback_status=?,error_message=?,attempt_count=?,
              parent_metrics_json=?,parent_previous_metrics_json=?,parent_delta_metrics_json=?
              WHERE feedback_id=?""",
              (now, current["likes"], current["replies"], current["reposts"],
               current["views"], current["follows"], _json(previous) if previous_row else None,
               _json(delta) if previous_row else None, interval,
               _json({"raw_response": metrics, "error": error,
                      "parent_metrics": parent_metrics,
                      "parent_feedback_error": item.get("parent_feedback_error"),
                      "parent_delta_is_not_attributed_to_reply": True}), status, error, attempt,
               _json(parent_current) if parent_metrics else None,
               _json(previous_parent) if previous_parent else None,
               _json(parent_delta) if parent_metrics else None,
               item["feedback_id"]))

    def pending_rewards(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("""SELECT a.action_id,a.action_type,a.status,a.remote_id,
              a.parameters_json,a.result_json FROM agent_actions a
              WHERE (a.status='SUCCEEDED' OR (a.action_type='WAIT' AND a.status='WAIT'))
                AND EXISTS(SELECT 1 FROM action_feedback_snapshots f
                  WHERE f.action_id=a.action_id) ORDER BY a.executed_at""").fetchall()
            out=[]
            for row in rows:
                snaps=db.execute("""SELECT window_name,scheduled_at,observed_at,feedback_status,
                  likes,replies,reposts,views,follows,raw_metrics_json,error_message,
                  delta_metrics_json,parent_remote_id,parent_metrics_json,parent_delta_metrics_json
                  FROM action_feedback_snapshots WHERE action_id=? ORDER BY scheduled_at""",
                  (row["action_id"],)).fetchall()
                assessment=db.execute("SELECT features_json FROM content_assessments WHERE action_id=?",
                                      (row["action_id"],)).fetchone()
                out.append({"action_id":row["action_id"],"action_type":row["action_type"],
                  "status":row["status"],"remote_id":row["remote_id"],
                  "parameters":json.loads(row["parameters_json"] or "{}"),
                  "result":json.loads(row["result_json"] or "{}"),
                  "assessment":json.loads(assessment[0]) if assessment else {},
                  "snapshots":[dict(s) for s in snaps]})
        return out

    def save_reward(self, action_id: str, version: str, feedback_window: str | None,
                    components: dict, total: float | None, status: str,
                    calculation: dict) -> str | None:
        payload = _json(calculation)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        with self.connect() as db:
            existing = db.execute("""SELECT reward_id FROM action_rewards WHERE
              action_id=? AND reward_version=? AND feedback_window IS ? AND calculation_hash=?""",
              (action_id, version, feedback_window, digest)).fetchone()
            if existing:
                return existing[0]
            reward_id = uuid4().hex
            db.execute("""INSERT INTO action_rewards
              (reward_id,action_id,reward_version,feedback_window,execution_component,
               factual_component,novelty_component,compliance_component,context_relevance_component,
               observed_feedback_component,reward_total,reward_status,computed_at,calculation_json,calculation_hash)
              VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
              (reward_id, action_id, version, feedback_window,
               components.get("execution_component"), components.get("factual_component"),
               components.get("novelty_component"), components.get("compliance_component"),
               components.get("context_relevance_component"),
               components.get("observed_feedback_component"), total, status, _now(), payload, digest))
        return reward_id

    def add_policy_update(self, policy_version: str, action_id: str, reward_id: str | None,
                          before: dict, after: dict, reason: str, learning_rate: float | None,
                          applied: bool):
        with self.connect() as db:
            db.execute("""INSERT INTO policy_updates
              (update_id,policy_version,action_id,reward_id,parameters_before_json,
               parameters_after_json,update_reason,learning_rate,created_at,update_applied)
              VALUES (?,?,?,?,?,?,?,?,?,?)""",
              (uuid4().hex, policy_version, action_id, reward_id, _json(before), _json(after),
               reason, learning_rate, _now(), int(applied)))

    def has_policy_update(self, reward_id: str | None) -> bool:
        if not reward_id:
            return False
        with self.connect() as db:
            row = db.execute(
                "SELECT 1 FROM policy_updates WHERE reward_id=? LIMIT 1",
                (reward_id,)).fetchone()
        return row is not None

    def has_policy_update_for_action(self, action_id: str | None) -> bool:
        if not action_id:
            return False
        with self.connect() as db:
            row = db.execute(
                "SELECT 1 FROM policy_updates WHERE action_id=? LIMIT 1",
                (action_id,)).fetchone()
        return row is not None

    def action_created_at(self, action_id: str | None) -> str | None:
        if not action_id:
            return None
        with self.connect() as db:
            row = db.execute(
                "SELECT created_at FROM agent_actions WHERE action_id=?",
                (action_id,)).fetchone()
        return row[0] if row else None

    def load_policy_state(self, version: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT parameters_json FROM policy_states WHERE policy_version=?",
                             (version,)).fetchone()
        return json.loads(row[0]) if row else None

    def save_policy_state(self, version: str, parameters: dict):
        with self.connect() as db:
            db.execute("""INSERT INTO policy_states VALUES (?,?,?)
              ON CONFLICT(policy_version) DO UPDATE SET parameters_json=excluded.parameters_json,
              updated_at=excluded.updated_at""", (version, _json(parameters), _now()))
