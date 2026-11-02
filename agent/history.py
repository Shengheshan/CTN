"""SQLite action journal. Reserve before any write to prevent duplicates."""

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from math import isfinite
from pathlib import Path

from agent.state import AgentState
from agent.content_features import extract_content_features
from policy.models import ActionPlan, utc_now


BLOCKING_STATUSES = ("PENDING", "SUCCEEDED", "UNKNOWN")


def content_hash(text: str | None) -> str | None:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest() if text else None


class ActionHistory:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS agent_actions (
                action_id TEXT PRIMARY KEY, snapshot_id INTEGER NOT NULL,
                action_type TEXT NOT NULL, target_id TEXT,
                source_event_id TEXT, action_key TEXT NOT NULL,
                content_hash TEXT, score REAL NOT NULL, status TEXT NOT NULL,
                created_at TEXT NOT NULL, executed_at TEXT, remote_id TEXT,
                error_message TEXT, parameters_json TEXT NOT NULL DEFAULT '{}',
                result_json TEXT NOT NULL DEFAULT '{}')""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_agent_actions_key ON agent_actions(action_key,status)")
            db.execute("""CREATE TABLE IF NOT EXISTS action_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT, action_id TEXT NOT NULL,
                observed_at TEXT NOT NULL, execution_success INTEGER,
                task_completed INTEGER, observed_metrics TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY(action_id) REFERENCES agent_actions(action_id))""")
            db.execute("""CREATE TABLE IF NOT EXISTS policy_rewards (
                action_id TEXT PRIMARY KEY, reward REAL NOT NULL,
                source TEXT NOT NULL, observed_at TEXT NOT NULL,
                trained INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(action_id) REFERENCES agent_actions(action_id))""")
            db.execute("""CREATE TABLE IF NOT EXISTS policy_models (
                name TEXT PRIMARY KEY, payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS policy_training_history (
                action_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                action_type TEXT NOT NULL,
                trained_at TEXT NOT NULL,
                model_sample_count INTEGER NOT NULL,
                reward REAL NOT NULL,
                source TEXT NOT NULL,
                recorded_mean_reward REAL NOT NULL,
                rolling_mean_reward_10 REAL NOT NULL,
                predicted_before REAL,
                uncertainty_before REAL,
                predicted_after REAL,
                FOREIGN KEY(action_id) REFERENCES agent_actions(action_id))""")

    def _connect(self):
        db = sqlite3.connect(self.db_path, timeout=10)
        db.execute("PRAGMA busy_timeout=10000")
        return db

    def load_state(self, include_dry_runs: bool = False) -> AgentState:
        failed_cutoff = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        with self._connect() as db:
            rows = db.execute("""SELECT action_type,target_id,action_key,content_hash,
                                     score,status,created_at,parameters_json FROM agent_actions
                                     ORDER BY created_at DESC LIMIT 200""").fetchall()
            statuses = BLOCKING_STATUSES + (("DRY_RUN",) if include_dry_runs else ())
            placeholders = ",".join("?" for _ in statuses)
            active = db.execute(f"""SELECT action_type,target_id,action_key,content_hash
                FROM agent_actions WHERE status IN ({placeholders})""", statuses).fetchall()
            recent_failures = db.execute("""SELECT DISTINCT action_key FROM agent_actions
                WHERE status='FAILED' AND created_at>=?""", (failed_cutoff,)).fetchall()
            published = db.execute("""SELECT remote_id,parameters_json FROM agent_actions
                WHERE status='SUCCEEDED' AND action_type IN ('CREATE_POST','REPLY')
                  AND remote_id IS NOT NULL ORDER BY executed_at DESC LIMIT 20""").fetchall()
            published_news = db.execute("""SELECT parameters_json FROM agent_actions
                WHERE status='SUCCEEDED' AND action_type='CREATE_POST'
                  AND remote_id IS NOT NULL ORDER BY executed_at DESC""").fetchall()
            mentions = db.execute("""SELECT target_id,parameters_json FROM agent_actions
                WHERE action_type='MENTION' AND status IN ('PENDING','SUCCEEDED','UNKNOWN')""").fetchall()
        state = AgentState()
        for action_type, target_id, key, digest, score, status, created_at, parameters_json in rows:
            try:
                parameters = json.loads(parameters_json or "{}")
                generation_profile = parameters.get("generation_profile")
                selected_tags = parameters.get("selected_tags")
            except (TypeError, json.JSONDecodeError, AttributeError):
                generation_profile = None
                selected_tags = None
            state.recent_actions.append({"action_type": action_type, "target_id": target_id,
                                         "action_key": key,
                                         "score": score, "status": status,
                                         "created_at": created_at,
                                         "generation_profile": generation_profile,
                                         "selected_tags": selected_tags
                                         if isinstance(selected_tags, list) else []})
        for action_type, target_id, key, digest in active:
            state.completed_keys.add(key)
            if digest:
                state.content_hashes.add(digest)
            if target_id and str(target_id).isdigit():
                target = int(target_id)
                if action_type == "LIKE": state.liked_post_ids.add(target)
                if action_type == "REPOST": state.reposted_post_ids.add(target)
                if action_type == "FOLLOW": state.followed_account_ids.add(target)
        state.completed_keys.update(key for (key,) in recent_failures)
        for remote_id, parameters_json in published:
            if not str(remote_id).isdigit():
                continue
            context = json.loads(parameters_json).get("context", {})
            articles = context.get("articles", [])
            source_text = " ".join(
                [str(context.get("post_body") or "")] +
                [" ".join(str(article.get(key) or "") for key in ("title", "summary"))
                 for article in articles if isinstance(article, dict)])
            state.own_published_posts.append({"post_id": int(remote_id),
                                              "source_text": source_text})
        for (parameters_json,) in published_news:
            try:
                parameters = json.loads(parameters_json)
                context = parameters.get("context", {})
            except (TypeError, json.JSONDecodeError):
                continue
            articles = context.get("articles", [])
            source_text = " ".join(
                " ".join(str(article.get(key) or "") for key in ("title", "summary"))
                for article in articles if isinstance(article, dict))
            generated_content = parameters.get("generated_content") or ""
            combined_text = f"{source_text} {generated_content}".strip()
            if combined_text:
                state.published_news_sources.append(combined_text)
        for target_id, parameters_json in mentions:
            if target_id and str(target_id).isdigit():
                state.mentioned_account_ids.add(int(target_id))
            own_post_id = json.loads(parameters_json).get("context", {}).get("mention_on_post_id")
            if isinstance(own_post_id, int):
                state.mentioned_post_ids.add(own_post_id)
        return state

    def reserve(self, plan: ActionPlan) -> bool:
        """Return false if action or content may already have been sent."""
        if plan.generated_content:
            plan.parameters["content_features"] = extract_content_features(
                plan.generated_content)
        digest = content_hash(plan.generated_content)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("""SELECT 1 FROM agent_actions WHERE action_key=?
                AND status IN ('PENDING','SUCCEEDED','UNKNOWN') LIMIT 1""",
                (plan.action_key,)).fetchone()
            if digest:
                existing = existing or db.execute("""SELECT 1 FROM agent_actions
                    WHERE content_hash=? AND status IN ('PENDING','SUCCEEDED','UNKNOWN') LIMIT 1""",
                    (digest,)).fetchone()
            if existing:
                return False
            db.execute("""INSERT INTO agent_actions
                (action_id,snapshot_id,action_type,target_id,source_event_id,action_key,
                 content_hash,score,status,created_at,parameters_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(action_id) DO UPDATE SET
                content_hash=excluded.content_hash, score=excluded.score,
                status='PENDING', error_message=NULL, parameters_json=excluded.parameters_json""",
                (plan.action_id, plan.snapshot_id, plan.action_type,
                 str(plan.target_id) if plan.target_id is not None else None,
                 plan.source_event_id, plan.action_key, digest, plan.score,
                 "PENDING", plan.created_at, json.dumps(plan.parameters, ensure_ascii=False)))
        return True

    def save_result(self, plan: ActionPlan, status: str,
                    remote_id: str | None = None, error: str | None = None,
                    result: dict | None = None) -> None:
        if status not in ("DRY_RUN", "SUCCEEDED", "FAILED", "UNKNOWN", "WAIT", "BLOCKED"):
            raise ValueError("invalid action status")
        if plan.generated_content:
            plan.parameters["content_features"] = extract_content_features(
                plan.generated_content)
        with self._connect() as db:
            db.execute("""INSERT INTO agent_actions
                (action_id,snapshot_id,action_type,target_id,source_event_id,action_key,
                 content_hash,score,status,created_at,executed_at,remote_id,error_message,
                 parameters_json,result_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(action_id) DO UPDATE SET
                status=excluded.status, executed_at=excluded.executed_at,
                remote_id=excluded.remote_id, error_message=excluded.error_message,
                parameters_json=excluded.parameters_json,
                result_json=excluded.result_json""",
                (plan.action_id, plan.snapshot_id, plan.action_type,
                 str(plan.target_id) if plan.target_id is not None else None,
                 plan.source_event_id, plan.action_key, content_hash(plan.generated_content),
                 plan.score, status, plan.created_at, utc_now(), remote_id,
                 error, json.dumps(plan.parameters, ensure_ascii=False),
                 json.dumps(result or {}, ensure_ascii=False)))

    def save_feedback(self, action_id: str, execution_success: bool | None,
                      task_completed: bool | None, metrics: dict) -> None:
        with self._connect() as db:
            db.execute("""INSERT INTO action_feedback
                (action_id,observed_at,execution_success,task_completed,observed_metrics)
                VALUES (?,?,?,?,?)""",
                (action_id, utc_now(), execution_success, task_completed,
                 json.dumps(metrics, ensure_ascii=False)))

    def pending_delayed_feedback(self, cutoff: str) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("""SELECT a.action_id,a.action_type,a.remote_id
                FROM agent_actions a LEFT JOIN policy_rewards r ON a.action_id=r.action_id
                WHERE a.status='SUCCEEDED'
                  AND a.action_type IN ('CREATE_POST','REPLY')
                  AND a.remote_id IS NOT NULL
                  AND a.executed_at<=? AND r.action_id IS NULL""", (cutoff,)).fetchall()
        return [{"action_id": row[0], "action_type": row[1], "remote_id": row[2]}
                for row in rows]

    def save_reward(self, action_id: str, reward: float, source: str) -> bool:
        if not isfinite(reward) or not 0 <= reward <= 1 or not source:
            raise ValueError("invalid reward")
        with self._connect() as db:
            row = db.execute("SELECT status FROM agent_actions WHERE action_id=?",
                             (action_id,)).fetchone()
            if row != ("SUCCEEDED",):
                raise ValueError("only successful real actions can receive a reward")
            cursor = db.execute("""INSERT OR IGNORE INTO policy_rewards
                (action_id,reward,source,observed_at) VALUES (?,?,?,?)""",
                (action_id, reward, source, utc_now()))
        return cursor.rowcount == 1

    def load_untrained_rewards(self, source: str | None = None) -> list[tuple[str, tuple[float, ...], float]]:
        with self._connect() as db:
            query = """SELECT r.action_id,a.parameters_json,r.reward
                FROM policy_rewards r JOIN agent_actions a ON r.action_id=a.action_id
                WHERE r.trained=0"""
            params = ()
            if source is not None:
                query += " AND r.source=?"
                params = (source,)
            query += " ORDER BY r.observed_at,r.action_id"
            rows = db.execute(query, params).fetchall()
        results = []
        for action_id, raw, reward in rows:
            features = json.loads(raw).get("policy_features")
            if isinstance(features, list):
                results.append((action_id, tuple(features), reward))
        return results

    def load_policy_model(self, name: str) -> dict | None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM policy_models WHERE name=?",
                             (name,)).fetchone()
        return json.loads(row[0]) if row else None

    def commit_policy_model(self, name: str, model: dict,
                            trained_action_ids: list[str],
                            training_metrics: dict[str, dict] | None = None) -> None:
        training_metrics = training_metrics or {}
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("""INSERT INTO policy_models(name,payload_json,updated_at)
                VALUES (?,?,?) ON CONFLICT(name) DO UPDATE SET
                payload_json=excluded.payload_json,updated_at=excluded.updated_at""",
                (name, json.dumps(model), utc_now()))
            prior_rewards = [row[0] for row in db.execute(
                """SELECT reward FROM policy_training_history WHERE model_name=?
                   ORDER BY model_sample_count,trained_at""", (name,)).fetchall()]
            for action_id in trained_action_ids:
                reward_row = db.execute("""SELECT r.reward,r.source,a.action_type
                    FROM policy_rewards r JOIN agent_actions a ON a.action_id=r.action_id
                    WHERE r.action_id=? AND r.trained=0""", (action_id,)).fetchone()
                if reward_row is None:
                    continue
                reward, source, action_type = reward_row
                prior_rewards.append(float(reward))
                metrics = training_metrics.get(action_id, {})
                recorded_mean = sum(prior_rewards) / len(prior_rewards)
                rolling = prior_rewards[-10:]
                db.execute("""INSERT OR IGNORE INTO policy_training_history
                    (action_id,model_name,action_type,trained_at,model_sample_count,
                     reward,source,recorded_mean_reward,rolling_mean_reward_10,
                     predicted_before,uncertainty_before,predicted_after)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (action_id, name, action_type, utc_now(),
                     int(metrics.get("model_sample_count", model.get("count", len(prior_rewards)))),
                     reward, source, recorded_mean, sum(rolling) / len(rolling),
                     metrics.get("predicted_before"), metrics.get("uncertainty_before"),
                     metrics.get("predicted_after")))
                db.execute("UPDATE policy_rewards SET trained=1 WHERE action_id=?",
                           (action_id,))

    def load_policy_learning_history(self, model_name: str,
                                     limit: int = 100) -> list[dict]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._connect() as db:
            rows = db.execute("""SELECT action_id,action_type,trained_at,
                    model_sample_count,reward,source,recorded_mean_reward,
                    rolling_mean_reward_10,predicted_before,uncertainty_before,
                    predicted_after
                FROM policy_training_history WHERE model_name=?
                ORDER BY model_sample_count DESC,trained_at DESC LIMIT ?""",
                (model_name, limit)).fetchall()
        fields = ("action_id", "action_type", "trained_at", "model_sample_count",
                  "reward", "source", "recorded_mean_reward",
                  "rolling_mean_reward_10", "predicted_before",
                  "uncertainty_before", "predicted_after")
        return [dict(zip(fields, row)) for row in reversed(rows)]
