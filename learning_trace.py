"""Read-only trace of CTN policy decisions, rewards, and learning updates."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB = BASE_DIR / "data" / "state.db"
LEARNABLE_ACTIONS = ("CREATE_POST", "REPLY")
BANDIT_VERSION = "ctn-bandit-features-v1"


def _learning_flag() -> str:
    value = os.getenv("CTN_ENABLE_LEARNING")
    if value is None:
        env_file = BASE_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if line.startswith("CTN_ENABLE_LEARNING="):
                    value = line.split("=", 1)[1].strip().strip("\"'")
                    break
    if value is None:
        return "unknown"
    return "true" if value.lower() in {"1", "true", "yes", "on"} else "false"


def _connect_readonly(path: Path) -> sqlite3.Connection:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Database not found: {resolved}")
    connection = sqlite3.connect(resolved.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _json(value: str | None) -> dict:
    try:
        parsed = json.loads(value or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _since(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


def _summary(db: sqlite3.Connection, hours: float) -> None:
    since = _since(hours)
    print(f"Configured learning flag (CLI env/.env): {_learning_flag()} "
          "(runtime updates below are the proof of actual learning)")
    print(f"Summary window: last {hours:g} hour(s), since {since}")

    model = db.execute(
        "SELECT parameters_json,updated_at FROM policy_states WHERE policy_version=?",
        (BANDIT_VERSION,),
    ).fetchone()
    if model:
        state = _json(model["parameters_json"])
        print(f"Persisted LinUCB: samples={state.get('count', 'unknown')}, "
              f"updated_at={model['updated_at']}")
    else:
        print("Persisted LinUCB: no saved state")

    methods = db.execute("""SELECT selection_method,COUNT(*) AS n
        FROM policy_decisions WHERE created_at>=? GROUP BY selection_method
        ORDER BY n DESC""", (since,)).fetchall()
    print("Decision methods: " + (", ".join(
        f"{row['selection_method'] or 'unknown'}={row['n']}" for row in methods
    ) or "none"))

    reward_rows = db.execute("""SELECT reward_status,COUNT(*) AS n
        FROM action_rewards WHERE computed_at>=? GROUP BY reward_status
        ORDER BY reward_status""", (since,)).fetchall()
    print("Rewards: " + (", ".join(
        f"{row['reward_status']}={row['n']}" for row in reward_rows
    ) or "none"))

    updates = db.execute("""SELECT update_applied,update_reason,COUNT(*) AS n
        FROM policy_updates WHERE created_at>=?
        GROUP BY update_applied,update_reason ORDER BY n DESC""", (since,)).fetchall()
    applied = sum(row["n"] for row in updates if row["update_applied"])
    total = sum(row["n"] for row in updates)
    print(f"Parameter updates actually applied: {applied}/{total} update records")
    for row in updates:
        marker = "APPLIED" if row["update_applied"] else "SKIPPED"
        print(f"  {marker}: {row['update_reason']} ({row['n']})")

    transitions = db.execute("""SELECT created_at,parameters_before_json,parameters_after_json
        FROM policy_updates WHERE update_applied=1 AND update_reason LIKE '%linucb%'
        ORDER BY created_at""").fetchall()
    count_pairs = []
    for row in transitions:
        before = _json(row["parameters_before_json"]).get("contextual_bandit", {}).get("count")
        after = _json(row["parameters_after_json"]).get("contextual_bandit", {}).get("count")
        if isinstance(before, int) and isinstance(after, int):
            count_pairs.append((row["created_at"], before, after))
    discontinuities = sum(
        previous_after != current_before
        for (_, _, previous_after), (_, current_before, _) in zip(count_pairs, count_pairs[1:])
    )
    if model and count_pairs:
        persisted_count = _json(model["parameters_json"]).get("count")
        last_update_count = count_pairs[-1][2]
        if persisted_count != last_update_count:
            print("WARNING: persisted LinUCB sample count differs from the latest update trace.")
        if discontinuities:
            print(f"WARNING: {discontinuities} LinUCB count-chain discontinuity(ies); "
                  "multiple or stale agent processes may have overwritten each other's model state.")
        print(f"LinUCB trace records={len(count_pairs)}, latest transition="
              f"{count_pairs[-1][1]}->{count_pairs[-1][2]}, persisted={persisted_count}")


def _parameter_change(before_json: str, after_json: str, action_type: str) -> str:
    before = _json(before_json)
    after = _json(after_json)
    changes = []

    old_bandit = before.get("contextual_bandit", {})
    new_bandit = after.get("contextual_bandit", {})
    old_count, new_count = old_bandit.get("count"), new_bandit.get("count")
    if old_count is not None and new_count is not None and old_count != new_count:
        changes.append(f"LinUCB samples {old_count}->{new_count}")

    old_priors = before.get("action_priors", {})
    new_priors = after.get("action_priors", {})
    old_counts = old_priors.get("counts", {})
    new_counts = new_priors.get("counts", {})
    old_means = old_priors.get("means", {})
    new_means = new_priors.get("means", {})
    if action_type in old_counts and action_type in new_counts:
        if old_counts[action_type] != new_counts[action_type]:
            changes.append(f"{action_type} prior samples "
                           f"{old_counts[action_type]}->{new_counts[action_type]}")
        if old_means.get(action_type) != new_means.get(action_type):
            changes.append(f"{action_type} mean "
                           f"{old_means.get(action_type)}->{new_means.get(action_type)}")
    return "; ".join(changes) or "serialized parameters unchanged"


def _action_state(row: sqlite3.Row) -> str:
    reward_status = row["reward_status"]
    update_applied = row["update_applied"]
    if update_applied == 1:
        return "LEARNED: " + _parameter_change(
            row["parameters_before_json"], row["parameters_after_json"], row["action_type"]
        )
    if update_applied == 0:
        return f"NOT_UPDATED: {row['update_reason']}"
    if reward_status in ("PENDING", "PROVISIONAL"):
        return f"WAITING_FOR_FINAL_REWARD: {reward_status}"
    if reward_status == "FINAL":
        return "FINAL_REWARD_WITHOUT_UPDATE_RECORD"
    if reward_status == "INVALID":
        return "INVALID_REWARD"
    return "WAITING_FOR_REWARD"


def _trace(db: sqlite3.Connection, limit: int, action_id: str | None) -> None:
    filters = "a.action_type IN ('CREATE_POST','REPLY')"
    args: list[object] = []
    if action_id:
        filters += " AND a.action_id=?"
        args.append(action_id)
    query = f"""SELECT a.action_id,a.action_type,a.status,a.target_id,a.created_at,
        pd.selection_method,cs.base_score,cs.final_score,cs.exploration_bonus,
        cs.components_json,ar.reward_status,ar.reward_total,ar.feedback_window,
        pu.update_applied,pu.update_reason,pu.parameters_before_json,pu.parameters_after_json
      FROM agent_actions a
      LEFT JOIN policy_decisions pd ON pd.selected_action_id=a.action_id
      LEFT JOIN candidate_scores cs ON cs.decision_id=pd.decision_id AND cs.selected=1
      LEFT JOIN action_rewards ar ON ar.reward_id=(
        SELECT r.reward_id FROM action_rewards r WHERE r.action_id=a.action_id
        ORDER BY r.computed_at DESC LIMIT 1)
      LEFT JOIN policy_updates pu ON pu.update_id=(
        SELECT u.update_id FROM policy_updates u WHERE u.action_id=a.action_id
        ORDER BY u.created_at DESC LIMIT 1)
      WHERE {filters}
      ORDER BY a.created_at DESC LIMIT ?"""
    args.append(limit)
    rows = db.execute(query, args).fetchall()
    print("\nRecent learning path (CREATE_POST / REPLY):")
    if not rows:
        print("  No matching actions.")
        return
    for row in rows:
        reward = row["reward_status"] or "NONE"
        reward_total = (f"{row['reward_total']:.4f}"
                        if isinstance(row["reward_total"], (int, float)) else "NULL")
        components = _json(row["components_json"])
        probability = components.get("selection_probability")
        probability_text = (f", p={probability:.3f}"
                            if isinstance(probability, (int, float)) else "")
        score = "n/a" if row["final_score"] is None else f"{row['final_score']:.4f}"
        learning = _action_state(row)
        print(f"  {row['created_at']} | {row['action_type']} {row['status']} "
              f"| id={row['action_id']} target={row['target_id'] or '-'}")
        print(f"    decision={row['selection_method'] or 'unrecorded'} "
              f"score={score}{probability_text} "
              f"exploration={row['exploration_bonus'] if row['exploration_bonus'] is not None else 'n/a'}")
        print(f"    reward={reward}:{reward_total} window={row['feedback_window'] or '-'} "
              f"| {learning}")


def run(db_path: Path, hours: float, limit: int, action_id: str | None) -> int:
    with _connect_readonly(db_path) as db:
        _summary(db, hours)
        _trace(db, limit, action_id)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only trace: policy decision -> feedback reward -> parameter update."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB,
                        help="SQLite path (default: data/state.db)")
    parser.add_argument("--hours", type=float, default=24,
                        help="summary window in hours (default: 24)")
    parser.add_argument("--limit", type=int, default=20,
                        help="recent CREATE_POST/REPLY rows to print (default: 20)")
    parser.add_argument("--action-id", help="show one action's learning path")
    args = parser.parse_args(argv)
    if args.hours <= 0 or args.limit < 1:
        parser.error("--hours must be positive and --limit must be at least 1")
    try:
        return run(args.db, args.hours, args.limit, args.action_id)
    except (OSError, sqlite3.Error) as exc:
        print(f"learning trace failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
