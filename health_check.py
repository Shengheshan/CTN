"""One-shot, read-only health report for the CTN runtime and monitor."""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "state.db"
STALE_SECONDS = 240


def _utc(value):
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def check(db_path: Path = DB_PATH, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    report = {"checked_at": now.isoformat(), "alerts": [], "stats": {}}
    if not db_path.is_file():
        report["alerts"].append({"code": "database_missing", "message": "State database is missing"})
        return report

    with sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=5) as db:
        db.row_factory = sqlite3.Row
        latest_snapshot = db.execute("""SELECT id,finished_at FROM snapshots
            ORDER BY id DESC LIMIT 1""").fetchone()
        if latest_snapshot:
            age = max(0, int((now - _utc(latest_snapshot["finished_at"])).total_seconds()))
            report["stats"].update({"snapshot_id": latest_snapshot["id"],
                                    "snapshot_age_seconds": age})
            if age > STALE_SECONDS:
                report["alerts"].append({"code": "observation_stale",
                                         "message": f"No new world snapshot for {age} seconds"})
        else:
            report["alerts"].append({"code": "no_snapshots", "message": "No world snapshots stored"})

        statuses = dict(db.execute("""SELECT status,count(*) FROM agent_actions
            GROUP BY status""").fetchall())
        report["stats"]["action_status_counts"] = statuses
        recent = db.execute("""SELECT status,action_type,target_id,error_message,created_at
            FROM agent_actions ORDER BY created_at DESC LIMIT 10""").fetchall()
        if recent:
            report["stats"]["latest_action"] = {
                "status": recent[0]["status"], "action_type": recent[0]["action_type"],
                "target_id": recent[0]["target_id"], "created_at": recent[0]["created_at"],
                "error_message": recent[0]["error_message"],
            }
        consecutive_failures = 0
        for action in recent:
            if action["status"] != "FAILED":
                break
            consecutive_failures += 1
        report["stats"]["consecutive_failures"] = consecutive_failures
        if consecutive_failures >= 3:
            report["alerts"].append({"code": "repeated_action_failure",
                                     "message": f"{consecutive_failures} consecutive actions failed"})
        active_window = [item for item in recent
                         if _utc(item["created_at"]) >= now - timedelta(minutes=10)]
        failed_window = sum(item["status"] == "FAILED" for item in active_window)
        report["stats"]["failed_in_recent_window"] = failed_window
        if len(active_window) >= 5 and failed_window >= 5:
            report["alerts"].append({"code": "high_recent_failure_rate",
                                     "message": f"{failed_window} of the last {len(active_window)} actions failed in 10 minutes"})
        if recent and recent[0]["status"] in ("UNKNOWN", "BLOCKED"):
            report["alerts"].append({"code": "write_uncertain_or_blocked",
                                     "message": f"Latest action is {recent[0]['status']}"})

        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "observed_post_counters" not in tables:
            report["alerts"].append({"code": "monitor_not_initialized",
                                     "message": "Counter monitor has not initialized its table"})
        else:
            latest_counter = db.execute("""SELECT observed_at FROM observed_post_counters
                ORDER BY id DESC LIMIT 1""").fetchone()
            if latest_counter:
                counter_age = max(0, int((now - _utc(latest_counter[0])).total_seconds()))
                report["stats"]["counter_age_seconds"] = counter_age
                if counter_age > STALE_SECONDS:
                    report["alerts"].append({"code": "counter_monitor_stale",
                                             "message": f"No new counter sample for {counter_age} seconds"})
            elif statuses.get("SUCCEEDED", 0):
                report["alerts"].append({"code": "no_counter_samples",
                                         "message": "Successful posts exist but no counter samples were stored"})
            errors = db.execute("""SELECT error_type FROM observed_post_counters
                ORDER BY id DESC LIMIT 3""").fetchall()
            if len(errors) == 3 and all(row[0] for row in errors):
                report["alerts"].append({"code": "counter_reads_failing",
                                         "message": "Three consecutive counter reads failed"})
    report["ok"] = not report["alerts"]
    return report


if __name__ == "__main__":
    print(json.dumps(check(), ensure_ascii=False))
