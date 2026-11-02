import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from agent.observation_monitor import ObservationMonitor


class FakeSocial:
    def __init__(self):
        self.calls = 0

    def counters(self, post_id):
        self.calls += 1
        return {"view_count": 2 * self.calls, "like_count": self.calls,
                "reply_count": 0, "repost_count": 0, "post_id": post_id}


class ObservationMonitorTests(unittest.TestCase):
    def test_records_only_successful_own_posts_without_rewards(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.db"
            with closing(sqlite3.connect(path)) as db:
                db.execute("""CREATE TABLE agent_actions (
                    action_id TEXT, action_type TEXT, target_id TEXT, status TEXT,
                    created_at TEXT, executed_at TEXT, remote_id TEXT,
                    parameters_json TEXT, error_message TEXT)""")
                db.executemany("INSERT INTO agent_actions VALUES (?,?,?,?,?,?,?,?,?)", [
                    ("ok", "REPLY", "11", "SUCCEEDED", "2026-11-01T00:00:00Z",
                     "2026-11-01T00:01:00Z", "22", '{"selection_method":"heuristic"}', None),
                    ("dry", "CREATE_POST", None, "DRY_RUN", "2026-11-01T00:00:00Z",
                     "2026-11-01T00:01:00Z", None, "{}", None),
                    ("repost", "REPOST", "11", "SUCCEEDED", "2026-11-01T00:00:00Z",
                     "2026-11-01T00:01:00Z", "11", "{}", None),
                ])
                db.commit()
            social = FakeSocial()
            monitor = ObservationMonitor(path, social)
            first = monitor.observe_once()
            second = monitor.observe_once()
            self.assertEqual(social.calls, 2)
            self.assertEqual(len(first), 1)
            self.assertEqual(first[0]["delta"], {})
            self.assertEqual(second[0]["delta"]["view_count"], 2)
            actions = {item["action_id"]: item for item in monitor.recent_actions(3)}
            self.assertEqual(actions["ok"]["selection_method"], "heuristic")
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute(
                    "SELECT count(*) FROM observed_post_counters").fetchone()[0], 2)
                self.assertIsNone(db.execute("SELECT name FROM sqlite_master "
                                             "WHERE name='policy_rewards'").fetchone())


if __name__ == "__main__":
    unittest.main()
