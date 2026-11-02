import json
import sqlite3
from pathlib import Path
from dataclasses import asdict
from datetime import datetime


class StateStore:
    

    def load_previous_post_observations(self, snapshot_id: int) -> dict:

        def parse_time(value):
            return datetime.fromisoformat(
                value.replace("Z", "+00:00")
            )

        with sqlite3.connect(self.db_path) as conn:

            # 获取目标快照
            target = conn.execute("""
                SELECT started_at, payload
                FROM snapshots
                WHERE id = ?
            """, (snapshot_id,)).fetchone()

            if target is None:
                raise ValueError("Snapshot not found")

            target_start = parse_time(target[0])
            current_posts = json.loads(target[1])["posts"]

            # 只寻找当前有热度数据的帖子
            target_ids = {
                p["post_id"] for p in current_posts
                if p.get("hot_score") is not None
            }

            rows = conn.execute("""
                SELECT id, finished_at, payload
                FROM snapshots
                WHERE id != ?
            """, (snapshot_id,)).fetchall()

        previous = {}

        # 按实际观测结束时间倒序
        rows = sorted(
            rows,
            key=lambda r: (parse_time(r[1]), r[0]),
            reverse=True
        )

        for old_id, finished_at, payload in rows:

            # 排除目标快照开始后才完成的观测
            if parse_time(finished_at) > target_start:
                continue

            old_posts = json.loads(payload)["posts"]

            for post in old_posts:

                post_id = post["post_id"]

                if (
                    post_id in target_ids
                    and post_id not in previous
                    and post.get("hot_score") is not None
                ):
                    previous[post_id] = {
                        "post": post,
                        "finished_at": finished_at,
                        "snapshot_id": old_id
                    }

            if len(previous) == len(target_ids):
                break

        return previous

    
    def load_historically_trending_posts(
        self, snapshot_id: int, after_snapshot_id: int = 0
    ) -> dict[int, dict]:
        """Return the latest saved observation of each post ever seen in Trending."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT id, payload FROM snapshots
                   WHERE id > ? AND id <= ? ORDER BY id DESC""",
                (after_snapshot_id, snapshot_id),
            ).fetchall()

        observed = {}
        for observed_snapshot_id, payload in rows:
            try:
                posts = json.loads(payload).get("posts", [])
            except (TypeError, json.JSONDecodeError):
                continue
            for post in posts:
                if not isinstance(post, dict):
                    continue
                post_id = post.get("post_id")
                sources = post.get("feed_sources")
                if (not isinstance(post_id, int) or isinstance(post_id, bool)
                        or post_id <= 0 or not isinstance(sources, (list, tuple))
                        or "trending" not in sources or post_id in observed):
                    continue
                observed[post_id] = {
                    **post,
                    "observed_snapshot_id": observed_snapshot_id,
                }
        return observed

    def load_snapshot_record(self, snapshot_id: int) -> dict:

        with sqlite3.connect(self.db_path) as conn:

            row = conn.execute("""
                SELECT started_at, finished_at, payload
                FROM snapshots
                WHERE id = ?
            """, (snapshot_id,)).fetchone()

        if row is None:
            raise ValueError(f"Snapshot {snapshot_id} not found")

        payload = json.loads(row[2])

        return {
            "snapshot_id": snapshot_id,
            "started_at": row[0],
            "finished_at": row[1],
            **payload
        }



    def __init__(self, db_path="data/state.db"):

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        self._init_database()

    def _init_database(self):

        with sqlite3.connect(self.db_path) as conn:

            conn.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL,
                    raw_path TEXT NOT NULL UNIQUE,
                    schema_version TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
            """)

    def save_snapshot(
        self,
        raw_path,
        raw_snapshot,
        posts,
        articles,
        events
    ):

        raw_path = str(Path(raw_path).resolve())

        payload = {
            "posts": [asdict(p) for p in posts],
            "articles": [asdict(a) for a in articles],
            "events": [asdict(e) for e in events]
        }

        with sqlite3.connect(self.db_path) as conn:

            conn.execute("""
                INSERT OR IGNORE INTO snapshots
                (started_at, finished_at, raw_path,
                 schema_version, payload)
                VALUES (?, ?, ?, ?, ?)
            """, (
                raw_snapshot["started_at"],
                raw_snapshot["finished_at"],
                raw_path,
                "1.0",
                json.dumps(payload, ensure_ascii=False)
            ))

            row = conn.execute("""
                SELECT id FROM snapshots
                WHERE raw_path = ?
            """, (raw_path,)).fetchone()

            return row[0]

    def load_snapshot(self, snapshot_id):

        with sqlite3.connect(self.db_path) as conn:

            row = conn.execute("""
                SELECT payload FROM snapshots
                WHERE id = ?
            """, (snapshot_id,)).fetchone()

        if row is None:
            raise ValueError(
                f"Snapshot {snapshot_id} not found"
            )

        return json.loads(row[0])
