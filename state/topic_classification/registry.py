
# state/topic_classification/registry.py

import hashlib
import sqlite3
from pathlib import Path


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def name_key(name: str) -> str:
    return " ".join(name.casefold().split())


def format_topic_id(value: int) -> str:
    return f"topic_{value:06d}"


def parse_topic_id(value: str) -> int:
    if not value.startswith("topic_"):
        raise ValueError(f"Invalid topic ID: {value}")
    return int(value[6:])


class TopicRegistry:

    def __init__(self, db_path="data/state.db"):

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        self._init_database()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def load_snapshot_assignments(
        self,
        snapshot_id: int,
        version: str
    ) -> dict[str, list[str]]:

        with self._connect() as conn:
            rows = conn.execute("""
                SELECT c.content_id, a.topic_id
                FROM classified_items c
                LEFT JOIN topic_assignments a
                    ON c.snapshot_id = a.snapshot_id
                    AND c.content_id = a.content_id
                WHERE c.snapshot_id = ?
                AND c.classifier_version = ?
                ORDER BY c.content_id, a.topic_id
            """, (snapshot_id, version)).fetchall()

        assignments = {}

        for content_id, topic_id in rows:
            assignments.setdefault(content_id, [])

            if topic_id is not None:
                assignments[content_id].append(
                    format_topic_id(topic_id)
                )

        return assignments

    def _init_database(self):

        with self._connect() as conn:

            conn.execute("""
                CREATE TABLE IF NOT EXISTS topics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    name_key TEXT NOT NULL UNIQUE
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS classified_items (
                    snapshot_id INTEGER NOT NULL,
                    content_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    classifier_version TEXT NOT NULL,
                    PRIMARY KEY (snapshot_id, content_id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS topic_assignments (
                    snapshot_id INTEGER NOT NULL,
                    content_id TEXT NOT NULL,
                    topic_id INTEGER NOT NULL,
                    PRIMARY KEY (
                        snapshot_id, content_id, topic_id
                    )
                )
            """)

    def get_all(self) -> list[dict]:

        with self._connect() as conn:
            rows = conn.execute("""
                SELECT id, name, description
                FROM topics
                ORDER BY id
            """).fetchall()

        return [
            {
                "topic_id": format_topic_id(row[0]),
                "name": row[1],
                "description": row[2]
            }
            for row in rows
        ]

    def get_cached(
        self,
        snapshot_id: int,
        item: dict,
        version: str
    ) -> list[str] | None:

        item_id = item["id"]
        digest = content_hash(item["text"])

        with self._connect() as conn:

            row = conn.execute("""
                SELECT content_hash, classifier_version
                FROM classified_items
                WHERE snapshot_id = ? AND content_id = ?
            """, (snapshot_id, item_id)).fetchone()

            if row != (digest, version):
                return None

            rows = conn.execute("""
                SELECT topic_id
                FROM topic_assignments
                WHERE snapshot_id = ? AND content_id = ?
                ORDER BY topic_id
            """, (snapshot_id, item_id)).fetchall()

        # [] 表示合法的无主题结果，None 表示未分类
        return [format_topic_id(r[0]) for r in rows]

    def save_batch(
        self,
        snapshot_id: int,
        items: list[dict],
        result: dict,
        version: str
    ):

        with self._connect() as conn:

            temp_to_real = {}

            # 1. 注册新主题
            for proposal in result["proposed_topics"]:

                key = name_key(proposal["name"])

                row = conn.execute("""
                    SELECT id FROM topics
                    WHERE name_key = ?
                """, (key,)).fetchone()

                if row:
                    topic_id = row[0]

                else:
                    cursor = conn.execute("""
                        INSERT INTO topics
                        (name, description, name_key)
                        VALUES (?, ?, ?)
                    """, (
                        proposal["name"],
                        proposal["description"],
                        key
                    ))

                    topic_id = cursor.lastrowid

                temp_to_real[proposal["temp_id"]] = topic_id

            # 2. 保存每条内容的主题对应关系
            for item in items:

                item_id = item["id"]
                classification = result["results"][item_id]

                matched = {
                    parse_topic_id(t)
                    for t in classification["matched_topic_ids"]
                }

                proposed = {
                    temp_to_real[t]
                    for t in classification["proposed_topic_ids"]
                }

                topic_ids = matched | proposed

                # 检查引用的正式 Topic 是否存在
                for topic_id in topic_ids:
                    exists = conn.execute(
                        "SELECT 1 FROM topics WHERE id = ?",
                        (topic_id,)
                    ).fetchone()

                    if not exists:
                        raise ValueError("Unknown registered topic")

                # 重新分类时替换该快照的旧关系
                conn.execute("""
                    DELETE FROM topic_assignments
                    WHERE snapshot_id = ? AND content_id = ?
                """, (snapshot_id, item_id))

                conn.execute("""
                    INSERT INTO classified_items
                    (
                        snapshot_id, content_id,
                        content_hash, classifier_version
                    )
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(snapshot_id, content_id)
                    DO UPDATE SET
                        content_hash = excluded.content_hash,
                        classifier_version = excluded.classifier_version
                """, (
                    snapshot_id,
                    item_id,
                    content_hash(item["text"]),
                    version
                ))

                for topic_id in sorted(topic_ids):
                    conn.execute("""
                        INSERT INTO topic_assignments
                        (snapshot_id, content_id, topic_id)
                        VALUES (?, ?, ?)
                    """, (snapshot_id, item_id, topic_id))
