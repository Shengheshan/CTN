
import sqlite3

with sqlite3.connect("data/state.db") as conn:

    rows = conn.execute("""
        SELECT
            a.content_id,
            t.id,
            t.name
        FROM topic_assignments a
        JOIN topics t ON a.topic_id = t.id
        WHERE a.snapshot_id = 1
        ORDER BY a.content_id, t.id
    """).fetchall()

for content_id, topic_id, topic_name in rows:
    print(
        f"{content_id} -> "
        f"topic_{topic_id:06d} ({topic_name})"
    )
