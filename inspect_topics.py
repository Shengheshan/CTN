
import sqlite3
from collections import defaultdict

from state.store import StateStore


snapshot = StateStore().load_snapshot(1)

posts = {
    f"post:{p['post_id']}": p.get("body", "")
    for p in snapshot["posts"]
}

with sqlite3.connect("data/state.db") as conn:
    rows = conn.execute("""
        SELECT
            t.id,
            t.name,
            a.content_id
        FROM topic_assignments a
        JOIN topics t ON a.topic_id = t.id
        WHERE a.snapshot_id = 1
        ORDER BY t.id, a.content_id
    """).fetchall()

groups = defaultdict(list)

for topic_id, name, content_id in rows:
    groups[(topic_id, name)].append(content_id)

for (topic_id, name), content_ids in groups.items():
    print(f"\n{'=' * 60}")
    print(f"topic_{topic_id:06d}: {name}")
    print(f"Assigned items: {len(content_ids)}")

    for content_id in content_ids:
        print(f"\n  {content_id}")
        print(f"  {posts.get(content_id, '[not a post]')[:350]}")
