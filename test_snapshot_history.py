
import sqlite3

from state.store import StateStore
from state.builder import StateBuilder


DB_PATH = "data/state.db"

with sqlite3.connect(DB_PATH) as conn:
    rows = conn.execute("""
        SELECT id, started_at, finished_at
        FROM snapshots
        ORDER BY id DESC
        LIMIT 5
    """).fetchall()

print("Recent snapshots:")
for row in rows:
    print(row)

if not rows:
    raise RuntimeError("No snapshots found")

latest_id = rows[0][0]

store = StateStore(db_path=DB_PATH)
world = StateBuilder(store).build(latest_id)

measured = []

for post in world.posts:
    velocity = post.features.get("hot_velocity")

    if velocity is not None:
        measured.append((
            post.post_id,
            post.features.get("hot_delta"),
            velocity
        ))

print("\nLatest snapshot:", latest_id)
print("Observed posts:", len(world.posts))
print("Posts with historical velocity:", len(measured))

for post_id, delta, velocity in measured[:10]:
    print(
        f"Post {post_id}: "
        f"delta={delta}, velocity={velocity}"
    )
