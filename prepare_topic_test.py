
import sqlite3
from pathlib import Path

source = Path("data/state.db")
target = Path("data/state_v5_test.db")

if target.exists():
    raise RuntimeError(
        "Test database already exists. "
        "Remove it manually only if you want to restart."
    )

with sqlite3.connect(source) as src:
    with sqlite3.connect(target) as dst:
        src.backup(dst)

        # 只清空测试副本里的主题相关数据
        dst.execute("DELETE FROM topic_assignments")
        dst.execute("DELETE FROM classified_items")
        dst.execute("DELETE FROM topics")
        dst.execute(
            "DELETE FROM sqlite_sequence WHERE name = 'topics'"
        )

print("Clean test database created:", target)
