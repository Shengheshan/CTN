
# test_pipeline_live.py

from pathlib import Path

from config import CTN_TEAM_CODE
from state.store import StateStore
from state.topic_classification.client import LLMClient
from state.topic_classification.pipeline import (
    TopicClassificationPipeline
)


# 使用独立测试数据库，避免污染原始数据
DB_PATH = "data/state_v5_test.db"

if not Path(DB_PATH).exists():
    raise FileNotFoundError(
        f"{DB_PATH} 不存在，请先运行 prepare_topic_test.py"
    )

store = StateStore(db_path=DB_PATH)
client = LLMClient(CTN_TEAM_CODE)

pipeline = TopicClassificationPipeline(
    store=store,
    llm_client=client,
    batch_size=3,
    min_interval_seconds=21.0
)

# 先处理 6 条，检查分类质量
result = pipeline.run(
    snapshot_id=1,
    max_items=6
)

print("\n=== Classification Result ===")
print(result)

print("\n=== Registered Topics ===")
for topic in pipeline.registry.get_all():
    print(topic)

print("\n=== Topic Assignments ===")
from state.topic_classification.classifier import PROMPT_VERSION

assignments = pipeline.registry.load_snapshot_assignments(
    snapshot_id=1,
    version=PROMPT_VERSION
)

for content_id, topic_ids in sorted(assignments.items()):
    print(content_id, "->", topic_ids)
