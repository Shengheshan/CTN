
# state/topic_classification/pipeline.py

import time

from openai import APIError

from .classifier import TopicClassifier, PROMPT_VERSION
from .validator import ClassificationError
from .registry import TopicRegistry


class RequestPacer:

    def __init__(self, interval=21.0):
        self.interval = interval
        self.last_request = None

    def wait(self):

        now = time.monotonic()

        if self.last_request is not None:
            elapsed = now - self.last_request
            delay = max(0, self.interval - elapsed)

            if delay > 0:
                time.sleep(delay)

        self.last_request = time.monotonic()


class TopicClassificationPipeline:

    def __init__(
        self,
        store,
        llm_client,
        batch_size=3,
        min_interval_seconds=21.0
    ):

        self.store = store
        self.registry = TopicRegistry(store.db_path)
        self.batch_size = batch_size

        pacer = RequestPacer(min_interval_seconds)

        self.classifier = TopicClassifier(
            llm_client,
            before_request=pacer.wait
        )

    def _load_items(self, snapshot_id):

        snapshot = self.store.load_snapshot(snapshot_id)

        items = []

        # 帖子
        for post in snapshot["posts"]:

            body = (post.get("body") or "").strip()

            if body:
                items.append({
                    "id": f"post:{post['post_id']}",
                    "text": body[:900]
                })

        # 新闻事件
        article_index = {
            article["article_id"]: article
            for article in snapshot["articles"]
        }

        for event in snapshot["events"]:

            texts = []

            for article_id in event["article_ids"]:

                article = article_index.get(article_id)

                if article:
                    texts.append(" ".join(
                        str(v) for v in (
                            article.get("title"),
                            article.get("summary")
                        )
                        if v
                    ))

            text = " ".join(texts).strip()

            if text:
                items.append({
                    "id": f"event:{event['event_id']}",
                    "text": text[:900]
                })

        return items

    def run(self, snapshot_id, max_items=None):

        items = self._load_items(snapshot_id)

        if max_items is not None:
            items = items[:max_items]

        pending = [
            item for item in items
            if self.registry.get_cached(
                snapshot_id,
                item,
                PROMPT_VERSION
            ) is None
        ]

        completed = 0
        initial_pending = len(pending)

        for start in range(0, len(pending), self.batch_size):

            batch = pending[start:start + self.batch_size]

            # 每次都重新读取最新 Registry
            existing_topics = self.registry.get_all()

            print(
                f"\nBatch {start // self.batch_size + 1}: "
                f"{len(batch)} items, "
                f"{len(existing_topics)} existing topics"
            )

            try:
                result = self.classifier.classify_batch(
                    items=batch,
                    existing_topics=existing_topics
                )

                # 必须先保存成功，再处理下一批
                self.registry.save_batch(
                    snapshot_id=snapshot_id,
                    items=batch,
                    result=result,
                    version=PROMPT_VERSION
                )

            except (ClassificationError, APIError) as e:

                print(
                    f"Batch stopped: {type(e).__name__}: {e}"
                )

                # 当前及后续批次保持待处理
                # 下次运行可从缓存处继续
                break

            completed += len(batch)

            print(
                "Registered topics:",
                len(self.registry.get_all())
            )

        return {
            "snapshot_id": snapshot_id,
            "total_items": len(items),
            "cached_items": len(items) - initial_pending,
            "newly_completed": completed,
            "pending_items": initial_pending - completed,
            "total_topics": len(self.registry.get_all())
        }
