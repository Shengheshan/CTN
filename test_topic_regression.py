
import json

from config import CTN_TEAM_CODE
from state.store import StateStore
from state.topic_classification.client import LLMClient
from state.topic_classification.classifier import TopicClassifier
from state.topic_classification.registry import TopicRegistry


SNAPSHOT_ID = 1

TEST_POST_IDS = [
    1039865,  # Cyclone / weather warning
    1036485,  # Privacy of certificate app
    1028230,  # Children / public-space requirements
]


def main():
    store = StateStore()
    registry = TopicRegistry(store.db_path)

    snapshot = store.load_snapshot(SNAPSHOT_ID)

    post_index = {
        post["post_id"]: post
        for post in snapshot["posts"]
    }

    items = [
        {
            "id": f"post:{post_id}",
            "text": post_index[post_id]["body"][:900]
        }
        for post_id in TEST_POST_IDS
    ]

    existing_topics = registry.get_all()

    print("Existing topics:", len(existing_topics))

    classifier = TopicClassifier(
        LLMClient(CTN_TEAM_CODE)
    )

    result = classifier.classify_batch(
        items=items,
        existing_topics=existing_topics
    )

    print("\nClassification result:")
    print(json.dumps(
        result, ensure_ascii=False, indent=2
    ))

    # 回归检查
    results = result["results"]

    checks = {
        "Cyclone not classified as voting rights":
            "topic_000004" not in
            results["post:1039865"]["matched_topic_ids"],

        "App privacy not classified as hospital staffing":
            "topic_000001" not in
            results["post:1036485"]["matched_topic_ids"],

        "Children/public-space rules not classified as privacy":
            "topic_000003" not in
            results["post:1028230"]["matched_topic_ids"],
    }

    checks.update({
        "Cyclone has a proposed new topic":
            bool(
                results["post:1039865"]["proposed_topic_ids"]
            ),

        "Privacy reuses topic_000003":
            "topic_000003" in
            results["post:1036485"]["matched_topic_ids"],

        "No duplicate existing topic names":
            all(
                topic["name"].casefold() not in {
                    existing["name"].casefold()
                    for existing in existing_topics
                }
                for topic in result["proposed_topics"]
            ),

        "Children/public-space post not voting rights":
            "topic_000004" not in
            results["post:1028230"]["matched_topic_ids"]
            and not any(
                "voting rights" in topic["name"].casefold()
                for topic in result["proposed_topics"]
                if topic["temp_id"] in
                results["post:1028230"]["proposed_topic_ids"]
            )
    })


    print("\nRegression checks:")

    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {name}")

    print(
        f"\nPassed: {sum(checks.values())}/{len(checks)}"
    )


if __name__ == "__main__":
    main()
