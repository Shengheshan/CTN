
# state/topic_classification/validator.py

import json


class ClassificationError(Exception):
    pass


def validate_response(
    response: str,
    expected_ids: list[str],
    existing_topic_ids: set[str]
) -> dict:

    try:
        data = json.loads(response)
    except json.JSONDecodeError as e:
        raise ClassificationError("Invalid JSON") from e

    if not isinstance(data, dict):
        raise ClassificationError("Expected JSON object")

    proposals = data.get("proposed_topics")
    items = data.get("results")

    if not isinstance(proposals, list):
        raise ClassificationError("Missing proposed_topics")

    if not isinstance(items, list):
        raise ClassificationError("Missing results")

    # 1. 验证批次级新主题
    proposal_map = {}
    names = set()

    for topic in proposals:

        if not isinstance(topic, dict):
            raise ClassificationError("Invalid topic proposal")

        temp_id = topic.get("temp_id")
        name = topic.get("name")
        description = topic.get("description")

        if not all(
            isinstance(x, str) and x.strip()
            for x in (temp_id, name, description)
        ):
            raise ClassificationError("Invalid topic fields")

        name_key = " ".join(name.casefold().split())

        if temp_id in proposal_map or name_key in names:
            raise ClassificationError("Duplicate topic proposal")

        if len(name) > 80 or len(description) > 300:
            raise ClassificationError("Topic text too long")

        proposal_map[temp_id] = {
            "temp_id": temp_id,
            "name": name.strip(),
            "description": description.strip()
        }

        names.add(name_key)

    # 2. 验证每条帖子的分类引用
    results = {}
    used_proposals = set()

    for item in items:

        if not isinstance(item, dict):
            raise ClassificationError("Invalid result item")

        item_id = item.get("id")
        matched = item.get("matched_topic_ids")
        proposed = item.get("proposed_topic_ids")

        if not isinstance(item_id, str) or item_id in results:
            raise ClassificationError("Invalid or duplicate item ID")

        if not isinstance(matched, list) or not isinstance(proposed, list):
            raise ClassificationError("Invalid topic ID lists")

        if any(
            not isinstance(t, str)
            or t not in existing_topic_ids
            for t in matched
        ):
            raise ClassificationError("Unknown existing topic ID")

        if any(
            not isinstance(t, str)
            or t not in proposal_map
            for t in proposed
        ):
            raise ClassificationError("Unknown proposed topic ID")

        used_proposals.update(proposed)

        results[item_id] = {
            "matched_topic_ids": list(dict.fromkeys(matched)),
            "proposed_topic_ids": list(dict.fromkeys(proposed))
        }

    if (
        len(expected_ids) != len(set(expected_ids))
        or set(results) != set(expected_ids)
    ):
        raise ClassificationError("Missing or unexpected input IDs")

    if set(proposal_map) != used_proposals:
        raise ClassificationError("Unused proposed topics")

    return {
        "proposed_topics": list(proposal_map.values()),
        "results": results
    }
