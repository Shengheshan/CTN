
# state/topic_classification/classifier.py

import json

from .validator import (
    ClassificationError,
    validate_response,
)

PROMPT_VERSION = "topic-batch-v5-two-stage"


EXTRACT_SYSTEM = """
You identify the main factual subjects of texts.

Analyze the ENTIRE batch together.

For each distinct subject:
- Give it a short, specific, reusable name.
- Write a concise description.
- List all input item IDs discussing that subject.
- Merge genuinely equivalent subjects across texts.
- Keep different subjects separate.
- A text may discuss multiple subjects.

Focus on what is actually being discussed.
Ignore incidental hashtags, links, voting positions,
and generic political statements.

Do not combine unrelated subjects into a broad label.
Do not infer a subject from unsupported assumptions.

Every substantive text should have at least one subject.
If a text genuinely has no identifiable subject,
it may have no subject assignment.

Return ONLY JSON:

{
  "subjects": [
    {
      "id": "s1",
      "name": "Subject Name",
      "description": "What this subject covers",
      "item_ids": ["post:123"]
    }
  ]
}

Temporary subject IDs must be unique.
"""


MATCH_SYSTEM = """
You match a list of subjects against an existing
registered topic library.

For EACH subject, decide:

A. Reuse ONE existing topic if its definition
   genuinely covers the subject.

B. Otherwise propose ONE new topic.

Rules:
- Compare the subject to ALL existing topics.
- Reuse semantically equivalent topics even when
  their wording is different.
- Shared keywords alone do not justify matching.
- Never propose a duplicate of an existing topic.
- Never combine unrelated subjects.
- Do not infer voting intentions.
- Return ONLY valid JSON.

Required format:

{
  "assignments": [
    {
      "subject_id": "s1",
      "existing_topic_id": null,
      "new_topic": {
        "name": "New Topic",
        "description": "Topic definition"
      }
    }
  ]
}

For an existing topic, set existing_topic_id to its
registered ID and new_topic to null.

Every subject must have exactly one assignment.
"""


def _name_key(value: str) -> str:
    return " ".join(value.casefold().split())


def _nonempty(value, max_length=300):
    return (
        isinstance(value, str)
        and bool(value.strip())
        and len(value) <= max_length
    )


class TopicClassifier:

    def __init__(self, llm_client, before_request=None):
        self.client = llm_client
        self.before_request = before_request

    def _request(self, payload, system, parser):
        base_prompt = json.dumps(
            payload,
            ensure_ascii=False
        )

        last_error = None

        for attempt in range(2):
            prompt = base_prompt

            if attempt:
                prompt += (
                    "\nPrevious output was invalid: "
                    + str(last_error)
                    + "\nReturn corrected complete JSON."
                )

            if self.before_request:
                self.before_request()

            response = self.client.ask(
                prompt=prompt,
                system=system,
                max_tokens=900
            )

            try:
                data = json.loads(response)
                return parser(data)

            except (
                json.JSONDecodeError,
                ClassificationError
            ) as exc:
                last_error = exc

        raise ClassificationError(
            f"Phase failed validation: {last_error}"
        )

    def _extract(self, items):

        expected = {item["id"] for item in items}

        def parse(data):
            if not isinstance(data, dict):
                raise ClassificationError(
                    "Extraction must be an object"
                )

            subjects = data.get("subjects")

            if not isinstance(subjects, list):
                raise ClassificationError(
                    "Missing subjects list"
                )

            seen_ids = set()
            seen_names = set()
            covered_items = set()

            for subject in subjects:
                if not isinstance(subject, dict):
                    raise ClassificationError(
                        "Invalid subject"
                    )

                sid = subject.get("id")
                name = subject.get("name")
                desc = subject.get("description")
                members = subject.get("item_ids")

                if not (
                    _nonempty(sid, 50)
                    and _nonempty(name, 80)
                    and _nonempty(desc)
                    and isinstance(members, list)
                    and bool(members)
                ):
                    raise ClassificationError(
                        "Incomplete subject"
                    )

                if sid in seen_ids:
                    raise ClassificationError(
                        "Duplicate subject ID"
                    )

                key = _name_key(name)

                if key in seen_names:
                    raise ClassificationError(
                        "Duplicate subject name"
                    )

                if (
                    any(
                        not isinstance(i, str)
                        or i not in expected
                        for i in members
                    )
                    or len(members) != len(set(members))
                ):
                    raise ClassificationError(
                        "Unknown or duplicate item reference"
                    )

                seen_ids.add(sid)
                seen_names.add(key)
                covered_items.update(members)

            # 真实的非空帖子不允许被随意丢弃。
            # substantive = {
            #     item["id"]
            #     for item in items
            #     if len(item["text"].strip()) >= 20
            # }

            # if not substantive.issubset(covered_items):
            #     raise ClassificationError(
            #         "Substantive items missing subjects: "
            #         + str(sorted(substantive - covered_items))
            #     )

            # return subjects
            return subjects

        # return self._request(
        #     {"items": items},
        #     EXTRACT_SYSTEM,
        #     parse
        # )
        
        subjects = self._request(
            {"items": items},
            EXTRACT_SYSTEM,
            parse
        )

        required_ids = {
            item["id"]
            for item in items
            if len(item["text"].strip()) >= 20
        }

        covered_ids = {
            item_id
            for subject in subjects
            for item_id in subject["item_ids"]
        }

        missing_ids = required_ids - covered_ids

        if missing_ids:
            print(
                "Missing subjects for:",
                sorted(missing_ids)
            )

            missing_items = [
                item for item in items
                if item["id"] in missing_ids
            ]

            def parse_recovery(data):
                recovered = parse(data)

                recovered_ids = {
                    item_id
                    for subject in recovered
                    for item_id in subject["item_ids"]
                }

                if not recovered_ids.issubset(missing_ids):
                    raise ClassificationError(
                        "Recovery contains unexpected items"
                    )

                if not missing_ids.issubset(recovered_ids):
                    raise ClassificationError(
                        "Recovery still missing subjects"
                    )

                return recovered

            recovered = self._request(
                {
                    "items": missing_items,
                    "existing_batch_subjects": [
                        {
                            "name": s["name"],
                            "description": s["description"]
                        }
                        for s in subjects
                    ]
                },
                EXTRACT_SYSTEM + """
Classify ONLY the supplied missing items.
Every supplied item has a substantive subject
and MUST receive at least one subject.

If an existing batch subject fits, copy its
exact name and description.
Otherwise create a distinct subject.
""",
                parse_recovery
            )

            # 将补充结果合并回原批次
            by_name = {
                _name_key(s["name"]): s
                for s in subjects
            }

            used_ids = {s["id"] for s in subjects}
            next_id = 1

            for subject in recovered:
                key = _name_key(subject["name"])

                if key in by_name:
                    target = by_name[key]

                    for item_id in subject["item_ids"]:
                        if item_id not in target["item_ids"]:
                            target["item_ids"].append(item_id)

                else:
                    while f"s{next_id}" in used_ids:
                        next_id += 1

                    new_subject = dict(subject)
                    new_subject["id"] = f"s{next_id}"

                    used_ids.add(new_subject["id"])
                    subjects.append(new_subject)
                    by_name[key] = new_subject

        return subjects


    def _match(self, subjects, existing_topics):

        existing_ids = {
            t["topic_id"]
            for t in existing_topics
        }

        existing_names = {
            _name_key(t["name"]): t["topic_id"]
            for t in existing_topics
        }

        subject_ids = {
            subject["id"]
            for subject in subjects
        }

        def parse(data):
            if not isinstance(data, dict):
                raise ClassificationError(
                    "Matching must be an object"
                )

            assignments = data.get("assignments")

            if not isinstance(assignments, list):
                raise ClassificationError(
                    "Missing assignments"
                )

            mapping = {}

            for item in assignments:
                if not isinstance(item, dict):
                    raise ClassificationError(
                        "Invalid assignment"
                    )

                sid = item.get("subject_id")
                matched = item.get("existing_topic_id")
                new = item.get("new_topic")

                if (
                    not isinstance(sid, str)
                    or sid not in subject_ids
                    or sid in mapping
                ):
                    raise ClassificationError(
                        "Unknown or duplicate subject"
                    )

                if matched is not None:
                    if (
                        not isinstance(matched, str)
                        or matched not in existing_ids
                        or new is not None
                    ):
                        raise ClassificationError(
                            "Invalid registered topic match"
                        )

                    mapping[sid] = {
                        "existing": matched,
                        "new": None
                    }

                else:
                    if not isinstance(new, dict):
                        raise ClassificationError(
                            "Missing new topic"
                        )

                    name = new.get("name")
                    desc = new.get("description")

                    if not (
                        _nonempty(name, 80)
                        and _nonempty(desc)
                    ):
                        raise ClassificationError(
                            "Invalid new topic"
                        )

                    # Exact-name duplicate:
                    # automatically reuse the registered ID.
                    duplicate_id = existing_names.get(
                        _name_key(name)
                    )

                    if duplicate_id:
                        mapping[sid] = {
                            "existing": duplicate_id,
                            "new": None
                        }
                    else:
                        mapping[sid] = {
                            "existing": None,
                            "new": {
                                "name": name.strip(),
                                "description": desc.strip()
                            }
                        }

            
            if set(mapping) != subject_ids:
                raise ClassificationError(
                    "Missing subject assignments"
                )

            # 对 LLM 提议的新主题进行保守复用检查
            # 只处理明确的名称包含关系，不强行合并一般相似主题
            for subject in subjects:
                sid = subject["id"]
                decision = mapping[sid]

                if decision["existing"] is not None:
                    continue

                subject_name = _name_key(subject["name"])

                if len(subject_name.split()) < 2:
                    continue

                candidates = [
                    topic["topic_id"]
                    for topic in existing_topics
                    if (
                        _name_key(topic["name"]) == subject_name
                        or _name_key(topic["name"]).startswith(
                            subject_name + " "
                        )
                    )
                ]

                if len(candidates) == 1:
                    decision["existing"] = candidates[0]
                    decision["new"] = None

            return mapping


        return self._request(
            {
                "existing_topics": existing_topics,
                "subjects": [
                    {
                        "id": s["id"],
                        "name": s["name"],
                        "description": s["description"]
                    }
                    for s in subjects
                ]
            },
            MATCH_SYSTEM,
            parse
        )

    def classify_batch(
        self,
        items: list[dict],
        existing_topics: list[dict]
    ) -> dict:

        if not 1 <= len(items) <= 3:
            raise ValueError(
                "Batch size must be 1-3"
            )

        ids = [item["id"] for item in items]

        if (
            len(ids) != len(set(ids))
            or any(
                not isinstance(i, str) or not i
                for i in ids
            )
            or any(
                not isinstance(item.get("text"), str)
                for item in items
            )
        ):
            raise ValueError("Invalid batch items")

        # Phase 1: recognize and merge subjects
        subjects = self._extract(items)
        print("\n=== STAGE 1: EXTRACTED SUBJECTS ===")
        print(json.dumps(subjects, ensure_ascii=False, indent=2))

        # Phase 2: reuse or propose registry topics
        matches = self._match(
            subjects,
            existing_topics
        ) if subjects else {}
        print("\n=== STAGE 2: TOPIC MATCHES ===")
        print(json.dumps(matches, ensure_ascii=False, indent=2))

        result = {
            "proposed_topics": [],
            "results": {
                item_id: {
                    "matched_topic_ids": [],
                    "proposed_topic_ids": []
                }
                for item_id in ids
            }
        }

        new_topic_ids = {}

        for subject in subjects:
            sid = subject["id"]
            decision = matches[sid]

            existing = decision["existing"]

            if existing:
                assigned_id = existing
                is_new = False
            else:
                new_topic = decision["new"]
                key = _name_key(new_topic["name"])

                if key not in new_topic_ids:
                    temp_id = (
                        f"new_{len(new_topic_ids) + 1}"
                    )

                    new_topic_ids[key] = temp_id

                    result["proposed_topics"].append({
                        "temp_id": temp_id,
                        **new_topic
                    })

                assigned_id = new_topic_ids[key]
                is_new = True

            for item_id in subject["item_ids"]:
                field = (
                    "proposed_topic_ids"
                    if is_new
                    else "matched_topic_ids"
                )

                values = result["results"][item_id][field]

                if assigned_id not in values:
                    values.append(assigned_id)

        validation_payload = {
        "proposed_topics": result["proposed_topics"],
        "results": [
            {
                "id": item_id,
                **classification
            }
            for item_id, classification
            in result["results"].items()
        ]
        }

        return validate_response(
            response=json.dumps(validation_payload),
            expected_ids=ids,
            existing_topic_ids={
                t["topic_id"]
                for t in existing_topics
            }
        )

