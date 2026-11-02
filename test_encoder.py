
# test_encoder.py

import math

from state.store import StateStore
from state.builder import StateBuilder
from state.encoder import (
    StateEncoder,
    GLOBAL_FEATURE_NAMES,
    POST_FEATURE_NAMES,
    EVENT_FEATURE_NAMES,
    TOPIC_FEATURE_NAMES,
)


store = StateStore(
    db_path="data/state_v5_test.db"
)

world = StateBuilder(store).build(1)

encoder = StateEncoder()
encoded = encoder.encode(world)

print("=== ENCODER ===")
print("Version:", encoded.version)
print("Snapshot:", encoded.snapshot_id)

print("\n=== GLOBAL FEATURES ===")

for name, value in zip(
    GLOBAL_FEATURE_NAMES,
    encoded.global_vector
):
    print(f"{name:28s}: {value:.4f}")


print("\n=== SAMPLE POST FEATURES ===")

for post_id, vector in list(
    encoded.post_vectors.items()
)[:3]:

    print(f"\nPost {post_id}")

    for name, value in zip(
        POST_FEATURE_NAMES, vector
    ):
        print(f"  {name:24s}: {value:.4f}")


print("\n=== VECTOR DIMENSIONS ===")

print("Global:", len(encoded.global_vector))
print("Posts:", len(encoded.post_vectors))
print("Events:", len(encoded.event_vectors))
print("Topics:", len(encoded.topic_vectors))


# 1. Fixed vector dimensions
assert len(encoded.global_vector) == len(
    GLOBAL_FEATURE_NAMES
)

assert all(
    len(v) == len(POST_FEATURE_NAMES)
    for v in encoded.post_vectors.values()
)

assert all(
    len(v) == len(EVENT_FEATURE_NAMES)
    for v in encoded.event_vectors.values()
)

assert all(
    len(v) == len(TOPIC_FEATURE_NAMES)
    for v in encoded.topic_vectors.values()
)


# 2. No NaN or infinity
all_vectors = (
    [encoded.global_vector]
    + list(encoded.post_vectors.values())
    + list(encoded.event_vectors.values())
    + list(encoded.topic_vectors.values())
)

assert all(
    math.isfinite(value)
    for vector in all_vectors
    for value in vector
)


# 3. Deterministic encoding
encoded_again = encoder.encode(world)

assert encoded == encoded_again


# 4. Correct counts
assert len(encoded.post_vectors) == len(world.posts)
assert len(encoded.event_vectors) == len(world.events)
assert len(encoded.topic_vectors) == len(world.topics)


print("\nStateEncoder tests passed!")
