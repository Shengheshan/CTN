from dataclasses import asdict
from pprint import pprint

from state.store import StateStore
from state.builder import StateBuilder

store = StateStore(
    db_path="data/state_v5_test.db"
)

world = StateBuilder(store).build(1)

pprint(asdict(world), depth=4)