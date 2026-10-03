"""Common event schema. Every adapter emits these tables; every layer and analysis reads them.

events   one row per observable act
agents   one row per node, with attributes
"""

EVENT_COLUMNS = [
    "event_id",  # str, unique
    "ts",  # UTC timestamp
    "day",  # str, the swarm's local calendar day (permutation block)
    "actor",  # agent id, or "human:<name>" / "system:<name>"
    "actor_class",  # agent | human | system
    "room",  # str
    "type",  # message | session_start
    "text",  # str
    "goal_id",  # str or null
    "era",  # int or null
    "active_n",  # agents with a message in the same room that day
]

AGENT_COLUMNS = [
    "agent",  # id used in events.actor
    "family",  # model family / vendor
    "release_date",  # capability proxy
    "joined",
    "left",
    "aliases",  # list[str]: every way the agent is named in text
]

EVENT_TYPES = ["message", "session_start"]
ACTOR_CLASSES = ["agent", "human", "system"]


def validate_events(df):
    missing = [c for c in EVENT_COLUMNS if c not in df.columns]
    assert not missing, f"events table is missing columns: {missing}"
    assert df.event_id.is_unique, "event_id must be unique"
    assert set(df.type.unique()) <= set(EVENT_TYPES), set(df.type.unique())
    assert set(df.actor_class.unique()) <= set(ACTOR_CLASSES), set(df.actor_class.unique())
    return df[EVENT_COLUMNS + [c for c in df.columns if c not in EVENT_COLUMNS]]
