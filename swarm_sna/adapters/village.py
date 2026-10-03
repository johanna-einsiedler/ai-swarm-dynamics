"""AI Village adapter (aidigestorg/ai-village) -> common event schema.

Reads the gzipped tables as downloaded from Hugging Face plus the curated
meta/ tables written by scripts/build_meta.py.
"""
import gzip
import json
import re
from pathlib import Path

import pandas as pd

from ..schema import validate_events

LOCAL_TZ = "America/Los_Angeles"
ERA_STARTS = [("2025-04-02", 1), ("2025-07-01", 2), ("2026-07-06", 3)]
SYSTEM_SPEAKERS = {"automated"}

# Names that the generic alias rules handle badly.
ALIAS_OVERRIDES = {
    "[Temporary] Fine-tuned Leader": ["Fine-tuned Leader", "Temporary Fine-tuned Leader"],
    "Fine-Tuned Leader": ["Fine-tuned Leader"],
    "Opus 4.5 (Claude Code)": ["Opus 4.5 (Claude Code)", "Opus 4.5 Claude Code", "Claude Code", "Opus 4.5", "Opus", "Claude"],
}
ALIAS_EXTRAS = {
    "GPT-5.6 Sol": ["Sol"],
    "GPT-5.6 Luna": ["Luna"],
    "GPT-5.6 Terra": ["Terra"],
    "GPT-6 Astra": ["Astra"],
    "Kimi K2.6": ["K2.6"],
    "Kimi K3": ["K3"],
}


def aliases_for(name):
    """Full name plus every shorter form: 'Claude Opus 4.5' -> Opus 4.5, Claude Opus, Opus, Claude."""
    if name in ALIAS_OVERRIDES:
        return ALIAS_OVERRIDES[name]
    tokens = re.split(r"[\s\-]+", name)
    out = [" ".join(tokens[:i]) for i in range(len(tokens), 0, -1)]
    if tokens[0] == "Claude" and len(tokens) > 1:
        out += [" ".join(tokens[1:i]) for i in range(len(tokens), 1, -1)]
    return list(dict.fromkeys(out + ALIAS_EXTRAS.get(name, [])))


def _human_names(data_dir):
    """messageId -> display name, from USER_TALK events (the users table is not exported)."""
    names = {}
    with gzip.open(data_dir / "events.jsonl.gz", "rt") as f:
        for line in f:
            if '"USER_TALK"' not in line:
                continue
            d = json.loads(line)["data"]
            if d.get("actionType") == "USER_TALK":
                names[d.get("messageId")] = d.get("speakerName") or "unknown"
    return names


def _era(ts):
    era = pd.Series(1, index=ts.index)
    for start, e in ERA_STARTS:
        era[ts >= pd.Timestamp(start)] = e
    return era


def load_agents(meta_dir):
    a = pd.read_csv(Path(meta_dir) / "agents.csv")
    a["agent"] = a["name"]
    a["aliases"] = a["name"].map(aliases_for)
    for c in ["release_date", "joined", "left"]:
        a[c] = pd.to_datetime(a[c])
    return a


def build(data_dir, meta_dir="meta"):
    data_dir = Path(data_dir)
    agents = load_agents(meta_dir)
    id_to_name = dict(zip(agents.agent_id, agents.agent))
    rooms = pd.read_json(data_dir / "chat_rooms.jsonl.gz", lines=True).set_index("id")["name"]

    cm = pd.read_json(data_dir / "chat_messages.jsonl.gz", lines=True)
    humans = _human_names(data_dir)
    is_agent = cm.speaker_type == "agent"
    human_name = cm.id.map(humans).fillna("unknown")
    is_system = ~is_agent & human_name.isin(SYSTEM_SPEAKERS)
    msgs = pd.DataFrame(
        {
            "event_id": cm.id,
            "ts": pd.to_datetime(cm.created_at),
            "actor": cm.agent_speaker_id.map(id_to_name).where(
                is_agent, ("system:" + human_name).where(is_system, "human:" + human_name)
            ),
            "actor_class": pd.Series("agent", index=cm.index).where(
                is_agent, pd.Series("system", index=cm.index).where(is_system, "human")
            ),
            "room": cm.room_id.map(rooms),
            "type": "message",
            "text": cm.content.fillna(""),
        }
    )

    cs = pd.read_json(data_dir / "computer_use_sessions.jsonl.gz", lines=True)
    sess = pd.DataFrame(
        {
            "event_id": cs.id,
            "ts": pd.to_datetime(cs.created_at),
            "actor": cs.agent_id.map(id_to_name),
            "actor_class": "agent",
            "room": None,
            "type": "session_start",
            "text": cs.session_goal.fillna(""),
        }
    ).dropna(subset=["actor"])

    ev = pd.concat([msgs, sess], ignore_index=True).sort_values("ts", kind="stable").reset_index(drop=True)
    # A session has no room of its own: it inherits the agent's latest chat room.
    ev["room"] = ev.groupby("actor")["room"].transform(lambda s: s.ffill().bfill())
    ev["room"] = ev["room"].fillna("general")
    ev["day"] = ev.ts.dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ).dt.strftime("%Y-%m-%d")

    goals = pd.read_csv(Path(meta_dir) / "goals.csv", parse_dates=["start_time"]).sort_values("start_time")
    ev = pd.merge_asof(ev, goals[["start_time", "goal_id"]], left_on="ts", right_on="start_time").drop(columns="start_time")
    ev["era"] = _era(ev.ts)

    talk = ev[(ev.type == "message") & (ev.actor_class == "agent")]
    active = talk.groupby(["room", "day"]).actor.nunique().rename("active_n").reset_index()
    ev = ev.merge(active, on=["room", "day"], how="left")
    ev["active_n"] = ev.active_n.fillna(0).astype(int)

    return validate_events(ev), agents[["agent", "family", "release_date", "joined", "left", "aliases", "model_string", "claude_code_scaffold", "imposed_leader"]]
