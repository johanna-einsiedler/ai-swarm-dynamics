"""Generic adapter: any transcript as JSON Lines (or CSV) -> common event schema.

One row per message. Required fields: `ts`, `actor`, `text`.
Optional: `room` (default "main"), `actor_class` (agent | human | system; default agent),
`event_id`, `type`, `goal_id`, `era`, `day`.

`--data` is the file, or a directory holding `events.jsonl` / `events.csv`.
An optional `agents.csv` next to it (or in `--meta`) adds node attributes:
`agent`, and any of `family`, `release_date`, `joined`, `left`, `aliases` (separated by |).
Without it every actor seen in the transcript becomes a node named by its actor id.
"""
from pathlib import Path

import pandas as pd

from ..schema import validate_events


def _read(path):
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_json(path, lines=True)


def _find(data_dir, meta_dir, stem):
    for d in [Path(data_dir)] + ([Path(meta_dir)] if meta_dir else []):
        for ext in (".jsonl", ".csv"):
            if d.is_dir() and (d / f"{stem}{ext}").exists():
                return d / f"{stem}{ext}"
    return None


def build(data_dir, meta_dir=None):
    data = Path(data_dir)
    path = data if data.is_file() else _find(data, None, "events")
    if path is None:
        raise FileNotFoundError(f"no events.jsonl or events.csv in {data}")
    ev = _read(path)
    missing = [c for c in ("ts", "actor", "text") if c not in ev.columns]
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}")

    ev["ts"] = pd.to_datetime(ev.ts, utc=True, format="mixed").dt.tz_localize(None)
    ev["actor"] = ev.actor.astype(str)
    ev["text"] = ev.text.fillna("").astype(str)
    defaults = {"room": "main", "actor_class": "agent", "type": "message", "goal_id": None, "era": 1}
    for col, value in defaults.items():
        ev[col] = ev[col].fillna(value) if col in ev.columns and value is not None else ev.get(col, value)
    if "event_id" not in ev.columns:
        ev["event_id"] = [f"e{i}" for i in range(len(ev))]
    ev["event_id"] = ev.event_id.astype(str)
    if "day" not in ev.columns:
        ev["day"] = ev.ts.dt.strftime("%Y-%m-%d")
    ev = ev.sort_values("ts", kind="stable").reset_index(drop=True)

    talk = ev[(ev.type == "message") & (ev.actor_class == "agent")]
    active = talk.groupby(["room", "day"]).actor.nunique().rename("active_n").reset_index()
    ev = ev.drop(columns="active_n", errors="ignore").merge(active, on=["room", "day"], how="left")
    ev["active_n"] = ev.active_n.fillna(0).astype(int)

    seen = talk.groupby("actor").ts.agg(["min", "max"])
    agents = pd.DataFrame({"agent": seen.index, "joined": seen["min"].to_numpy()})
    extra = _find(data if data.is_dir() else data.parent, meta_dir, "agents")
    if extra is not None:
        given = _read(extra)
        given["agent"] = given.agent.astype(str)
        agents = agents.drop(columns=[c for c in given.columns if c in agents.columns and c != "agent"]).merge(given, on="agent", how="left")
    for col, value in {"family": "unknown", "release_date": pd.NaT, "joined": pd.NaT, "left": pd.NaT}.items():
        if col not in agents.columns:
            agents[col] = value
    agents["family"] = agents.family.fillna("unknown")
    agents["joined"] = pd.to_datetime(agents.joined).fillna(agents.agent.map(seen["min"]))
    for col in ("release_date", "left"):
        agents[col] = pd.to_datetime(agents[col])
    if "aliases" in agents.columns:
        agents["aliases"] = [str(a).split("|") if isinstance(a, str) and a else [n] for a, n in zip(agents.aliases, agents.agent)]
    else:
        agents["aliases"] = [[n] for n in agents.agent]
    return validate_events(ev), agents[["agent", "family", "release_date", "joined", "left", "aliases"]]
