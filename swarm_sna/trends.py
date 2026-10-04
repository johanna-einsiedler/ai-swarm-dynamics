"""Time trends: the weekly series behind every pooled number, with era boundaries marked.

A pooled statistic over eighteen months mixes regimes. This prints and plots how
the basic rates move week by week, and breaks the response rate down by goal
type, so a reader can see whether a headline result is a property of the swarm
or of one period.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import figures
from .helping import checked_requests, helpful, load_all

MIN_REQUESTS = 20  # weeks with fewer judged requests show no rate

SERIES = [
    ("agents", "agents active"),
    ("requests_per_100", "requests per 100 messages"),
    ("answered", "requests answered by anyone"),
    ("each_agent_responds", "a given agent present answers"),
    ("compliance", "directives complied with"),
    ("backed", "answers backed by checkable evidence"),
]


def _week(day):
    return pd.to_datetime(day).dt.to_period("W").dt.start_time


def weekly(events, requests, responses, dyads):
    m = events[(events.type == "message") & (events.actor_class == "agent")]
    m = m.assign(week=_week(m.day))
    out = m.groupby("week").agg(messages=("event_id", "size"), agents=("actor", "nunique"), era=("era", "first"))

    req = requests.assign(week=_week(requests.day))
    out["requests"] = req.groupby("week").size()
    out["requests_per_100"] = 100 * out.requests / out.messages

    judged = req[req.judged.fillna(False).astype(bool)]
    checked = checked_requests(responses)
    if checked is not None:
        judged = judged[judged.request_id.isin(checked)]
    good = helpful(responses)
    judged = judged.assign(answered=judged.request_id.isin(good.request_id))
    n = judged.groupby("week").size()
    out["answered"] = judged.groupby("week").answered.mean().where(n >= MIN_REQUESTS)
    out["backed"] = good.merge(req[["request_id", "week"]], on="request_id").groupby("week").backed.mean().where(n >= MIN_REQUESTS)

    d = dyads.assign(week=_week(dyads.day))
    out["each_agent_responds"] = d.groupby("week").responded.mean().where(n >= MIN_REQUESTS)
    c = d[d.directive & d.addressed]
    nc = c.groupby("week").size()
    out["compliance"] = (c.responded | c.acted).groupby(c.week).mean().where(nc >= MIN_REQUESTS)
    return out.reset_index()


def by_goal_type(dyads, goals):
    """Response rates by what the swarm was told to do, within era."""
    d = dyads.merge(goals[["goal_id", "goal_type"]], on="goal_id", how="left")
    per_req = d.groupby("request_id").agg(any_response=("responded", "any"), era=("era", "first"), goal_type=("goal_type", "first"))
    t = pd.DataFrame(
        {
            "requests": per_req.groupby(["era", "goal_type"]).size(),
            "answered_by_anyone": per_req.groupby(["era", "goal_type"]).any_response.mean(),
            "each_agent_responds": d.groupby(["era", "goal_type"]).responded.mean(),
            "addressed_agent_responds": d[d.addressed].groupby(["era", "goal_type"]).responded.mean(),
        }
    )
    return t.reset_index()


def run(in_dir, meta=None):
    in_dir = Path(in_dir)
    events = pd.read_parquet(in_dir / "events.parquet")
    dyads = pd.read_parquet(in_dir / "dyads.parquet")
    w = weekly(events, load_all(in_dir, "requests"), load_all(in_dir, "responses"), dyads)
    w.to_csv(in_dir / "trends_weekly.csv", index=False)
    era_starts = w.groupby("era").week.min()
    figures.trends(w, SERIES, era_starts, in_dir / "fig_trends.png")

    print("\nmean of the weekly series, per era")
    cols = [c for c, _ in SERIES]
    print(w.groupby("era")[cols].mean().round(3).to_string())
    print("\nlinear trend per week WITHIN era (slope x 52 = change per year); a pooled trend would mix regimes")
    rows = {}
    for era, g in w.groupby("era"):
        x = (g.week - g.week.min()).dt.days / 7
        rows[f"era {era}"] = {c: (np.polyfit(x[g[c].notna()], g[c].dropna(), 1)[0] * 52 if g[c].notna().sum() > 4 else np.nan) for c in cols}
    print(pd.DataFrame(rows).round(3).to_string())

    goals_path = Path(meta) / "goals.csv" if meta else None
    if goals_path and goals_path.exists():
        t = by_goal_type(dyads, pd.read_csv(goals_path))
        t.to_csv(in_dir / "trends_goal_type.csv", index=False)
        print("\nresponse rates by goal type, within era")
        print(t.round(3).to_string(index=False))
    # The same two networks drawn per quarter: who mentions whom, and who answers whose requests.
    agents = pd.read_parquet(in_dir / "agents.parquet")
    m = pd.read_parquet(in_dir / "mentions.parquet")
    m = m[m.actor_class == "agent"].rename(columns={"actor": "source"})
    figures.network_timeline(m[["ts", "source", "target", "weight"]], agents, in_dir / "fig_network_timeline_mentions.png", "Who mentions whom, per quarter (node size: mentions received; agents in order of joining, clockwise)")
    resp = load_all(in_dir, "responses")
    req = load_all(in_dir, "requests")
    h = helpful(resp)
    h = h[h.actor_class == "agent"].merge(req[["request_id", "actor"]].rename(columns={"actor": "target"}), on="request_id")
    h = h.rename(columns={"actor": "source"}).assign(weight=1.0)
    figures.network_timeline(h[["ts", "source", "target", "weight"]], agents, in_dir / "fig_network_timeline_help.png", "Who answers whose requests, per quarter (node size: answers received; agents in order of joining, clockwise)")
    print(f"\nwrote {in_dir}/trends_weekly.csv, fig_trends.png, fig_network_timeline_mentions.png, fig_network_timeline_help.png" + (", trends_goal_type.csv" if goals_path else ""))
