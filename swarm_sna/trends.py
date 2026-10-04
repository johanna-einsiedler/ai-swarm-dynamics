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
from .helping import checked_requests, helpful, load_all, logit_fit

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


def by_goal_type(dyads, goals, n_boot=1000, seed=0):
    """Response rates by what the swarm was told to do, within era. Each rate comes with a 95% interval from
    resampling days within the era and kind of goal (requests cluster by day, so a plain binomial interval would be
    too narrow), and every kind is set against the era's commonest kind by the odds that a given agent answers once
    group size (log) and being addressed by name are held fixed: a logistic model, its interval again over days."""
    rng = np.random.default_rng(seed)
    d = dyads.merge(goals[["goal_id", "goal_type"]], on="goal_id", how="left").dropna(subset=["goal_type"])
    keys = ["era", "goal_type", "day"]
    per_req = d.groupby(keys + ["request_id"]).agg(any_response=("responded", "any"), active_n=("active_n", "first")).reset_index()
    cells = pd.DataFrame(
        {
            "req": per_req.groupby(keys).size(), "req_ans": per_req.groupby(keys).any_response.sum(),
            "pair": d.groupby(keys).size(), "pair_ans": d.groupby(keys).responded.sum(),
            "addr": d[d.addressed].groupby(keys).size(), "addr_ans": d[d.addressed].groupby(keys).responded.sum(),
        }
    ).fillna(0).reset_index()
    rates = [("answered_by_anyone", "req_ans", "req"), ("each_agent_responds", "pair_ans", "pair"), ("addressed_agent_responds", "addr_ans", "addr")]
    rows = []
    for (era, gt), c in cells.groupby(["era", "goal_type"], sort=True):
        a = c[["req", "req_ans", "pair", "pair_ans", "addr", "addr_ans"]].to_numpy(float)
        s = dict(zip(["req", "req_ans", "pair", "pair_ans", "addr", "addr_ans"], a.sum(0)))
        b = a[rng.integers(0, len(c), size=(n_boot, len(c)))].sum(1)
        bs = dict(zip(["req", "req_ans", "pair", "pair_ans", "addr", "addr_ans"], b.T))
        row = {"era": era, "goal_type": gt, "requests": int(s["req"]), "days": int(len(c)), "agents_a_day": float(per_req[(per_req.era == era) & (per_req.goal_type == gt)].active_n.mean())}
        for name, num, den in rates:
            row[name] = s[num] / s[den] if s[den] else np.nan
            draws = bs[num] / np.where(bs[den] > 0, bs[den], np.nan)
            row[name + "_lo"], row[name + "_hi"] = (np.nanpercentile(draws, [2.5, 97.5]) if np.isfinite(draws).any() else (np.nan, np.nan))
        rows.append(row)
    t = pd.DataFrame(rows)
    t["main"], t["odds_vs_main"], t["odds_lo"], t["odds_hi"] = "", np.nan, np.nan, np.nan
    for era, te in t.groupby("era"):
        kinds = te.sort_values("requests", ascending=False).goal_type.tolist()
        t.loc[te.index, "main"] = kinds[0]
        if len(kinds) < 2:
            continue
        e = d[d.era == era]
        X = np.c_[np.ones(len(e)), e.addressed.astype(float), np.log(e.active_n.clip(lower=1)), np.column_stack([(e.goal_type == k).astype(float) for k in kinds[1:]])]
        y = e.responded.to_numpy(float)
        fit = logit_fit(X, y)
        days = e.day.to_numpy()
        uniq = np.unique(days)
        where = {u: np.flatnonzero(days == u) for u in uniq}
        boots = []
        for _ in range(min(n_boot, 200)):
            ii = np.concatenate([where[u] for u in rng.choice(uniq, size=len(uniq), replace=True)])
            boots.append(logit_fit(X[ii], y[ii]))
        boots = np.array(boots)
        for j, k in enumerate(kinds[1:]):
            i = te.index[te.goal_type == k][0]
            t.loc[i, "odds_vs_main"] = np.exp(fit[3 + j])
            t.loc[i, ["odds_lo", "odds_hi"]] = np.exp(np.percentile(boots[:, 3 + j], [2.5, 97.5]))
    return t


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
        print("\nresponse rates by goal type, within era (95% intervals over days; odds against the era's commonest kind, group size and addressing held fixed)")
        print(t[["era", "goal_type", "requests", "agents_a_day", "answered_by_anyone", "each_agent_responds", "addressed_agent_responds", "odds_vs_main", "odds_lo", "odds_hi"]].round(3).to_string(index=False))
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
