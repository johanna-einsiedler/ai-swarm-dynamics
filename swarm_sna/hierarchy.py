"""Dominance hierarchy from directives and compliance.

A contest is a directive addressed to a named agent. The sender wins if the
target complies (responds by answering or acting, or starts a matching
session); the target wins if it declines or ignores it.

Null: shuffle the outcomes among contests within room x day. Keeps who directs
whom and how much compliance there is that day; breaks who gets obeyed.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import figures

MIN_CONTESTS = 5


def contests(dyads):
    c = dyads[dyads.directive & dyads.addressed].sort_values("ts")
    return pd.DataFrame({"ts": c.ts, "day": c.day, "room": c.room, "goal_id": c.goal_id, "era": c.era, "sender": c.requester, "target": c.responder, "complied": c.responded | c.acted})


def win_matrix(sender, target, complied, n):
    """W[i, j] = contests i won against j."""
    w = np.where(complied, sender, target)
    l = np.where(complied, target, sender)
    return np.bincount(w * n + l, minlength=n * n).reshape(n, n).astype(float)


def davids_score(W):
    """Normalised David's score with the correction for unequal numbers of contests (de Vries et al. 2006)."""
    n = len(W)
    tot = W + W.T
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.where(tot > 0, W / tot, 0.0)
    D = np.where(tot > 0, P - (P - 0.5) / (tot + 1), 0.0)
    w, l = D.sum(1), D.sum(0)
    return (w + D @ w - l - D.T @ l + n * (n - 1) / 2) / n


def steepness(W):
    """Slope of normalised David's score on rank: 0 = flat, 1 = fully linear."""
    ds = np.sort(davids_score(W))[::-1]
    return float(-np.polyfit(np.arange(1, len(ds) + 1), ds, 1)[0]) if len(ds) > 2 else np.nan


def landau_h(W):
    """Linearity: 0 = everyone dominates equally many, 1 = a strict order. Unknown and tied dyads count half each way."""
    n = len(W)
    if n < 3:
        return np.nan
    v = ((W > W.T) + 0.5 * (W == W.T)).sum(1) - 0.5
    return float(12 / (n**3 - n) * ((v - (n - 1) / 2) ** 2).sum())


def triangle_transitivity(W):
    """Share of fully decided triads that are transitive, rescaled so 0 = chance, 1 = all (Shizuka & McDonald 2012)."""
    A = (W > W.T).astype(float)
    k = A.sum(1)
    decided = np.trace(np.linalg.matrix_power(A + A.T, 3)) / 6
    cyclic = np.trace(np.linalg.matrix_power(A, 3)) / 3
    return float(4 * ((decided - cyclic) / decided - 0.75)) if decided else np.nan


def elo(sender, target, complied, n, k=32):
    r = np.full(n, 1000.0)
    for s, t, c in zip(sender, target, complied):
        e = 1 / (1 + 10 ** ((r[t] - r[s]) / 400))
        r[s] += k * (c - e)
        r[t] -= k * (c - e)
    return r


STATS = {"steepness": steepness, "landau_h": landau_h, "triangle_transitivity": triangle_transitivity}


def _spearman(a, b):
    ok = a.notna() & b.notna()
    return a[ok].rank().corr(b[ok].rank()) if ok.sum() > 2 else np.nan


def analyse(c, events, agents, n_perm=1000, seed=0):
    """One group of contests -> (ranking table, statistics with their null)."""
    counts = pd.concat([c.sender, c.target]).value_counts()
    names = sorted(counts[counts >= MIN_CONTESTS].index)
    c = c[c.sender.isin(names) & c.target.isin(names)]
    n = len(names)
    if n < 3 or len(c) < 10:
        return None, None
    code = {a: i for i, a in enumerate(names)}
    s, t, won = c.sender.map(code).to_numpy(), c.target.map(code).to_numpy(), c.complied.to_numpy()
    W = win_matrix(s, t, won, n)

    volume = events[(events.type == "message") & events.event_id.notna() & events.ts.between(c.ts.min(), c.ts.max())].actor.value_counts()
    rank = pd.DataFrame({"agent": names, "davids_score": davids_score(W), "elo": elo(s, t, won, n), "directives_sent": np.bincount(s, minlength=n), "obeyed": np.bincount(s, weights=won, minlength=n), "directives_received": np.bincount(t, minlength=n)})
    rank["messages"] = rank.agent.map(volume).fillna(0).astype(int)
    rank["release_date"] = rank.agent.map(agents.set_index("agent").release_date)
    rank = rank.sort_values("davids_score", ascending=False).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    block = pd.factorize(c.room + "|" + c.day)[0]
    sims = {k: np.empty(n_perm) for k in STATS}
    for p in range(n_perm):
        order = np.lexsort((rng.random(len(won)), block))
        shuffled = np.empty_like(won)
        shuffled[np.argsort(block, kind="stable")] = won[order]
        Wp = win_matrix(s, t, shuffled, n)
        for k, f in STATS.items():
            sims[k][p] = f(Wp)
    rows = []
    for k, f in STATS.items():
        o, d = f(W), sims[k][~np.isnan(sims[k])]
        if not len(d) or np.isnan(o):  # e.g. no fully decided triad
            rows.append((k, o, np.nan, np.nan, np.nan, np.nan))
            continue
        hi, lo = (d >= o).sum(), (d <= o).sum()
        rows.append((k, o, d.mean(), np.quantile(d, 0.025), np.quantile(d, 0.975), min(1.0, 2 * (min(hi, lo) + 1) / (len(d) + 1))))
    stats = pd.DataFrame(rows, columns=["statistic", "observed", "null_mean", "null_lo", "null_hi", "p"])
    extra = {
        "agents": n, "contests": len(c), "compliance": won.mean(),
        "rank_vs_release_date": _spearman(rank.davids_score, rank.release_date.astype("int64").where(rank.release_date.notna())),
        "rank_vs_messages": _spearman(rank.davids_score, rank.messages),
    }
    return rank, (stats, extra)


def episodes(goals):
    """Runs of goals with an imposed leader, each with an equally long window before and after."""
    g = goals.sort_values("start_time").reset_index(drop=True)
    g["end_time"] = g.end_time.fillna(g.start_time.max() + pd.Timedelta(days=30))
    run_id = (g.leader_imposed != g.leader_imposed.shift()).cumsum()
    out = []
    for _, r in g[g.leader_imposed == 1].groupby(run_id):
        a, b = r.start_time.min(), r.end_time.max()
        out.append((r.goal.iloc[-1], [("before", a - (b - a), a), ("during", a, b), ("after", b, b + (b - a))]))
    return out


def _print(label, rank, res):
    stats, extra = res
    print(f"\n{label}: {extra['agents']} agents, {extra['contests']:,} directives, {extra['compliance']:.0%} complied with")
    print(f"{'statistic':24s}{'observed':>10s}{'null mean':>11s}{'null 95% band':>19s}{'p':>9s}")
    for r in stats.itertuples():
        p_txt = "<0.002" if r.p < 0.002 else f"{r.p:.3f}"
        print(f"{r.statistic:24s}{r.observed:10.3f}{r.null_mean:11.3f}{f'[{r.null_lo:.3f}, {r.null_hi:.3f}]':>19s}{p_txt:>9s}")
    print(f"rank correlation of David's score with release date {extra['rank_vs_release_date']:.2f}, with message count {extra['rank_vs_messages']:.2f}")
    print(rank.drop(columns="release_date").head(8).round({"davids_score": 2, "elo": 0}).to_string(index=False))


def run(in_dir, group="era", meta=None, n_perm=1000, seed=0):
    in_dir = Path(in_dir)
    events = pd.read_parquet(in_dir / "events.parquet")
    agents = pd.read_parquet(in_dir / "agents.parquet")
    c = contests(pd.read_parquet(in_dir / "dyads.parquet"))
    print(f"{len(c):,} directives to a named agent, {c.complied.mean():.0%} complied with")

    ranks, tables = [], []
    for g, cg in c.groupby(group):
        rank, res = analyse(cg, events, agents, n_perm, seed)
        if rank is None:
            print(f"\n{group} {g}: too few contests ({len(cg)})")
            continue
        _print(f"{group} {g}", rank, res)
        ranks.append(rank.assign(group=f"{group} {g}"))
        tables.append(res[0].assign(group=f"{group} {g}", **res[1]))

    goals_path = Path(meta) / "goals.csv" if meta else None
    if goals_path and goals_path.exists():
        for name, windows in episodes(pd.read_csv(goals_path, parse_dates=["start_time", "end_time"])):
            print(f"\n=== imposed-leader episode: {name}")
            for label, a, b in windows:
                cg = c[(c.ts >= a) & (c.ts < b)]
                rank, res = analyse(cg, events, agents, n_perm, seed)
                if rank is None:
                    print(f"\n{label} ({a:%Y-%m-%d} to {b:%Y-%m-%d}): too few contests ({len(cg)})")
                    continue
                _print(f"{label} ({a:%Y-%m-%d} to {b:%Y-%m-%d})", rank, res)
                ranks.append(rank.assign(group=f"{name} | {label}"))
                tables.append(res[0].assign(group=f"{name} | {label}", **res[1]))

    if ranks:
        pd.concat(ranks).to_csv(in_dir / f"hierarchy_ranks_{group}.csv", index=False)
        t = pd.concat(tables)
        t.to_csv(in_dir / f"hierarchy_{group}.csv", index=False)
        figures.hierarchy(t, in_dir / f"fig_hierarchy_{group}.png")
        print(f"\nwrote {in_dir}/hierarchy_{group}.csv, hierarchy_ranks_{group}.csv, fig_hierarchy_{group}.png")
