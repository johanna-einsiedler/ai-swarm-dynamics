"""Does information travel along ties? Order-of-acquisition diffusion analysis.

An item is a distinctive token (a URL, a file or tool name, a backticked term)
that several agents come to use. For each item, the agents are ordered by first
use. The statistic is how much tie weight each adopter had to the agents who
adopted before it, summed over adopters (Franz & Nunn 2009; Hoppitt & Laland
2013, OADA). The null permutes the order of adoption among the same agents, so
who adopts is fixed and only the order is tested: if ties carry information,
later adopters should be the ones connected to earlier ones.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import figures

MIN_ADOPTERS = 4
MAX_ADOPTERS = 20
MIN_SPREAD_DAYS, MAX_SPREAD_DAYS = 1, 90
TIE_WINDOW_DAYS = 30  # ties measured on mentions in the month before the item first appears
MAX_ITEMS = 400
N_PERM = 1000
STOP = set("the and for with this that from http https com www org io md py js html json csv txt readme main true false none null".split())

CODE = re.compile(r"`([^`\n]{3,48})`")
URL = re.compile(r"https?://([a-z0-9.\-]+\.[a-z]{2,})(/[A-Za-z0-9_\-.]+)?", re.IGNORECASE)
IDENT = re.compile(r"(?<![\w/])([A-Za-z][A-Za-z0-9]*(?:[_\-.][A-Za-z0-9]+){1,4}\.(?:py|js|ts|md|json|csv|yaml|yml|sh|html|txt|ipynb))(?![\w/])")


def tokens(text):
    out = set()
    for m in CODE.finditer(text):
        t = m.group(1).strip()
        if " " not in t and len(t) >= 4 and t.lower() not in STOP:
            out.add(t)
    for m in URL.finditer(text):
        host = m.group(1).lower()
        if host.count(".") >= 1 and not host.endswith(("google.com", "github.com", "gitlab.com", "docs.google.com")):
            out.add(host + (m.group(2) or "").lower()[:40])
        elif m.group(2):
            out.add(host + m.group(2).lower()[:40])
    for m in IDENT.finditer(text):
        out.add(m.group(1))
    return out


def first_uses(events):
    """-> item, actor, first ts, for agent messages."""
    m = events[(events.type == "message") & (events.actor_class == "agent")].sort_values("ts")
    rows = []
    seen = set()
    for r in m.itertuples():
        for t in tokens(r.text):
            key = (t, r.actor)
            if key not in seen:
                seen.add(key)
                rows.append((t, r.actor, r.ts))
    return pd.DataFrame(rows, columns=["item", "actor", "ts"])


def select_items(fu):
    g = fu.groupby("item").agg(adopters=("actor", "nunique"), first=("ts", "min"), last=("ts", "max"))
    g["spread_days"] = (g["last"] - g["first"]).dt.total_seconds() / 86400
    keep = g[(g.adopters >= MIN_ADOPTERS) & (g.adopters <= MAX_ADOPTERS) & g.spread_days.between(MIN_SPREAD_DAYS, MAX_SPREAD_DAYS)]
    return keep.sort_values("adopters", ascending=False).head(MAX_ITEMS)


def tie_matrix(mentions, names, start, end):
    code = {a: i for i, a in enumerate(names)}
    m = mentions[(mentions.actor_class == "agent") & (mentions.ts >= start) & (mentions.ts < end) & mentions.actor.isin(code) & mentions.target.isin(code)]
    W = np.zeros((len(names), len(names)))
    np.add.at(W, (m.actor.map(code).to_numpy(), m.target.map(code).to_numpy()), m.weight.to_numpy())
    return W


def oada(order, W, rng, n_perm=N_PERM):
    """order: adopter indices in adoption order. -> (observed, null mean, null sd, p one-sided)."""
    n = len(order)

    def stat(o):
        return sum(W[o[k], o[:k]].sum() + W[o[:k], o[k]].sum() for k in range(1, n))  # ties in either direction to earlier adopters

    obs = stat(order)
    null = np.array([stat(rng.permutation(order)) for _ in range(n_perm)])
    sd = null.std()
    return obs, null.mean(), sd, ((null >= obs).sum() + 1) / (n_perm + 1), (obs - null.mean()) / sd if sd > 0 else np.nan


def run(in_dir, n_perm=N_PERM, seed=0):
    in_dir = Path(in_dir)
    events = pd.read_parquet(in_dir / "events.parquet")
    mentions = pd.read_parquet(in_dir / "mentions.parquet")
    names = sorted(events[events.actor_class == "agent"].actor.unique())
    code = {a: i for i, a in enumerate(names)}
    fu = first_uses(events)
    items = select_items(fu)
    print(f"{len(fu):,} first uses of {fu.item.nunique():,} tokens; {len(items)} items with {MIN_ADOPTERS}-{MAX_ADOPTERS} adopters spread over {MIN_SPREAD_DAYS}-{MAX_SPREAD_DAYS} days")
    rng = np.random.default_rng(seed)
    rows = []
    for item, meta in items.iterrows():
        order = fu[fu.item == item].sort_values("ts")
        W = tie_matrix(mentions, names, meta["first"] - pd.Timedelta(days=TIE_WINDOW_DAYS), meta["first"])
        idx = order.actor.map(code).to_numpy()
        if W[np.ix_(idx, idx)].sum() == 0:
            continue
        obs, mu, sd, p, z = oada(idx, W, rng, n_perm)
        rows.append({"item": item, "adopters": len(idx), "first_adopter": order.actor.iloc[0], "first": meta["first"], "spread_days": meta.spread_days, "observed": obs, "null_mean": mu, "null_sd": sd, "z": z, "p": p})
    t = pd.DataFrame(rows)
    t.to_csv(in_dir / "diffusion_items.csv", index=False)
    z = t.z.dropna()
    share = (t.p < 0.05).mean()
    pooled = z.mean() * np.sqrt(len(z))  # Stouffer
    print(f"\n{len(t)} items tested against {n_perm} random adoption orders each")
    print(f"items where adoption order follows ties (p < 0.05): {share:.1%} (5% expected by chance)")
    print(f"mean z {z.mean():+.2f}; Stouffer combined z {pooled:+.1f}")
    print("\nstrongest cascades:")
    print(t.nsmallest(8, "p")[["item", "adopters", "first_adopter", "spread_days", "z", "p"]].round(3).to_string(index=False))
    figures.diffusion(t, in_dir / "fig_diffusion.png")
    (in_dir / "diffusion_summary.csv").write_text(f"items,share_p_below_05,mean_z,stouffer_z\n{len(t)},{share:.4f},{z.mean():.4f},{pooled:.4f}\n")
    print(f"\nwrote {in_dir}/diffusion_items.csv, diffusion_summary.csv, fig_diffusion.png")
