"""Does information travel along ties? Order-of-acquisition diffusion analysis.

An item is a distinctive token (a URL, a file or tool name, a backticked term)
that several agents come to use. For each item, the agents are ordered by first
use. At each adoption after the first, the adopter is compared with the agents
that had not yet adopted: its rank among them by tie weight to those who already
had (0 = least tied, 1 = most tied). The statistic is the mean rank over
adoptions, 0.5 under random order. This is the order-of-acquisition idea of
Franz & Nunn (2009) and Hoppitt & Laland (2013, OADA) as a rank test. The null
permutes the order of adoption among the same agents, so who adopts is fixed
and only the order is tested: if ties carry information, the next adopter should
be among the candidates most tied to the earlier ones.

(The total tie weight among adopters, which an earlier version summed over
adopters, is the same for every order; it cannot be tested.)

The second half builds the adoption network itself: who picks things up first
and who follows, against two nulls (random order, and order drawn in proportion
to posting volume while the item spread).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import figures, redact

MIN_ADOPTERS = 4
MAX_ADOPTERS = 20
MIN_SPREAD_DAYS, MAX_SPREAD_DAYS = 1, 90
TIE_WINDOW_DAYS = 30  # ties measured on mentions in the month before the item first appears
MAX_ITEMS = 1500
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
    """-> item, actor, first ts and the message it first appeared in, for agent messages."""
    m = events[(events.type == "message") & (events.actor_class == "agent")].sort_values("ts")
    rows = []
    seen = set()
    for r in m.itertuples():
        for t in tokens(r.text):
            key = (t, r.actor)
            if key not in seen:
                seen.add(key)
                rows.append((t, r.actor, r.ts, r.event_id))
    return pd.DataFrame(rows, columns=["item", "actor", "ts", "event_id"])


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


def mean_rank(orders, S):
    """orders: (P, n) adopter indices, one row per adoption order. -> (P,) mean rank of each adopter
    among the agents still to adopt, by tie weight to those who already had; 0.5 under random order."""
    n = orders.shape[1]
    M = S[orders[:, :, None], orders[:, None, :]]  # ties between adopters, rows and columns in adoption order
    C = np.cumsum(M, axis=2)  # C[:, i, k - 1] = ties of adopter i to the first k adopters
    ranks = []
    for k in range(1, n - 1):  # the last adoption has no rival left
        c = C[:, k:, k - 1]  # candidates still to adopt; column 0 is the one that did
        me, others = c[:, :1], c[:, 1:]
        ranks.append(((others < me).sum(1) + 0.5 * (others == me).sum(1)) / others.shape[1])
    return np.mean(ranks, axis=0)


def oada(order, W, rng, n_perm=N_PERM):
    """order: adopter indices in adoption order. -> (observed, null mean, null sd, p one-sided, z)."""
    S = W + W.T  # a tie in either direction counts
    np.fill_diagonal(S, 0)
    orders = np.vstack([order] + [rng.permutation(order) for _ in range(n_perm)])
    vals = mean_rank(orders, S)
    obs, null = vals[0], vals[1:]
    sd = null.std()
    return obs, null.mean(), sd, ((null >= obs).sum() + 1) / (n_perm + 1), (obs - null.mean()) / sd if sd > 0 else np.nan


# ---------- the adoption network: who picks things up from whom

def adoption_orders(fu, items, agents, events):
    """Per item: adopters in order, restricted to agents already present when it first appeared
    (a later arrival cannot have been first), with each adopter's message count over the item's span."""
    joined = dict(zip(agents.agent, agents.joined))
    m = events[(events.type == "message") & (events.actor_class == "agent")]
    volume = m.groupby("actor").ts.apply(lambda t: np.sort(t.to_numpy())).to_dict()
    out = []
    for item, meta in items.iterrows():
        o = fu[fu.item == item].sort_values("ts")
        o = o[[pd.isna(joined.get(a)) or joined[a] <= meta["first"] for a in o.actor]]
        if len(o) < 3:
            continue
        lo, hi = np.datetime64(meta["first"]), np.datetime64(meta["last"])
        w = np.array([np.searchsorted(volume[a], hi, "right") - np.searchsorted(volume[a], lo, "left") + 1 for a in o.actor], float)
        out.append((item, meta["first"], o.actor.to_numpy(), w))
    return out


def _credit(order_idx, n):
    """Each adopter after the first hands one unit of credit, split evenly, to those who had it before."""
    W = np.zeros((n, n))
    for k in range(1, len(order_idx)):
        W[order_idx[:k], order_idx[k]] += 1.0 / k
    return W


def adoption_network(orders, names, rng, n_perm=300, by_volume=True):
    """-> (W observed [source, adopter], null draws of each agent's lead score, null draws of steepness).
    Null: adoption order redrawn with each agent's chance of going next proportional to how much it
    posted while the item spread. So leading has to beat talkativeness, not just a coin flip."""
    from .hierarchy import steepness

    code = {a: i for i, a in enumerate(names)}
    n = len(names)
    idx = [np.array([code[a] for a in actors]) for _, _, actors, _ in orders]
    W = sum(_credit(i, n) for i in idx)

    def lead(M):
        out, inn = M.sum(1), M.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(out + inn > 0, (out - inn) / (out + inn), np.nan)

    null_lead, null_steep = np.empty((n_perm, n)), np.empty(n_perm)
    active = (W.sum(0) + W.sum(1)) > 0
    for p in range(n_perm):
        Wp = np.zeros((n, n))
        for i, (_, _, _, w) in zip(idx, orders):
            Wp += _credit(i[np.argsort(rng.exponential(size=len(i)) / (w if by_volume else 1.0))], n)  # Plackett-Luce order with weights w
        null_lead[p] = lead(Wp)
        null_steep[p] = steepness(Wp[np.ix_(active, active)])
    return W, lead(W), null_lead, steepness(W[np.ix_(active, active)]), null_steep


def leaders(in_dir, fu, items, events, agents, rng, n_perm=300):
    orders = adoption_orders(fu, items, agents, events)
    if not orders:
        return None
    names = sorted({a for _, _, actors, _ in orders for a in actors})
    W, lead, null_lead, steep, null_steep = adoption_network(orders, names, rng, n_perm)
    mu, sd = np.nanmean(null_lead, 0), np.nanstd(null_lead, 0)
    t = pd.DataFrame({"agent": names, "led": W.sum(1), "followed": W.sum(0), "lead_score": lead, "null_mean": mu, "z": (lead - mu) / np.where(sd > 0, sd, np.nan)})
    t["p"] = [min(1.0, 2 * (min((null_lead[:, i] >= lead[i]).sum(), (null_lead[:, i] <= lead[i]).sum()) + 1) / (n_perm + 1)) for i in range(len(names))]
    # The same against plain random orders. The volume null assumes an agent that posts ten times
    # as much picks things up ten times sooner; plain random order assumes volume is irrelevant.
    # The truth lies between, so both are reported.
    _, _, null_u, steep_u, null_steep_u = adoption_network(orders, names, rng, n_perm, by_volume=False)
    sd_u = np.nanstd(null_u, 0)
    t["z_random_order"] = (lead - np.nanmean(null_u, 0)) / np.where(sd_u > 0, sd_u, np.nan)
    a = agents.set_index("agent")
    t["release_date"] = t.agent.map(a.release_date)
    t["family"] = t.agent.map(a.family)
    t["messages"] = t.agent.map(events[(events.type == "message")].actor.value_counts()).fillna(0).astype(int)
    t = t[(t.led + t.followed) >= 10].sort_values("z", ascending=False).reset_index(drop=True)
    edges = pd.DataFrame([(names[i], names[j], W[i, j]) for i in range(len(names)) for j in range(len(names)) if W[i, j] > 0], columns=["source", "adopter", "weight"])
    def test(obs, null):
        return obs, null.mean(), np.quantile(null, 0.025), np.quantile(null, 0.975), min(1.0, 2 * (min((null >= obs).sum(), (null <= obs).sum()) + 1) / (n_perm + 1))

    return t, edges, test(steep, null_steep), test(steep_u, null_steep_u), len(orders)


def _spearman(x, y):
    ok = x.notna() & y.notna()
    return x[ok].rank().corr(y[ok].rank()) if ok.sum() > 3 else np.nan


ITEM_COLUMNS = ["item", "adopters", "first_adopter", "first", "spread_days", "observed", "null_mean", "null_sd", "z", "p"]
FILE_EXT = re.compile(r"\.(?:py|js|ts|md|json|csv|yaml|yml|sh|html|txt|ipynb)$")
HOST = re.compile(r"^[a-z0-9.\-]+\.[a-z]{2,}(?:/|$)")


def item_kind(item):
    """link (a host, maybe with a path), file (a name with an extension) or term (a backticked token).
    A masked name carries a '#hash' suffix (redact.label); it is ignored here."""
    item = re.sub(r"#[0-9a-f]{6}$", "", str(item))
    return "file" if FILE_EXT.search(item) else "link" if HOST.match(item) else "term"


def _nothing(in_dir, why):
    """Nothing spread: write the item table and the summary empty, so the card says 'not testable' rather than 'not run'."""
    pd.DataFrame(columns=ITEM_COLUMNS).to_csv(in_dir / "diffusion_items.csv", index=False)
    pd.DataFrame([{"items": 0}]).to_csv(in_dir / "diffusion_summary.csv", index=False)
    print(f"{why}; wrote {in_dir}/diffusion_items.csv, diffusion_summary.csv (empty)")


def run(in_dir, n_perm=N_PERM, seed=0):
    in_dir = Path(in_dir)
    events = pd.read_parquet(in_dir / "events.parquet")
    mentions = pd.read_parquet(in_dir / "mentions.parquet")
    agents = pd.read_parquet(in_dir / "agents.parquet")
    names = sorted(events[events.actor_class == "agent"].actor.unique())
    code = {a: i for i, a in enumerate(names)}
    fu = first_uses(events)
    if fu.empty:
        return _nothing(in_dir, "no traceable tokens (links, file names, backticked terms) in this transcript")
    items = select_items(fu)
    print(f"{len(fu):,} first uses of {fu.item.nunique():,} tokens; {len(items)} items with {MIN_ADOPTERS}-{MAX_ADOPTERS} adopters spread over {MIN_SPREAD_DAYS}-{MAX_SPREAD_DAYS} days")
    if items.empty:
        return _nothing(in_dir, "no token spread to enough agents to test")
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
    t = pd.DataFrame(rows, columns=ITEM_COLUMNS)
    # Item names are strings lifted from messages, so they go out masked like any other text.
    t.assign(item=redact.label(t.item)).to_csv(in_dir / "diffusion_items.csv", index=False)
    # The adoption events themselves: with these and the mention edges the analysis can be redone without the text.
    a = fu[fu.item.isin(items.index)].sort_values(["item", "ts"])
    a = a.assign(order=a.groupby("item").cumcount() + 1, item=redact.label(a.item)).rename(columns={"actor": "agent", "ts": "first_use", "event_id": "first_use_event_id"})
    a[["item", "order", "agent", "first_use", "first_use_event_id"]].to_csv(in_dir / "diffusion_adoptions.csv", index=False)
    summary = {"items": len(t), "mean_rank": np.nan, "share_p_below_05": np.nan, "mean_z": np.nan, "stouffer_z": np.nan}
    if t.empty:
        print("\n1. Does adoption follow existing ties? No item had adopters with prior ties, so there is nothing to test.")
    else:
        z = t.z.dropna()
        share = (t.p < 0.05).mean()
        pooled = z.mean() * np.sqrt(len(z))  # Stouffer
        summary.update(mean_rank=t.observed.mean(), share_p_below_05=share, mean_z=z.mean(), stouffer_z=pooled)
        print(f"\n1. Does adoption follow existing ties? {len(t)} items tested against {n_perm} random adoption orders each")
        print(f"   mean rank of the next adopter among the agents still to adopt, by ties to earlier adopters: {t.observed.mean():.3f} (0.5 by chance)")
        print(f"   items where adoption order follows ties (p < 0.05): {share:.1%} (5% expected by chance)")
        print(f"   mean z {z.mean():+.2f}; Stouffer combined z {pooled:+.1f}")
        print("   strongest cascades:")
        print(t.nsmallest(8, "p")[["item", "adopters", "first_adopter", "spread_days", "observed", "z", "p"]].round(3).to_string(index=False))
        figures.diffusion(t, in_dir / "fig_diffusion.png")

    # 2. The adoption network itself: who leads, who follows. It needs no ties, so it is built even when nothing above was testable.
    res = leaders(in_dir, fu, items, events, agents, rng, n_perm=min(n_perm, 500))
    if res is None:
        pd.DataFrame([summary]).to_csv(in_dir / "diffusion_summary.csv", index=False)
        print(f"\n2. The adoption network: no item with three or more adopters present from the start; nothing to build\n\nwrote {in_dir}/diffusion_items.csv, diffusion_summary.csv")
        return
    lead, edges, (steep, s_mu, s_lo, s_hi, s_p), (_, u_mu, u_lo, u_hi, u_p), n_items = res
    lead.to_csv(in_dir / "diffusion_leaders.csv", index=False)
    edges.to_csv(in_dir / "diffusion_edges.csv", index=False)
    rd = lead.release_date.map(lambda d: np.nan if pd.isna(d) else pd.Timestamp(d).value)
    msgs = lead.messages.astype(float)
    dom = pd.Series(np.nan, index=lead.index)
    ranks = sorted(in_dir.glob("hierarchy_ranks_*.csv"))
    if ranks:
        r = pd.read_csv(ranks[0])
        dom = lead.agent.map(r[r.group.str.startswith("era")].groupby("agent").davids_score.mean())
    corr = pd.DataFrame(
        {"vs release date": [_spearman(lead[c], rd) for c in ("lead_score", "z_random_order", "z")], "vs message count": [_spearman(lead[c], msgs) for c in ("lead_score", "z_random_order", "z")],
         "vs dominance rank": [_spearman(lead[c], dom) for c in ("lead_score", "z_random_order", "z")]},
        index=["raw lead score", "z against random order", "z against volume-weighted order"],
    )
    corr.to_csv(in_dir / "diffusion_correlates.csv")
    print(f"\n2. The adoption network: {n_items} items, {len(lead)} agents with 10+ adoption events")
    print(f"   leader-follower steepness {steep:.3f}; against random order {u_mu:.3f} [{u_lo:.3f}, {u_hi:.3f}] p = {u_p:.3f}; against volume-weighted order {s_mu:.3f} [{s_lo:.3f}, {s_hi:.3f}] p = {s_p:.3f}")
    print("   rank correlation of leading with:")
    print(corr.round(2).to_string())
    cols = ["agent", "led", "followed", "lead_score", "z_random_order", "z"]
    print("   most leading (by raw lead score):")
    print(lead.sort_values("lead_score", ascending=False).head(6)[cols].round(2).to_string(index=False))
    print("   most following:")
    print(lead.sort_values("lead_score").head(5)[cols].round(2).to_string(index=False))
    figures.leaders(lead, in_dir / "fig_diffusion_leaders.png")
    summary.update(network_items=n_items, steepness=steep, steepness_random_null=u_mu, steepness_random_p=u_p, steepness_volume_null=s_mu, steepness_volume_p=s_p)
    pd.DataFrame([summary]).to_csv(in_dir / "diffusion_summary.csv", index=False)
    print(f"\nwrote {in_dir}/diffusion_items.csv, diffusion_adoptions.csv, diffusion_leaders.csv, diffusion_edges.csv, diffusion_correlates.csv, diffusion_summary.csv, fig_diffusion.png, fig_diffusion_leaders.png")
