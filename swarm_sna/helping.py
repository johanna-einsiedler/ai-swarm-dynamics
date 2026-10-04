"""Who answers whose requests: the dyad table, the reciprocity ladder and the bystander curve.

dyads      one row per (judged request, agent who could have responded)
ladder     nested logistic models of `responded`, scored against a lookup table
bystander  response probability against the number of agents present
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import figures
from .layers import mentions

HELP_TYPES = ("answers", "acts")
MEMORY_DAYS = 7
ACT_MIN = 60
BUSY_MIN = 30
STOP = set("the a an and or of to in on for with is are was were be been it this that i you we they my your our at as by from have has had will would can could should not no do does did if so but about just now me us them their its please".split())

# Each rung adds one covariate to the ones before it.
LADDER = [
    ("base rate", ["addressed"]),
    ("+ direct reciprocity", ["prior_help"]),
    ("+ indirect reciprocity", ["reputation"]),
    ("+ cost", ["busy"]),
    ("+ bystanders", ["log_active_n"]),
]


def helpful(responses):
    """Answers and reported actions that pass every check available: the quote is verbatim, and,
    where the link check has been run, a second model confirmed the message responds to that request."""
    ok = responses.type.isin(HELP_TYPES) & responses.quote_verified
    if "link_verified" in responses.columns:
        ok &= responses.link_verified.map(lambda v: v is True or v == True)  # noqa: E712
    return responses[ok]


def _words(t):
    return {w for w in re.findall(r"[a-z0-9][a-z0-9\-_.]{2,}", t.lower()) if w not in STOP}


def _count_between(times, lo, hi):
    return np.searchsorted(times, hi, "left") - np.searchsorted(times, lo, "left") if times is not None else 0


def checked_requests(responses):
    """Once a link check exists, a request counts only if every helpful link it has was checked.
    -> set of request ids, or None when no check has been run."""
    if "link_verified" not in responses.columns:
        return None
    h = responses[responses.type.isin(HELP_TYPES) & responses.quote_verified]
    done = h.groupby("request_id").link_verified.agg(lambda v: v.notna().all())
    return set(done[done].index) | (set(responses.request_id) - set(h.request_id))


def dyads(events, requests, responses):
    req = requests[requests.judged.fillna(False).astype(bool) & (requests.addressed != "human")].sort_values("ts")
    checked = checked_requests(responses)
    if checked is not None:
        before = len(req)
        req = req[req.request_id.isin(checked)]
        print(f"  link check: {len(req):,} of {before:,} judged requests fully checked ({len(req) / before:.0%}); the rest are left out")
    active = mentions.active_sets(events)
    good = helpful(responses)
    good = good[good.actor_class == "agent"]
    first = good.groupby(["request_id", "actor"]).latency_s.min().to_dict()

    # Past help, as sorted timestamps: per (helper, helped) and per helper.
    h = good.merge(req[["request_id", "actor"]].rename(columns={"actor": "helped"}), on="request_id")
    pair = {k: np.sort(g.to_numpy()) for k, g in h.groupby(["actor", "helped"]).ts}
    gave = {k: np.sort(g.to_numpy()) for k, g in h.groupby("actor").ts}

    ev = events[events.actor_class == "agent"].sort_values("ts")
    log = {a: (g.ts.to_numpy(), (g.type == "session_start").to_numpy(), g.text.to_numpy()) for a, g in ev.groupby("actor")}

    rows = []
    for r in req.itertuples():
        t = np.datetime64(r.ts)
        since = t - np.timedelta64(MEMORY_DAYS, "D")
        want = _words(r.quote)
        targets = set(r.targets)
        for c in (active.get((r.room, r.day), set()) | targets) - {r.actor}:
            ts, is_sess, text = log.get(c, (np.array([], "M8[ns]"), np.array([], bool), np.array([], object)))
            i = np.searchsorted(ts, t)
            busy = i > 0 and is_sess[i - 1] and t - ts[i - 1] < np.timedelta64(BUSY_MIN, "m")
            j = np.searchsorted(ts, t + np.timedelta64(ACT_MIN, "m"))
            acted = any(is_sess[k] and len(want & _words(text[k])) >= 2 for k in range(i, j))
            lat = first.get((r.request_id, c))
            rows.append(
                (r.request_id, r.ts, r.day, r.room, r.goal_id, r.era, r.actor, c, r.kind, r.directive, r.beneficiary, c in targets, r.addressed == "broadcast",
                 r.active_n, lat is not None, lat, acted, _count_between(pair.get((r.actor, c)), since, t), _count_between(gave.get(r.actor), since, t), bool(busy),
                 _count_between(pair.get((c, r.actor)), since, t))
            )
    return pd.DataFrame(
        rows,
        columns=["request_id", "ts", "day", "room", "goal_id", "era", "requester", "responder", "kind", "directive", "beneficiary", "addressed", "broadcast",
                 "active_n", "responded", "latency_s", "acted", "prior_help_from_requester", "requester_reputation", "responder_busy", "prior_help_to_requester"],
    )


# ---------- models

def logit_fit(X, y, ridge=1e-3, iters=40, w=None):
    """`w`: optional row weights."""
    w = np.ones(len(y)) if w is None else w
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        step = np.linalg.solve((X * (w * p * (1 - p))[:, None]).T @ X + ridge * np.eye(len(b)), X.T @ (w * (y - p)) - ridge * b)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def _loss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _features(d):
    return pd.DataFrame(
        {
            "addressed": d.addressed.astype(float), "prior_help": np.log1p(d.prior_help_from_requester), "reputation": np.log1p(d.requester_reputation),
            "busy": d.responder_busy.astype(float), "log_active_n": np.log(d.active_n.clip(lower=1)),
        }
    )


def _cells(d):
    """The same covariates, discretised: the lookup table that bounds what any model of them can explain."""
    return (
        d.addressed.astype(int).astype(str) + "|" + pd.cut(d.prior_help_from_requester, [-1, 0, 2, np.inf], labels=False).astype(str) + "|"
        + pd.cut(d.requester_reputation, [-1, 0, 4, np.inf], labels=False).astype(str) + "|" + d.responder_busy.astype(int).astype(str) + "|"
        + pd.cut(d.active_n, [-1, 4, 8, 15, np.inf], labels=False).astype(str)
    )


def ladder(d, folds=5, n_boot=500, seed=0):
    """Cross-validated log loss per rung, held-out by responder. Completeness = share of the
    lookup table's gain over the base rate of responding that the rung reaches; bootstrap over responders."""
    rng = np.random.default_rng(seed)
    y = d.responded.to_numpy(float)
    F, cells = _features(d), _cells(d).to_numpy()
    who, block = np.unique(d.responder.to_numpy(), return_inverse=True)
    fold = rng.permutation(len(who))[block] % folds
    names = ["no covariates"] + [n for n, _ in LADDER] + ["lookup table"]
    loss = np.zeros((len(names), len(d)))
    for k in range(folds):
        tr, te = fold != k, fold == k
        if not te.any() or not tr.any():
            continue
        loss[0, te] = _loss(np.full(te.sum(), y[tr].mean()), y[te])
        cols = []
        for i, (_, add) in enumerate(LADDER, 1):
            cols += add
            X = np.c_[np.ones(len(d)), F[cols].to_numpy()]
            loss[i, te] = _loss(1 / (1 + np.exp(-X[te] @ logit_fit(X[tr], y[tr]))), y[te])
        cell = pd.DataFrame({"c": cells[tr], "y": y[tr]}).groupby("c").y.agg(["sum", "count"])
        rate = ((cell["sum"] + y[tr].mean()) / (cell["count"] + 1)).to_dict()  # one pseudo-observation at the base rate
        loss[-1, te] = _loss(np.array([rate.get(c, y[tr].mean()) for c in cells[te]]), y[te])

    per = np.stack([np.bincount(block, weights=row, minlength=len(who)) for row in loss])  # summed loss per responder

    def completeness(tot):
        return (tot[0] - tot) / (tot[0] - tot[-1]) if tot[0] != tot[-1] else np.full(len(tot), np.nan)

    boots = np.array([completeness(per[:, rng.integers(0, len(who), len(who))].sum(1)) for _ in range(n_boot)])
    table = pd.DataFrame(
        {"model": names, "log_loss": loss.mean(1), "completeness": completeness(per.sum(1)), "ci_lo": np.nanquantile(boots, 0.025, 0), "ci_hi": np.nanquantile(boots, 0.975, 0)}
    )
    X = np.c_[np.ones(len(d)), F.to_numpy()]
    coef = pd.Series(logit_fit(X, y), index=["intercept"] + list(F.columns), name="coefficient")
    return table, coef


SIZE_BINS, SIZE_LABELS = [0, 4, 8, 15, np.inf], ["2-4", "5-8", "9-15", "16+"]


def bystander(d, n_boot=200, seed=0, min_requests=30):
    """Broadcast requests only, and always within era: the village grew over time, so a pooled
    curve would compare eras, not group sizes.
    -> (rates by era x group size, slope of log-odds on log(active_n) within goal, 95% CI over responders)"""
    b = d[d.broadcast]
    b = b.assign(size=pd.cut(b.active_n, SIZE_BINS, labels=SIZE_LABELS))
    per_req = b.groupby("request_id").agg(any_response=("responded", "any"), era=("era", "first"), size=("size", "first"))
    curve = pd.DataFrame(
        {
            "requests": per_req.groupby(["era", "size"], observed=True).size(),
            "p_each_agent_responds": b.groupby(["era", "size"], observed=True).responded.mean(),
            "p_any_agent_responds": per_req.groupby(["era", "size"], observed=True).any_response.mean(),
        }
    ).reset_index()
    curve = curve[curve.requests >= min_requests]

    def slope(x):
        goal = pd.get_dummies(x.goal_id, drop_first=True, dtype=float).to_numpy()
        X = np.c_[np.ones(len(x)), np.log(x.active_n.clip(lower=1)), goal]
        return logit_fit(X, x.responded.to_numpy(float))[1]

    rng = np.random.default_rng(seed)
    groups = [g for _, g in b.groupby("responder")]
    boots = [slope(pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])) for _ in range(n_boot)] if len(groups) > 1 else [np.nan]
    return curve, (slope(b) if len(b) else np.nan, np.nanquantile(boots, 0.025), np.nanquantile(boots, 0.975))


def coefficients(d):
    """The full model refitted every way a pooled fit could mislead: per era, within goal,
    with every requester weighted equally (a few agents ask most of the questions), and
    within requester and responder (fixed effects)."""
    y = d.responded.to_numpy(float)
    F = _features(d)
    X = np.c_[np.ones(len(d)), F.to_numpy()]
    k = F.shape[1]
    cols = {"pooled": logit_fit(X, y)[1 : k + 1]}
    for e, g in d.groupby("era"):
        Fg = _features(g)
        cols[f"era {e}"] = logit_fit(np.c_[np.ones(len(g)), Fg.to_numpy()], g.responded.to_numpy(float))[1 : k + 1]
    dummies = lambda c: pd.get_dummies(d[c], drop_first=True, dtype=float).to_numpy()  # noqa: E731
    cols["within goal"] = logit_fit(np.c_[X, dummies("goal_id")], y)[1 : k + 1]
    w = (1 / d.groupby("requester").request_id.transform("size")).to_numpy()
    cols["requesters weighted equally"] = logit_fit(X, y, w=w / w.mean())[1 : k + 1]
    cols["within requester and responder"] = logit_fit(np.c_[X, dummies("requester"), dummies("responder")], y)[1 : k + 1]
    if "prior_help_to_requester" in d.columns:  # is it reciprocity, or just the habit of answering this asker?
        cols["controlling for own past answers to the asker"] = logit_fit(np.c_[X, np.log1p(d.prior_help_to_requester)], y)[1 : k + 1]
    return pd.DataFrame(cols, index=F.columns)


def load_all(in_dir, name):
    """Both screens: requests.parquet and requests_wide.parquet."""
    files = sorted(Path(in_dir).glob(f"{name}*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {name}*.parquet in {in_dir}")
    print(f"  {name}: " + ", ".join(f.name for f in files))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def run(in_dir, n_boot=500, seed=0, reuse_dyads=False):
    in_dir = Path(in_dir)
    cached = in_dir / "dyads.parquet"
    sources = [f for name in ("requests", "responses") for f in in_dir.glob(f"{name}*.parquet")]
    if cached.exists() and (reuse_dyads or (sources and cached.stat().st_mtime > max(f.stat().st_mtime for f in sources))):
        d = pd.read_parquet(cached)
        print("  dyads: reusing dyads.parquet")
    else:
        events = pd.read_parquet(in_dir / "events.parquet")
        d = dyads(events, load_all(in_dir, "requests"), load_all(in_dir, "responses"))
        d.to_parquet(cached, index=False)
    print(f"{d.request_id.nunique():,} judged requests x agents present = {len(d):,} dyad rows; an agent responds in {d.responded.mean():.1%} of them")
    print(f"  addressed by name: responds {d[d.addressed].responded.mean():.1%} (n={d.addressed.sum():,});  not addressed: {d[~d.addressed].responded.mean():.1%}")

    table, coef = ladder(d, n_boot=n_boot, seed=seed)
    table.to_csv(in_dir / "helping_ladder.csv", index=False)
    figures.ladder(table, in_dir / "fig_helping_ladder.png")
    print(f"\nreciprocity ladder (5-fold, held out by responder; completeness = share of the lookup table's gain, 95% bootstrap over responders)")
    print(f"{'model':26s}{'log loss':>10s}{'completeness':>14s}{'95% CI':>18s}")
    for r in table.itertuples():
        print(f"{r.model:26s}{r.log_loss:10.4f}{r.completeness:14.2f}{f'[{r.ci_lo:.2f}, {r.ci_hi:.2f}]':>18s}")

    by_era = coefficients(d)
    by_era.to_csv(in_dir / "helping_coefficients.csv")
    top = d.drop_duplicates("request_id").requester.value_counts(normalize=True)
    print(f"\nfull model, log-odds coefficients, refitted each way a pooled fit could mislead ({top.index[0]} alone asks {top.iloc[0]:.0%} of all requests)")
    print(by_era.round(3).T.to_string())

    curve, (s, lo, hi) = bystander(d, n_boot=min(n_boot, 200), seed=seed)
    curve.to_csv(in_dir / "helping_bystander.csv", index=False)
    figures.bystander(curve, in_dir / "fig_bystander.png")
    print("\nbystander curve (broadcast requests, within era, by agents present in the room that day)")
    print(curve.round(3).to_string(index=False))
    print(f"slope of log-odds that an agent responds on log(agents present), within goal: {s:.2f} [{lo:.2f}, {hi:.2f}]")
    summary = {
        "requests": int(d.request_id.nunique()), "dyad_rows": int(len(d)), "respond_rate": float(d.responded.mean()),
        "addressed_rate": float(d[d.addressed].responded.mean()), "not_addressed_rate": float(d[~d.addressed].responded.mean()),
        "bystander_slope": [float(s), float(lo), float(hi)], "top_asker": str(top.index[0]), "top_asker_share": float(top.iloc[0]),
    }
    (in_dir / "helping_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {in_dir}/dyads.parquet, helping_ladder.csv, helping_coefficients.csv, helping_bystander.csv, helping_summary.json, fig_helping_ladder.png, fig_bystander.png")
