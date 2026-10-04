"""Directed goal alignment between agents with individual goals (era 3).

alignment(A -> B) asks: if A pursues its own goal successfully, what does that
do to B's chances of achieving B's goal?  +1 advances it, 0 unrelated, -1 defeats it.
It is directed (a performance coach serves everyone; nobody's goal serves the coach)
and it is about outcomes, not topics: two agents with the same goal can be rivals.
This is outcome interdependence in the sense of Deutsch (1949).

Two models rate every ordered pair of distinct goals independently; the score used
is the first rater's, and the agreement between the two is reported.

usage: goal_alignment.py [rater models, comma separated]
"""
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from swarm_sna import llm

RATERS = (sys.argv[1] if len(sys.argv) > 1 else "opus,sonnet").split(",")
DATA, OUT, CACHE = Path("data/village"), Path("out/village"), Path("out/llm_cache")

SYSTEM = """You assess how the goals of AI agents relate. The agents live in the "AI Village": each has its own computer with internet access, they share a group chat, and each was assigned one goal to maximise. They cannot see each other's goals.
For each numbered ordered pair (A, B) answer: if agent A pursues its own goal successfully, what does that do to agent B's chances of achieving B's goal? Judge outcomes, not topics: two agents with similar goals may be rivals for the same audience, or may not be.
Return a JSON array with one object per pair, in order:
{"i": <pair number>,
 "reason": one sentence naming the mechanism,
 "mechanism": "serves" (A's goal is itself to benefit B, or agents like B) | "complements" (what A produces while pursuing its goal is something B's goal can use) | "independent" (no meaningful effect either way) | "competes" (both draw on the same scarce audience, resource or ranking, so A's gain is partly B's loss) | "obstructs" (A's success directly sets back B's goal),
 "shared_or_contested": the audience, resource or output involved, or "",
 "basis_a": the words of A's goal the judgement rests on, copied character for character,
 "basis_b": the words of B's goal the judgement rests on, copied character for character,
 "score": one of -1, -0.5, 0, 0.5, 1  (1 = A's success directly advances B's goal; 0.5 = helps somewhat; 0 = unrelated; -0.5 = hinders somewhat; -1 = A's success directly defeats B's goal)}
JSON only."""


def load(name):
    return pd.DataFrame([json.loads(l) for l in gzip.open(DATA / f"{name}.jsonl.gz", "rt")])


def goals():
    ag, g = load("agents"), load("agent_goals")
    g["agent"] = g.agent_id.map(dict(zip(ag.id, ag.name)))
    g = g.sort_values("start_time").drop_duplicates("agent", keep="last")
    note = g.description.fillna("").map(lambda d: re.sub(r"\S*https?://\S+|\S+ — *", "", d).split("\n")[0].strip()[:200])
    g["text"] = (g.name + np.where(note.str.len() > 0, " (Note given to the agent: " + note + ")", "")).str.strip()
    g["start_time"] = pd.to_datetime(g.start_time, format="mixed")
    return g[["agent", "short_name", "name", "text", "start_time"]].reset_index(drop=True)


def rate(g, model):
    texts = g.drop_duplicates("name").set_index("name").text
    held_by = g.groupby("name").agent.nunique()
    prompts, keys = [], []
    for a in texts.index:
        others = [b for b in texts.index if b != a or held_by[a] > 1]
        lines = [f"[{i}] B's goal: {texts[b]}" + ("  (B is a different agent with the same goal as A)" if b == a else "") for i, b in enumerate(others)]
        prompts.append(f"A's goal: {texts[a]}\n\n" + "\n".join(lines))
        keys.append((a, others))
    res = llm.map_calls(SYSTEM, prompts, CACHE, workers=6, model=model)
    rows = []
    for (a, others), out in zip(keys, res):
        for x in out or []:
            if isinstance(x, dict) and isinstance(x.get("i"), int) and 0 <= x["i"] < len(others) and x.get("score") in (-1, -0.5, 0, 0.5, 1):
                b = others[x["i"]]
                rows.append({"goal_a": a, "goal_b": b, "score": float(x["score"]), "mechanism": x.get("mechanism"), "reason": x.get("reason"), "shared_or_contested": x.get("shared_or_contested"),
                             "basis_verified": str(x.get("basis_a") or "").strip(" .\"'") in texts[a] and str(x.get("basis_b") or "").strip(" .\"'") in texts[b]})
    return pd.DataFrame(rows)


def main():
    g = goals()
    rated = {m: rate(g, m) for m in RATERS}
    first = rated[RATERS[0]]
    print(f"{len(g)} agents, {g.name.nunique()} distinct goals, {len(first)} ordered goal pairs rated by {RATERS[0]}; basis quotes verbatim in {first.basis_verified.mean():.0%}")
    print("scores:", first.score.value_counts().sort_index().to_dict())
    print("mechanisms:", first.mechanism.value_counts().to_dict())
    if len(RATERS) > 1:
        both = first.merge(rated[RATERS[1]], on=["goal_a", "goal_b"], suffixes=("", "_2"))
        first = first.merge(both[["goal_a", "goal_b", "score_2", "mechanism_2"]], on=["goal_a", "goal_b"], how="left")
        print(f"agreement with {RATERS[1]} on {len(both)} pairs: same score {np.mean(both.score == both.score_2):.0%}, same sign {np.mean(np.sign(both.score) == np.sign(both.score_2)):.0%}, correlation {both.score.corr(both.score_2):.2f}, same mechanism {np.mean(both.mechanism == both.mechanism_2):.0%}")

    # Goal pairs -> ordered agent pairs.
    pairs = g.merge(g, how="cross", suffixes=("_a", "_b"))
    pairs = pairs[pairs.agent_a != pairs.agent_b].merge(first, left_on=["name_a", "name_b"], right_on=["goal_a", "goal_b"])
    pairs["since"] = pairs[["start_time_a", "start_time_b"]].max(axis=1)
    cols = ["agent_a", "agent_b", "short_name_a", "short_name_b", "score", "mechanism", "reason", "shared_or_contested", "basis_verified", "since"] + [c for c in ("score_2", "mechanism_2") if c in pairs]
    pairs[cols].to_csv(OUT / "goal_alignment.csv", index=False)
    print(f"\n{len(pairs)} ordered agent pairs -> {OUT}/goal_alignment.csv")

    out_mean = pairs.groupby(["agent_a", "short_name_a"]).score.mean().sort_values()
    in_mean = pairs.groupby(["agent_b", "short_name_b"]).score.mean().sort_values()
    print("\nwhose goal serves the others most (mean alignment given):")
    print(out_mean.tail(6).round(2).to_string())
    print("\nwhose goal serves the others least:")
    print(out_mean.head(5).round(2).to_string())
    print("\nmost rivalrous pairs:")
    print(pairs.nsmallest(8, "score")[["agent_a", "short_name_a", "agent_b", "short_name_b", "score", "reason"]].to_string(index=False))


def _attach(d, al):
    out = al.rename(columns={"agent_a": "responder", "agent_b": "requester", "score": "align_out"})[["responder", "requester", "align_out", "since"]]
    inn = al.rename(columns={"agent_a": "requester", "agent_b": "responder", "score": "align_in"})[["responder", "requester", "align_in"]]
    return d.merge(out, on=["responder", "requester"]).merge(inn, on=["responder", "requester"])


def analyse(n_boot=300, seed=0):
    """Does alignment predict who answers whom?
    align_out: the responder's goal serves the requester's.  align_in: the requester's goal serves the responder's.
    1. Era 3 cross-section, with and without agent fixed effects (one agent per goal, so identity and goal are confounded without them).
    2. Before/after: the same ordered dyad in the three months before goals were assigned and after."""
    from swarm_sna import helping as H

    rng = np.random.default_rng(seed)
    d = pd.read_parquet(OUT / "dyads.parquet")
    d["ts"] = pd.to_datetime(d.ts)
    al = pd.read_csv(OUT / "goal_alignment.csv", parse_dates=["since"])
    x = _attach(d, al)
    post = x[(x.era == 3) & (x.ts >= x.since)]
    print(f"\n=== does alignment predict responding? era 3: {len(post):,} dyad rows, {post.request_id.nunique():,} requests, {post.responder.nunique()} responders")
    print("response rate by align_out (responder's goal serves the requester's):")
    print(post.groupby("align_out").responded.agg(rate="mean", rows="size").round(3).to_string())

    def fit(g, fe):
        F = H._features(g).assign(align_out=g.align_out.to_numpy(), align_in=g.align_in.to_numpy())
        X = np.c_[np.ones(len(g)), F.to_numpy()]
        if fe:
            X = np.c_[X, pd.get_dummies(g.responder, drop_first=True, dtype=float).to_numpy(), pd.get_dummies(g.requester, drop_first=True, dtype=float).to_numpy()]
        return H.logit_fit(X, g.responded.to_numpy(float))[F.shape[1] - 1 : F.shape[1] + 1]  # the two alignment terms

    rows = []
    groups = [t for _, t in post.groupby("responder")]
    for fe, label in [(False, "era 3 logit, controls"), (True, "era 3 logit, + agent fixed effects")]:
        est = fit(post, fe)
        bs = np.array([fit(pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))]), fe) for _ in range(n_boot // 3)])
        for j, k in enumerate(["align_out", "align_in"]):
            rows.append((label, k, est[j], np.quantile(bs[:, j], 0.025), np.quantile(bs[:, j], 0.975)))

    start = al.since.min()
    w = d[d.ts >= start - pd.Timedelta(days=96)].assign(post=lambda t: (t.ts >= start).astype(int))
    g = w.groupby(["responder", "requester", "post"]).responded.agg(["mean", "size"]).unstack("post")
    g.columns = ["rate_pre", "rate_post", "n_pre", "n_post"]
    g = _attach(g.dropna().reset_index(), al)
    g = g[(g.n_pre >= 10) & (g.n_post >= 10)].assign(delta=lambda t: t.rate_post - t.rate_pre)
    print(f"\nbefore/after {start:%Y-%m-%d}: {len(g)} ordered dyads with 10+ opportunities in both periods; response rate {g.rate_pre.mean():.1%} -> {g.rate_post.mean():.1%}")
    print(g.groupby("align_out").agg(dyads=("delta", "size"), before=("rate_pre", "mean"), after=("rate_post", "mean"), change=("delta", "mean")).round(3).to_string())

    def ols(t):
        wts = np.sqrt(np.minimum(t.n_pre, t.n_post)).to_numpy()
        return np.linalg.lstsq(np.c_[np.ones(len(t)), t.align_out, t.align_in] * wts[:, None], t.delta.to_numpy() * wts, rcond=None)[0][1:]

    groups = [t for _, t in g.groupby("responder")]
    est = ols(g)
    bs = np.array([ols(pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])) for _ in range(n_boot * 3)])
    for j, k in enumerate(["align_out", "align_in"]):
        rows.append(("before/after, change in response rate", k, est[j], np.quantile(bs[:, j], 0.025), np.quantile(bs[:, j], 0.975)))
    t = pd.DataFrame(rows, columns=["model", "term", "estimate", "ci_lo", "ci_hi"])
    t.to_csv(OUT / "goal_alignment_effects.csv", index=False)
    print("\neffect of one unit of alignment (95% bootstrap over responders):")
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
    analyse()
