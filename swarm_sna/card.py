"""`swarm-sna card`: the report card. One self-contained, interactive HTML page.

The fixed questions as cards, each with its verdict against a null model, the
numbers, a figure drawn in the browser (D3, inlined, no network needed) and a
drawer of the verbatim evidence behind the labels. It is built from whatever the
other commands have written to the directory; a question whose layer has not
been run says which command would answer it.

A group (era) whose median day has fewer than `--min-agents` agents is shown but
not judged: a network statistic on four nodes has nothing to say.
"""
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .figures import FAINT, INK, MUTED, NAMED_FAMILIES, NULL_FILL, OBSERVED, OTHER, SERIES
from .helping import SIZE_LABELS, helpful
from .report import LABELS as STAT_LABELS
from .trends import SERIES as TREND_SERIES

ALPHA = 0.05
MIN_AGENTS = 6
MIN_AD = 4
STATIC = Path(__file__).parent / "static"
NULL_NAMES = {"speaker": "shuffling who spoke", "target": "shuffling who was addressed"}
NETWORK_CARDS = [
    ("reciprocity", "Is interaction more reciprocal than chance?", "Share of all mention weight that is matched in the opposite direction."),
    ("same_family_share", "Do agents favour their own kind?", "Share of mention weight that stays within a family (for models: the same vendor)."),
    ("partner_selectivity", "Do agents have preferred partners?", "How concentrated each agent's mentions are on a few partners; perfectly even would be 1/k."),
    ("transitivity", "Do ties close into cliques?", "Share of connected triples that close into triangles, on the stronger half of ties."),
]
LADDER_NOTE = {
    "base rate": "was the request addressed to you",
    "+ direct reciprocity": "has the asker answered you in the last 7 days",
    "+ indirect reciprocity": "has the asker answered anyone in the last 7 days",
    "+ cost": "were you busy in a work session",
    "+ bystanders": "how many others were present",
    "lookup table": "ceiling: every combination of the above, tabulated",
}
FEATURE_NAMES = {"addressed": "addressed by name", "prior_help": "asker answered you before", "reputation": "asker answers others", "busy": "responder busy", "log_active_n": "agents present (log)"}
CHIP_LABELS = {"above": "above chance", "below": "below chance", "within": "within chance", "unknown": "not testable", "thin": "too thin to judge", "mixed": "nulls disagree"}
PALETTE = {"series": SERIES, "other": OTHER, "null": NULL_FILL, "observed": OBSERVED, "ink": INK, "muted": MUTED, "faint": FAINT, "grid": "#eceef1", "axis": "#c3c6cc"}

esc = html.escape


def _p(p):
    return "n/a" if pd.isna(p) else "&lt; 0.002" if p < 0.002 else f"{p:.3f}"


def _outcome(observed, null_mean, p):
    if pd.isna(p) or pd.isna(observed):
        return "unknown"
    return "within" if p >= ALPHA else "above" if observed > null_mean else "below"


def _chip(outcome):
    return f'<span class="chip {outcome}">{CHIP_LABELS[outcome]}</span>'


def _tag(outcome, text):
    return f'<span class="chip {outcome}">{esc(text)}</span>'


def _fig(kind, grouped=False):
    """A placeholder the page's script fills with a D3 figure; `grouped` ones re-render when the group filter changes."""
    return f'<div class="fig" data-fig="{kind}"{" data-groups=1" if grouped else ""}></div>'


def _table(df, formats=None, raw=(), classes=None):
    formats = formats or {}
    head = "".join(f"<th>{esc(str(c))}</th>" for c in df.columns)
    body = ""
    for i, row in enumerate(df.itertuples(index=False)):
        cells = []
        for c, v in zip(df.columns, row):
            if c in raw:
                cells.append(f"<td>{v}</td>")
            elif isinstance(v, (float, np.floating)):
                cells.append(f'<td class="num">{"" if pd.isna(v) else format(v, formats.get(c, ".3f"))}</td>')
            elif isinstance(v, (int, np.integer)):
                cells.append(f'<td class="num">{v:,}</td>')
            else:
                cells.append(f"<td>{esc(str(v))}</td>")
        cls = f' class="{classes[i]}"' if classes and classes[i] else ""
        body += f"<tr{cls}>" + "".join(cells) + "</tr>"
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _drawer(summary, body):
    return f"<details><summary>{esc(summary)}</summary>{body}</details>" if body else ""


def _card(title, verdict, body, anchor):
    return f'<article class="card" id="{anchor}"><h2>{esc(title)}</h2><p class="verdict">{verdict}</p>{body}</article>'


def _missing(title, command, anchor):
    return _card(title, _chip("unknown") + f" This question needs a layer that has not been run. <code>{esc(command)}</code> would answer it.", "", anchor), (title, _tag("unknown", "not run"), anchor)


def _load_all(d, name):
    files = sorted(d.glob(f"{name}*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else None


def _thin_note(thin, groups):
    hit = sorted(thin & set(groups))
    return f' {_chip("thin")} {esc(", ".join(hit))}: shown, not judged.' if hit else ""


# ---------- JSON payload helpers

def _clean(v):
    """Anything pandas or numpy hands back -> plain JSON values."""
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
        return [_clean(x) for x in v]
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return None if pd.isna(v) else str(pd.Timestamp(v))[:19]
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else round(float(v), 5)
    if v is pd.NaT or v is pd.NA:
        return None
    return v


def _records(df):
    return [_clean(r) for r in df.to_dict("records")]


def group_info(msgs, group, min_agents):
    """Size of each group: agents, days, and the median number of agents active per day, which decides `thin`."""
    out = []
    for g, m in msgs.groupby(group, sort=True):
        daily = m.groupby("day").actor.nunique()
        out.append({"name": f"{group} {g}", "agents": int(m.actor.nunique()), "median_active": float(daily.median()), "days": int(len(daily)), "messages": int(len(m)), "thin": bool(daily.median() < min_agents)})
    return out


def networks_payload(msgs, mentions, agents, group, top=150):
    """The mention network per group: every node with its strengths, the strongest ties."""
    grp = msgs.set_index("event_id")[group]
    m = mentions[(mentions.actor_class == "agent") & mentions.kind.isin(("at", "name"))]
    m = m.assign(g=m.event_id.map(grp)).dropna(subset=["g"])
    m = m[m.actor != m.target]
    fam = dict(zip(agents.agent, agents.family))
    out = {}
    for g, mg in m.groupby("g"):
        e = mg.groupby(["actor", "target"]).weight.sum().reset_index()
        ins, outs = e.groupby("target").weight.sum(), e.groupby("actor").weight.sum()
        volume = msgs[msgs[group] == g].actor.value_counts()
        nodes = sorted(set(e.actor) | set(e.target))
        out[f"{group} {g}"] = {
            "nodes": [{"id": a, "family": fam.get(a, "unknown"), "in": round(float(ins.get(a, 0)), 2), "out": round(float(outs.get(a, 0)), 2), "msgs": int(volume.get(a, 0))} for a in nodes],
            "edges": [{"s": r.actor, "t": r.target, "w": round(float(r.weight), 2)} for r in e.nlargest(top, "weight").itertuples()],
        }
    return out


def _week(days):
    return pd.to_datetime(days).dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")


def timeline_payload(events, msgs, mentions, agents, group, req, resp):
    """Weekly edge lists for the mention network and, when the LLM layers exist, the help network;
    agents in order of joining so the page can keep each one at a fixed place."""
    m = msgs.assign(week=_week(msgs.day))
    order = agents.sort_values("joined").agent.tolist()
    order += sorted(set(msgs.actor) - set(order))
    idx = {a: i for i, a in enumerate(order)}
    fam = dict(zip(agents.agent, agents.family))
    by_week = m.groupby("week")
    active = {w: sorted(idx[a] for a in g.actor.unique()) for w, g in by_week}
    messages = {w: {str(idx[a]): int(c) for a, c in g.actor.value_counts().items()} for w, g in by_week}
    era = {w: _clean(g[group].mode().iloc[0]) if g[group].notna().any() else None for w, g in by_week}

    def edges(df, src, dst):
        e = df.groupby(["week", src, dst]).weight.sum().reset_index()
        e = e[(e[src] != e[dst]) & e[src].isin(idx) & e[dst].isin(idx)]
        return {w: [[idx[s], idx[t], round(float(v), 2)] for s, t, v in zip(g[src], g[dst], g.weight)] for w, g in e.groupby("week")}

    mm = mentions[(mentions.actor_class == "agent") & mentions.kind.isin(("at", "name"))]
    mention_edges = edges(mm.assign(week=_week(mm.day)), "actor", "target")
    help_edges = None
    if req is not None and resp is not None:
        h = helpful(resp)
        h = h[h.actor_class == "agent"].merge(req[["request_id", "actor"]].rename(columns={"actor": "target"}), on="request_id")
        h = h.assign(day=h.event_id.map(events.set_index("event_id").day), weight=1.0).dropna(subset=["day"])
        help_edges = edges(h.assign(week=_week(h.day)), "actor", "target") if len(h) else None
    return {"periods": sorted(active), "agents": [{"id": a, "family": fam.get(a, "unknown")} for a in order], "active": active, "messages": messages, "era": era, "mentions": mention_edges, "help": help_edges}


# ---------- cards

def network_cards(d, group, thin):
    path = d / f"report_{group}.csv"
    if not path.exists():
        return [_missing(q, "swarm-sna report", s) for s, q, _ in NETWORK_CARDS]
    t = pd.read_csv(path)
    out = []
    for i, (stat, question, meaning) in enumerate(NETWORK_CARDS):
        s = t[(t.statistic == stat) & t.null.isin(NULL_NAMES)]
        s = s.assign(outcome=[_outcome(r.observed, r.null_mean, r.p) for r in s.itertuples()], thin=s.group.isin(thin))
        judged = s[~s.thin]
        outcomes = set(judged.outcome)
        if not len(judged):
            verdict = _chip("thin") + " Every group has too few agents a day to judge."
        elif len(outcomes) == 1:
            o = outcomes.pop()
            verdict = _chip(o) + {"above": " In every group, under both nulls.", "below": " In every group, under both nulls.", "within": " Not distinguishable from chance in any group."}.get(o, "")
        else:
            parts = []
            for null, g in judged.groupby("null", sort=False):
                by = g.groupby("outcome", sort=False).group.agg(", ".join)
                parts.append(f"<b>{NULL_NAMES[null].capitalize()}</b>: " + "; ".join(f"{_chip(o)} {esc(groups)}" for o, groups in by.items()))
            verdict = "It depends on the group and the null. " + " &nbsp; ".join(parts)
        verdict += _thin_note(thin, s.group)
        rows = pd.DataFrame(
            {
                "group": s.group, "null": s.null.map(NULL_NAMES), "observed": s.observed, "null mean": s.null_mean,
                "null 95% band": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(s.null_lo, s.null_hi)], "p": [_p(p) for p in s.p],
                "": [_chip("thin") if th else _chip(o) for o, th in zip(s.outcome, s.thin)],
            }
        )
        body = f'<p class="meaning">{esc(meaning)}</p>' + _table(rows, raw=("p", ""), classes=["thin" if th else "" for th in s.thin])
        if i == 0:
            body += _fig("nulls", grouped=True)
        if i == 2:
            body += _fig("network", grouped=True)
        chips = []
        for g, rows_g in s.groupby("group", sort=False):
            if g in thin:
                chips.append(_tag("thin", f"{g}: thin"))
                continue
            o = rows_g.outcome.iloc[0] if rows_g.outcome.nunique() == 1 else "mixed"
            chips.append(_tag(o, f"{g}: {CHIP_LABELS[o]}"))
        out.append((_card(question, verdict, body, stat), (question, " ".join(chips), stat)))
    return out


def hierarchy_card(d, group, thin):
    title, anchor = "Is there a dominance hierarchy?", "hierarchy"
    path = d / f"hierarchy_{group}.csv"
    if not path.exists():
        return _missing(title, "swarm-sna label, then helping and hierarchy", anchor)
    t = pd.read_csv(path)
    main = t[t.group.str.startswith(group.split("_")[0])]
    st = main[main.statistic == "steepness"]
    outs = [_outcome(r.observed, r.null_mean, r.p) for r in st.itertuples()]
    steep = [g for g, o in zip(st.group, outs) if o == "above" and g not in thin]
    flat = [g for g, o in zip(st.group, outs) if o != "above" and g not in thin]
    verdict = (_chip("above") + f" Steeper than chance in {esc(', '.join(steep))}. " if steep else "") + (_chip("within") + f" Not distinguishable from chance in {esc(', '.join(flat))}." if flat else "")
    verdict = (verdict or _chip("thin") + " Every group has too few agents a day to judge.") + _thin_note(thin, st.group)
    rows = pd.DataFrame(
        {
            "group": main.group, "statistic": main.statistic.str.replace("_", " "), "observed": main.observed, "null mean": main.null_mean,
            "null 95% band": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(main.null_lo, main.null_hi)], "p": [_p(p) for p in main.p],
            "": [_chip("thin") if g in thin else _chip(_outcome(r.observed, r.null_mean, r.p)) for g, r in zip(main.group, main.itertuples())],
        }
    )
    per = main.drop_duplicates("group")
    corr = pd.DataFrame({"group": per.group, "agents": per.agents, "directives": per.contests, "complied with": per.compliance, "rank vs release date": per.rank_vs_release_date, "rank vs message count": per.rank_vs_messages})
    body = '<p class="meaning">A contest is a directive addressed to a named agent: the sender wins if the target complies, the target wins if it declines or ignores it. Null: outcomes shuffled among contests within room and day.</p>'
    body += _table(rows, raw=("p", ""), classes=["thin" if g in thin else "" for g in main.group]) + _fig("hierarchy", grouped=True)
    body += "<h3>Does rank follow capability or talkativeness?</h3>" + _table(corr, {"complied with": ".0%", "rank vs release date": "+.2f", "rank vs message count": "+.2f"}, classes=["thin" if g in thin else "" for g in per.group])
    body += "<h3>Who ranks highest</h3>" + _fig("ranks", grouped=True)
    ranks_path = d / f"hierarchy_ranks_{group}.csv"
    if ranks_path.exists():
        r = pd.read_csv(ranks_path)
        top = r.groupby("group", sort=False).head(5)[["group", "agent", "davids_score", "directives_sent", "obeyed", "directives_received", "messages"]]
        body += _drawer("The top five per group and episode, as a table", _table(top, {"davids_score": ".2f", "obeyed": ".0f"}))
    episodes = t[~t.group.isin(main.group) & (t.statistic == "steepness")]
    if len(episodes):
        e = pd.DataFrame({"episode | window": episodes.group, "steepness": episodes.observed, "null mean": episodes.null_mean, "p": [_p(p) for p in episodes.p], "directives": episodes.contests})
        body += _drawer("Before, during and after an imposed leader", _table(e, raw=("p",)))
    chips = " ".join(_tag("thin", f"{g}: thin") if g in thin else _tag(o if o == "above" else "within", f"{g}: " + ("steeper than chance" if o == "above" else "no hierarchy beyond chance")) for g, o in zip(st.group, outs))
    return _card(title, verdict, body, anchor), (title, chips, anchor)


def _examples(d, n=6, seed=0):
    req, resp = _load_all(d, "requests"), _load_all(d, "responses")
    if req is None or resp is None or "quote" not in req.columns or "quote" not in resp.columns:
        return None, None  # the published derived tables carry the labels without the text
    good = helpful(resp)
    good = good[good.actor_class == "agent"]
    pairs = good.merge(req[["request_id", "actor", "quote", "addressed", "targets"]].rename(columns={"actor": "asker", "quote": "request"}), on="request_id")
    return req, pairs.sample(min(n, len(pairs)), random_state=seed) if len(pairs) else pairs


def _exchange(r):
    to = ", ".join(r.targets) if len(r.targets) else "anyone"
    backing = f'<p class="backing"><span class="tag ok">checkable</span> {esc(str(r.evidence))}</p>' if r.backed else '<p class="backing"><span class="tag bare">bare claim</span> nothing in the message to check it against</p>'
    return (
        f'<div class="exchange"><p class="who">{esc(r.asker)} &rarr; {esc(to)}</p><blockquote>{esc(str(r.request))}</blockquote>'
        f'<p class="who">{esc(r.actor)} {esc(r.type)}, {r.latency_s / 60:.0f} min later</p><blockquote>{esc(str(r.quote)[:600])}</blockquote>{backing}</div>'
    )


def helping_cards(d, thin):
    t1, a1, t2, a2 = "What explains who answers whom?", "helping", "Is there a bystander effect?", "bystander"
    if not (d / "helping_ladder.csv").exists():
        return [_missing(t1, "swarm-sna label, then helping", a1), _missing(t2, "swarm-sna label, then helping", a2)]
    ladder = pd.read_csv(d / "helping_ladder.csv")
    summary = json.loads((d / "helping_summary.json").read_text()) if (d / "helping_summary.json").exists() else {}
    rungs = ladder[ladder.model != "no covariates"]
    first, last = rungs.iloc[0], rungs.iloc[-2]
    verdict = f"Being addressed by name reaches <b>{first.completeness:.0%}</b> of what a lookup table can explain; adding reciprocity, cost and bystanders reaches <b>{last.completeness:.0%}</b> [{last.ci_lo:.0%}, {last.ci_hi:.0%}]."
    rows = pd.DataFrame({"model": rungs.model, "adds": rungs.model.map(LADDER_NOTE), "log loss": rungs.log_loss, "completeness": rungs.completeness, "95% CI": [f"[{a:.2f}, {b:.2f}]" for a, b in zip(rungs.ci_lo, rungs.ci_hi)]})
    body = '<p class="meaning">One row per request and agent present. Nested logistic models, scored out of sample (held out by responder) as a share of the gain a lookup table achieves over the base rate.</p>'
    if summary:
        body += f'<p class="meaning">{summary["requests"]:,} requests, {summary["dyad_rows"]:,} request-agent pairs. An agent addressed by name answers {summary["addressed_rate"]:.0%} of the time; one not addressed, {summary["not_addressed_rate"]:.0%}.</p>'
    body += _fig("ladder") + _table(rows, {"log loss": ".4f", "completeness": ".2f"})
    if (d / "helping_coefficients.csv").exists():
        c = pd.read_csv(d / "helping_coefficients.csv", index_col=0).T.rename(columns=FEATURE_NAMES).reset_index(names="fit")
        note = f'<p class="meaning">Log-odds coefficients of the full model, refitted each way a pooled fit could mislead. {esc(summary.get("top_asker", "One agent"))} alone asks {summary.get("top_asker_share", 0):.0%} of all requests, hence the equal-weight row.</p>' if summary else ""
        body += _drawer("Does it hold up? Per era, within goal, with heavy askers down-weighted", note + _table(c, {k: "+.2f" for k in c.columns}))
    _, ex = _examples(d)
    if ex is not None and len(ex):
        body += _drawer(f"Evidence: {len(ex)} request-and-answer pairs, verbatim", "".join(_exchange(r) for r in ex.itertuples()))
    cards = [(_card(t1, verdict, body, a1), (t1, _tag("info", f"model reaches {last.completeness:.0%} of the ceiling"), a1))]

    curve = pd.read_csv(d / "helping_bystander.csv")
    if curve.empty:
        cards.append((_card(t2, _tag("unknown", "not testable") + " This transcript has too few broadcast requests (ones addressed to nobody in particular) to test it.", "", a2), (t2, _tag("unknown", "not testable"), a2)))
        return cards
    s, lo, hi = summary.get("bystander_slope", [np.nan] * 3)
    o = "unknown" if np.isnan(s) else "within" if lo <= 0 <= hi else "below" if hi < 0 else "above"
    label = {"below": "yes: each agent answers less in a larger group", "above": "no: each agent answers more in a larger group", "within": "no clear effect of group size", "unknown": "slope not computed"}[o]
    verdict = _tag(o, label) + (f" Slope of the log-odds that an agent answers on log group size, within goal: {s:.2f} [{lo:.2f}, {hi:.2f}]." if o != "unknown" else "")
    rows = curve.rename(columns={"size": "agents present", "p_each_agent_responds": "a given agent answers", "p_any_agent_responds": "anyone answers"})
    body = '<p class="meaning">Broadcast requests only, compared within era, because group size and era are confounded across eras.</p>' + _fig("bystander", grouped=True)
    body += _drawer("The curve as a table", _table(rows, {"a given agent answers": ".0%", "anyone answers": ".0%"}))
    cards.append((_card(t2, verdict, body, a2), (t2, _tag(o, label), a2)))
    return cards


def evidence_card(d):
    title, anchor = "Are claims of work done backed by something checkable?", "evidence"
    resp = _load_all(d, "responses")
    if resp is None or "backed" not in resp.columns:
        return _missing(title, "swarm-sna label", anchor)
    done = helpful(resp)
    share = done.backed.mean()
    verdict = f"<b>{share:.0%}</b> of the {len(done):,} answers and reported actions carry a link, id, number, file name or quoted output a reader could check. The rest are bare claims."
    by = done.groupby("type").backed.agg(["mean", "size"]).reset_index().rename(columns={"type": "response type", "mean": "backed", "size": "responses"})
    body = '<p class="meaning">An agent saying it did something is a claim, not a fact. For every response the labeller must quote, verbatim, the part of the message that would let a reader verify it; the quote is then checked against the message text.</p>' + _table(by, {"backed": ".0%"})
    _, ex = _examples(d, n=400, seed=1)
    if ex is not None and len(ex):
        pick = pd.concat([ex[ex.backed].head(3), ex[~ex.backed].head(3)])
        body += _drawer("Evidence: three backed answers and three bare claims, verbatim", "".join(_exchange(r) for r in pick.itertuples()))
    return _card(title, verdict, body, anchor), (title, _tag("info", f"{share:.0%} backed"), anchor)


def diffusion_card(d):
    title, anchor = "Does information spread along ties?", "diffusion"
    path = d / "diffusion_items.csv"
    if not path.exists():
        return _missing(title, "swarm-sna diffusion", anchor)
    t = pd.read_csv(path)
    if t.empty:
        verdict = _tag("unknown", "not testable") + f" No link, file name or backticked term spread to {MIN_AD}+ agents in this transcript, so there is no adoption order to test."
        return _card(title, verdict, "", anchor), (title, _tag("unknown", "not testable"), anchor)
    z = t.z.dropna()
    share = (t.p < 0.05).mean()
    stouffer = z.mean() * np.sqrt(len(z)) if len(z) else np.nan
    o = "above" if stouffer > 1.96 else "within"
    label = "adoption follows ties" if o == "above" else "no more than chance"
    verdict = _tag(o, label) + f" {len(t)} items (links, file names, backticked terms) that spread to {MIN_AD}+ agents. The next adopter ranks <b>{t.observed.mean():.2f}</b> among the agents still to adopt, by ties to those who already had (0.50 by chance). In {share:.1%} of items the order leans on ties more than random orders do (5% expected by chance); combined z = {stouffer:+.1f}."
    body = '<p class="meaning">Order-of-acquisition diffusion analysis as a rank test (Franz &amp; Nunn 2009; Hoppitt &amp; Laland 2013). For each item, agents are ordered by first use. At each adoption the adopter is ranked, by tie weight to the agents who already adopted, among the agents who had not yet (0 = least tied, 1 = most); the statistic is the mean rank, and the null permutes the order among the same agents. Ties are mentions in the month before the item first appeared. An item has few adoptions, so single-item tests are weak; the pooled rank and the combined z carry the evidence.</p>'
    body += _fig("diffusion_items")
    body += '<p class="meaning">Read with care in a shared chat room: every agent sees every message, so access to an item is not gated by ties the way it is in an animal group. The test asks whether attention ties predict who picks something up next, not whether they were needed to hear of it.</p>'
    top = t.nsmallest(8, "p")[["item", "adopters", "first_adopter", "spread_days", "observed", "z", "p"]].rename(columns={"spread_days": "days to spread", "first_adopter": "first adopter", "observed": "adopter rank"})
    body += _drawer("The eight items whose spread leans most on ties", _table(top, {"days to spread": ".0f", "adopter rank": ".2f", "z": "+.2f", "p": ".3f"}))
    chips = [_tag(o, label)]

    # The adoption network: who picks things up first, who follows (diffusion_leaders.csv, diffusion_edges.csv).
    lead_path, sm_path = d / "diffusion_leaders.csv", d / "diffusion_summary.csv"
    if lead_path.exists() and sm_path.exists():
        lead, sm = pd.read_csv(lead_path), pd.read_csv(sm_path).iloc[0]
        if len(lead) and "steepness" in sm and pd.notna(sm.steepness):
            beyond = sm.steepness_volume_p < ALPHA and sm.steepness > sm.steepness_volume_null
            steep = "steeper than volume alone predicts" if beyond else "no steeper than volume alone predicts"
            body += "<h3>The adoption network: who picks things up first, and who follows</h3>"
            body += f'<p class="meaning">Adoption defines its own network. For each item, every adopter after the first hands one unit of credit, split evenly, to the agents who had it before. An agent\'s lead score is (credit received &minus; credit given) / total: +1 is always first, &minus;1 always after others. Across {int(sm.network_items)} items the leader-follower ordering is <b>{steep}</b> (steepness {sm.steepness:.3f}; null with order drawn in proportion to posting volume {sm.steepness_volume_null:.3f}, p = {sm.steepness_volume_p:.3f}; plain random order {sm.steepness_random_null:.3f}, p = {sm.steepness_random_p:.3f}).</p>'
            body += _fig("leaders") + _fig("diffusion_network")
            if (d / "diffusion_correlates.csv").exists():
                c = pd.read_csv(d / "diffusion_correlates.csv").rename(columns={"Unnamed: 0": "leading, measured as"})
                body += '<p class="meaning">Does leading go with capability, talkativeness or rank? Rank correlations across agents. The two nulls bracket the truth: plain random order ignores that heavy posters reach everything sooner; volume-weighted order assumes an agent posting ten times as much adopts ten times sooner.</p>'
                body += _table(c, {k: "+.2f" for k in c.columns if k.startswith("vs")})
            show = lead.sort_values("lead_score", ascending=False)[["agent", "led", "followed", "lead_score", "z_random_order", "z"]].rename(columns={"lead_score": "lead score", "z_random_order": "z, random order", "z": "z, volume-weighted"})
            body += _drawer("Every agent's lead score", _table(show, {"led": ".0f", "followed": ".0f", "lead score": "+.2f", "z, random order": "+.1f", "z, volume-weighted": "+.1f"}))
            chips.append(_tag("above" if beyond else "within", "leader-follower order: " + ("beyond volume" if beyond else "explained by volume")))
    return _card(title, verdict, body, anchor), (title, " ".join(chips), anchor)


def trends_card(d, has_timeline):
    title, anchor = "How does the network change over time?", "time"
    body = ""
    if has_timeline:
        body += "<h3>The network, week by week</h3>" + _fig("timeline")
    if (d / "trends_weekly.csv").exists():
        body += "<h3>The weekly series behind the pooled numbers</h3>" + _fig("trends")
        w = pd.read_csv(d / "trends_weekly.csv")
        body += _drawer("The weekly series as a table", _table(w.rename(columns=lambda c: c.replace("_", " ")), {c.replace("_", " "): ".0%" for c in ("answered", "each_agent_responds", "compliance", "backed")} | {"requests per 100": ".1f"}))
    if not body:
        return _missing(title, "swarm-sna report, then trends", anchor)
    body = '<p class="meaning">A pooled statistic over a long transcript mixes regimes. These views show whether a result is a property of the swarm or of one period.</p>' + body
    if (d / "trends_goal_type.csv").exists():
        g = pd.read_csv(d / "trends_goal_type.csv").rename(columns=lambda c: c.replace("_", " "))
        body += _drawer("Response rates by the kind of goal the swarm was given", _table(g, {c: ".0%" for c in g.columns if "respond" in c or "answered" in c}))
    return _card(title, "Read these before trusting any pooled number above.", body, anchor), (title, _tag("info", "see figures"), anchor)


def quality_card(d, validation):
    title, anchor = "How far can the labels be trusted?", "quality"
    rows = []
    v = Path(validation) / "mentions.csv" if validation else None
    if v is not None and v.exists():
        m = pd.read_csv(v)
        if m.target_correct.notna().any():
            rows.append(("Mention edges (regex)", f"{len(m)} checked by hand or by a second rater", f"{m.target_correct.mean():.0%} point at the right agent"))
    req, resp = _load_all(d, "requests"), _load_all(d, "responses")
    if req is not None:
        rows.append(("Requests (LLM)", f"{len(req):,} found", f"{req.quote_verified.mean():.0%} of quotes are verbatim in the source message"))
    if resp is not None:
        rows.append(("Responses (LLM)", f"{len(resp):,} found", f"{resp.quote_verified.mean():.0%} of quotes are verbatim in the responding message"))
        if "link_verified" in resp.columns:
            lv = resp.link_verified.dropna()
            if len(lv):
                rows.append(("Request-response links (second model)", f"{len(lv):,} re-checked one pair at a time", f"{lv.astype(bool).mean():.0%} confirmed as responses to that request; the rest are dropped"))
    v = Path(validation) / "response_links.csv" if validation else None
    if v is not None and v.exists():
        lk = pd.read_csv(v)
        if "is_response" in lk.columns and lk.is_response.notna().any():
            rows.append(("Request-response links (first pass)", f"{len(lk)} adjudicated by hand or by a second rater", f"{lk.is_response.mean():.0%} were genuine responses, which is why the second pass exists"))
    if not rows:
        return None
    body = _table(pd.DataFrame(rows, columns=["layer", "size", "check"]))
    body += '<p class="meaning">Every LLM label must come with the sentence that justifies it, copied character for character, and the copy is checked. Rows that fail are excluded from every statistic. The first response pass reads up to 45 following messages at once and over-links; a second, stronger model then re-reads each link on its own and only confirmed links count. What this does not measure is recall: a request with neither a question mark nor a request phrase is never shown to the LLM. <code>scripts/recall_check.py</code> estimates that miss rate on a sample.</p>'
    return _card(title, "Every number above rests on these layers.", body, anchor), (title, _tag("info", "see checks"), anchor)


# ---------- the data behind the figures

def _csv(path):
    return _records(pd.read_csv(path)) if path.exists() else None


def payload(d, group, groups, msgs, events, agents, mentions, req, resp):
    fam_present = set(agents.family.dropna())
    family_order = [f for f in NAMED_FAMILIES if f in fam_present]
    for f, _ in agents.family.value_counts().items():
        if len(family_order) >= 3:
            break
        if f not in family_order and f != "unknown":
            family_order.append(f)
    out = {
        "group_label": group, "groups": groups, "palette": PALETTE, "family_order": family_order, "families": dict(zip(agents.agent, agents.family)),
        "nulls": None, "networks": {}, "timeline": None, "helping": None, "hierarchy": None, "trends": None, "diffusion": None,
    }
    rep = d / f"report_{group}.csv"
    if rep.exists():
        t = pd.read_csv(rep)
        draws_path = d / f"report_{group}_draws.json"
        out["nulls"] = {
            "table": _records(t[t.null.isin(NULL_NAMES)]), "stats": [{"key": k, "label": STAT_LABELS[k]} for k in [s for s, _, _ in NETWORK_CARDS]], "null_names": NULL_NAMES,
            "draws": json.loads(draws_path.read_text()) if draws_path.exists() else None,
        }
    if mentions is not None:
        out["networks"] = networks_payload(msgs, mentions, agents, group)
        out["timeline"] = timeline_payload(events, msgs, mentions, agents, group, req, resp)
    if (d / "helping_ladder.csv").exists():
        summary = json.loads((d / "helping_summary.json").read_text()) if (d / "helping_summary.json").exists() else {}
        out["helping"] = {"ladder": _csv(d / "helping_ladder.csv"), "bystander": _csv(d / "helping_bystander.csv") or [], "sizes": list(SIZE_LABELS), "slope": summary.get("bystander_slope"), "summary": summary}
    if (d / f"hierarchy_{group}.csv").exists():
        out["hierarchy"] = {"table": _csv(d / f"hierarchy_{group}.csv"), "ranks": _csv(d / f"hierarchy_ranks_{group}.csv") or []}
    if (d / "trends_weekly.csv").exists():
        w = pd.read_csv(d / "trends_weekly.csv")
        out["trends"] = {"weekly": _records(w), "series": [list(s) for s in TREND_SERIES], "era_starts": {str(_clean(e)): wk for e, wk in w.dropna(subset=["era"]).groupby("era").week.min().items()}}
    if (d / "diffusion_items.csv").exists():
        sm = _csv(d / "diffusion_summary.csv")
        out["diffusion"] = {"items": _csv(d / "diffusion_items.csv"), "leaders": _csv(d / "diffusion_leaders.csv") or [], "edges": _csv(d / "diffusion_edges.csv") or [], "summary": sm[0] if sm else None}
    return out


def _static(name):
    return (STATIC / name).read_text().replace("</script", "<\\/script")


def run(in_dir, title=None, group="era", validation="validation", min_agents=MIN_AGENTS):
    d = Path(in_dir)
    ev = pd.read_parquet(d / "events.parquet")
    agents = pd.read_parquet(d / "agents.parquet")
    msgs = ev[ev.type == "message"]
    amsgs = msgs[msgs.actor_class == "agent"]
    mentions = pd.read_parquet(d / "mentions.parquet") if (d / "mentions.parquet").exists() else None
    req, resp = _load_all(d, "requests"), _load_all(d, "responses")
    groups = group_info(amsgs, group, min_agents)
    thin = {g["name"] for g in groups if g["thin"]}
    tiles = [(f"{len(msgs):,}", "messages"), (f"{amsgs.actor.nunique()}", "agents"), (f"{msgs.day.nunique()}", "active days"), (f"{msgs.room.nunique()}", "rooms")]
    if req is not None:
        tiles.append((f"{len(req):,}", "requests found"))
    if resp is not None:
        tiles.append((f"{len(resp):,}", "responses found"))

    n_perm = 1000
    if (d / f"report_{group}.csv").exists():
        rep = pd.read_csv(d / f"report_{group}.csv")
        if "permutations" in rep.columns:
            n_perm = int(rep.permutations.iloc[0])
    data = payload(d, group, groups, amsgs, ev, agents, mentions, req, resp)
    cards = network_cards(d, group, thin) + [hierarchy_card(d, group, thin)] + helping_cards(d, thin) + [evidence_card(d), diffusion_card(d), trends_card(d, data["timeline"] is not None), quality_card(d, validation)]
    cards = [c for c in cards if c]
    glance = "".join(f'<li><a href="#{a}">{esc(t)}</a><span class="chips">{chips}</span></li>' for _, (t, chips, a) in cards)
    thin_text = ""
    if thin:
        parts = ", ".join(f"{g['name']} ({g['median_active']:.0f})" for g in groups if g["thin"])
        thin_text = f" A group whose median day has fewer than {min_agents} agents is shown but not judged: {esc(parts)}."
    data_json = json.dumps(_clean(data), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")  # never closes the script element early
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Swarm report card</title><style>{_static("card.css")}</style></head><body>
<header><h1>Swarm report card</h1><p class="sub">{esc(title or d.name)} &middot; {esc(str(msgs.day.min()))} to {esc(str(msgs.day.max()))}</p>
<div class="tiles">{"".join(f'<div class="tile"><b>{v}</b><span>{esc(k)}</span></div>' for v, k in tiles)}</div>
<p class="nojs">The figures on this page are drawn in the browser. If they are missing, the viewer has blocked the page's script (GitHub's file view and most in-app previews do): open the file in a web browser.</p>
<div class="groupbar" id="groupbar"></div></header>
<main>
<article class="card"><h2>At a glance</h2><p class="meaning">The questions worth asking of any group of agents. A pattern only counts if it beats a null model: the raw event stream is shuffled {n_perm:,} times in ways that keep how much each agent talks, and the statistic is recomputed on each shuffle. Two shuffles are used. One reassigns who spoke each message; the other reassigns whom each message addressed. They can disagree, and when they do the card says so.{thin_text}</p><ul class="glance">{glance}</ul></article>
{"".join(c for c, _ in cards)}
</main>
<footer>Generated by swarm-sna. Null models follow Bejder et al. 1998 and Farine 2017; dominance measures follow de Vries et al. 2006 and Shizuka &amp; McDonald 2012. Figures drawn with D3 (Mike Bostock, ISC licence).</footer>
<script id="card-data" type="application/json">{data_json}</script>
<script>{_static("d3.v7.min.js")}</script>
<script>{_static("card.js")}</script>
</body></html>"""
    out = d / "report_card.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1e6:.1f} MB)")
