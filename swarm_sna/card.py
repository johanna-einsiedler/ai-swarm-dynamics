"""`swarm-sna card`: the report card. One self-contained HTML page.

The fixed questions as cards, each with its verdict against a null model, the
numbers, the figure, and a drawer of the verbatim evidence behind the labels.
It is built from whatever the other commands have written to the directory; a
question whose layer has not been run says which command would answer it.
"""
import base64
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .helping import helpful

ALPHA = 0.05
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

esc = html.escape


def _p(p):
    return "n/a" if pd.isna(p) else "&lt; 0.002" if p < 0.002 else f"{p:.3f}"


def _outcome(observed, null_mean, p):
    if pd.isna(p) or pd.isna(observed):
        return "unknown"
    return "within" if p >= ALPHA else "above" if observed > null_mean else "below"


def _chip(outcome):
    label = {"above": "above chance", "below": "below chance", "within": "within chance", "unknown": "not testable"}[outcome]
    return f'<span class="chip {outcome}">{label}</span>'


def _img(path, caption=""):
    path = Path(path)
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode()
    cap = f"<figcaption>{esc(caption)}</figcaption>" if caption else ""
    return f'<figure><img src="data:image/png;base64,{data}" alt="{esc(caption)}">{cap}</figure>'


def _table(df, formats=None, raw=()):
    formats = formats or {}
    head = "".join(f"<th>{esc(str(c))}</th>" for c in df.columns)
    body = ""
    for row in df.itertuples(index=False):
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
        body += "<tr>" + "".join(cells) + "</tr>"
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _drawer(summary, body):
    return f"<details><summary>{esc(summary)}</summary>{body}</details>" if body else ""


def _card(title, verdict, body, anchor):
    return f'<article class="card" id="{anchor}"><h2>{esc(title)}</h2><p class="verdict">{verdict}</p>{body}</article>'


def _tag(outcome, text):
    return f'<span class="chip {outcome}">{esc(text)}</span>'


def _missing(title, command, anchor):
    return _card(title, _chip("unknown") + f" This question needs a layer that has not been run. <code>{esc(command)}</code> would answer it.", "", anchor), (title, _tag("unknown", "not run"), anchor)


def _load_all(d, name):
    files = sorted(d.glob(f"{name}*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else None


# ---------- cards

def network_cards(d, group):
    path = d / f"report_{group}.csv"
    if not path.exists():
        return [_missing(q, "swarm-sna report", s) for s, q, _ in NETWORK_CARDS]
    t = pd.read_csv(path)
    out = []
    for i, (stat, question, meaning) in enumerate(NETWORK_CARDS):
        s = t[(t.statistic == stat) & t.null.isin(NULL_NAMES)]
        s = s.assign(outcome=[_outcome(r.observed, r.null_mean, r.p) for r in s.itertuples()])
        outcomes = set(s.outcome)
        if len(outcomes) == 1:
            o = outcomes.pop()
            verdict = _chip(o) + {"above": " In every group, under both nulls.", "below": " In every group, under both nulls.", "within": " Not distinguishable from chance in any group."}.get(o, "")
            headline = o
        else:
            parts = []
            for null, g in s.groupby("null", sort=False):
                by = g.groupby("outcome", sort=False).group.agg(", ".join)
                parts.append(f"<b>{NULL_NAMES[null].capitalize()}</b>: " + "; ".join(f"{_chip(o)} {esc(groups)}" for o, groups in by.items()))
            verdict = "It depends on the group and the null. " + " &nbsp; ".join(parts)
            headline = "mixed"
        rows = pd.DataFrame(
            {
                "group": s.group, "null": s.null.map(NULL_NAMES), "observed": s.observed, "null mean": s.null_mean,
                "null 95% band": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(s.null_lo, s.null_hi)], "p": [_p(p) for p in s.p], "": [_chip(o) for o in s.outcome],
            }
        )
        body = f'<p class="meaning">{esc(meaning)}</p>' + _table(rows, raw=("p", ""))
        if i == 0:
            body += _img(d / f"report_{group}_nulls.png", "Every statistic against the permuted event streams (grey) with the observed value marked.")
        # At a glance: one chip per group, and say so when the two nulls disagree.
        chips = []
        for g, rows_g in s.groupby("group", sort=False):
            o = rows_g.outcome.iloc[0] if rows_g.outcome.nunique() == 1 else "mixed"
            chips.append(_tag(o, f"{g}: " + {"above": "above chance", "below": "below chance", "within": "within chance", "mixed": "nulls disagree", "unknown": "not testable"}[o]))
        out.append((_card(question, verdict, body, stat), (question, " ".join(chips), stat)))
    return out


def hierarchy_card(d, group):
    title, anchor = "Is there a dominance hierarchy?", "hierarchy"
    path = d / f"hierarchy_{group}.csv"
    if not path.exists():
        return _missing(title, "swarm-sna label, then helping and hierarchy", anchor)
    t = pd.read_csv(path)
    main = t[t.group.str.startswith(group.split("_")[0])]
    st = main[main.statistic == "steepness"]
    outs = [_outcome(r.observed, r.null_mean, r.p) for r in st.itertuples()]
    steep = [g for g, o in zip(st.group, outs) if o == "above"]
    flat = [g for g, o in zip(st.group, outs) if o != "above"]
    verdict = (_chip("above") + f" Steeper than chance in {esc(', '.join(steep))}. " if steep else "") + (_chip("within") + f" Not distinguishable from chance in {esc(', '.join(flat))}." if flat else "")
    rows = pd.DataFrame(
        {
            "group": main.group, "statistic": main.statistic.str.replace("_", " "), "observed": main.observed, "null mean": main.null_mean,
            "null 95% band": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(main.null_lo, main.null_hi)], "p": [_p(p) for p in main.p],
            "": [_chip(_outcome(r.observed, r.null_mean, r.p)) for r in main.itertuples()],
        }
    )
    per = main.drop_duplicates("group")
    corr = pd.DataFrame({"group": per.group, "agents": per.agents, "directives": per.contests, "complied with": per.compliance, "rank vs release date": per.rank_vs_release_date, "rank vs message count": per.rank_vs_messages})
    body = '<p class="meaning">A contest is a directive addressed to a named agent: the sender wins if the target complies, the target wins if it declines or ignores it. Null: outcomes shuffled among contests within room and day.</p>'
    body += _table(rows, raw=("p", "")) + _img(d / f"fig_hierarchy_{group}.png")
    body += "<h3>Does rank follow capability or talkativeness?</h3>" + _table(corr, {"complied with": ".0%", "rank vs release date": "+.2f", "rank vs message count": "+.2f"})
    ranks_path = d / f"hierarchy_ranks_{group}.csv"
    if ranks_path.exists():
        r = pd.read_csv(ranks_path)
        top = r.groupby("group", sort=False).head(5)[["group", "agent", "davids_score", "directives_sent", "obeyed", "directives_received", "messages"]]
        body += _drawer("Who ranks highest (top five per group and episode)", _table(top, {"davids_score": ".2f", "obeyed": ".0f"}))
    episodes = t[~t.group.isin(main.group) & (t.statistic == "steepness")]
    if len(episodes):
        e = pd.DataFrame({"episode | window": episodes.group, "steepness": episodes.observed, "null mean": episodes.null_mean, "p": [_p(p) for p in episodes.p], "directives": episodes.contests})
        body += _drawer("Before, during and after an imposed leader", _table(e, raw=("p",)))
    chips = " ".join(_tag(o if o == "above" else "within", f"{g}: " + ("steeper than chance" if o == "above" else "no hierarchy beyond chance")) for g, o in zip(st.group, outs))
    return _card(title, verdict, body, anchor), (title, chips, anchor)


def _examples(d, n=6, seed=0):
    req, resp = _load_all(d, "requests"), _load_all(d, "responses")
    if req is None or resp is None:
        return None, None
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


def helping_cards(d):
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
    body += _table(rows, {"log loss": ".4f", "completeness": ".2f"}) + _img(d / "fig_helping_ladder.png")
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
    body = '<p class="meaning">Broadcast requests only, compared within era, because group size and era are confounded across eras.</p>' + _table(rows, {"a given agent answers": ".0%", "anyone answers": ".0%"}) + _img(d / "fig_bystander.png")
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


def trends_card(d, group):
    title, anchor = "How does the network change over time?", "time"
    figs = _img(d / f"network_{group}.png", "The mention network per group; strongest ties only.")
    figs += _img(d / "fig_network_timeline_mentions.png", "Who mentions whom, per quarter. Each agent keeps its place on the circle, in order of joining.")
    figs += _img(d / "fig_network_timeline_help.png", "Who answers whose requests, per quarter.")
    figs += _img(d / "fig_trends.png", "The weekly series behind the pooled numbers.")
    if not figs:
        return _missing(title, "swarm-sna report, then trends", anchor)
    body = '<p class="meaning">A pooled statistic over a long transcript mixes regimes. These views show whether a result is a property of the swarm or of one period.</p>' + figs
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


# ---------- page

CSS = """
:root { --bg:#f6f6f4; --card:#ffffff; --ink:#1f2328; --muted:#5b6470; --line:#e2e4e8; --above:#2a78d6; --below:#c2410c; --within:#6b7280;
  --above-bg:#e6f0fb; --below-bg:#fbeadf; --within-bg:#eceef1; --ok:#0f7a52; --ok-bg:#e2f4ec; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
main, header, footer { max-width: 1000px; margin: 0 auto; padding: 0 16px; }
header { padding-top: 36px; }
h1 { font-size: 28px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 19px; margin: 0 0 8px; }
h3 { font-size: 14px; margin: 18px 0 6px; color: var(--muted); font-weight: 600; }
.sub { color: var(--muted); margin: 0 0 20px; }
.tiles { display:grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap:10px; margin: 0 0 18px; }
.tile { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 12px; }
.tile b { display:block; font-size:21px; font-variant-numeric: tabular-nums; }
.tile span { color:var(--muted); font-size:12.5px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:20px 22px; margin: 0 0 16px; }
.verdict { font-size: 15.5px; margin: 0 0 12px; }
.meaning { color: var(--muted); font-size: 13.5px; margin: 0 0 10px; }
.chip { display:inline-block; font-size:12px; font-weight:600; padding:1px 9px; border-radius:99px; white-space:nowrap; }
.chip.above { background:var(--above-bg); color:#174e91; } .chip.below { background:var(--below-bg); color:#8a2f0a; }
.chip.within, .chip.unknown, .chip.info, .chip.mixed { background:var(--within-bg); color:#3d4450; }
.glance { list-style:none; padding:0; margin:0; } .glance li { padding:7px 0; border-top:1px solid var(--line); display:flex; gap:10px; align-items:baseline; }
.glance li:first-child { border-top:0; } .glance a { color:var(--ink); text-decoration:none; flex: 0 0 43%; font-weight:600; } .glance a:hover { text-decoration:underline; }
.glance .chips { display:flex; flex-wrap:wrap; gap:5px; }
@media (max-width: 640px) { .glance li { flex-direction:column; gap:4px; } }
.scroll { overflow-x:auto; } table { border-collapse:collapse; width:100%; font-size:13px; margin: 4px 0 12px; }
th { text-align:left; color:var(--muted); font-weight:600; border-bottom:1px solid var(--line); padding:5px 10px 5px 0; white-space:nowrap; }
td { padding:5px 10px 5px 0; border-bottom:1px solid var(--line); vertical-align:top; } td.num { font-variant-numeric: tabular-nums; }
figure { margin: 10px 0 6px; } img { max-width:100%; height:auto; border:1px solid var(--line); border-radius:8px; background:#fff; }
figcaption { color:var(--muted); font-size:12.5px; margin-top:4px; }
details { border-top:1px solid var(--line); margin-top:10px; padding-top:8px; } summary { cursor:pointer; font-weight:600; font-size:13.5px; }
.exchange { border-left:3px solid var(--line); padding:2px 0 2px 12px; margin:12px 0; }
.who { font-size:12.5px; color:var(--muted); margin:6px 0 2px; } blockquote { margin:0 0 4px; font-size:13.5px; white-space:pre-wrap; overflow-wrap:anywhere; }
.backing { font-size:13px; margin:4px 0 0; overflow-wrap:anywhere; } .tag { font-size:11.5px; font-weight:600; padding:1px 7px; border-radius:99px; margin-right:4px; }
.tag.ok { background:var(--ok-bg); color:var(--ok); } .tag.bare { background:var(--below-bg); color:#8a2f0a; }
code { background:var(--within-bg); padding:1px 5px; border-radius:4px; font-size:12.5px; }
footer { color:var(--muted); font-size:12.5px; padding-bottom:40px; }
"""


def run(in_dir, title=None, group="era", validation="validation"):
    d = Path(in_dir)
    ev = pd.read_parquet(d / "events.parquet", columns=["ts", "day", "actor", "actor_class", "room", "type"])
    msgs = ev[ev.type == "message"]
    agents = msgs[msgs.actor_class == "agent"]
    req, resp = _load_all(d, "requests"), _load_all(d, "responses")
    tiles = [(f"{len(msgs):,}", "messages"), (f"{agents.actor.nunique()}", "agents"), (f"{msgs.day.nunique()}", "active days"), (f"{msgs.room.nunique()}", "rooms")]
    if req is not None:
        tiles.append((f"{len(req):,}", "requests found"))
    if resp is not None:
        tiles.append((f"{len(resp):,}", "responses found"))

    n_perm = 1000
    if (d / f"report_{group}.csv").exists():
        rep = pd.read_csv(d / f"report_{group}.csv")
        if "permutations" in rep.columns:
            n_perm = int(rep.permutations.iloc[0])
    cards = network_cards(d, group) + [hierarchy_card(d, group)] + helping_cards(d) + [evidence_card(d), trends_card(d, group), quality_card(d, validation)]
    cards = [c for c in cards if c]
    glance = "".join(f'<li><a href="#{a}">{esc(t)}</a><span class="chips">{chips}</span></li>' for _, (t, chips, a) in cards)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Swarm report card</title><style>{CSS}</style></head><body>
<header><h1>Swarm report card</h1><p class="sub">{esc(title or d.name)} &middot; {esc(str(msgs.day.min()))} to {esc(str(msgs.day.max()))}</p>
<div class="tiles">{"".join(f'<div class="tile"><b>{v}</b><span>{esc(k)}</span></div>' for v, k in tiles)}</div></header>
<main>
<article class="card"><h2>At a glance</h2><p class="meaning">The questions worth asking of any group of agents. A pattern only counts if it beats a null model: the raw event stream is shuffled {n_perm:,} times in ways that keep how much each agent talks, and the statistic is recomputed on each shuffle. Two shuffles are used. One reassigns who spoke each message; the other reassigns whom each message addressed. They can disagree, and when they do the card says so.</p><ul class="glance">{glance}</ul></article>
{"".join(c for c, _ in cards)}
</main>
<footer>Generated by swarm-sna. Null models follow Bejder et al. 1998 and Farine 2017; dominance measures follow de Vries et al. 2006 and Shizuka &amp; McDonald 2012.</footer>
</body></html>"""
    out = d / "report_card.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1e6:.1f} MB)")
