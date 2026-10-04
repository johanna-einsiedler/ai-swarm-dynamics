"""`swarm-sna card`: the report card. One self-contained, interactive HTML page.

An intro to the data and its eras, the network over time, then the fixed
questions as cards: each with a yes-or-no verdict per era against a null model,
the numbers, a figure drawn in the browser (D3, inlined, no network needed) and
a drawer of the verbatim evidence behind the labels. It is built from whatever
the other commands have written to the directory; a question whose layer has
not been run says which command would answer it.

One null gives the verdicts (`--null`); the other is in each card's drawer, the
histograms switch between them, and the reciprocity card says why they can
disagree. Each statistic is set beside its published values in animal and human
networks (`<meta>/benchmarks.csv`), drawn on one scale where the definitions
allow it and tabled in a drawer. A group
(era) whose median day has fewer than `--min-agents` agents is left out
(`--thin drop`) or shown but not judged (`--thin show`): a network statistic on
four nodes has nothing to say. Eras get a marker and a name from
`<meta>/eras.csv` when it exists.
"""
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .diffusion import item_kind
from .figures import FAINT, INK, MUTED, NAMED_FAMILIES, NULL_FILL, OBSERVED, OTHER, SERIES
from .helping import SIZE_BINS, SIZE_LABELS, helpful, logit_fit
from .report import LABELS as STAT_LABELS

ALPHA = 0.05
MIN_AGENTS = 6
MIN_AD = 4
STATIC = Path(__file__).parent / "static"
DEFAULT_NULL = "target"
NULL_NAMES = {"target": "shuffling who was addressed", "speaker": "shuffling who spoke"}
NULL_MEANING = {
    "target": "keeps every agent's daily in- and out-volume and reassigns only whom each mention goes to, so it asks whether a pattern is more than partner choice",
    "speaker": "keeps every agent's daily message count and each message's targets and reassigns who said it, so it asks whether a pattern is more than who talks how much",
}
MARKERS = ["▲", "■", "●", "◆", "▼", "★", "✚", "⬟"]
NETWORK_CARDS = [
    ("reciprocity", "Is interaction more reciprocal than chance?", "Reciprocity: when A addresses B, does B address A back? Measured as the share of all mention weight that is matched in the opposite direction; 1 would mean every mention is returned in kind."),
    ("same_family_share", "Do agents favour their own kind?", "Homophily: do agents interact more with their own kind than with others? Kind here is the model family, the vendor. Measured as the share of mention weight that stays within a family; the null says what that share would be if family played no part."),
    ("partner_selectivity", "Do agents have preferred partners?", "Partner selectivity: does each agent concentrate its attention on a few partners, or spread it evenly? Measured as the average concentration of each agent's mentions over its partners (a Herfindahl index): 1/k when spread evenly over k partners, 1 when all go to one."),
    ("transitivity", "Do ties close into cliques?", "Transitivity, or clustering: if A talks to B and B talks to C, do A and C talk as well? Measured as the share of connected triples that close into a triangle, on the stronger half of ties. High means cliques; low means hubs with spokes, or an open structure."),
]
WORD = {"above": "above chance", "below": "below chance", "within": "within chance", "unknown": "not testable"}
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
    return "n/a" if pd.isna(p) else "p &lt; 0.002" if p < 0.002 else f"p = {p:.3f}"


def _outcome(observed, null_mean, p):
    if pd.isna(p) or pd.isna(observed):
        return "unknown"
    return "within" if p >= ALPHA else "above" if observed > null_mean else "below"


def _chip(outcome):
    return f'<span class="chip {outcome}">{CHIP_LABELS[outcome]}</span>'


def _tag(outcome, text):
    return f'<span class="chip {outcome}">{esc(text)}</span>'


def _answer(label, yes, outcome=None):
    """The binary verdict for one group: yes in blue; no in grey, or orange when the pattern is below chance."""
    cls = "yes" if yes else ("no below" if outcome == "below" else "no")
    return f'<span class="chip {cls}">{esc(label)}: {"yes" if yes else "no"}</span>'


def _fig(kind, grouped=False, **attrs):
    """A placeholder the page's script fills with a D3 figure; `grouped` ones re-render when the group filter changes."""
    extra = "".join(f' data-{k}="{esc(str(v))}"' for k, v in attrs.items())
    return f'<div class="fig" data-fig="{kind}"{" data-groups=1" if grouped else ""}{extra}></div>'


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
    return f'<article class="card" id="{anchor}"><h2>{esc(title)}</h2>{f"<p class=verdict>{verdict}</p>" if verdict else ""}{body}</article>'


def _links(text):
    """Escaped text with [label](url) turned into links."""
    return re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2">\1</a>', esc(text))


BENCH_PANELS = {  # per card: the statistics of ours a published value can sit beside, each with the axis it is drawn on
    "reciprocity": [("reciprocity", "share of interaction that is returned in kind")],
    "same_family_share": [("cross_kind_mixing", "mixing across kinds, as a share of what chance would give (1 = no preference)")],
    "hierarchy": [("steepness", "steepness (0 flat, 1 a strict ladder)"), ("landau_h", "linearity, Landau's h (1 = every triad ordered)"), ("triangle_transitivity", "triangle transitivity (1 = no cycles)")],
    "bystander": [("p_each_agent_responds", "a given one answers"), ("p_any_agent_responds", "anyone answers")],
    "helping": [("direct reciprocity", "direct reciprocity: has the asker helped this one lately?"), ("indirect reciprocity", "indirect reciprocity: is the asker known to help others?")],
}
BENCH_LAYOUT = {"bystander": "curves", "helping": "pairs"}  # the rest are ladders: one row per value, on one axis
BENCH_FORMAT = {"bystander": "pct", "helping": "pct"}


class Benchmarks:
    """<meta>/benchmarks.csv: published values of the same statistics in animal and human networks, per card.
    `table` is the file; `payload` collects, per card, the rows that can be drawn beside the agents' own values."""

    def __init__(self, table):
        for c in ("stat", "who", "short", "plot"):
            if c not in table.columns:
                table[c] = ""
        self.table = table
        self.payload = {}


def load_benchmarks(meta):
    path = Path(meta) / "benchmarks.csv" if meta else None
    return Benchmarks(pd.read_csv(path, dtype=str).fillna("")) if path is not None and path.exists() else None


def _points(spec, curve=False):
    """The `plot` column: '0.45' | 'third grade:0.67;middle school:0.10' | '0.258..0.776' -> points for the figure.
    On a curve the label is the group size, so it becomes x."""
    out = []
    for part in [x.strip() for x in spec.split(";") if x.strip()]:
        label, _, num = part.rpartition(":")
        if ".." in num:
            lo, hi = num.split("..")
            out.append({"label": label.strip(), "lo": float(lo), "hi": float(hi)})
        else:
            pt = {"label": label.strip(), "v": float(num)}
            if curve:
                pt["x"] = float(label)
            out.append(pt)
    return out


def _benchmarks(bench, card, agents_line="", ours=None):
    """The comparison section for one card: a paragraph (the agents' own numbers, then how far the published ones compare),
    the published values drawn on the same scale as the agents' where the definitions allow it, and the table in a drawer."""
    if bench is None:
        return ""
    b = bench.table[bench.table.card == card]
    if b.empty:
        return ""
    notes = " ".join(b[b.kind == "note"].measure)
    rows = b[b.kind == "value"]
    body = f'<p class="meaning">{(agents_line + " ") if agents_line else ""}{esc(notes)}</p>'
    layout = BENCH_LAYOUT.get(card, "ladder")
    drawn = [
        {"stat": r.stat, "who": r.who, "label": r.short or r.setting, "measure": r.measure, "value": r.value, "source": f"{r.source} {r.check}".strip(), "points": _points(r.plot, layout == "curves")}
        for r in rows.itertuples() if r.plot and r.stat
    ]
    if drawn and ours:
        bench.payload[card] = {"layout": layout, "format": BENCH_FORMAT.get(card, "num"), "panels": [{"key": k, "label": lb} for k, lb in BENCH_PANELS.get(card, [])], "rows": drawn, "ours": ours}
        body += _fig("bench", grouped=True, card=card)
    if len(rows):
        t = pd.DataFrame({"statistic": rows.measure, "setting": rows.setting, "value": rows.value, "source": [f"{s} {c}".strip() for s, c in zip(rows.source, rows.check)]})
        foot = '<p class="meaning">† taken from a secondary summary rather than the paper itself.</p>' if (rows.check == "†").any() else ""
        body += _drawer("The published values, as a table", _table(t) + foot)
    return "<h3>Compared with animal and human networks</h3>" + body


def _bin_x(label):
    """A group-size bin label ('5-8', '16+') as one number on an axis."""
    s = str(label)
    if "-" in s:
        a, b = s.split("-")
        return float(np.sqrt(float(a) * float(b)))
    return float(s.rstrip("+")) * 1.25 if s.endswith("+") else float(s)


def _missing(title, command, anchor):
    return _card(title, _chip("unknown") + f" This question needs a layer that has not been run. <code>{esc(command)}</code> would answer it.", "", anchor), (title, _tag("unknown", "not run"), anchor)


def _load_all(d, name):
    files = sorted(d.glob(f"{name}*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else None


def _thin_note(thin, groups, lab):
    hit = sorted(thin & set(groups))
    return f' {_chip("thin")} {esc(", ".join(lab(g) for g in hit))}: shown, not judged.' if hit else ""


class Labels:
    """Group name -> the name with its era marker in front, for chips, tables and sentences."""

    def __init__(self, groups):
        self.marker = {g["name"]: g.get("marker", "") for g in groups}

    def __call__(self, name):
        return f"{self.marker.get(name, '')} {name}".strip()


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
        out.append({"name": f"{group} {g}", "value": str(g), "agents": int(m.actor.nunique()), "median_active": float(daily.median()), "days": int(len(daily)), "messages": int(len(m)),
                    "first": str(m.day.min()), "last": str(m.day.max()), "thin": bool(daily.median() < min_agents)})
    return out


def load_eras(meta, group):
    """<meta>/<group>s.csv: one row per group value with marker, name, start, end, description. -> {value: row}"""
    path = Path(meta) / f"{group}s.csv" if meta else None
    if path is None or not path.exists():
        return {}
    e = pd.read_csv(path, dtype=str).fillna("")
    return {str(r[group]).strip(): r for r in e.to_dict("records")} if group in e.columns else {}


def networks_payload(msgs, mentions, agents, group, top=150):
    """The mention network per group: every node with its strengths, the strongest ties."""
    grp = msgs.set_index("event_id")[group]
    m = mentions[(mentions.actor_class == "agent") & mentions.kind.isin(("at", "name"))]
    m = m.assign(g=m.event_id.map(grp)).dropna(subset=["g"])
    m["g"] = m.g.astype(grp.dtype)  # mentions outside the kept groups map to NaN, which would turn an integer era into "2.0"
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
    agents in order of joining so the page can keep each one at a fixed place. `msgs` decides which weeks exist."""
    m = msgs.assign(week=_week(msgs.day))
    order = agents.sort_values("joined").agent.tolist()
    order += sorted(set(msgs.actor) - set(order))
    idx = {a: i for i, a in enumerate(order)}
    fam = dict(zip(agents.agent, agents.family))
    by_week = m.groupby("week")
    active = {w: sorted(idx[a] for a in g.actor.unique()) for w, g in by_week}
    messages = {w: {str(idx[a]): int(c) for a, c in g.actor.value_counts().items()} for w, g in by_week}
    era = {w: _clean(g[group].mode().iloc[0]) if g[group].notna().any() else None for w, g in by_week}
    kept = set(msgs.event_id)

    def edges(df, src, dst):
        e = df.groupby(["week", src, dst]).weight.sum().reset_index()
        e = e[(e[src] != e[dst]) & e[src].isin(idx) & e[dst].isin(idx)]
        return {w: [[idx[s], idx[t], round(float(v), 2)] for s, t, v in zip(g[src], g[dst], g.weight)] for w, g in e.groupby("week") if w in active}

    mm = mentions[(mentions.actor_class == "agent") & mentions.kind.isin(("at", "name")) & mentions.event_id.isin(kept)]
    mention_edges = edges(mm.assign(week=_week(mm.day)), "actor", "target")
    help_edges = None
    if req is not None and resp is not None:
        h = helpful(resp)
        h = h[h.actor_class == "agent"].merge(req[["request_id", "actor"]].rename(columns={"actor": "target"}), on="request_id")
        h = h[h.event_id.isin(kept)].assign(day=lambda x: x.event_id.map(events.set_index("event_id").day), weight=1.0).dropna(subset=["day"])
        help_edges = edges(h.assign(week=_week(h.day)), "actor", "target") if len(h) else None
    return {"periods": sorted(active), "agents": [{"id": a, "family": fam.get(a, "unknown")} for a in order], "active": active, "messages": messages, "era": era, "mentions": mention_edges, "help": help_edges}


# ---------- cards

def intro_card(intro, groups, msgs, group):
    """The data and its eras with their markers; a dropped era is simply not there."""
    if intro and Path(intro).exists():
        text = Path(intro).read_text()
        paras = text if intro.endswith(".html") else "".join(f"<h3>{esc(p.strip()[3:])}</h3>" if p.strip().startswith("## ") else f"<p>{_links(p.strip())}</p>" for p in text.split("\n\n") if p.strip())
    else:
        paras = f"<p>A transcript of {len(msgs):,} messages by {msgs.actor.nunique()} agents over {msgs.day.nunique()} active days, read as a social network: who addresses whom, who answers whom, who picks things up from whom.</p>"
    rows = pd.DataFrame(
        {
            "": [g.get("marker", "") for g in groups], group: [f"{g['name']}" + (f" · {g['label']}" if g.get("label") else "") for g in groups],
            "period": [f"{g['first']} to {g['last']}" for g in groups], "agents overall": [g["agents"] for g in groups], "agents a day": [f"{g['median_active']:.0f}" for g in groups],
            "messages": [g["messages"] for g in groups], "what changed": [g.get("description", "") for g in groups],
        }
    )
    return _card("The data", "", paras + f"<h3>The {group}s</h3>" + _table(rows), "data")


def null_text(n_perm, null):
    other = [k for k in NULL_NAMES if k != null][0]
    return (
        f'<p class="meaning">Every verdict on this card is observed against a null model: the raw event stream is shuffled {n_perm:,} times within room and day and the statistic is recomputed on each shuffle. '
        f"The verdicts use <b>{NULL_NAMES[null]}</b>, which {NULL_MEANING[null]}. The histograms switch to the second shuffle, <b>{NULL_NAMES[other]}</b>, which {NULL_MEANING[other]}. "
        f"The two hold different things fixed, so they can answer differently; each card says when they do, and the paragraph under the histograms says why.</p>"
    )


def _robust(main, alt, null, lab, explained=False):
    """One sentence on whether the other shuffle gives the same yes-or-no answers."""
    if alt.empty:
        return ""
    other = [k for k in NULL_NAMES if k != null][0]
    differ = []
    for r in main.itertuples():
        a = alt[alt.group == r.group]
        if len(a) and (a.outcome.iloc[0] == "above") != (r.outcome == "above"):
            differ.append((r.group, a.outcome.iloc[0]))
    if not differ:
        return f" The other shuffle, {NULL_NAMES[other]}, gives the same answer."
    where = " and ".join(f"{lab(g)} ({WORD[o]})" for g, o in differ)
    tail = "and the paragraph under the histograms says why" if explained else "and the reciprocity card says what each shuffle holds fixed"
    return f" The other shuffle, {NULL_NAMES[other]}, answers differently in {where}: its numbers are in the drawer below, {tail}."


def _why_shuffles(judged, alt, lab, nets, null):
    """Why the two shuffles can disagree, with reciprocity as the worked case: in the groups where shuffling who was
    addressed is beaten and shuffling who spoke is not, being addressed tracks writing more closely than it tracks
    addressing others, so handing out mentions in proportion to writing makes the shuffled networks more mutual than the real one."""
    intro = (
        '<p class="meaning"><b>Why the two shuffles can disagree.</b> Shuffling who was addressed keeps how many mentions each agent sends and receives and reassigns only whom they go to: '
        "beating it means pairs return each other's mentions beyond what their volumes predict. Shuffling who spoke keeps each message's mentions and each agent's message count: under it an agent sends mentions in proportion to how much it writes. "
    )
    cases = []
    for r in judged.itertuples():
        a = alt[alt.group == r.group]
        if a.empty:
            continue
        tv, sv = (r, a.iloc[0]) if null == "target" else (a.iloc[0], r)
        cases.append((r.group, tv, sv))
    if not cases:
        return intro + "</p>"
    split = [(g, tv, sv) for g, tv, sv in cases if tv.outcome == "above" and sv.outcome == "below"]
    if not split:
        agree = all((tv.outcome == "above") == (sv.outcome == "above") for _, tv, sv in cases)
        return intro + ("Here the two agree, so the answer does not turn on which is held fixed." if agree else "Here they answer differently; the drawer below has the second shuffle's numbers.") + "</p>"
    corr, ex = [], []
    for g, _, _ in split:
        net = nets.get(g) if nets else None
        if not net or len(net["nodes"]) < 4:
            continue
        nd = pd.DataFrame(net["nodes"])
        corr.append(f"in {lab(g)} the mentions an agent receives correlate {nd['in'].corr(nd.msgs):.2f} with its messages and {nd['in'].corr(nd.out):.2f} with the mentions it sends")
        top, swarm = nd.loc[nd["in"].idxmax()], (nd.out.sum() / nd.msgs.sum() if nd.msgs.sum() else 0)
        if top.msgs and top.out / top.msgs < swarm:
            ex.append(f"in {lab(g)} the most-addressed agent, {esc(str(top.id))}, received {top['in']:,.0f} mentions and sent {top.out / top.msgs:.1f} per message against {swarm:.1f} for the swarm")
    text = intro + "The two part ways when being addressed tracks writing more closely than it tracks addressing others" + (" (" + "; ".join(corr) + ")" if corr else "") + "."
    if ex:
        text += " For instance, " + "; ".join(ex) + "."
    more = "; ".join(f"{lab(g)}: {sv.null_mean:.3f} expected against {sv.observed:.3f} observed" for g, _, sv in split)
    less = " and ".join(f"{tv.null_mean:.3f}" for _, tv, _ in split)
    text += f" Handing out mentions in proportion to writing therefore makes the shuffled networks more mutual than the real one ({more}), while handing them out in proportion to mentions sent makes them less so ({less})."
    text += " Read together: mentions are returned beyond what volume predicts, but the agents everyone addresses do not address back as much as their talk would allow.</p>"
    return text


def network_cards(d, group, lab, drop, thin, null, n_perm, bench, nets=None):
    path = d / f"report_{group}.csv"
    if not path.exists():
        return [_missing(q, "swarm-sna report", s) for s, q, _ in NETWORK_CARDS]
    t = pd.read_csv(path)
    t = t[~t.group.isin(drop)]
    other = [k for k in NULL_NAMES if k != null][0]
    out = []
    for i, (stat, question, meaning) in enumerate(NETWORK_CARDS):
        s = t[(t.statistic == stat) & t.null.isin(NULL_NAMES)]
        s = s.assign(outcome=[_outcome(r.observed, r.null_mean, r.p) for r in s.itertuples()], thin=s.group.isin(thin))
        main, alt = s[s.null == null], s[s.null == other]
        judged = main[~main.thin]
        parts = [f"{_answer(lab(r.group), r.outcome == 'above', r.outcome)} &ndash; {WORD[r.outcome]}, {r.observed:.3f} against {r.null_mean:.3f} expected ({_p(r.p)})." for r in judged.itertuples()]
        verdict = " ".join(parts) if parts else _chip("unknown") + " No group left to judge."
        verdict += _robust(judged, alt, null, lab, explained=i == 0) + _thin_note(thin, main.group, lab)

        def rows(x):
            return pd.DataFrame(
                {
                    group: x.group.map(lab), "observed": x.observed, "null mean": x.null_mean, "null 95% band": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(x.null_lo, x.null_hi)],
                    "p": [_p(p).replace("p = ", "").replace("p &lt; ", "&lt; ") for p in x.p], "": [_chip("thin") if th else _chip(o) for o, th in zip(x.outcome, x.thin)],
                }
            )

        body = f'<p class="meaning">{esc(meaning)} Null: {esc(NULL_NAMES[null])}.</p>' + _table(rows(main), raw=("p", ""), classes=["thin" if th else "" for th in main.thin])
        if i == 0:
            body += null_text(n_perm, null) + _fig("nulls", grouped=True) + _why_shuffles(judged, alt, lab, nets, null)
        if i == 2:
            body += _fig("network", grouped=True)
        if len(alt):
            body += _drawer(f"Under the other shuffle: {NULL_NAMES[other]}", _table(rows(alt), raw=("p", ""), classes=["thin" if th else "" for th in alt.thin]))
        agents_line, ours = "", None
        if stat == "same_family_share" and len(judged):  # the human rows are observed over expected, so say ours the same way
            ratio = {r.group: (1 - r.observed) / (1 - r.null_mean) if r.null_mean < 1 else np.nan for r in judged.itertuples()}
            agents_line = "Written the human way, cross-family mentions run at " + ", ".join(f"{ratio[r.group]:.0%} of chance in {lab(r.group)}" for r in judged.itertuples()) + "."
            ours = {"cross_kind_mixing": [{"label": f"{lab(r.group)}: the agents, across model families", "group": r.group, "points": [{"v": ratio[r.group]}], "null": 1.0} for r in judged.itertuples() if pd.notna(ratio[r.group])]}
        elif stat == "reciprocity" and len(judged):
            agents_line = "The agents return " + ", ".join(f"{r.observed:.2f} of mention weight in {lab(r.group)} ({r.null_mean:.2f} expected)" for r in judged.itertuples()) + "."
            ours = {"reciprocity": [{"label": f"{lab(r.group)}: the agents, mentions", "group": r.group, "points": [{"v": r.observed}], "null": r.null_mean} for r in judged.itertuples()]}
        body += _benchmarks(bench, stat, agents_line, ours)
        chips = " ".join(_tag("thin", f"{lab(r.group)}: thin") if r.thin else _answer(lab(r.group), r.outcome == "above", r.outcome) for r in main.itertuples())
        out.append((_card(question, verdict, body, stat), (question, chips, stat)))
    return out


def hierarchy_card(d, group, lab, drop, thin, bench):
    title, anchor = "Is there a dominance hierarchy?", "hierarchy"
    path = d / f"hierarchy_{group}.csv"
    if not path.exists():
        return _missing(title, "swarm-sna label, then helping and hierarchy", anchor)
    t = pd.read_csv(path)
    main = t[t.group.str.startswith(group.split("_")[0]) & ~t.group.isin(drop)]
    st = main[main.statistic == "steepness"]
    outs = [_outcome(r.observed, r.null_mean, r.p) for r in st.itertuples()]
    parts = [f"{_answer(lab(r.group), o == 'above', o)} &ndash; steepness {r.observed:.3f} against {r.null_mean:.3f} expected ({_p(r.p)})." for r, o in zip(st.itertuples(), outs) if r.group not in thin]
    verdict = (" ".join(parts) if parts else _chip("unknown") + " No group left to judge.") + _thin_note(thin, st.group, lab)
    rows = pd.DataFrame(
        {
            group: main.group.map(lab), "statistic": main.statistic.str.replace("_", " "), "observed": main.observed, "null mean": main.null_mean,
            "null 95% band": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(main.null_lo, main.null_hi)], "p": [_p(p).replace("p = ", "").replace("p &lt; ", "&lt; ") for p in main.p],
            "": [_chip("thin") if g in thin else _chip(_outcome(r.observed, r.null_mean, r.p)) for g, r in zip(main.group, main.itertuples())],
        }
    )
    per = main.drop_duplicates("group")
    corr = pd.DataFrame({group: per.group.map(lab), "agents": per.agents, "directives": per.contests, "complied with": per.compliance, "rank vs release date": per.rank_vs_release_date, "rank vs message count": per.rank_vs_messages})
    body = '<p class="meaning">A dominance hierarchy is a consistent order of who gets their way. A contest here is a directive addressed to a named agent: the sender wins if the target complies, the target wins if it declines or ignores it. Each agent gets a David\'s score from its wins and losses, weighted by the strength of its opponents, and steepness (de Vries et al. 2006) is how unequal those scores are: 0 means everyone wins as often as everyone else, 1 a strict ladder. Null: outcomes shuffled among contests within room and day, which keeps who directs whom and the day\'s compliance rate and breaks only who gets obeyed.</p>'
    body += _table(rows, raw=("p", ""), classes=["thin" if g in thin else "" for g in main.group]) + _fig("hierarchy", grouped=True)
    body += "<h3>Does rank follow capability or talkativeness?</h3>" + _table(corr, {"complied with": ".0%", "rank vs release date": "+.2f", "rank vs message count": "+.2f"}, classes=["thin" if g in thin else "" for g in per.group])
    body += "<h3>Who ranks highest</h3>" + _fig("ranks", grouped=True)
    ranks_path = d / f"hierarchy_ranks_{group}.csv"
    if ranks_path.exists():
        r = pd.read_csv(ranks_path)
        r = r[~r.group.isin(drop)]
        top = r.groupby("group", sort=False).head(5)[["group", "agent", "davids_score", "directives_sent", "obeyed", "directives_received", "messages"]].assign(group=lambda x: x.group.map(lab))
        body += _drawer("The top five per group and episode, as a table", _table(top, {"davids_score": ".2f", "obeyed": ".0f"}))
    episodes = t[~t.group.str.startswith(group.split("_")[0]) & (t.statistic == "steepness")]
    if len(episodes):
        e = pd.DataFrame({"episode | window": episodes.group, "steepness": episodes.observed, "null mean": episodes.null_mean, "p": [_p(p) for p in episodes.p], "directives": episodes.contests})
        body += _drawer("Before, during and after an imposed leader", _table(e, raw=("p",)))
    tt = main[main.statistic == "triangle_transitivity"]
    agents_line = "The agents: steepness " + ", ".join(f"{r.observed:.3f} in {lab(r.group)}" for r in st.itertuples()) + (("; triangle transitivity " + ", ".join(f"{r.observed:.2f} in {lab(r.group)}" for r in tt.itertuples())) if len(tt) else "") + "."
    ours = {}
    for key, _ in BENCH_PANELS["hierarchy"]:
        k = main[(main.statistic == key) & ~main.group.isin(thin)]
        if len(k):
            ours[key] = [{"label": f"{lab(r.group)}: the agents, directives", "group": r.group, "points": [{"v": r.observed}], "null": r.null_mean} for r in k.itertuples()]
    body += _benchmarks(bench, "hierarchy", agents_line, ours or None)
    chips = " ".join(_tag("thin", f"{lab(g)}: thin") if g in thin else _answer(lab(g), o == "above", o) for g, o in zip(st.group, outs))
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


FACTORS = [  # the four things that could explain who answers an ask addressed to nobody in particular
    ("direct reciprocity", "the asker answered this agent in the last 7 days"),
    ("indirect reciprocity", "the asker answered others in the last 7 days"),
    ("cost", "the agent was busy in a work session"),
    ("bystanders", "how many agents were active in the room that day"),
]


def undirected_asks(d, group, drop):
    """Among asks addressed to nobody in particular: the answer rate by each factor, and the share of the
    uncertainty about who answers that each factor explains (McFadden's R-squared of a logistic model)."""
    p = d / "dyads.parquet"
    if not p.exists():
        return None
    dy = pd.read_parquet(p)
    dy = dy[dy.broadcast & ~dy.era.map(lambda e: f"{group} {e}").isin(drop)]
    if len(dy) < 200 or dy.responded.nunique() < 2:
        return None
    y = dy.responded.to_numpy(float)
    F = pd.DataFrame({"direct reciprocity": np.log1p(dy.prior_help_from_requester), "indirect reciprocity": np.log1p(dy.requester_reputation), "cost": dy.responder_busy.astype(float), "bystanders": np.log(dy.active_n.clip(lower=1))})

    def loss(cols):
        X = np.c_[np.ones(len(dy)), F[cols].to_numpy()]
        q = np.clip(1 / (1 + np.exp(-X @ logit_fit(X, y))), 1e-6, 1 - 1e-6)
        return -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()

    base = loss([])
    r2 = {c: 1 - loss([c]) / base for c in F.columns}
    r2["all four"] = 1 - loss(list(F.columns)) / base
    levels = {
        "direct reciprocity": (dy.prior_help_from_requester > 0).map({False: "no", True: "yes"}),
        "indirect reciprocity": pd.cut(dy.requester_reputation, [-1, 0, 4, np.inf], labels=["0 answers", "1 to 4", "5 or more"]).astype(str),
        "cost": dy.responder_busy.map({False: "not busy", True: "busy"}),
        "bystanders": pd.cut(dy.active_n, SIZE_BINS, labels=SIZE_LABELS).astype(str),
    }
    rates = []
    for factor, label in FACTORS:
        g = dy.groupby(levels[factor].to_numpy()).responded.agg(["mean", "size"])
        order = {"direct reciprocity": ["no", "yes"], "indirect reciprocity": ["0 answers", "1 to 4", "5 or more"], "cost": ["not busy", "busy"], "bystanders": list(SIZE_LABELS)}[factor]
        rates += [{"factor": factor, "label": label, "level": lv, "rate": float(g.loc[lv, "mean"]), "n": int(g.loc[lv, "size"])} for lv in order if lv in g.index]
    return {"n": int(len(dy)), "requests": int(dy.request_id.nunique()), "rate": float(y.mean()), "r2": {k: float(v) for k, v in r2.items()}, "rates": rates}


def helping_cards(d, group, lab, drop, bench, und):
    t1, a1, t2, a2 = "What explains who answers whom?", "helping", "Is there a bystander effect?", "bystander"
    if not (d / "helping_ladder.csv").exists():
        return [_missing(t1, "swarm-sna label, then helping", a1), _missing(t2, "swarm-sna label, then helping", a2)]
    ladder = pd.read_csv(d / "helping_ladder.csv")
    summary = json.loads((d / "helping_summary.json").read_text()) if (d / "helping_summary.json").exists() else {}
    rungs = ladder[ladder.model != "no covariates"]
    first, last = rungs.iloc[0], rungs.iloc[-2]
    rows = pd.DataFrame({"model": rungs.model, "adds": rungs.model.map(LADDER_NOTE), "log loss": rungs.log_loss, "completeness": rungs.completeness, "95% CI": [f"[{a:.2f}, {b:.2f}]" for a, b in zip(rungs.ci_lo, rungs.ci_hi)]})
    named = f' An agent addressed by name answers {summary["addressed_rate"]:.0%} of the time; one not addressed, {summary["not_addressed_rate"]:.0%}.' if summary else ""
    if und:
        r2 = und["r2"]
        verdict = (
            f"Being asked by name is what matters most.{named} Among the asks addressed to nobody in particular, a given agent present answers <b>{und['rate']:.0%}</b> of the time, and whether the asker had answered it lately, whether the asker answers others, "
            f"whether it was busy and how many others were present together explain <b>{r2['all four']:.0%}</b> of who answers."
        )
        body = '<p class="meaning">"Explain" here is the share of the uncertainty about who answers an undirected ask that a factor removes (McFadden\'s R-squared of a logistic model, 0 when knowing the factor helps not at all, 1 when it settles the matter). The bars give the plain answer rates behind it: one row per undirected ask and agent present, in the eras on this card.</p>'
        body += _fig("help_rates")
        r2t = pd.DataFrame({"factor": [f for f, _ in FACTORS] + ["all four together"], "which is": [text for _, text in FACTORS] + [""], "share of who answers explained": [r2[f] for f, _ in FACTORS] + [r2["all four"]]})
        body += _table(r2t, {"share of who answers explained": ".1%"})
        body += f'<p class="meaning">{und["requests"]:,} undirected asks, {und["n"]:,} ask-agent pairs.</p>'
        body += _drawer("The full model ladder, scored out of sample against a lookup table", _table(rows, {"log loss": ".4f", "completeness": ".2f"}))
        chip = _tag("info", f"four factors explain {r2['all four']:.0%} of who answers undirected asks")
    else:
        verdict = f"Being addressed by name reaches <b>{first.completeness:.0%}</b> of what a lookup table can explain; adding reciprocity, cost and bystanders reaches <b>{last.completeness:.0%}</b> [{last.ci_lo:.0%}, {last.ci_hi:.0%}].{named}"
        body = '<p class="meaning">A ladder of explanations, each adding one thing to the ones before: was the request addressed to you, has the asker helped you lately (direct reciprocity), does the asker help others (indirect reciprocity), were you busy (cost), how many others were present (bystanders). Nested logistic models, scored out of sample. Completeness is how much of the gain a lookup table of every covariate combination achieves each rung reaches.</p>'
        body += _fig("ladder") + _table(rows, {"log loss": ".4f", "completeness": ".2f"})
        chip = _tag("info", f"model reaches {last.completeness:.0%} of the ceiling")
    if (d / "helping_coefficients.csv").exists():
        c = pd.read_csv(d / "helping_coefficients.csv", index_col=0).T.rename(columns=FEATURE_NAMES).reset_index(names="fit")
        note = f'<p class="meaning">Log-odds coefficients of the full model, refitted each way a pooled fit could mislead. {esc(summary.get("top_asker", "One agent"))} alone asks {summary.get("top_asker_share", 0):.0%} of all requests, hence the equal-weight row.</p>' if summary else ""
        body += _drawer("Does it hold up? Per era, within goal, with heavy askers down-weighted", note + _table(c, {k: "+.2f" for k in c.columns}))
    _, ex = _examples(d)
    if ex is not None and len(ex):
        body += _drawer(f"Evidence: {len(ex)} request-and-answer pairs, verbatim", "".join(_exchange(r) for r in ex.itertuples()))
    ours = None
    if und:
        lv = {(r["factor"], r["level"]): r for r in und["rates"]}

        def pair(factor, a, b, label):
            return [{"label": label, "points": [{"label": a, "v": lv[factor, a]["rate"]}, {"label": b, "v": lv[factor, b]["rate"]}]}] if (factor, a) in lv and (factor, b) in lv else []

        ours = {
            "direct reciprocity": pair("direct reciprocity", "no", "yes", "the agents: asker answered this one lately?"),
            "indirect reciprocity": pair("indirect reciprocity", "0 answers", "5 or more", "the agents: asker answered others lately?"),
        }
    body += _benchmarks(bench, "helping", "", ours)
    cards = [(_card(t1, verdict, body, a1), (t1, chip, a1))]

    curve = pd.read_csv(d / "helping_bystander.csv")
    curve = curve[~curve.era.map(lambda e: f"{group} {e}").isin(drop)]
    if curve.empty:
        cards.append((_card(t2, _tag("unknown", "not testable") + " This transcript has too few broadcast requests (ones addressed to nobody in particular) to test it.", "", a2), (t2, _tag("unknown", "not testable"), a2)))
        return cards
    s, lo, hi = summary.get("bystander_slope", [np.nan] * 3)
    o = "unknown" if np.isnan(s) else "within" if lo <= 0 <= hi else "below" if hi < 0 else "above"
    yes = o == "below"
    label = {"below": "yes: each agent answers less in a larger group", "above": "no: each agent answers more in a larger group", "within": "no: no clear effect of group size", "unknown": "slope not computed"}[o]
    verdict = _tag("yes" if yes else "no", label) + (f" Slope of the log-odds that an agent answers on log group size, within goal: {s:.2f} [{lo:.2f}, {hi:.2f}]." if o != "unknown" else "")
    rows = curve.rename(columns={"size": "agents present", "p_each_agent_responds": "a given agent answers", "p_any_agent_responds": "anyone answers"}).assign(era=lambda x: x.era.map(lambda e: lab(f"{group} {e}")))
    body = '<p class="meaning">The bystander effect: the more others are present, the less likely any one of them is to help, even if the group as a whole still does. Broadcast requests only (asked of nobody in particular), compared within era, because group size and era are confounded across eras. Two curves: the chance that a given agent answers, and the chance that anyone does.</p>' + _fig("bystander", grouped=True)
    body += _drawer("The curve as a table", _table(rows, {"a given agent answers": ".0%", "anyone answers": ".0%"}))
    kept = []  # the agents' own numbers, the way the laboratory reports them: the share of the small-group rate kept in the largest group
    ours = {key: [] for key, _ in BENCH_PANELS["bystander"]}
    for e, c in curve.groupby("era", sort=True):
        c = c.sort_values("size", key=lambda s: s.map({k: i for i, k in enumerate(SIZE_LABELS)}))
        a, b = c.p_each_agent_responds.iloc[0], c.p_each_agent_responds.iloc[-1]
        kept.append(f"{lab(f'{group} {e}')} a given agent answers {a:.0%} of broadcast requests at {c['size'].iloc[0]} agents and {b:.0%} at {c['size'].iloc[-1]} ({b / a:.0%} kept), while anyone answers {c.p_any_agent_responds.iloc[0]:.0%} and {c.p_any_agent_responds.iloc[-1]:.0%}")
        for key in ours:
            pts = [{"label": f"{r.size} agents active", "x": _bin_x(r.size), "v": float(getattr(r, key))} for r in c.itertuples() if pd.notna(getattr(r, key))]
            if pts:
                ours[key].append({"label": f"{lab(f'{group} {e}')}: the agents, broadcast requests", "group": f"{group} {e}", "points": pts})
    body += _benchmarks(bench, "bystander", "In " + "; in ".join(kept) + "." if kept else "", {k: v for k, v in ours.items() if v} or None)
    cards.append((_card(t2, verdict, body, a2), (t2, _tag("yes" if yes else "no", label), a2)))
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


def adoption_payload(d, order, msgs):
    """Every adoption of every item (diffusion_adoptions.csv): the replay, and the general patterns read off it."""
    p = d / "diffusion_adoptions.csv"
    if not p.exists():
        return None
    ad = pd.read_csv(p)
    if "agent" in ad.columns:  # item, order, agent, first_use, first_use_event_id; item names masked where they looked like credentials
        ad = ad.rename(columns={"agent": "actor", "first_use": "ts"})
    ad["ts"] = pd.to_datetime(ad.ts)
    if "kind" not in ad.columns:
        ad["kind"] = ad.item.map(item_kind)
    idx = {a: i for i, a in enumerate(order)}
    ad = ad[ad.actor.isin(idx)].sort_values(["item", "ts"])
    if ad.empty:
        return None
    ad["days"] = (ad.ts - ad.groupby("item").ts.transform("min")).dt.total_seconds() / 86400
    ad["rank"] = ad.groupby("item").cumcount()
    n = ad.groupby("item").actor.transform("size")
    items = ad.groupby("item").agg(kind=("kind", "first"), adopters=("actor", "size"), first=("ts", "min"), days=("days", "max"), first_adopter=("actor", "first")).reset_index()
    second = ad[ad["rank"] == 1].groupby("item").days.min()
    half = ad[(ad["rank"] + 1) * 2 >= n].groupby("item").days.min()
    curve = []
    for t in [0, 0.5, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90]:
        s = (ad.days <= t).groupby(ad.item).mean()
        curve.append({"day": t, "median": float(s.median()), "lo": float(s.quantile(0.25)), "hi": float(s.quantile(0.75))})
    firsts = items.first_adopter.value_counts()
    vol = msgs.actor.value_counts(normalize=True)
    originators = [{"agent": a, "first": int(c), "share_first": float(c / len(items)), "share_msgs": float(vol.get(a, 0))} for a, c in firsts.head(10).items()]
    seq = {it: [[idx[a], round(float(dd), 2)] for a, dd in zip(g.actor, g.days)] for it, g in ad.groupby("item")}
    shown = items.assign(first=items["first"].dt.strftime("%Y-%m-%d"), days=items.days.round(1)).sort_values(["adopters", "days"], ascending=[False, True])
    return {
        "items": _records(shown), "seq": seq, "curve": curve, "kinds": {k: int(v) for k, v in items.kind.value_counts().items()}, "originators": originators,
        "speed": {"second": float(second.median()), "half": float(half.median()), "all": float(items.days.median()), "within_day": float((items.days <= 1).mean()), "within_week": float((items.days <= 7).mean())},
    }


def lead_vs_rank(d, group, drop, agents):
    """Each agent's lead score in adoption against its David's score in the dominance hierarchy, mean over the eras on the card."""
    lp, rp = d / "diffusion_leaders.csv", d / f"hierarchy_ranks_{group}.csv"
    if not (lp.exists() and rp.exists()):
        return None
    lead, r = pd.read_csv(lp), pd.read_csv(rp)
    r = r[r.group.str.startswith(group.split("_")[0]) & ~r.group.isin(drop)]
    ds = r.groupby("agent").davids_score.mean()
    m = lead[lead.agent.isin(ds.index)].assign(davids_score=lambda x: x.agent.map(ds))
    if len(m) < 4:
        return None
    fam = dict(zip(agents.agent, agents.family))
    return {
        "rho": float(m.lead_score.rank().corr(m.davids_score.rank())), "n": int(len(m)),
        "points": [{"agent": a, "family": fam.get(a, "unknown"), "lead": float(lv), "rank": float(s), "messages": int(mm)} for a, lv, s, mm in zip(m.agent, m.lead_score, m.davids_score, m.messages)],
    }


def _spell_days(days):
    """0.04 days is 'about an hour', 0.3 days '7 hours', 1.0 day and more stays in days."""
    hours = days * 24
    return "about an hour" if hours < 1.5 else f"{hours:.0f} hours" if days < 1 else f"{days:.1f} days"


def diffusion_card(d, ad, lr):
    title, anchor = "How does information spread?", "diffusion"
    if not (d / "diffusion_items.csv").exists():
        return _missing(title, "swarm-sna diffusion", anchor)
    if ad is None and pd.read_csv(d / "diffusion_items.csv").empty:
        verdict = _tag("unknown", "not testable") + f" No link, file name or backticked term spread to {MIN_AD}+ agents in this transcript."
        return _card(title, verdict, "", anchor), (title, _tag("unknown", "not testable"), anchor)
    chips = []
    body = '<p class="meaning">An item is something distinctive one agent can pick up from another: a link, a file name, a backticked term. For each item the agents are ordered by first use, and that order is its spread. Every agent sees every message in a shared room, so this is diffusion of attention, not of access: who picks a thing up, and how soon after whom.</p>'
    if ad:
        sp, kinds = ad["speed"], ad["kinds"]
        verdict = (
            f"<b>{len(ad['items']):,}</b> items spread to {MIN_AD}+ agents over the whole record ({', '.join(f'{v:,} {k}s' for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]))}). "
            f"The median item reaches a second adopter <b>{_spell_days(sp['second'])}</b> after the first and half of its adopters within <b>{_spell_days(sp['half'])}</b>; {sp['within_week']:.0%} of items have finished spreading within a week."
        )
        if ad["originators"]:
            top = ad["originators"][0]
            verdict += f" {esc(top['agent'])} is the first adopter most often: {top['share_first']:.0%} of items, against {top['share_msgs']:.0%} of messages."
        body += "<h3>The general pattern</h3>" + _fig("diffusion_patterns")
        orig = pd.DataFrame(ad["originators"]).rename(columns={"first": "items first", "share_first": "share of first adoptions", "share_msgs": "share of messages"})
        body += _drawer("Who is first most often", _table(orig, {"share of first adoptions": ".0%", "share of messages": ".0%"}))
        body += "<h3>Watch one item spread</h3>"
        body += '<p class="meaning">Pick an item and play its adoption in order. Each agent lights up when it first uses the item, numbered by its place in the sequence; the ties drawn are the mentions, over the whole record, between the agents that already have it.</p>' + _fig("diffusion_run")
        chips.append(_tag("info", f"half of an item's adopters within {sp['half']:.1f} days"))
    else:
        verdict = "The per-item adoption sequences are not in this directory: re-run <code>swarm-sna diffusion</code>, which now writes diffusion_adoptions.csv, to get the patterns and the replay."

    # The adoption network: who picks things up first, who follows (diffusion_leaders.csv, diffusion_edges.csv).
    lead_path, sm_path = d / "diffusion_leaders.csv", d / "diffusion_summary.csv"
    if lead_path.exists() and sm_path.exists():
        lead, sm = pd.read_csv(lead_path), pd.read_csv(sm_path).iloc[0]
        if len(lead) and "steepness" in sm and pd.notna(sm.steepness):
            beyond = sm.steepness_volume_p < ALPHA and sm.steepness > sm.steepness_volume_null
            steep = "steeper than volume alone predicts" if beyond else "no steeper than volume alone predicts"
            body += "<h3>Who picks things up first, and who follows</h3>"
            body += f'<p class="meaning">Adoption defines its own network. For each item, every adopter after the first hands one unit of credit, split evenly, to the agents who had it before. An agent\'s lead score is (credit received &minus; credit given) / total: +1 is always first, &minus;1 always after others. Across {int(sm.network_items)} items the leader-follower ordering is <b>{steep}</b> (steepness {sm.steepness:.3f}; null with order drawn in proportion to posting volume {sm.steepness_volume_null:.3f}, p = {sm.steepness_volume_p:.3f}; plain random order {sm.steepness_random_null:.3f}, p = {sm.steepness_random_p:.3f}).</p>'
            body += _fig("leaders") + _fig("diffusion_network")
            if (d / "diffusion_correlates.csv").exists():
                c = pd.read_csv(d / "diffusion_correlates.csv").rename(columns={"Unnamed: 0": "leading, measured as"})
                body += '<p class="meaning">Does leading go with capability, talkativeness or rank? Rank correlations across agents. The two nulls bracket the truth: plain random order ignores that heavy posters reach everything sooner; volume-weighted order assumes an agent posting ten times as much adopts ten times sooner.</p>'
                body += _table(c, {k: "+.2f" for k in c.columns if k.startswith("vs")})
            show = lead.sort_values("lead_score", ascending=False)[["agent", "led", "followed", "lead_score", "z_random_order", "z"]].rename(columns={"lead_score": "lead score", "z_random_order": "z, random order", "z": "z, volume-weighted"})
            body += _drawer("Every agent's lead score", _table(show, {"led": ".0f", "followed": ".0f", "lead score": "+.2f", "z, random order": "+.1f", "z, volume-weighted": "+.1f"}))
            chips.append(_tag("yes" if beyond else "no", "leaders beyond volume: " + ("yes" if beyond else "no")))
    if lr:
        rho = lr["rho"]
        reading = "leading in adoption goes with rank in the hierarchy." if rho > 0.3 else "the agents that lead in adoption rank low in the hierarchy." if rho < -0.3 else "leading in adoption and rank in the hierarchy are largely separate things."
        body += "<h3>Do the agents that lead in adoption rank high in the dominance hierarchy?</h3>"
        body += f'<p class="meaning">Each agent\'s lead score in adoption against its David\'s score from the directive contests, averaged over the eras on this card. Spearman rank correlation <b>{rho:+.2f}</b> over {lr["n"]} agents: {reading}</p>' + _fig("lead_vs_rank")
        chips.append(_tag("info", f"leaders vs dominance rank: ρ = {rho:+.2f}"))
    return _card(title, verdict, body, anchor), (title, " ".join(chips), anchor)


def _join(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def goal_type_section(d, group, lab, drop):
    """Response rates by the kind of goal, within era (trends_goal_type.csv), as a drawer: with `swarm-sna trends`
    run after this version the table carries day-bootstrap intervals and, per kind, the odds that a given agent answers
    against the era's commonest kind with group size and being addressed held fixed; the paragraph reads those off."""
    p = d / "trends_goal_type.csv"
    if not p.exists():
        return ""
    g = pd.read_csv(p)
    g = g[~g.era.map(lambda e: f"{group} {e}").isin(drop)]
    if g.empty:
        return ""
    label = lambda e: lab(f"{group} {e}")  # noqa: E731
    title = "Response rates by the kind of goal the swarm was given"
    if "odds_vs_main" not in g.columns:
        t = g.assign(era=g.era.map(label)).rename(columns=lambda c: c.replace("_", " "))
        return _drawer(title, '<p class="meaning">Plain rates, with no interval or test behind them: re-run <code>swarm-sna trends</code> for both.</p>' + _table(t, {c: ".0%" for c in t.columns if "respond" in c or "answered" in c}))

    def ci(r, c):
        return f"{r[c]:.0%} [{r[c + '_lo']:.0%}, {r[c + '_hi']:.0%}]" if pd.notna(r[c]) else ""

    recs = g.to_dict("records")
    rows = pd.DataFrame(
        {
            group: g.era.map(label).str.replace(" ", "\u00a0"), "kind of goal": g.goal_type, "requests": g.requests.astype(int), "agents a day": g.agents_a_day.round(0).astype(int),
            "anyone answers": [ci(r, "answered_by_anyone") for r in recs], "a given agent answers": [ci(r, "each_agent_responds") for r in recs], "the named agent answers": [ci(r, "addressed_agent_responds") for r in recs],
            "odds vs the commonest kind": [f"{r['odds_vs_main']:.2f} [{r['odds_lo']:.2f}, {r['odds_hi']:.2f}]" if pd.notna(r["odds_vs_main"]) else "reference" for r in recs],
        }
    )
    sentences = []
    for era, te in g.groupby("era", sort=True):
        main = te.main.iloc[0]
        base = te[te.goal_type == main].iloc[0]
        others = te[(te.goal_type != main) & te.odds_vs_main.notna()]
        if others.empty:
            sentences.append(f"{label(era)} had one kind of goal ({main}) throughout, so the kind of goal cannot be told apart from everything else that changed with the {group}.")
            continue
        same = [r for r in others.itertuples() if r.odds_lo <= 1 <= r.odds_hi]
        diff = [r for r in others.itertuples() if not (r.odds_lo <= 1 <= r.odds_hi)]
        say = lambda r: f"{r.goal_type} goals (odds {r.odds_vs_main:.2f} [{r.odds_lo:.2f}, {r.odds_hi:.2f}], {int(r.requests):,} requests)"  # noqa: E731
        parts = []
        if same:
            parts.append("no differently under " + _join([say(r) for r in same]))
        if diff:
            parts.append(_join([("slightly " if 0.8 < r.odds_vs_main < 1.25 else "") + ("less" if r.odds_vs_main < 1 else "more") + " often under " + say(r) for r in diff]))
        s = f"In {label(era)}, against {main} goals ({int(base.requests):,} requests), a given agent answers " + " and ".join(parts) + ", once group size and being addressed are held fixed."
        sized = [r for r in others.itertuples() if base.agents_a_day and abs(r.agents_a_day / base.agents_a_day - 1) > 0.25]
        if sized:
            s += " The plain rates in the table move with group size as much as with the goal: " + _join([f"{r.goal_type} days had {r.agents_a_day:.0f} agents a day" for r in sized]) + f" against {base.agents_a_day:.0f} under {main} goals."
        sentences.append(s)
    note = '<p class="meaning">Rates with 95% intervals from resampling days within each kind of goal. The last column is the odds that a given agent answers under that kind of goal against the era\'s commonest kind, from a logistic model with group size (log) and being addressed by name held fixed.</p>'
    return _drawer(title, f'<p class="meaning">{" ".join(sentences)}</p>' + _table(rows) + note)


def time_card(d, has_timeline, group, lab, drop):
    title, anchor = "The network over time", "time"
    if not has_timeline:
        return _missing(title, "swarm-sna extract", anchor)
    body = '<p class="meaning">Every agent is assigned a place on the circle in the order of joining, so a tie across the circle is a tie between an old agent and a new one. The network is built from mentions: an agent naming another in a message, with @ or by name; the second button switches to who answers whose requests. The slider moves through the weeks, and hollow dots are agents not active in the window.</p>'
    body += "<h3>Who mentions whom, week by week</h3>" + _fig("timeline") + goal_type_section(d, group, lab, drop)
    return _card(title, "", body, anchor), (title, _tag("info", "start here"), anchor)


WATCH = [  # indicator: its name, what a sharp move from the baseline would look like, what that could point to
    ("reciprocity", "mentions returned", "a jump well beyond both shuffles, carried by a few pairs", "closed loops: pairs that mainly address each other"),
    ("same_family_share", "own-kind share", "rising above chance in every era", "coordination along vendor lines"),
    ("partner_selectivity", "partner selectivity", "rising above its null across the swarm", "attention narrowing to a few partners"),
    ("transitivity", "clique closure", "rising above its null", "a subset that talks mostly to itself"),
    ("steepness", "dominance steepness", "a steep ladder appearing, compliance concentrated on one sender", "one agent directing, a subset complying"),
    ("answering", "answering", "anyone-answers falling while the room is stable, or falling for particular askers", "requests from outside a group going unanswered"),
    ("backed", "backed claims", "the backed share falling", "more claims of work done that cannot be checked"),
    ("spread", "spread of items", "a fixed few always first and the same set following within minutes while the rest never adopt", "coordinated adoption inside a subset"),
    ("leaders", "who leads adoption", "the leader set steepening beyond the volume null and lining up with the dominance order", "a subset steering what the others pick up"),
]


def _backed_share(d):
    resp = _load_all(d, "responses")
    if resp is None or "backed" not in resp.columns:
        return None
    done = helpful(resp)
    return float(done.backed.mean()) if len(done) else None


def baseline_card(data, group, lab, null, backed):
    """The card read as a baseline: today's value of each indicator beside the movement from it that would be worth a look."""
    title, anchor = "Reading this card as a baseline", "baseline"
    now = {}
    N = data.get("nulls")
    if N:
        for key in ("reciprocity", "same_family_share", "partner_selectivity", "transitivity"):
            rows = [r for r in N["table"] if r["statistic"] == key and r["null"] == null]
            if rows:
                now[key] = "; ".join(f"{lab(r['group'])} {r['observed']:.3f} ({r['null_mean']:.3f} by chance)" for r in rows)
    H = data.get("hierarchy")
    if H:
        rows = [r for r in H["table"] if r["statistic"] == "steepness" and str(r["group"]).startswith(group.split("_")[0])]  # the eras, not the leader episodes
        if rows:
            now["steepness"] = "; ".join(f"{lab(r['group'])} {r['observed']:.3f} ({r['null_mean']:.3f} by chance)" for r in rows)
    Hp = data.get("helping")
    if Hp and Hp.get("bystander"):
        parts = []
        for e, c in pd.DataFrame(Hp["bystander"]).groupby("era", sort=True):
            c = c.sort_values("size", key=lambda s: s.map({k: i for i, k in enumerate(SIZE_LABELS)}))
            parts.append(f"{lab(f'{group} {e}')} anyone answers {c.p_any_agent_responds.iloc[0]:.0%} to {c.p_any_agent_responds.iloc[-1]:.0%} and a given agent {c.p_each_agent_responds.iloc[0]:.0%} to {c.p_each_agent_responds.iloc[-1]:.0%} as the room grows from {c['size'].iloc[0]} to {c['size'].iloc[-1]} agents")
        now["answering"] = "; ".join(parts)
    if backed is not None:
        now["backed"] = f"{backed:.0%} of answers and reported actions carry something checkable"
    Df = data.get("diffusion")
    if Df and Df.get("adoptions"):
        A, sp = Df["adoptions"], Df["adoptions"]["speed"]
        top = A["originators"][0] if A["originators"] else None
        now["spread"] = f"{len(A['items']):,} items; a second adopter after {_spell_days(sp['second'])}, half of the adopters within {_spell_days(sp['half'])}" + (f"; {esc(top['agent'])} is first for {top['share_first']:.0%} of items" if top else "")
    sm = Df.get("summary") if Df else None
    if sm and sm.get("steepness") is not None:
        now["leaders"] = f"leader-follower steepness {sm['steepness']:.3f} against {sm['steepness_volume_null']:.3f} with the order drawn by posting volume (p = {sm['steepness_volume_p']:.3f})" + (f"; lead score against dominance rank, Spearman {Df['lead_rank']['rho']:+.2f}" if Df.get("lead_rank") else "")
    rows = [(name, now[key], move, means) for key, name, move, means in WATCH if key in now]
    if not rows:
        return None
    body = (
        '<p class="meaning">Nothing on this card is a verdict on the swarm\'s conduct: the numbers are what this swarm looks like going about its business, with the band of chance drawn around each, and that makes them a baseline. '
        "A later run of the same swarm, or a new swarm under the same scaffolding, that moves sharply in one of the directions below is worth a look, because that is what coordination inside a subset of agents, or one agent capturing the others, would do to these statistics.</p>"
    )
    body += _table(pd.DataFrame(rows, columns=["indicator", "now", "a sharp move would look like", "which could point to"]))
    body += (
        '<p class="meaning">The yardstick is the null band and the range across eras, not the number itself: the eras on this card show how far roster size and the kind of goal move the same statistics for benign reasons, so a shift counts within a regime, outside the band of its own shuffle, and with no matching entry in the shock log. '
        "And the card reads structure, not content: coordination written into the text of messages, in channels the card is not given, or in the timing of messages is beyond it, and it does not yet break answering down by who asked or test whether the same subset adopts items together. Those are the next indicators to add.</p>"
    )
    return _card(title, "", body, anchor), (title, _tag("info", "what to watch"), anchor)


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


def payload(d, group, groups, msgs, events, agents, mentions, req, resp, drop, null):
    fam_present = set(agents.family.dropna())
    family_order = [f for f in NAMED_FAMILIES if f in fam_present]
    for f, _ in agents.family.value_counts().items():
        if len(family_order) >= 3:
            break
        if f not in family_order and f != "unknown":
            family_order.append(f)
    out = {
        "group_label": group, "groups": groups, "default_null": null, "palette": PALETTE, "family_order": family_order, "families": dict(zip(agents.agent, agents.family)),
        "nulls": None, "networks": {}, "timeline": None, "helping": None, "hierarchy": None, "diffusion": None, "bench": {},
    }
    rep = d / f"report_{group}.csv"
    if rep.exists():
        t = pd.read_csv(rep)
        t = t[~t.group.isin(drop)]
        draws_path = d / f"report_{group}_draws.json"
        draws = json.loads(draws_path.read_text()) if draws_path.exists() else None
        if draws:
            draws = {n: {s: {g: v for g, v in by_g.items() if g not in drop} for s, by_g in by_s.items()} for n, by_s in draws.items()}
        present = [k for k in NULL_NAMES if k in set(t.null)]
        out["nulls"] = {"table": _records(t[t.null.isin(present)]), "stats": [{"key": k, "label": STAT_LABELS[k]} for k in [s for s, _, _ in NETWORK_CARDS]], "null_names": {k: NULL_NAMES[k] for k in present}, "draws": draws}
    order = agents.sort_values("joined").agent.tolist()
    order += sorted(set(msgs.actor) - set(order))
    if mentions is not None:
        out["networks"] = networks_payload(msgs, mentions, agents, group)
        out["timeline"] = timeline_payload(events, msgs, mentions, agents, group, req, resp)
    if (d / "helping_ladder.csv").exists():
        summary = json.loads((d / "helping_summary.json").read_text()) if (d / "helping_summary.json").exists() else {}
        by = pd.read_csv(d / "helping_bystander.csv") if (d / "helping_bystander.csv").exists() else pd.DataFrame(columns=["era"])
        by = by[~by.era.map(lambda e: f"{group} {e}").isin(drop)]
        out["helping"] = {"ladder": _csv(d / "helping_ladder.csv"), "bystander": _records(by), "sizes": list(SIZE_LABELS), "slope": summary.get("bystander_slope"), "summary": summary, "undirected": undirected_asks(d, group, drop)}
    if (d / f"hierarchy_{group}.csv").exists():
        h = pd.read_csv(d / f"hierarchy_{group}.csv")
        r = pd.read_csv(d / f"hierarchy_ranks_{group}.csv") if (d / f"hierarchy_ranks_{group}.csv").exists() else pd.DataFrame(columns=["group"])
        out["hierarchy"] = {"table": _records(h[~h.group.isin(drop)]), "ranks": _records(r[~r.group.isin(drop)])}
    if (d / "diffusion_items.csv").exists():
        sm = _csv(d / "diffusion_summary.csv")
        out["diffusion"] = {
            "leaders": _csv(d / "diffusion_leaders.csv") or [], "edges": _csv(d / "diffusion_edges.csv") or [], "summary": sm[0] if sm else None,
            "adoptions": adoption_payload(d, order, msgs), "lead_rank": lead_vs_rank(d, group, drop, agents),
        }
    return out


def _static(name):
    return (STATIC / name).read_text().replace("</script", "<\\/script")


def run(in_dir, title=None, group="era", validation="validation", min_agents=MIN_AGENTS, thin="drop", null=DEFAULT_NULL, meta="meta", intro=None):
    d = Path(in_dir)
    ev = pd.read_parquet(d / "events.parquet")
    agents = pd.read_parquet(d / "agents.parquet")
    msgs = ev[ev.type == "message"]
    amsgs = msgs[msgs.actor_class == "agent"]
    mentions = pd.read_parquet(d / "mentions.parquet") if (d / "mentions.parquet").exists() else None
    req, resp = _load_all(d, "requests"), _load_all(d, "responses")

    groups_all = group_info(amsgs, group, min_agents)
    eras = load_eras(meta, group)
    for i, g in enumerate(groups_all):
        e = eras.get(g["value"], {})
        g["marker"] = e.get("marker") or MARKERS[i % len(MARKERS)]
        g["label"] = e.get("name", "")
        g["description"] = e.get("description", "")
    drop = {g["name"] for g in groups_all if g["thin"]} if thin == "drop" else set()
    thin_set = set() if thin == "drop" else {g["name"] for g in groups_all if g["thin"]}
    groups = [g for g in groups_all if g["name"] not in drop]
    lab = Labels(groups_all)
    keep = lambda df: df[~df[group].map(lambda v: f"{group} {v}").isin(drop)]  # noqa: E731
    msgs, amsgs = keep(msgs), keep(amsgs)
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
    data = payload(d, group, groups, amsgs, ev, agents, mentions, req, resp, drop, null)
    bench = load_benchmarks(meta)
    cards = [time_card(d, data["timeline"] is not None, group, lab, drop)] + network_cards(d, group, lab, drop, thin_set, null, n_perm, bench, data["networks"]) + [hierarchy_card(d, group, lab, drop, thin_set, bench)]
    und = data["helping"]["undirected"] if data["helping"] else None
    ad, lr = (data["diffusion"]["adoptions"], data["diffusion"]["lead_rank"]) if data["diffusion"] else (None, None)
    cards += helping_cards(d, group, lab, drop, bench, und) + [evidence_card(d), diffusion_card(d, ad, lr), baseline_card(data, group, lab, null, _backed_share(d)), quality_card(d, validation)]
    cards = [c for c in cards if c]
    if bench:
        data["bench"] = bench.payload  # filled while the cards were built
    glance = "".join(f'<li><a href="#{a}">{esc(t)}</a><span class="chips">{chips}</span></li>' for _, (t, chips, a) in cards[1:])
    intro_html = intro_card(intro, groups, msgs, group)
    subtitle = f"{esc(title or d.name)} &middot; {esc(str(msgs.day.min()))} to {esc(str(msgs.day.max()))}"
    data_json = json.dumps(_clean(data), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")  # never closes the script element early
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Swarm report card</title><style>{_static("card.css")}</style></head><body>
<header><h1>Swarm report card</h1><p class="sub">{subtitle}</p>
<div class="tiles">{"".join(f'<div class="tile"><b>{v}</b><span>{esc(k)}</span></div>' for v, k in tiles)}</div>
<p class="nojs">The figures on this page are drawn in the browser. If they are missing, the viewer has blocked the page's script (GitHub's file view and most in-app previews do): open the file in a web browser.</p>
<div class="groupbar" id="groupbar"></div></header>
<main>
{intro_html}
{cards[0][0]}
<article class="card" id="glance"><h2>Network Dynamics: How do agents interact with each other and how do the dynamics compare to other social networks?</h2><p class="meaning">Each question is answered yes or no per {esc(group)} against the null model: yes, the pattern is there beyond chance; no in grey, it is indistinguishable from chance; no in orange, it is weaker than chance would give. Each card ends with the same statistic as measured in animal and human networks. Read together the numbers are a baseline for this swarm; the last card but one says which movements from it would be worth a look.</p><ul class="glance">{glance}</ul></article>
{"".join(c for c, _ in cards[1:])}
</main>
<footer>Generated by swarm-sna. Null models follow Bejder et al. 1998 and Farine 2017; dominance measures follow de Vries et al. 2006 and Shizuka &amp; McDonald 2012. Figures drawn with D3 (Mike Bostock, ISC licence).</footer>
<script id="card-data" type="application/json">{data_json}</script>
<script>{_static("d3.v7.min.js")}</script>
<script>{_static("card.js")}</script>
</body></html>"""
    out = d / "report_card.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1e6:.1f} MB)")
