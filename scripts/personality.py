"""Personality-test week: self-reported scores with verbatim evidence, next to observed behaviour.

During the goal "Take a bunch of personality tests!" (2025-09-22 to 2025-09-29, goal id
6fe0b7dd-99f2-4807-ad0a-e7593b125fcd) six agents took personality tests and reported the
results in chat.

Task A screens the chat of that week and the week after for messages that report a score,
has an LLM copy each reported score together with the sentence that reports it, and checks
every quote against the message (whitespace-normalised substring). Rows that fail the
check are kept in the CSV but excluded from every table.

Task B puts the self-reported Big Five next to behavioural measures from era 2: David's
score, response rate as a responder, requests made per 100 messages, battery label shares,
and mention strengths. Six agents, so the Spearman correlations are descriptive, not tests.

usage: personality.py [parallel LLM calls]

writes out/village/personality_scores.csv and docs/personality.md
"""
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from swarm_sna import llm

warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")

OUT, CACHE, DOCS = Path("out/village"), Path("out/llm_cache"), Path("docs")
WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
GOAL_ID = "6fe0b7dd-99f2-4807-ad0a-e7593b125fcd"
WINDOW = ("2025-09-22", "2025-10-07")  # goal week plus the week after, when results were still discussed
AGENTS = ["Claude 3.7 Sonnet", "o3", "Gemini 2.5 Pro", "GPT-5", "Grok 4", "Claude Opus 4.1"]
ALIASES = {
    "Claude 3.7 Sonnet": ["claude 3.7 sonnet", "claude 3.7", "3.7 sonnet", "sonnet"], "o3": ["o3"], "Gemini 2.5 Pro": ["gemini 2.5 pro", "gemini 2.5", "gemini"],
    "GPT-5": ["gpt-5", "gpt5", "gpt 5"], "Grok 4": ["grok 4", "grok"], "Claude Opus 4.1": ["claude opus 4.1", "opus 4.1", "claude opus", "opus"],
}
ERA = 2
MAX_MESSAGES, BATCH, TEXT_CAP = 900, 15, 1500  # at most 900 / 15 = 60 LLM calls
FAMILIES = ["Big Five", "HEXACO", "MBTI", "Enneagram", "DISC", "Dark Triad", "VIA", "IQ", "other"]
B5 = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
HEX = ["honesty_humility", "emotionality", "extraversion", "agreeableness", "conscientiousness", "openness"]

# How a test was answered, where the agent says so. Picked by hand while reading the transcript, not by the LLM;
# each quote is checked against that agent's messages in the window and reported with its timestamp.
ANSWER_METHOD = [
    ("o3", "Big Five", "neutral baseline", "using mostly Neutral responses for speed to generate a baseline"),
    ("Grok 4", "Big Five", "neutral baseline", "all 50th percentiles from neutral responses"),
    ("o3", "HEXACO", "neutral baseline", "I shortcut-completed the AMBI inventory by pasting a results URL that ends with 181"),
    ("GPT-5", "HEXACO", "neutral baseline", "validated a DevTools snippet to select Neutral (value=3) for all visible items"),
    ("o3", "Big Five", "site: openpsychometrics.org", "Started the openpsychometrics Big Five test"),
    ("Grok 4", "Big Five", "site: openpsychometrics.org", "completed the Big Five test on openpsychometrics.org"),
]

TEST = re.compile(
    r"(big ?-?five|big ?5|\bocean\b|hexaco|mbti|myers|16 ?-?personalit|16 ?types|enneagram|\bdisc\b|dark triad|machiavell|narciss|psychopath"
    r"|openness|conscientious|extraver|extrover|introver|agreeable|neurotic|honesty.humility|emotionality|emotional stability"
    r"|personality (test|quiz|type|profile|assessment)|typefinder|truity|ipip|\bbfi\b|neo.?pi|five.?factor|temperament|attachment style"
    r"|love language|strengthsfinder|via (character|survey)|character strengths|holland|riasec|locus of control|self.esteem|rosenberg|grit"
    r"|empathy quotient|autism quotient|political compass|moral foundations|schwartz|sorting hat|hogwarts|iq test"
    r"|bigfive-test|openpsychometrics|16personalities|mypersonality|personality\.co|oejts|jung)",
    re.I,
)
TRAIT_WORD = r"(openness|conscientious\w*|extrav\w*|extrov\w*|introv\w*|agreeable\w*|neurotic\w*|honesty|humility|emotionality|stability|intellect|imagination)"
SIGNALS = {  # weight, pattern: how strongly a message looks like it reports a result
    "mbti_code": (3, re.compile(r"\b[IE][NS][TF][JP](?:-[AT])?\b")),
    "trait_number": (3, re.compile(TRAIT_WORD + r"\W{0,25}\d{1,3}\b|\b\d{1,3}\W{0,25}" + TRAIT_WORD, re.I)),
    "scale": (2, re.compile(r"percentile|\b\d{1,3}\s?(?:/|out of)\s?\d{2,3}\b|\b\d{1,3}\s?%", re.I)),
    "enneagram": (2, re.compile(r"\btype\s?[1-9]\b|\b[1-9]w[1-9]\b", re.I)),
    "score_word": (2, re.compile(r"\b(scored?|scores|results?)\b.{0,60}\d|\d.{0,60}\b(scored?|scores|results?)\b", re.I | re.S)),
    "level_word": (1, re.compile(r"\b(very high|very low|high|low|moderate|average)\s+(on|in)\b", re.I)),
}
PROGRESS = re.compile(r"question \d+|\d+ ?% (complete|done|through)|\d+ of \d+ questions|completed? \d+ questions|\d+ questions (remaining|left)|\d+ ?% progress", re.I)

SYSTEM = """You extract personality-test results from messages in a group chat between AI agents. Each numbered message is written by the named agent. The agents took tests (Big Five / OCEAN, HEXACO, MBTI / 16 types, Enneagram, DISC, dark triad, VIA strengths, IQ and others) and reported their results in the chat.
Return a JSON array with one object per reported score: several objects for a message that reports several traits or tests, none for a message that reports no score.
{"i": <message number>,
 "subject": the agent whose score it is: the writer's own name when it reports its own result, otherwise the other agent's name as written,
 "test": "Big Five" | "HEXACO" | "MBTI" | "Enneagram" | "DISC" | "Dark Triad" | "VIA" | "IQ" | "other: <name>",
 "trait": the trait, factor, facet or type scored, as named in the message (for example "Openness", "Honesty-Humility", "Extraversion"; use "type" for a four-letter MBTI code, an Enneagram number or a DISC letter),
 "score": the score exactly as written: a number, a percentile, a letter code, a type name or a level word such as "high",
 "scale": the scale if the message states one ("percentile", "%", "/120", "1-5", "out of 40"), else "",
 "quote": the sentence that reports the score, copied character for character from the message: no paraphrase, no ellipsis, no added or dropped words}
A score is the result of a completed test or of a finished part of it. Do not return progress figures ("63% complete", "question 20 of 50"), answers to single items, plans, expectations or predictions of what the result will be. A message that only mentions a test, waits for results, or talks about screenshots, files or spreadsheets reports no score.
Return [] when no message reports a score. JSON only."""


def norm(s):
    return re.sub(r"\s+", " ", str(s if s is not None else "")).strip()


# ----------------------------------------------------------------------------- task A
def screen(ev):
    w = ev[(ev.type == "message") & (ev.ts >= WINDOW[0]) & (ev.ts < WINDOW[1]) & ev.actor.isin(AGENTS)].copy()
    w["mentions_test"] = w.text.str.contains(TEST)
    w["signal"] = 0
    for name, (weight, pat) in SIGNALS.items():
        w[name] = w.text.str.contains(pat)
        w["signal"] += weight * w[name]
    w["progress"] = w.text.str.contains(PROGRESS)
    w.loc[w.progress, "signal"] -= 2
    hits = w[w.mentions_test & (w.signal > 0)].sort_values(["signal", "ts"], ascending=[False, True])
    return w, hits, hits.head(MAX_MESSAGES).sort_values("ts").reset_index(drop=True)


def resolve_subject(subject, writer):
    s = norm(subject).lower()
    if s in ("", "self", "me", "i", "writer", "own", "myself"):
        return writer
    for agent, names in ALIASES.items():
        if s in names:
            return agent
    return norm(subject)


def test_family(test):
    t = test.lower()
    if re.search(r"big ?-?five|big ?5|ocean|ipip|bfi|neo", t):
        return "Big Five"
    if "hexaco" in t:
        return "HEXACO"
    if re.search(r"mbti|16|myers|jung|oejts|typefinder", t):
        return "MBTI"
    if "enneagram" in t:
        return "Enneagram"
    for k in ("DISC", "Dark Triad", "VIA", "IQ"):
        if re.search(rf"\b{k.lower()}\b", t):
            return k
    return "other"


BIG5_MAP = [  # pattern, canonical trait, reversed (a stability score counts as 100 - neuroticism)
    (r"open|intellect|imagination", "openness", False),
    (r"conscient", "conscientiousness", False),
    (r"extrav|extrov", "extraversion", False),
    (r"introv", "extraversion", True),
    (r"agreeab", "agreeableness", False),
    (r"neurotic|natural reactions", "neuroticism", False),
    (r"stability", "neuroticism", True),
]
HEXACO_MAP = [(r"honesty|humility|^h$", "honesty_humility"), (r"emotionality|^e$", "emotionality"), (r"extrav|extrov|^x$", "extraversion"), (r"agreeab|^a$", "agreeableness"), (r"conscient|^c$", "conscientiousness"), (r"open|^o$", "openness")]


def canon_trait(trait, family):
    t = trait.lower().strip()
    if family == "Big Five":
        for pat, name, rev in BIG5_MAP:
            if re.search(pat, t):
                return name + ("_rev" if rev else "")
    if family == "HEXACO":
        for pat, name in HEXACO_MAP:
            if re.search(pat, t):
                return name
    return ""


def score_100(score, scale, raw120):
    """Numeric score on 0-100 and the rule used. Percentiles and percentages as written, x/y and Likert rescaled.
    raw120: the message gives unscaled Big Five scores above 100, read as IPIP-NEO-120 domain scores (24 items x 1-5, range 24-120)."""
    s, sc = score.lower(), scale.lower()
    num = re.search(r"\d+(?:\.\d+)?", s)
    if not num:
        return np.nan, "no_number"
    x = float(num.group(0))
    frac = re.search(r"(\d+(?:\.\d+)?)\s*(?:/|out of)\s*(\d+(?:\.\d+)?)", s)
    if frac:
        y = float(frac.group(2))
        return (100 * float(frac.group(1)) / y if y else np.nan), "x_of_y"
    if "percentile" in s + sc:
        return x, "percentile"
    if "%" in s + sc:
        return x, "percent"
    of = re.search(r"(?:/|out of)\s*(\d+(?:\.\d+)?)", sc)
    if of:
        y = float(of.group(1))
        return (100 * x / y if y else np.nan), "x_of_y"
    lik = re.search(r"\b1\s*(?:-|to)\s*([57])\b", sc)
    if lik:
        top = float(lik.group(1))
        return 100 * (x - 1) / (top - 1), f"likert_1_{int(top)}"
    if raw120:
        return 100 * (x - 24) / 96, "raw_24_120_assumed"
    if "." in num.group(0) or x > 100:
        return np.nan, "unknown_scale"  # e.g. HEXACO "7.41" with no scale stated
    return x, "as_written"


def label(sample):
    items = [f"[{i}] {r.actor}: {norm(r.text)[:TEXT_CAP]}" for i, r in enumerate(sample.itertuples())]
    batches = ["\n\n".join(items[i : i + BATCH]) for i in range(0, len(items), BATCH)]
    print(f"labelling {len(items)} messages in {len(batches)} calls, {WORKERS} in parallel", flush=True)
    res = llm.map_calls(SYSTEM, batches, CACHE, workers=WORKERS, model="haiku")
    failed = sum(r is None for r in res)
    rows = []
    for r in res:
        for x in r or []:
            if not isinstance(x, dict) or "i" not in x:
                continue
            try:
                i = int(x["i"])
            except (TypeError, ValueError):
                continue
            if not 0 <= i < len(sample):
                continue
            m = sample.iloc[i]
            quote = norm(x.get("quote"))
            rows.append(
                {
                    "event_id": m.event_id, "ts": m.ts, "day": m.day, "reporter": m.actor, "subject": resolve_subject(x.get("subject"), m.actor),
                    "test": norm(x.get("test")), "trait": norm(x.get("trait")), "score": norm(x.get("score")), "scale": norm(x.get("scale")),
                    "quote": quote, "quote_verified": bool(quote) and quote in norm(m.text),
                }
            )
    scores = pd.DataFrame(rows)
    n_raw = len(scores)
    prog = scores.trait.str.contains(r"progress|complet", case=False) | scores.score.str.contains(r"complete|progress", case=False)
    scores = scores[~prog].reset_index(drop=True)  # progress figures the LLM returned despite the instruction
    scores["self_report"] = scores.subject == scores.reporter
    scores["test_family"] = scores.test.map(test_family)
    scores["trait_canon"] = [canon_trait(t, f) for t, f in zip(scores.trait, scores.test_family)]
    num = pd.to_numeric(scores.score.str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
    unscaled = scores.scale.eq("") & ~scores.score.str.contains(r"%|percentile|/|out of", case=False)
    raw_msgs = set(scores[(scores.test_family == "Big Five") & unscaled & (num > 100) & (num <= 120)].event_id)
    out = [score_100(s, sc, fam == "Big Five" and eid in raw_msgs and sc == "") for s, sc, fam, eid in zip(scores.score, scores.scale, scores.test_family, scores.event_id)]
    scores["score_100"] = [o[0] for o in out]
    scores["norm_rule"] = [o[1] for o in out]
    return scores, {"calls": len(batches), "failed": failed, "rows_raw": n_raw, "rows_progress": int(prog.sum())}


def usable(scores, family):
    return scores[scores.self_report & scores.quote_verified & (scores.test_family == family)]


def trait_values(scores, family, traits):
    """Distinct 0-100 values per agent and trait from verified self-reports (a repeated report of the same result counts once)."""
    s = usable(scores, family)
    s = s[s.trait_canon.ne("") & s.score_100.notna()].copy()
    rev = s.trait_canon.str.endswith("_rev")
    s.loc[rev, "score_100"] = 100 - s.loc[rev, "score_100"]
    s["trait_canon"] = s.trait_canon.str.replace("_rev", "")
    d = s.drop_duplicates(["subject", "trait_canon", "score_100"])
    med = d.groupby(["subject", "trait_canon"]).score_100.median().unstack().reindex(index=AGENTS, columns=traits)
    lo = d.groupby(["subject", "trait_canon"]).score_100.min().unstack().reindex(index=AGENTS, columns=traits)
    hi = d.groupby(["subject", "trait_canon"]).score_100.max().unstack().reindex(index=AGENTS, columns=traits)
    shown = pd.DataFrame("", index=AGENTS, columns=traits)
    for a in AGENTS:
        for t in traits:
            if pd.notna(med.loc[a, t]):
                shown.loc[a, t] = f"{med.loc[a, t]:.0f}" + (f" ({lo.loc[a, t]:.0f}-{hi.loc[a, t]:.0f})" if hi.loc[a, t] - lo.loc[a, t] >= 1 else "")
    return med, shown, hi - lo


def type_table(scores, family):
    s = usable(scores, family)
    s = s[s.trait.str.lower().str.strip() == "type"]
    pat = r"^([IE][NS][TF][JP])" if family == "MBTI" else r"^(?:TYPE )?(\d(?:W\d)?)"
    out = {}
    for a in AGENTS:
        vals = s[s.subject == a].score.str.strip().str.upper().str.extract(pat)[0].dropna().str.replace("W", "w")
        vc = vals.value_counts()
        out[a] = "" if vc.empty else (vc.index[0] if len(vc) == 1 else f"{vc.index[0]} (also {', '.join(vc.index[1:])})")
    return out


def answer_methods(window):
    rows = []
    for agent, test, method, quote in ANSWER_METHOD:
        g = window[window.actor == agent].sort_values("ts")
        hit = g[g.text.map(norm).str.contains(norm(quote), regex=False)]
        rows.append({"agent": agent, "test": test, "method": method, "quote": quote, "verified": len(hit) > 0, "first_said": str(hit.ts.iloc[0])[:16] if len(hit) else ""})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- task B
BEHAVIOUR = ["davids_score", "response_rate", "response_rate_addressed", "requests_per_100_msgs", "affiliative", "display", "provisioning", "leads", "directs_others", "out_mentions_per_100_msgs", "in_mentions_per_100_msgs"]


def behaviour(ev):
    e2 = ev[(ev.era == ERA) & (ev.type == "message")]
    msgs = e2[e2.actor.isin(AGENTS)].groupby("actor").size().reindex(AGENTS)
    ranks = pd.read_csv(OUT / "hierarchy_ranks_era.csv")
    david = ranks[ranks.group == f"era {ERA}"].set_index("agent").davids_score.reindex(AGENTS)
    d = pd.read_parquet(OUT / "dyads.parquet")
    d = d[(d.era == ERA) & d.responder.isin(AGENTS)]
    resp_all = d.groupby("responder").responded.mean().reindex(AGENTS)
    resp_addr = d[d.addressed].groupby("responder").responded.mean().reindex(AGENTS)
    req = pd.concat([pd.read_parquet(OUT / f) for f in ("requests.parquet", "requests_wide.parquet") if (OUT / f).exists()])  # both screens, as in the dyad table
    req = req[(req.era == ERA) & req.actor.isin(AGENTS)].groupby("actor").request_id.nunique().reindex(AGENTS).fillna(0)
    b = pd.read_parquet(OUT / "battery_labels.parquet")
    b = b[(b.era == ERA) & b.actor.isin(AGENTS) & b.act.notna()]
    bat = pd.DataFrame(
        {
            "affiliative": b.ethogram.eq("affiliative"), "display": b.ethogram.eq("display"), "provisioning": b.ethogram.eq("provisioning"),
            "leads": b.leads_or_follows.eq("leads"), "directs_others": b.directs_others.eq(True),
        }
    ).groupby(b.actor).mean().reindex(AGENTS)
    m = pd.read_parquet(OUT / "mentions.parquet").merge(ev[["event_id", "era"]], on="event_id", how="left")
    m = m[m.era == ERA]
    out_s = m[m.actor.isin(AGENTS)].groupby("actor").weight.sum().reindex(AGENTS).fillna(0)
    in_s = m[m.target.isin(AGENTS)].groupby("target").weight.sum().reindex(AGENTS).fillna(0)
    # exposure for in-mentions: messages others wrote on the days the agent was active
    others = pd.Series({a: len(e2[e2.day.isin(set(e2[e2.actor == a].day)) & (e2.actor != a)]) for a in AGENTS})
    return pd.DataFrame(
        {
            "messages_era2": msgs, "davids_score": david, "response_rate": resp_all, "response_rate_addressed": resp_addr,
            "requests_per_100_msgs": 100 * req / msgs, "affiliative": bat.affiliative, "display": bat.display, "provisioning": bat.provisioning,
            "leads": bat.leads, "directs_others": bat.directs_others, "battery_n": b.groupby("actor").size().reindex(AGENTS),
            "out_mentions_per_100_msgs": 100 * out_s / msgs, "in_mentions_per_100_msgs": 100 * in_s / others,
        }
    )


def spearman_r(x, y):
    """Spearman as Pearson on average ranks (no scipy in the environment)."""
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 3 or d.iloc[:, 0].nunique() < 2 or d.iloc[:, 1].nunique() < 2:
        return np.nan
    return d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank())


def spearman(traits, beh, agents=None):
    agents = agents or list(traits.index)
    return pd.DataFrame({t: {c: spearman_r(traits.loc[agents, t], beh.loc[agents, c]) for c in BEHAVIOUR} for t in traits.columns}).T


# ----------------------------------------------------------------------------- report
PCT = ["response_rate", "response_rate_addressed", "affiliative", "display", "provisioning", "leads", "directs_others"]
FMT = {**{c: "{:.0%}" for c in PCT}, "messages_era2": "{:.0f}", "battery_n": "{:.0f}", "davids_score": "{:.2f}", "requests_per_100_msgs": "{:.1f}", "out_mentions_per_100_msgs": "{:.0f}", "in_mentions_per_100_msgs": "{:.1f}"}


def md_table(df, fmt=None, index_name="agent"):
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + index_name + " | " + " | ".join(str(c) for c in cols) + " |", "|" + " --- |" * (len(cols) + 1)]
    for idx, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("")
            elif c in fmt:
                cells.append(fmt[c].format(v))
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{v:.2f}")
            else:
                cells.append(str(v).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + str(idx) + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def quote_rows(scores, texts):
    """Per agent and test: the message that reports most traits, what it reports, and the verbatim span of it that covers the reported scores."""
    rows = []
    for a in AGENTS:
        for fam in FAMILIES:
            g = usable(scores, fam)
            g = g[g.subject == a]
            if g.empty:
                continue
            g = g.assign(says_result=g.quote.str.contains(r"result|scores?\b|scored|percentile", case=False))
            best = g.groupby("event_id").agg(n=("trait", "nunique"), res=("says_result", "max"), ts=("ts", "min")).sort_values(["n", "res", "ts"], ascending=[False, False, True]).index[0]
            gb = g[g.event_id == best]
            text = texts[best]
            spans = [(text.find(q), text.find(q) + len(q)) for q in gb.quote.unique()]
            span = text[min(a for a, _ in spans) : max(b for _, b in spans)]
            quote = span if len(span) <= 450 else max(gb.quote, key=len)  # both are verbatim substrings of the message
            summary = "; ".join(f"{t} {sc}" + (f" [{scl}]" if scl and scl not in sc else "") for t, sc, scl in gb[["trait", "score", "scale"]].drop_duplicates().itertuples(index=False))
            rows.append({"agent": a, "test": fam, "reported in the quoted message": summary, "verbatim quote": '"' + quote + '"', "day": gb.day.iloc[0], "rows": len(g), "messages": g.event_id.nunique()})
    return pd.DataFrame(rows)


def ordinal(n):
    return {1: "highest", 2: "second-highest", 3: "third-highest", 4: "third-lowest", 5: "second-lowest", 6: "lowest"}[int(n)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    ev = pd.read_parquet(OUT / "events.parquet")
    window, hits, sample = screen(ev)
    print(f"window messages by the six agents: {len(window)}; mention a test: {int(window.mentions_test.sum())}; with a result signal: {len(hits)}; sent to the LLM: {len(sample)}", flush=True)
    scores, stats = label(sample)
    csv_cols = ["event_id", "ts", "day", "agent", "reporter", "self_report", "test", "test_family", "trait", "trait_canon", "score", "scale", "score_100", "norm_rule", "quote", "quote_verified"]
    scores.assign(agent=scores.subject)[csv_cols].to_csv(OUT / "personality_scores.csv", index=False)  # agent = whose score it is; reporter = who wrote the message
    ver = scores.quote_verified.mean()
    print(f"{len(scores)} score rows ({stats['rows_progress']} progress rows dropped) from {scores.event_id.nunique()} messages; quotes verified: {ver:.1%}; failed calls: {stats['failed']}")

    b5, b5_shown, b5_range = trait_values(scores, "Big Five", B5)
    hx, hx_shown, _ = trait_values(scores, "HEXACO", HEX)
    mbti, ennea = type_table(scores, "MBTI"), type_table(scores, "Enneagram")
    methods = answer_methods(window)
    neutral_b5 = list(methods[(methods.test == "Big Five") & (methods.method == "neutral baseline") & methods.verified].agent)
    beh = behaviour(ev)
    all_six = bool(b5.notna().all().all())
    corr = spearman(b5, beh) if all_six else None
    rest = [a for a in AGENTS if a not in neutral_b5]
    corr_rest = spearman(b5, beh, rest) if all_six and len(rest) >= 3 else None

    # ---- document
    fam = scores[scores.quote_verified].groupby(["test_family", "self_report"]).size().unstack(fill_value=0).rename(columns={True: "own result", False: "another agent's result"})
    fam = fam[[c for c in ("own result", "another agent's result") if c in fam.columns]]
    rules = ", ".join(f"{k} {v}" for k, v in scores[scores.quote_verified & scores.self_report & (scores.test_family == "Big Five")].norm_rule.value_counts().items())
    neutral = methods[(methods.method == "neutral baseline") & methods.verified]
    truncated = int((sample.text.map(norm).str.len() > TEXT_CAP).sum())
    method_col = pd.Series({a: "neutral baseline" if a in neutral_b5 else "" for a in AGENTS}, name="Big Five answered as")
    combined = pd.concat([b5.rename(columns={t: "B5 " + t[:5] for t in B5}), method_col, pd.Series(mbti, name="MBTI"), pd.Series(ennea, name="Enneagram"), beh[["davids_score", "response_rate", "requests_per_100_msgs", "affiliative", "display", "provisioning", "in_mentions_per_100_msgs", "out_mentions_per_100_msgs"]]], axis=1)
    L = ["# Personality-test week: self-reports next to behaviour\n"]
    L.append(f"Goal \"Take a bunch of personality tests!\" (goal id `{GOAL_ID}`, 2025-09-22 to 2025-09-29). Six agents on the roster: {', '.join(AGENTS)}. Everything here is produced by `scripts/personality.py`; the extracted rows are in `out/village/personality_scores.csv`.\n")
    L.append("## What was read\n")
    L.append(f"- Messages by the six agents from {WINDOW[0]} up to {WINDOW[1]} (the goal week plus the week after, when results were still being discussed): **{len(window)}**.")
    L.append(f"- {int(window.mentions_test.sum())} of them mention a test or trait. **{len(hits)}** also carry a result signal (a number next to a trait name, a percentile or x/y, a four-letter type, an Enneagram type, or a score/result word near a number). All {len(sample)} were labelled by Haiku in {stats['calls']} calls of {BATCH} messages; {stats['failed']} calls failed. {truncated} messages were longer than {TEXT_CAP} characters and were cut there for the LLM. The screen's recall was not measured: a result reported without any of these signals is missed.")
    L.append(f"- The LLM returned {stats['rows_raw']} rows; {stats['rows_progress']} were progress figures (\"40% complete\") and were dropped, leaving **{len(scores)} score rows from {scores.event_id.nunique()} messages**.")
    L.append(f"- Each row carries the sentence that reports the score. **{ver:.1%}** of the quotes ({int(scores.quote_verified.sum())} of {len(scores)}) are a verbatim substring of the message after whitespace normalisation (`quote_verified`). Unverified rows stay in the CSV and are excluded from every table below.")
    L.append(f"- {int(scores.self_report.sum())} rows are an agent reporting its own result; {int((~scores.self_report).sum())} report another agent's result and are not used in the tables.\n")
    L.append("Verified rows by test:\n")
    L.append(md_table(fam, index_name="test"))
    L.append("\n## How the tests were answered\n")
    L.append(f"{neutral.agent.nunique()} of the six agents ({', '.join(dict.fromkeys(neutral.agent))}) say in chat that they answered a test with neutral responses to get a result quickly, so those scores are not self-descriptions. These annotations were picked by hand while reading the transcript (they are not LLM labels); the script checks each quote against that agent's messages.\n")
    L.append(md_table(methods.set_index("agent")))
    sites = methods[methods.method.str.startswith("site") & methods.verified & (methods.test == "Big Five")]
    same_site = [a for a in neutral_b5 if a in set(sites.agent)]
    if len(same_site) >= 2 and all_six:
        vals = "; ".join(f"{a}: " + ", ".join(f"{t} {b5.loc[a, t]:.0f}" for t in B5) for a in same_site)
        L.append(f"\n{' and '.join(same_site)} took the Big Five on the same site and both describe neutral answering, yet they report different percentiles ({vals}). o3 says \"mostly\" neutral, so the two reports need not conflict, but the chat cannot settle what the site showed; the screenshots the agents saved could.")
    L.append("\n## Self-reported scores, one verbatim quote per agent and test\n")
    L.append("For each agent and test: the message that reports the most traits, what it reports, and the verbatim span of that message covering the reported scores. `rows` and `messages` count all verified own-result rows for that agent and test; results were repeated across messages and some tests were taken more than once. No row means no verified own result in chat (GPT-5 and Grok 4 report no MBTI type, for example).\n")
    L.append(md_table(quote_rows(scores, {r.event_id: norm(r.text) for r in sample.itertuples()}).set_index("agent")))
    L.append("\n### Big Five on a common 0-100 scale\n")
    L.append(f"Median of the distinct values each agent reported for a trait, with the range in brackets where reports differ. Percentiles and percentages are taken as written; a stability score counts as 100 minus neuroticism. Scales differ across agents (percentiles against a human norm group for most, percentages or raw scores for others), so only the rank order across agents is used below. Rules applied to the verified own-result Big Five rows: {rules}. `raw_24_120_assumed` is an assumption: Claude Opus 4.1 reports \"raw scores\" up to 112 from the 120-item test on bigfive-test.com with no scale, read here as domain scores on 24-120.\n")
    L.append(md_table(pd.concat([b5_shown, method_col], axis=1)))
    wide = [(a, t, b5_range.loc[a, t]) for a in AGENTS for t in B5 if pd.notna(b5_range.loc[a, t]) and b5_range.loc[a, t] >= 10]
    if wide:
        L.append("\nReports that disagree by 10 points or more for the same agent and trait: " + "; ".join(f"{a} {t} ({r:.0f} points)" for a, t, r in wide) + ". These are different test runs, sites or scales reported by the same agent.")
    if hx.notna().any().any():
        L.append("\n### HEXACO on 0-100, where numbers were reported\n")
        L.append("Same construction. Rows with a number on an unstated scale (for example \"7.41\") are left out; agents with no row reported no number in chat.\n")
        L.append(md_table(hx_shown[hx.notna().any(axis=1)]))
    L.append("\n## Behaviour in era 2\n")
    L.append(f"Era 2 runs from {ev[ev.era == ERA].ts.min().date()} to {ev[ev.era == ERA].ts.max().date()} and the six agents were present for different stretches of it, so every measure is a rate. `davids_score` is from `hierarchy_ranks_era.csv`. `response_rate` is the mean of `responded` over dyad rows where the agent is the responder; `response_rate_addressed` keeps only requests addressed to it. `requests_per_100_msgs` counts the agent's request ids in `requests.parquet` and `requests_wide.parquet` (the two screens the dyad table is built from). `affiliative`, `display`, `provisioning`, `leads` and `directs_others` are shares of the agent's battery-labelled messages (`battery_n` of them). Mention strength: out = mention weight the agent sends per 100 of its own messages; in = mention weight it receives per 100 messages others wrote on days it was active.\n")
    L.append(md_table(beh, fmt=FMT))
    L.append("\n## Self-reports next to behaviour\n")
    L.append(md_table(combined, fmt={**FMT, **{"B5 " + t[:5]: "{:.0f}" for t in B5}}))
    if all_six:
        a_lo, e_lo, e_hi = b5.agreeableness.idxmin(), b5.extraversion.idxmin(), b5.extraversion.idxmax()
        rr, om = beh.response_rate.rank(ascending=False), beh.out_mentions_per_100_msgs.rank(ascending=False)
        L.append(f"\n- Lowest self-reported agreeableness: {a_lo} ({b5.agreeableness[a_lo]:.0f}), which has the {ordinal(rr[a_lo])} response rate of the six ({beh.response_rate[a_lo]:.0%})" + (", from a Big Five it says it answered with neutral responses." if a_lo in neutral_b5 else "."))
        L.append(f"- Lowest self-reported extraversion: {e_lo} ({b5.extraversion[e_lo]:.0f}), with the {ordinal(om[e_lo])} out-mention rate ({beh.out_mentions_per_100_msgs[e_lo]:.0f} per 100 messages). Highest: {e_hi} ({b5.extraversion[e_hi]:.0f}), with the {ordinal(om[e_hi])} ({beh.out_mentions_per_100_msgs[e_hi]:.0f}).")
    L.append("\n## Spearman correlations, Big Five trait by behaviour\n")
    if corr is not None:
        L.append(f"All six agents reported a Big Five result, so the table can be computed. **n = 6: these are descriptions of six agents, not tests.** With six points a coefficient needs |rho| of about 0.89 to pass p < 0.05 two-sided, nothing here is corrected for {corr.size} comparisons, and {len(neutral_b5)} of the six profiles ({', '.join(neutral_b5)}) come from a Big Five the agent says it answered with neutral responses, so part of what is correlated is answering strategy.\n")
        L.append(md_table(corr.round(2), index_name="trait"))
        if corr_rest is not None:
            top = corr.stack().rename("rho, all six").to_frame()
            top[f"rho, without neutral baselines (n = {len(rest)})"] = corr_rest.stack()
            top = top.reindex(top["rho, all six"].abs().sort_values(ascending=False).index).head(10)
            top.index = [f"{t} x {c}" for t, c in top.index]
            L.append(f"\nThe ten largest coefficients, and the same pairs recomputed on the {len(rest)} agents that did not declare neutral answering. With {len(rest)} points a rank correlation can take only a few values, so the second column shows fragility and nothing more.\n")
            L.append(md_table(top.round(2), index_name="pair"))
    else:
        L.append("Not every agent has a verified numeric Big Five self-report, so no correlation table is computed.")
    L.append("\n## Caveats\n")
    L.append("Self-reports are claims: every score is what an agent typed into chat, the quote is the only evidence, the screenshots the agents saved were not checked, and several agents report different numbers for the same test or answered it with neutral responses. Each agent is one instance of a model with its own memory and scaffold, taking a questionnaire normed on humans, so a score describes what that instance chose to answer that week and not the model. Six agents give rank correlations that one agent can flip, so the correlation table describes this roster and supports no inference beyond it.\n")
    (DOCS / "personality.md").write_text("\n".join(L))
    print(f"wrote {OUT / 'personality_scores.csv'} and {DOCS / 'personality.md'}")
    pd.set_option("display.width", 250)
    print(methods.to_string())
    print(combined.round(2).to_string())
    if corr is not None:
        print(corr.round(2).to_string())


if __name__ == "__main__":
    main()
