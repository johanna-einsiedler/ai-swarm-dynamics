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
ERA = 2
MAX_MESSAGES, BATCH, TEXT_CAP = 900, 15, 1500  # 900 / 15 = 60 LLM calls

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
    return re.sub(r"\s+", " ", str(s or "")).strip()


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


def is_self(subject, writer):
    s, a = norm(subject).lower(), writer.lower()
    if s in ("", "self", "me", "i", "writer", "own") or s == a:
        return True
    toks = [t for t in re.split(r"[\s\-]+", s) if t]
    return bool(toks) and all(t in a for t in toks)  # "opus 4.1" for "Claude Opus 4.1"


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
                    "event_id": m.event_id, "ts": m.ts, "day": m.day, "reporter": m.actor,
                    "subject": norm(x.get("subject")) or m.actor, "test": norm(x.get("test")), "trait": norm(x.get("trait")),
                    "score": norm(x.get("score")), "scale": norm(x.get("scale")), "quote": quote,
                    "quote_verified": bool(quote) and quote in norm(m.text),
                }
            )
    scores = pd.DataFrame(rows)
    scores["self_report"] = [is_self(s, a) for s, a in zip(scores.subject, scores.reporter)]
    scores.loc[scores.self_report, "subject"] = scores.loc[scores.self_report, "reporter"]
    scores["test_family"] = scores.test.map(test_family)
    scores["trait_canon"] = [canon_trait(t, f) for t, f in zip(scores.trait, scores.test_family)]
    scores["score_100"] = [score_100(s, sc, t) for s, sc, t in zip(scores.score, scores.scale, scores.trait)]
    return scores, len(batches), failed


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
        if k.lower() in t:
            return k
    return "other"


BIG5 = [  # pattern, canonical trait, reversed (a stability score is 100 - neuroticism)
    (r"open|intellect|imagination", "openness", False),
    (r"conscient", "conscientiousness", False),
    (r"extrav|extrov", "extraversion", False),
    (r"introv", "extraversion", True),
    (r"agreeab", "agreeableness", False),
    (r"neurotic|natural reactions|emotionality", "neuroticism", False),
    (r"stability", "neuroticism", True),
]
HEXACO = [(r"honesty|humility", "honesty_humility"), (r"emotionality", "emotionality"), (r"extrav|extrov", "extraversion"), (r"agreeab", "agreeableness"), (r"conscient", "conscientiousness"), (r"open", "openness")]


def canon_trait(trait, family):
    t = trait.lower()
    if family == "Big Five":
        for pat, name, rev in BIG5:
            if re.search(pat, t):
                return name + ("_rev" if rev else "")
    if family == "HEXACO":
        for pat, name in HEXACO:
            if re.search(pat, t):
                return name
    return ""


def score_100(score, scale, trait):
    """Put a numeric score on 0-100: percentiles and percentages as is, x/y and 1-5 rescaled."""
    s, sc = score.lower(), scale.lower()
    frac = re.search(r"(\d+(?:\.\d+)?)\s*(?:/|out of)\s*(\d+(?:\.\d+)?)", s + " " + sc)
    num = re.search(r"\d+(?:\.\d+)?", s)
    if frac and frac.group(0) in s:
        x, y = float(frac.group(1)), float(frac.group(2))
        return 100 * x / y if y else np.nan
    if not num:
        return np.nan
    x = float(num.group(0))
    if re.search(r"percentile|%", s + sc) or re.search(r"\b(0|1)\s*-\s*100\b|/ ?100", sc):
        return x
    if frac:  # scale given as "/120" or "out of 40"
        y = float(frac.group(2))
        return 100 * x / y if y else np.nan
    if re.search(r"1\s*-\s*5|/ ?5\b|likert", sc) or (x <= 5 and not sc):
        return 100 * (x - 1) / 4
    if re.search(r"1\s*-\s*7|/ ?7\b", sc):
        return 100 * (x - 1) / 6
    return x if x <= 100 else np.nan


def trait_table(scores):
    """One Big Five value per agent and trait: the median of verified, numeric self-reports."""
    s = scores[scores.self_report & scores.quote_verified & (scores.test_family == "Big Five") & scores.trait_canon.ne("") & scores.score_100.notna()].copy()
    rev = s.trait_canon.str.endswith("_rev")
    s.loc[rev, "score_100"] = 100 - s.loc[rev, "score_100"]
    s["trait_canon"] = s.trait_canon.str.replace("_rev", "")
    med = s.groupby(["subject", "trait_canon"]).score_100.median().unstack().reindex(AGENTS)
    n = s.groupby(["subject", "trait_canon"]).size().unstack().reindex(AGENTS).fillna(0).astype(int)
    rng = s.groupby(["subject", "trait_canon"]).score_100.agg(lambda v: v.max() - v.min()).unstack().reindex(AGENTS)
    return med, n, rng, s


def type_table(scores, family):
    s = scores[scores.self_report & scores.quote_verified & (scores.test_family == family) & scores.trait.str.lower().str.contains("type|overall|result|^$")]
    out = {}
    for a in AGENTS:
        vals = s[s.subject == a].score.str.strip()
        if family == "MBTI":
            vals = vals[vals.str.contains(r"^[IE][NS][TF][JP]", case=False)]
        out[a] = ", ".join(f"{v} ({c})" for v, c in vals.value_counts().items()) if len(vals) else ""
    return out


# ----------------------------------------------------------------------------- task B
def behaviour(ev):
    e2 = ev[(ev.era == ERA) & (ev.type == "message")]
    msgs = e2[e2.actor.isin(AGENTS)].groupby("actor").size().reindex(AGENTS)
    ranks = pd.read_csv(OUT / "hierarchy_ranks_era.csv")
    david = ranks[ranks.group == f"era {ERA}"].set_index("agent").davids_score.reindex(AGENTS)
    d = pd.read_parquet(OUT / "dyads.parquet")
    d = d[(d.era == ERA) & d.responder.isin(AGENTS)]
    resp_all = d.groupby("responder").responded.mean().reindex(AGENTS)
    resp_addr = d[d.addressed].groupby("responder").responded.mean().reindex(AGENTS)
    req = pd.read_parquet(OUT / "requests.parquet")
    req = req[(req.era == ERA) & req.actor.isin(AGENTS)].groupby("actor").request_id.nunique().reindex(AGENTS).fillna(0)
    b = pd.read_parquet(OUT / "battery_labels.parquet")
    b = b[(b.era == ERA) & b.actor.isin(AGENTS) & b.act.notna()]
    bat = pd.DataFrame(
        {
            "affiliative": b.ethogram.eq("affiliative"), "display": b.ethogram.eq("display"), "provisioning": b.ethogram.eq("provisioning"),
            "leads": b.leads_or_follows.eq("leads"), "directs_others": b.directs_others.eq(True),
        }
    ).groupby(b.actor).mean().reindex(AGENTS)
    bat_n = b.groupby("actor").size().reindex(AGENTS)
    m = pd.read_parquet(OUT / "mentions.parquet").merge(ev[["event_id", "era"]], on="event_id", how="left")
    m = m[m.era == ERA]
    out_s = m[m.actor.isin(AGENTS)].groupby("actor").weight.sum().reindex(AGENTS).fillna(0)
    in_s = m[m.target.isin(AGENTS)].groupby("target").weight.sum().reindex(AGENTS).fillna(0)
    others = {}  # messages others wrote on the days the agent was active: the exposure for in-mentions
    for a in AGENTS:
        days = set(e2[e2.actor == a].day)
        others[a] = len(e2[e2.day.isin(days) & (e2.actor != a)])
    others = pd.Series(others)
    return pd.DataFrame(
        {
            "messages_era2": msgs, "davids_score": david, "response_rate": resp_all, "response_rate_addressed": resp_addr,
            "requests_per_100_msgs": 100 * req / msgs, "affiliative": bat.affiliative, "display": bat.display, "provisioning": bat.provisioning,
            "leads": bat.leads, "directs_others": bat.directs_others, "battery_n": bat_n,
            "out_mentions_per_100_msgs": 100 * out_s / msgs, "in_mentions_per_100_msgs": 100 * in_s / others,
        }
    )


def spearman(traits, beh):
    cols = ["davids_score", "response_rate", "response_rate_addressed", "requests_per_100_msgs", "affiliative", "display", "provisioning", "leads", "directs_others", "out_mentions_per_100_msgs", "in_mentions_per_100_msgs"]
    rows = {}
    for t in traits.columns:
        rows[t] = {c: traits[t].corr(beh[c], method="spearman") for c in cols}
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------------------- report
def md_table(df, fmt=None, index_name="agent"):
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + index_name + " | " + " | ".join(str(c) for c in cols) + " |", "|" + " --- |" * (len(cols) + 1)]
    for idx, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float) and np.isnan(v):
                cells.append("")
            elif c in fmt:
                cells.append(fmt[c].format(v))
            elif isinstance(v, float):
                cells.append(f"{v:.2f}")
            else:
                cells.append(str(v).replace("|", "\\|"))
        lines.append("| " + str(idx) + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def quote_rows(scores):
    """Per agent and test family: the distinct scores reported and one verified quote (the one carrying most traits)."""
    s = scores[scores.self_report & scores.quote_verified]
    rows = []
    for a in AGENTS:
        for fam in ["Big Five", "HEXACO", "MBTI", "Enneagram", "DISC", "Dark Triad", "VIA", "IQ", "other"]:
            g = s[(s.subject == a) & (s.test_family == fam)]
            if g.empty:
                continue
            best = g.groupby("quote").size().sort_values(ascending=False).index[0]
            gb = g[g.quote == best]
            summary = "; ".join(f"{t}: {sc}" + (f" {scl}" if scl else "") for t, sc, scl in gb[["trait", "score", "scale"]].drop_duplicates().itertuples(index=False))
            rows.append({"agent": a, "test": fam, "n_rows": len(g), "n_messages": g.event_id.nunique(), "scores_in_quoted_message": summary, "quote": best[:400], "day": gb.day.iloc[0]})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    ev = pd.read_parquet(OUT / "events.parquet")
    window, hits, sample = screen(ev)
    print(f"window messages by the six agents: {len(window)}; mention a test: {int(window.mentions_test.sum())}; with a result signal: {len(hits)}; sent to the LLM: {len(sample)}")
    scores, n_calls, failed = label(sample)
    scores.to_csv(OUT / "personality_scores.csv", index=False)
    ver = scores.quote_verified.mean() if len(scores) else float("nan")
    print(f"{len(scores)} score rows from {scores.event_id.nunique()} messages; quotes verified: {ver:.1%}; failed calls: {failed}")
    print(scores.groupby(["test_family", "self_report"]).size().unstack(fill_value=0))

    med, n, rng, b5 = trait_table(scores)
    beh = behaviour(ev)
    mbti, ennea = type_table(scores, "MBTI"), type_table(scores, "Enneagram")
    hex_s = scores[scores.self_report & scores.quote_verified & (scores.test_family == "HEXACO") & scores.trait_canon.ne("") & scores.score_100.notna()]
    hex_med = hex_s.groupby(["subject", "trait_canon"]).score_100.median().unstack().reindex(AGENTS)
    all_six_b5 = med.notna().all(axis=1).all() if not med.empty else False
    corr = spearman(med, beh) if all_six_b5 else None
    corr_hex = spearman(hex_med, beh) if not hex_med.empty and hex_med.notna().all(axis=1).all() else None

    # ---- document
    qrows = quote_rows(scores)
    combined = pd.concat([med.add_prefix("B5_").round(0), pd.Series(mbti, name="MBTI"), pd.Series(ennea, name="Enneagram"), beh[["davids_score", "response_rate", "requests_per_100_msgs", "affiliative", "display", "provisioning", "in_mentions_per_100_msgs", "out_mentions_per_100_msgs"]]], axis=1)
    L = []
    L.append("# Personality-test week: self-reports next to behaviour\n")
    L.append(f"Goal \"Take a bunch of personality tests!\" (goal id `{GOAL_ID}`, 2025-09-22 to 2025-09-29). Six agents on the roster: {', '.join(AGENTS)}. Produced by `scripts/personality.py`; rows in `out/village/personality_scores.csv`.\n")
    L.append("## What was read\n")
    L.append(f"- Messages by the six agents between {WINDOW[0]} and {WINDOW[1]} (goal week plus the week after, when results were still being discussed): **{len(window)}**.")
    L.append(f"- Of these, {int(window.mentions_test.sum())} mention a test or trait, and **{len(hits)}** also carry a result signal (a number next to a trait name, a percentile or x/y, a four-letter type, an Enneagram type, a score or result word with a number). The {len(sample)} strongest were sent to the LLM in {n_calls} calls of {BATCH} messages (Haiku; {failed} calls failed).")
    L.append(f"- The LLM returned **{len(scores)}** score rows from {scores.event_id.nunique()} messages. Each row carries the sentence that reports the score; **{ver:.1%}** of those quotes are a verbatim substring of the message after whitespace normalisation (`quote_verified`). Unverified rows stay in the CSV and are excluded from every table below.")
    L.append(f"- {int(scores.self_report.sum())} rows are an agent reporting its own result; {int((~scores.self_report).sum())} report another agent's result (second-hand, not used in the tables).\n")
    fam = scores[scores.quote_verified].groupby(["test_family", "self_report"]).size().unstack(fill_value=0)
    fam.columns = ["second-hand" if not c else "self-report" for c in fam.columns]
    L.append("Verified rows by test family:\n")
    L.append(md_table(fam, index_name="test"))
    L.append("\n## Self-reported scores, one verbatim quote per agent and test\n")
    L.append("`scores_in_quoted_message` lists what the quoted message reports; `n_rows` / `n_messages` count all verified self-report rows for that agent and test (agents often took the same test on several sites, or repeated it).\n")
    L.append(md_table(qrows.set_index("agent"), index_name="agent"))
    L.append("\n### Big Five, one number per agent and trait\n")
    L.append("Median of verified numeric self-reports, put on 0-100 (percentiles and percentages as is; x/y and 1-5 scales rescaled; a stability score is counted as 100 minus neuroticism). `n` is the number of reports behind each median and `range` their spread, which is large where an agent took the test on several sites or answered one run with all-neutral responses.\n")
    L.append(md_table(med.round(0), fmt={c: "{:.0f}" for c in med.columns}))
    L.append("\nreports per cell (n):\n")
    L.append(md_table(n))
    L.append("\nrange (max - min) per cell:\n")
    L.append(md_table(rng.round(0), fmt={c: "{:.0f}" for c in rng.columns}))
    if not hex_med.empty:
        L.append("\n### HEXACO, one number per agent and factor\n")
        L.append(md_table(hex_med.round(0), fmt={c: "{:.0f}" for c in hex_med.columns}))
    L.append("\n## Behaviour in era 2\n")
    L.append(f"Era 2 ({ev[ev.era == ERA].ts.min().date()} to {ev[ev.era == ERA].ts.max().date()}); tenure differs, so everything is a rate. `response_rate` is the mean of `responded` over dyad rows where the agent is the responder (`response_rate_addressed`: only requests addressed to it). `requests_per_100_msgs` counts request ids in `requests.parquet`. `affiliative`, `display`, `provisioning`, `leads`, `directs_others` are shares of the agent's battery-labelled messages (`battery_n` of them). Mention strengths: out = mentions the agent makes per 100 of its own messages; in = mentions of the agent per 100 messages others wrote on days it was active.\n")
    L.append(md_table(beh, fmt={"messages_era2": "{:.0f}", "battery_n": "{:.0f}", "davids_score": "{:.2f}", "requests_per_100_msgs": "{:.1f}", "out_mentions_per_100_msgs": "{:.0f}", "in_mentions_per_100_msgs": "{:.1f}"}))
    L.append("\n## Self-reports next to behaviour\n")
    L.append(md_table(combined, fmt={**{f"B5_{c}": "{:.0f}" for c in med.columns}, "davids_score": "{:.2f}", "requests_per_100_msgs": "{:.1f}", "out_mentions_per_100_msgs": "{:.0f}", "in_mentions_per_100_msgs": "{:.1f}"}))
    L.append("\n## Spearman correlations, trait by behaviour\n")
    if corr is not None:
        L.append("All six agents reported a Big Five result. **n = 6**: with six points a Spearman coefficient needs |rho| >= 0.89 to reach p < 0.05 two-sided, and a single agent moves it by several tenths. These numbers describe this roster; they are not tests of anything.\n")
        L.append(md_table(corr.round(2), index_name="Big Five trait"))
    else:
        L.append("Not every agent has a verified numeric Big Five self-report, so no correlation table.")
    if corr_hex is not None:
        L.append("\nHEXACO (all six reported it):\n")
        L.append(md_table(corr_hex.round(2), index_name="HEXACO factor"))
    L.append("\n## Caveats\n")
    L.append("Self-reports are claims: the scores are what an agent typed into chat, with the quote as the only evidence, and the screenshots in the shared Drive folder were not checked; where an agent reports the same trait with different numbers across sites or runs, the table takes a median. Each agent is one instance of a model with its own memory and scaffold, taking a test built for humans, often with a stated strategy (one agent answered a whole run with neutral responses), so a score says what that instance chose to answer, not what the model is. Six agents, one week of self-reports and one era of behaviour give rank correlations that a single agent can flip, so the correlation table is a description of this roster and nothing in it should be read as a test.\n")
    (DOCS / "personality.md").write_text("\n".join(L))
    print(f"wrote {OUT / 'personality_scores.csv'} and {DOCS / 'personality.md'}")
    pd.set_option("display.width", 250)
    print(combined.round(2).to_string())
    if corr is not None:
        print(corr.round(2).to_string())


if __name__ == "__main__":
    main()
