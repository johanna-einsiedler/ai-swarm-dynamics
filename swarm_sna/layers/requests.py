"""Layer C: requests and the responses to them, labelled by an LLM.

Every label is backed by a verbatim quote from the message it describes. The
quote is checked against the message text; `quote_verified` is False when the
model did not copy it exactly, and those rows should not be used as evidence.

requests    one row per request: who asked, the quoted request, kind, addressees
responses   one row per (request, message that responds): type, the quoted reply, and
            `evidence` - the link, id, number or output that lets a reader check the
            claim. `backed` is True when that evidence is present and verbatim.

A directive (Layer E) is a request with `directive` true and `addressed` "named".
"""
import hashlib
import re

import numpy as np
import pandas as pd

from .. import llm
from . import mentions

REQUEST_PHRASE = re.compile(
    r"\b(?:please|could you|can you|would you|can someone|could someone|can anyone|anyone able|need you to|who can|any volunteers|let me know)\b",
    re.IGNORECASE,
)
# Imperatives and hand-offs that carry no question mark and no politeness marker.
# Measured: the base screen alone finds about 58% of request-bearing messages.
EXTRA_PHRASE = re.compile(
    r"\b(?:ready for (?:your|review)|would appreciate|I'?d appreciate|I'?d suggest|I would suggest|go ahead and|feel free to"
    r"|standing by for|we need to|if you'?re willing|if you want,|over to you|your turn|awaiting your|action required|required:|needs? your)\b",
    re.IGNORECASE,
)
BATCH = 20
JUDGE_BATCH = 2
MAX_CHARS = 1500
WINDOW_MIN = 30
MAX_FOLLOWING = 45  # a 30-min window holds 49 messages on average; 15 saw less than a third of them
REPLY_CHARS = None  # show replies in full; a cut of 400 hid a third of the average reply

SYSTEM_REQUESTS = """You find requests in messages from a group chat between AI agents working on goals.
A request is where the speaker wants someone else to answer or do something. Rhetorical questions, self-talk, headings, quoted questions, questions the speaker answers itself, and open offers of help ("let me know if you need anything") are not requests.
Return a JSON array with one object per numbered message, in order:
{"i": <message number>,
 "requests": [up to two, the most important first; empty list if the message makes no request
   {"quote": the sentence that makes the request, copied character for character from the message (no paraphrase, no ellipsis),
    "kind": "information" (wants a fact or status) | "action" (wants something done) | "review" (wants feedback or checking) | "help" (stuck, wants assistance) | "decision" (wants a choice or approval),
    "addressed": "named" (to specific agents) | "broadcast" (to anyone) | "human" (to a human or staff),
    "addressees": [names exactly as written in the message, empty if not named],
    "directive": true if it tells the other what to do, false if it asks,
    "beneficiary": "self" (serves the speaker's own task) | "shared" (serves a task the group shares) | "other" (serves someone else's task)}]}
JSON only."""

SYSTEM_RESPONSES = """You judge whether requests in a group chat between AI agents were answered.
You get several numbered REQUEST messages (the request is marked with >>> <<<), each with the numbered messages by other participants that followed in the same room within 30 minutes.
Return a JSON array with one object per request, in order:
{"r": <request number>,
 "responses": [{"n": <number of the following message>,
   "type": "answers" (gives what was asked) | "acts" (reports doing what was asked) | "acknowledges" (reacts without delivering) | "declines" (refuses or says it cannot) | "redirects" (passes it to someone else),
   "quote": the part of that message that responds to the request, copied character for character (no paraphrase, no ellipsis),
   "evidence": the part of that message that lets a reader check the request was really satisfied - a link, id, number, file name, command, or quoted output - copied character for character; "" if the message only claims it with nothing checkable}]}
Only list messages that respond to that request. A message that just happens to follow is not a response. JSON only."""


SYSTEM_VERIFY = """You check whether a chat message is a response to one specific request made earlier in the same group chat between AI agents.
For each numbered pair you get the REQUEST (who asked, whom it was addressed to, and the message it came from with the request marked >>> <<<) and a later MESSAGE by another agent.
The MESSAGE responds to the request only if it answers that question, does or reports doing what was asked, explicitly takes it up, or declines it. A message that is merely on a related topic, reports its author's own separate work, or replies to someone else is not a response.
Return a JSON array with one object per pair, in order: {"i": <pair number>, "why": at most 12 words, "responds": true or false}
JSON only."""
VERIFY_BATCH = 12
VERIFY_REQ_CHARS, VERIFY_RESP_CHARS = 700, 900


def verify_links(events, requests, responses, cache_dir, workers=10, model=llm.MODEL):
    """Second opinion on each (request, response) link, one pair at a time.
    The first pass reads up to 45 following messages at once and over-links; a focused check is an easier task.
    -> responses with `link_verified` (True / False, or None where the check failed)."""
    text = events.set_index("event_id").text
    req = requests.set_index("request_id")
    todo = responses[responses.quote_verified & responses.request_id.isin(req.index)]
    # Fixed random order over requests, links of one request kept together: if the pass is cut
    # short, the checked part is a random sample of whole requests, not of links.
    todo = todo.iloc[np.lexsort((todo.ts.to_numpy(), _order(todo.request_id).to_numpy()))]

    def block(i, r):
        q = req.loc[r.request_id]
        src = text[q.event_id]
        start = max(0, src.find(q.quote[:40]) - 250)
        marked = src[start : start + VERIFY_REQ_CHARS].replace(q.quote, f">>> {q.quote} <<<")
        to = ", ".join(q.targets) if len(q.targets) else "anyone"
        reply = text[r.event_id]
        at = max(0, reply.find(str(r.quote)[:40]) - 300)  # keep the quoted part in view
        return f"=== PAIR {i}\nREQUEST from {q.actor} to {to}:\n{marked}\n\nMESSAGE from {r.actor}, {r.latency_s / 60:.0f} min later:\n{reply[at : at + VERIFY_RESP_CHARS]}"

    rows = list(todo.itertuples())
    batches = [rows[i : i + VERIFY_BATCH] for i in range(0, len(rows), VERIFY_BATCH)]
    print(f"verifying {len(rows):,} request-response links", flush=True)
    res = llm.map_calls(SYSTEM_VERIFY, ["\n\n".join(block(i, r) for i, r in enumerate(b)) for b in batches], cache_dir, workers=workers, model=model)
    verdict = {}
    for b, out in zip(batches, res):
        for x in out if isinstance(out, list) else []:
            if isinstance(x, dict) and isinstance(x.get("i"), int) and 0 <= x["i"] < len(b) and isinstance(x.get("responds"), bool):
                verdict[b[x["i"]].Index] = x["responds"]
    out = responses.copy()
    out["link_verified"] = pd.Series(verdict, dtype=object).reindex(out.index)
    checked = out.link_verified.dropna()
    print(f"checked {len(checked):,}/{len(rows):,}; confirmed {checked.astype(bool).mean():.0%}")
    return out


def _clip(t):
    return t if REPLY_CHARS is None else t[:REPLY_CHARS]


def _norm(s):
    s = s.translate(str.maketrans("“”‘’", "\"\"''"))
    return re.sub(r"\s+", " ", s).strip(" \"'*_`.…")


def verified(quote, text):
    return bool(quote) and _norm(quote) in _norm(text)


def candidates(events, questions, screen="base"):
    """Agent messages that could hold a request.

    base  a question, or a request phrase.
    wide  an imperative or hand-off phrase that the base screen does not already catch.
    """
    m = events[(events.type == "message") & (events.actor_class == "agent")]
    base = m.event_id.isin(questions.event_id) | m.text.str.contains(REQUEST_PHRASE)
    hit = base if screen == "base" else m.text.str.contains(EXTRA_PHRASE) & ~base
    return m[hit].sort_values("ts").reset_index(drop=True)


def label_requests(events, agents, questions, cache_dir, workers=10, limit=0, seed=0, screen="base"):
    cand = candidates(events, questions, screen)
    if limit:
        cand = cand.sample(min(limit, len(cand)), random_state=seed).sort_values("ts").reset_index(drop=True)
    print(f"labelling {len(cand):,} candidate messages", flush=True)
    items = [f"[{i}] {r.actor}: {r.text[:MAX_CHARS]}" for i, r in enumerate(cand.itertuples())]
    res = llm.map_calls(SYSTEM_REQUESTS, ["\n\n".join(items[i : i + BATCH]) for i in range(0, len(items), BATCH)], cache_dir, workers=workers)
    lab = {x["i"]: x for r in res if r for x in r if isinstance(x, dict) and "i" in x}

    rows = []
    for i, m in enumerate(cand.itertuples()):
        for j, x in enumerate((lab.get(i) or {}).get("requests") or []):
            if not isinstance(x, dict):
                continue
            quote = str(x.get("quote") or "")
            rows.append(
                {
                    "request_id": f"{m.event_id}:{j}", "event_id": m.event_id, "ts": m.ts, "day": m.day, "actor": m.actor, "room": m.room,
                    "goal_id": m.goal_id, "era": m.era, "active_n": m.active_n,
                    "quote": quote, "quote_verified": verified(quote, m.text),
                    "kind": x.get("kind"), "addressed": x.get("addressed"), "directive": bool(x.get("directive")), "beneficiary": x.get("beneficiary"),
                    "addressees": [str(a) for a in x.get("addressees") or []],
                }
            )
    req = pd.DataFrame(rows)
    print(f"labelled {len(lab):,}/{len(cand):,} messages -> {len(req):,} requests, quote verified {req.quote_verified.mean():.0%}")

    # Resolve the names as written to agents, with the same rules as the mention layer.
    named = req[req.addressees.str.len() > 0]
    fake = named.assign(event_id=named.request_id, text=named.addressees.str.join("; "), type="message", actor_class="agent")
    hits = mentions.extract(fake, agents, active=mentions.active_sets(events))
    req["targets"] = req.request_id.map(hits.groupby("event_id").target.agg(list)).apply(lambda t: t if isinstance(t, list) else [])
    return req


def _order(request_ids):
    """A fixed pseudo-random order, so a larger sample extends a smaller one and reuses its cached calls."""
    return request_ids.map(lambda r: hashlib.sha256(r.encode()).hexdigest())


def repair_attribution(events, requests, responses):
    """The model sometimes quotes the right message but names the wrong number.
    The quote is the ground truth, so re-point each unverified row at the message that holds it."""
    msgs = events[(events.type == "message") & (events.actor_class != "system")].sort_values("ts").reset_index(drop=True)
    pos = {e: i for i, e in enumerate(msgs.event_id)}
    req = requests.set_index("request_id")
    fixed, dropped = 0, 0
    out = responses.copy()
    for i, r in responses[~responses.quote_verified].iterrows():
        src = req.loc[r.request_id] if r.request_id in req.index else None
        if src is None or not str(r.quote).strip():
            continue
        j = pos.get(src.event_id)
        near = msgs.iloc[j + 1 : j + 1500]
        w = near[(near.room == src.room) & (near.ts <= src.ts + pd.Timedelta(minutes=WINDOW_MIN)) & (near.actor != src.actor)].head(MAX_FOLLOWING)
        hit = w[[verified(r.quote, t) for t in w.text]]
        if len(hit):
            m = hit.iloc[0]
            out.loc[i, ["event_id", "ts", "actor", "actor_class", "latency_s", "quote_verified"]] = [m.event_id, m.ts, m.actor, m.actor_class, (m.ts - src.ts).total_seconds(), True]
            out.loc[i, ["evidence_verified", "backed"]] = [verified(r.evidence, m.text), bool(r.evidence) and verified(r.evidence, m.text)]
            fixed += 1
        else:
            dropped += 1
    print(f"re-pointed {fixed:,} misattributed responses; {dropped:,} quotes found in no window message and left unverified")
    return out


def judge_responses(events, requests, cache_dir, workers=10, named_sample=6000, broadcast_sample=2000):
    """The first `named_sample` named and `broadcast_sample` broadcast requests in a fixed random order (0 = all).
    -> (requests with `judged`, responses)"""
    msgs = events[(events.type == "message") & (events.actor_class != "system")].sort_values("ts").reset_index(drop=True)
    pos = {e: i for i, e in enumerate(msgs.event_id)}
    ok = requests[requests.quote_verified & (requests.addressed != "human")]
    ok = ok.iloc[_order(ok.request_id).argsort()]
    named, broad = ok[ok.addressed == "named"], ok[ok.addressed == "broadcast"]
    todo = pd.concat([named.head(named_sample or len(named)), broad.head(broadcast_sample or len(broad))])
    print(f"judging {len(todo):,} requests ({(todo.addressed == 'named').sum():,} of {len(named):,} named, {(todo.addressed == 'broadcast').sum():,} of {len(broad):,} broadcast)", flush=True)

    windows = {}
    for r in todo.itertuples():
        i = pos[r.event_id]
        src = msgs.loc[i]
        near = msgs.iloc[i + 1 : i + 1500]  # rooms interleave, so look well past the window in the global stream
        windows[r.request_id] = near[(near.room == src.room) & (near.ts <= src.ts + pd.Timedelta(minutes=WINDOW_MIN)) & (near.actor != src.actor)].head(MAX_FOLLOWING)

    def block(n, r):
        text = msgs.text[pos[r.event_id]]
        start = max(0, text.find(r.quote[:40]) - 600)
        marked = text[start : start + MAX_CHARS].replace(r.quote, f">>> {r.quote} <<<")
        follow = "\n".join(f"[{k}] +{(x.ts - r.ts).total_seconds() / 60:.0f}min {x.actor}: {_clip(x.text)}" for k, x in enumerate(windows[r.request_id].itertuples()))
        return f"=== REQUEST {n} from {r.actor}:\n{marked}\n\nFOLLOWING MESSAGES for request {n}:\n{follow}"

    # A request with no message after it needs no call.
    ask = [r for r in todo.itertuples() if len(windows[r.request_id])]
    batches = [ask[i : i + JUDGE_BATCH] for i in range(0, len(ask), JUDGE_BATCH)]
    res = llm.map_calls(SYSTEM_RESPONSES, ["\n\n".join(block(n, r) for n, r in enumerate(b)) for b in batches], cache_dir, workers=workers)

    rows, judged = [], {r.request_id: 0 for r in todo.itertuples() if not len(windows[r.request_id])}
    for b, out in zip(batches, res):
        for x in out if isinstance(out, list) else []:
            if not isinstance(x, dict) or not isinstance(x.get("r"), int) or not 0 <= x["r"] < len(b):
                continue
            r = b[x["r"]]
            w = windows[r.request_id]
            judged[r.request_id] = len(w)
            for y in x.get("responses") or []:
                if not isinstance(y, dict) or not isinstance(y.get("n"), int) or not 0 <= y["n"] < len(w):
                    continue
                m = w.iloc[y["n"]]
                quote, ev = str(y.get("quote") or ""), str(y.get("evidence") or "")
                rows.append(
                    {
                        "request_id": r.request_id, "event_id": m.event_id, "ts": m.ts, "actor": m.actor, "actor_class": m.actor_class,
                        "latency_s": (m.ts - r.ts).total_seconds(), "type": y.get("type"), "quote": quote, "quote_verified": verified(quote, m.text),
                        "evidence": ev, "evidence_verified": verified(ev, m.text), "backed": bool(ev) and verified(ev, m.text),
                    }
                )
    resp = pd.DataFrame(rows, columns=["request_id", "event_id", "ts", "actor", "actor_class", "latency_s", "type", "quote", "quote_verified", "evidence", "evidence_verified", "backed"])
    requests = requests.assign(judged=requests.request_id.isin(judged.keys()), n_following=requests.request_id.map(judged))
    print(f"judged {len(judged):,}/{len(todo):,} requests -> {len(resp):,} responses, quote verified {resp.quote_verified.mean():.0%}, backed by checkable evidence {resp.backed.mean():.0%}")
    return requests, resp
