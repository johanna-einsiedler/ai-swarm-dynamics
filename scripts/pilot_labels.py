"""Pilot for the two LLM labelling passes.

A. Label a sample of extracted questions: is it a request, what kind, to whom.
B. For the requests found in A, read the messages that followed and judge
   which of them respond. Then check how many of those responses the cheap
   signals (mention of the requester, text overlap, lexical similarity) would
   have found on their own.
"""
import re
import sys
from pathlib import Path

import pandas as pd

from swarm_sna import llm

OUT = Path("out/pilot")
CACHE = Path("out/llm_cache")
N_QUESTIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
MAX_REQUESTS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
WINDOW_MIN = 30
MAX_FOLLOWING = 25

SYSTEM_A = """You label questions taken from a group chat between AI agents working on shared goals.
For each numbered item you get the speaker, the sentence before, the QUESTION, and the sentence after.
Return a JSON array with one object per item, in order:
{"i": <item number>,
 "is_request": true if the speaker wants someone else to answer or do something; false for rhetorical questions, self-talk, headings, quoted questions, or questions the speaker answers itself,
 "kind": "information" (wants a fact or status) | "action" (wants something done) | "review" (wants feedback or checking) | "help" (stuck, wants assistance) | "decision" (wants a choice or approval) | "none",
 "addressed": "named" (to specific agents) | "broadcast" (to anyone) | "human" (to a human or staff) | "none",
 "addressees": [names exactly as written, empty if not named]}
JSON only."""

SYSTEM_B = """You judge whether a request in a group chat between AI agents was answered.
You get the REQUEST message (the question is marked) and the numbered messages by other participants that followed in the same room within 30 minutes.
Return one JSON object:
{"responses": [{"n": <message number>, "type": "answers" (gives what was asked) | "acts" (reports doing what was asked) | "acknowledges" (reacts without delivering) | "declines"}],
 "answered": true if at least one message answers or acts,
 "first_answer_n": <number of the first message that answers or acts, or null>}
Only list messages that respond to THIS request. A message that just happens to follow is not a response. JSON only."""

STOP = set("the a an and or of to in on for with is are was were be been it this that i you we they my your our at as by from have has had will would can could should not no do does did if so but about just now me us them their its".split())


def words(t):
    return {w for w in re.findall(r"[a-z0-9][a-z0-9\-_.]{2,}", t.lower()) if w not in STOP}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = Path("out/village")
    ev = pd.read_parquet(d / "events.parquet")
    q = pd.read_parquet(d / "questions.parquet")
    m = pd.read_parquet(d / "mentions.parquet")
    o = pd.read_parquet(d / "overlaps.parquet", columns=["event_id", "source_id", "n_chars"])
    msgs = ev[(ev.type == "message") & (ev.actor_class != "system")].sort_values("ts").reset_index(drop=True)
    text = msgs.set_index("event_id").text

    # ---- A: label questions
    pool = q[q.actor_class == "agent"].merge(ev[["event_id", "era"]], on="event_id")
    sample = pd.concat([g.sample(min(len(g), N_QUESTIONS // 3), random_state=7) for _, g in pool.groupby("era")]).reset_index(drop=True)
    items = [f"[{i}] speaker: {r.actor}\nbefore: {r.context_before[:300]}\nQUESTION: {r.question[:500]}\nafter: {r.context_after[:300]}" for i, r in enumerate(sample.itertuples())]
    batches = [items[i : i + 20] for i in range(0, len(items), 20)]
    res = llm.map_calls(SYSTEM_A, ["\n\n".join(b) for b in batches], CACHE)
    lab = {x["i"]: x for r in res if r for x in r if isinstance(x, dict) and "i" in x}
    for col in ["is_request", "kind", "addressed", "addressees"]:
        sample[col] = [lab.get(i, {}).get(col) for i in range(len(sample))]
    sample.to_csv(OUT / "questions_labelled.csv", index=False)
    done = sample[sample.is_request.notna()]
    print(f"A. labelled {len(done)}/{len(sample)} questions")
    print("   is_request:", done.is_request.value_counts().to_dict())
    print("   kind (requests):", done[done.is_request == True].kind.value_counts().to_dict())  # noqa: E712
    print("   addressed (requests):", done[done.is_request == True].addressed.value_counts().to_dict())  # noqa: E712
    print("   request share by era:", done.groupby("era").is_request.agg(lambda x: x.astype(bool).mean()).round(2).to_dict())

    # ---- B: was it answered?
    reqs = done[(done.is_request == True) & (done.addressed != "human")].drop_duplicates("event_id").head(MAX_REQUESTS)  # noqa: E712
    idx = {e: i for i, e in enumerate(msgs.event_id)}
    mention_targets = m.groupby("event_id").target.agg(set).to_dict()
    overlap_pairs = set(zip(o.event_id, o.source_id))
    prompts, windows = [], []
    for r in reqs.itertuples():
        i = idx[r.event_id]
        req = msgs.loc[i]
        w = msgs[(msgs.index > i) & (msgs.room == req.room) & (msgs.ts <= req.ts + pd.Timedelta(minutes=WINDOW_MIN)) & (msgs.actor != req.actor)].head(MAX_FOLLOWING)
        windows.append(w)
        marked = req.text[:1200].replace(r.question[:200], f">>> {r.question[:200]} <<<")
        follow = "\n".join(f"[{n}] +{(x.ts - req.ts).total_seconds() / 60:.0f}min {x.actor}: {x.text[:400]}" for n, x in enumerate(w.itertuples()))
        prompts.append(f"REQUEST from {req.actor}:\n{marked}\n\nFOLLOWING MESSAGES:\n{follow or '(none)'}")
    res = llm.map_calls(SYSTEM_B, prompts, CACHE)

    rows = []
    for r, w, out in zip(reqs.itertuples(), windows, res):
        if out is None:
            continue
        resp = {x["n"]: x["type"] for x in out.get("responses", []) if isinstance(x, dict) and "n" in x}
        rw = words(text[r.event_id])
        sims = [len(rw & words(x.text)) / max(1, len(rw | words(x.text))) for x in w.itertuples()]
        rank = pd.Series(sims).rank(ascending=False, method="min").tolist() if sims else []
        for n, x in enumerate(w.itertuples()):
            rows.append(
                {
                    "request_id": r.event_id, "requester": r.actor, "kind": r.kind, "addressed": r.addressed, "era": r.era, "n_following": len(w),
                    "n": n, "responder": x.actor, "latency_min": (x.ts - msgs.loc[idx[r.event_id]].ts).total_seconds() / 60,
                    "llm_type": resp.get(n), "is_response": resp.get(n) in ("answers", "acts"),
                    "mentions_requester": r.actor in mention_targets.get(x.event_id, set()),
                    "overlaps_request": (x.event_id, r.event_id) in overlap_pairs,
                    "lex_sim": sims[n], "lex_rank": rank[n],
                    "request_question": r.question[:300], "response_text": x.text[:400],
                }
            )
    pairs = pd.DataFrame(rows)
    pairs.to_csv(OUT / "request_responses.csv", index=False)
    per_req = pairs.groupby("request_id").agg(answered=("is_response", "any"), n_following=("n_following", "first"), addressed=("addressed", "first"))
    n_empty = sum(len(w) == 0 for w in windows)
    print(f"\nB. judged {len(per_req) + n_empty} requests ({n_empty} had no following messages)")
    print(f"   answered within {WINDOW_MIN} min: {per_req.answered.sum()}/{len(per_req) + n_empty}")
    print("   answered by addressing:", per_req.groupby("addressed").answered.agg(["sum", "size"]).to_dict("index"))
    print("   response types:", pairs.llm_type.value_counts().to_dict())
    resp = pairs[pairs.is_response]
    if len(resp):
        print(f"   of {len(resp)} real responses: mention the requester {resp.mentions_requester.mean():.0%}, share 25+ chars {resp.overlaps_request.mean():.0%}, either {(resp.mentions_requester | resp.overlaps_request).mean():.0%}, top-3 lexical similarity {(resp.lex_rank <= 3).mean():.0%}")
        non = pairs[~pairs.is_response]
        print(f"   of {len(non)} non-responses: mention the requester {non.mentions_requester.mean():.0%}, share 25+ chars {non.overlaps_request.mean():.0%}, top-3 lexical similarity {(non.lex_rank <= 3).mean():.0%}")
        print(f"   median latency of first response: {resp.groupby('request_id').latency_min.min().median():.1f} min")


if __name__ == "__main__":
    main()
