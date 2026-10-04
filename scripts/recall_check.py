"""How many requests does the cheap screen miss?

The request layer only sends messages with a question mark or a request phrase
to the LLM. This labels a random sample of the messages it skipped, so the
miss rate is measured rather than assumed.

usage: recall_check.py [sample size] [parallel calls]
"""
import sys
from pathlib import Path

import pandas as pd

from swarm_sna import llm
from swarm_sna.layers import requests as R

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
OUT, CACHE = Path("out/village"), Path("out/llm_cache")


def main():
    ev = pd.read_parquet(OUT / "events.parquet")
    q = pd.read_parquet(OUT / "questions.parquet")
    msgs = ev[(ev.type == "message") & (ev.actor_class == "agent")]
    screened = set(R.candidates(ev, q).event_id)
    skipped = msgs[~msgs.event_id.isin(screened)]
    print(f"{len(msgs):,} agent messages: {len(screened):,} screened in, {len(skipped):,} skipped")

    sample = skipped.sample(min(N, len(skipped)), random_state=3).reset_index(drop=True)
    items = [f"[{i}] {r.actor}: {r.text[:R.MAX_CHARS]}" for i, r in enumerate(sample.itertuples())]
    res = llm.map_calls(R.SYSTEM_REQUESTS, ["\n\n".join(items[i : i + R.BATCH] ) for i in range(0, len(items), R.BATCH)], CACHE, workers=WORKERS)
    lab = {x["i"]: x for r in res if r for x in r if isinstance(x, dict) and "i" in x}

    rows = []
    for i, m in enumerate(sample.itertuples()):
        for x in (lab.get(i) or {}).get("requests") or []:
            if isinstance(x, dict):
                rows.append({"event_id": m.event_id, "actor": m.actor, "text": m.text[:400], "quote": x.get("quote"), "kind": x.get("kind"), "addressed": x.get("addressed"), "verified": R.verified(str(x.get("quote") or ""), m.text)})
    hits = pd.DataFrame(rows)
    hits.to_csv("validation/recall_unscreened.csv", index=False)

    n_lab = len(lab)
    n_hit = hits[hits.verified].event_id.nunique() if len(hits) else 0
    rate = n_hit / n_lab if n_lab else 0
    print(f"labelled {n_lab}/{len(sample)} skipped messages; {n_hit} held a request with a verified quote ({rate:.1%})")
    missed = rate * len(skipped)
    found = pd.read_parquet(OUT / "requests.parquet").event_id.nunique()
    print(f"-> estimated {missed:,.0f} request-bearing messages missed by the screen, against {found:,} found")
    print(f"-> estimated recall of the screen: {found / (found + missed):.1%}")
    if len(hits):
        print("\nexamples of what the screen misses:")
        for h in hits[hits.verified].head(6).itertuples():
            print(f"  [{h.kind}/{h.addressed}] {h.quote[:140]}")


if __name__ == "__main__":
    main()
