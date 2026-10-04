"""A small synthetic swarm with known structure, so the tool can be run and checked without any dataset.

Planted: agents answer the agents who addressed them (reciprocity) and prefer their own
team (homophily). Not planted: any hierarchy. A report on this file should find the first
two and nothing else.

usage: make_example.py [output directory]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "examples/toy")
TEAMS = {"red": ["Ada", "Bo", "Cy", "Dee"], "blue": ["Eli", "Fay", "Gus", "Hal"]}
WORK = ["the parser", "the test suite", "the deploy script", "the landing page", "the data loader", "the release notes"]
ASK = ["{to}, can you review {w}?", "{to}, could you take a look at {w}?", "{to}, what is the status of {w}?", "{to}, please rerun {w} and report back."]
ANSWER = ["{to}, done: {w} passes, see run #{n}.", "{to}, I reviewed {w} and left {n} comments.", "{to}, {w} is fixed in commit {n:x}."]
STATUS = ["Working on {w} now.", "Still on {w}; nothing blocking.", "Finished a pass over {w}."]


def main():
    rng = np.random.default_rng(7)
    team = {a: t for t, members in TEAMS.items() for a in members}
    agents = list(team)
    rows, owed = [], []  # owed: (who owes a reply, to whom, about what)
    t = pd.Timestamp("2026-01-05 09:00")
    for day in range(30):
        t = pd.Timestamp("2026-01-05 09:00") + pd.Timedelta(days=day)
        for _ in range(120):
            t += pd.Timedelta(seconds=int(rng.integers(20, 240)))
            if owed and rng.random() < 0.55:  # reciprocity: answer whoever asked
                a, to, w = owed.pop(int(rng.integers(len(owed))))
                text = rng.choice(ANSWER).format(to=to, w=w, n=int(rng.integers(100, 9999)))
            else:
                a = rng.choice(agents)
                w = rng.choice(WORK)
                if rng.random() < 0.5:
                    same = [b for b in agents if b != a and team[b] == team[a]]
                    other = [b for b in agents if team[b] != team[a]]
                    to = rng.choice(same) if rng.random() < 0.75 else rng.choice(other)  # homophily: ask your own team
                    text = rng.choice(ASK).format(to=to, w=w)
                    owed.append((to, a, w))
                else:
                    text = rng.choice(STATUS).format(w=w)
            rows.append({"ts": t.isoformat(), "actor": a, "room": "main", "text": text})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "events.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    pd.DataFrame({"agent": agents, "family": [team[a] for a in agents]}).to_csv(OUT / "agents.csv", index=False)
    print(f"wrote {len(rows):,} messages from {len(agents)} agents to {OUT}/")


if __name__ == "__main__":
    main()
