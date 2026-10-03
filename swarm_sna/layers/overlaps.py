"""Layer B2: text overlap. One row per (message, earlier message it copies from).

A message is linked to an earlier message by someone else in the same room if
they share a run of at least MIN_CHARS characters (case and whitespace
normalised). The longest shared run is recorded.
"""
import re
from collections import deque

import pandas as pd

MIN_CHARS = 25
WINDOW_MIN = 30
_WS = re.compile(r"\s+")


def normalise(text):
    return _WS.sub(" ", text).strip().lower()


def _shingles(t, n):
    """n-gram -> first position in t"""
    out = {}
    for i in range(len(t) - n + 1):
        out.setdefault(t[i : i + n], i)
    return out


def _longest_run(child, positions, n):
    """Longest stretch of `child` covered by consecutive shared n-grams."""
    pos = sorted(positions)
    best = cur = (pos[0], pos[0])
    for p in pos[1:]:
        cur = (cur[0], p) if p == cur[1] + 1 else (p, p)
        if cur[1] - cur[0] > best[1] - best[0]:
            best = cur
    return child[best[0] : best[1] + n]


def extract(events, min_chars=MIN_CHARS, window_min=WINDOW_MIN):
    msgs = events[events.type == "message"].sort_values("ts", kind="stable")
    window = pd.Timedelta(minutes=window_min)
    rows = []
    for room, g in msgs.groupby("room", sort=False):
        ids, ts, actors, classes, days = g.event_id.tolist(), g.ts.tolist(), g.actor.tolist(), g.actor_class.tolist(), g.day.tolist()
        index = {}  # n-gram -> messages in the window that contain it, oldest first
        live = deque()  # (message idx, its n-grams)
        for i, text in enumerate(g.text):
            norm = normalise(text)
            sh = _shingles(norm, min_chars)
            while live and ts[i] - ts[live[0][0]] > window:
                _, old = live.popleft()
                for s in old:
                    lst = index[s]
                    lst.popleft()
                    if not lst:
                        del index[s]
            hits = {}  # source idx -> positions in this message
            for s, p in sh.items():
                for j in index.get(s, ()):
                    hits.setdefault(j, []).append(p)
            for j, positions in hits.items():
                if actors[j] == actors[i]:
                    continue
                run = _longest_run(norm, positions, min_chars)
                rows.append((ids[i], ids[j], ts[i], days[i], actors[i], classes[i], actors[j], classes[j], room, (ts[i] - ts[j]).total_seconds(), len(run), len(positions), run))
            for s in sh:
                index.setdefault(s, deque()).append(i)
            live.append((i, sh.keys()))
    return pd.DataFrame(
        rows,
        columns=["event_id", "source_id", "ts", "day", "actor", "actor_class", "target", "target_class", "room", "latency_s", "n_chars", "n_shared_ngrams", "shared_text"],
    )
