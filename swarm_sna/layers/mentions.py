"""Layer A: mentions. One row per (message, mentioned agent).

`kind` is "at" for an @-mention and "name" for a plain string match on an
agent's name or short name; `pos` is the character offset of the first hit.

An alias shared by several agents ("Opus", "Gemini") resolves to the agents
that posted in that room that day; k candidates each get weight 1/k.
"""
import re

import pandas as pd

SEP = r"[\s\-_]?"


def _alias_pattern(alias):
    tokens = [re.escape(t) for t in re.split(r"[\s\-]+", alias) if t]
    return SEP.join(tokens)


def compile_aliases(agents):
    """-> (regex, {normalised alias: [agents]})"""
    owners = {}
    for agent, aliases in zip(agents.agent, agents.aliases):
        for al in aliases:
            owners.setdefault(_norm(al), []).append(agent)
    by_len = sorted({al for als in agents.aliases for al in als}, key=len, reverse=True)
    body = "|".join(_alias_pattern(al) for al in by_len)
    # No match inside a longer word or version: "GPT-5" must not fire on "GPT-5.1".
    rx = re.compile(rf"(@)?(?<![\w.\-])({body})(?![\w]|\.\d|[\-]\d)", re.IGNORECASE)
    return rx, owners


def _norm(s):
    return re.sub(r"[\s\-_()\[\]]+", "", s).lower()


def active_sets(events):
    talk = events[(events.type == "message") & (events.actor_class == "agent")]
    return talk.groupby(["room", "day"]).actor.agg(set).to_dict()


def extract(events, agents, active=None):
    """`active` ({(room, day): agents}) defaults to who posted in `events`; pass it when `events` is not the full stream."""
    rx, owners = compile_aliases(agents)
    msgs = events[events.type == "message"]
    if active is None:
        active = active_sets(events)
    joined = dict(zip(agents.agent, agents.joined))
    left = dict(zip(agents.agent, agents.left))

    rows = []
    for m in msgs.itertuples():
        seen = {}
        for hit in rx.finditer(m.text):
            key = _norm(hit.group(2))
            if key not in seen:
                seen[key] = (hit.group(2), bool(hit.group(1)), hit.start())
            elif hit.group(1):
                seen[key] = (seen[key][0], True, seen[key][2])
        if not seen:
            continue
        here = active.get((m.room, m.day), set())
        targets = {}
        for key, (alias, explicit, pos) in seen.items():
            cands = [a for a in owners.get(key, [])]
            pool = [a for a in cands if a in here]
            if not pool:  # nobody of that name spoke today: fall back to the roster
                pool = [a for a in cands if joined[a] <= m.ts and (pd.isna(left[a]) or m.ts <= left[a] + pd.Timedelta(days=1))]
            pool = [a for a in pool if a != m.actor]
            for a in pool:
                w = 1.0 / len(pool)
                # keep the most specific reading of each target within one message
                if a not in targets or w > targets[a][0]:
                    targets[a] = (w, len(pool), explicit, alias, pos)
        for a, (w, k, explicit, alias, pos) in targets.items():
            rows.append((m.event_id, m.ts, m.day, m.actor, m.actor_class, a, m.room, w, k, explicit, alias, pos))
    out = pd.DataFrame(
        rows, columns=["event_id", "ts", "day", "actor", "actor_class", "target", "room", "weight", "k", "explicit", "alias", "pos"]
    )
    out["kind"] = out.pop("explicit").map({True: "at", False: "name"})
    return out
