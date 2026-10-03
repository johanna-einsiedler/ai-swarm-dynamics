"""Layer B3: time mentions. One row per time expression in a message.

kind       what the expression is
  ago        elapsed time, looking back      "3 minutes ago", "for the past hour", "yesterday"
  eta        expected time, looking forward  "ETA ~5 min", "in about 10 minutes", "will take 2 hours"
  duration   a length of time, direction unclear   "a 30-minute session"
  clock      a clock time                    "1:47 PM", "11:13:05 AM PT"
  day_ref    a numbered day of the swarm     "Day 239"
seconds    the quantity in seconds (midpoint of a range), null for clock and day_ref
"""
import re

import pandas as pd

UNIT_S = {"second": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600, "hr": 3600, "day": 86400, "week": 604800, "month": 2592000}
WORD_N = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "ten": 10, "a few": 3, "few": 3, "a couple of": 2, "couple of": 2, "several": 4, "half an": 0.5}
_NUM = r"(?P<n1>\d+(?:\.\d+)?|" + "|".join(sorted(WORD_N, key=len, reverse=True)) + r")"
_UNIT = r"(?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?)"
QTY = rf"(?:~|≈|about |around |roughly |approx\.? |approximately |another |under |over |at least |up to )?{_NUM}(?:\s?(?:-|–|—|to)\s?(?P<n2>\d+(?:\.\d+)?))?[\s\-]?{_UNIT}\b"

PATTERNS = [
    ("ago", re.compile(rf"\b{QTY}\s+(?:ago|earlier|back)\b", re.I)),
    ("ago", re.compile(rf"\b(?:for|over|in|during|within)\s+the\s+(?:last|past|previous)\s+{QTY}", re.I)),
    ("eta", re.compile(rf"\b(?:ETA|estimated time)\b[^.\n]{{0,25}}?{QTY}", re.I)),
    ("eta", re.compile(rf"\b(?:will|should|would|might|may|could|going to)\s+(?:only\s+)?(?:take|need|require|be done in|be ready in|finish in)\s+{QTY}", re.I)),
    ("eta", re.compile(rf"\b(?:in|within)\s+(?:the\s+next\s+)?{QTY}(?!\s+(?:ago|earlier))", re.I)),
    ("eta", re.compile(rf"\b{QTY}\s+(?:left|remaining|remain|from now|until|to go)\b", re.I)),
    ("duration", re.compile(rf"\b{QTY}", re.I)),
]
RELATIVE = [
    ("ago", re.compile(r"\b(yesterday|last (?:week|night|session|month))\b", re.I)),
    ("eta", re.compile(r"\b(tomorrow|next (?:week|session|month))\b", re.I)),
]
RELATIVE_S = {"yesterday": 86400, "tomorrow": 86400, "week": 604800, "night": 86400, "session": None, "month": 2592000}
CLOCK = re.compile(r"(?<![\d:])(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?\s?(?:[AaPp]\.?[Mm]\.?)?(?:\s?(?:PT|PST|PDT|UTC|ET))?")
DAY_REF = re.compile(r"\bDay[\s\-‑]?(\d{1,4})\b")


def _seconds(m):
    n1 = m.group("n1").lower()
    a = WORD_N[n1] if n1 in WORD_N else float(n1)
    b = float(m.group("n2")) if m.group("n2") else a
    unit = m.group("unit").lower().rstrip("s")
    return (a + b) / 2 * UNIT_S[unit]


def find(text):
    """-> [(kind, seconds, matched text, position)], each stretch of text counted once."""
    out, taken = [], []

    def free(a, b):
        return all(b <= s or a >= e for s, e in taken)

    for kind, rx in PATTERNS:
        for m in rx.finditer(text):
            if free(*m.span()):
                taken.append(m.span())
                out.append((kind, _seconds(m), m.group(0), m.start()))
    for kind, rx in RELATIVE:
        for m in rx.finditer(text):
            if free(*m.span()):
                taken.append(m.span())
                out.append((kind, RELATIVE_S[m.group(1).lower().split()[-1]], m.group(0), m.start()))
    for m in CLOCK.finditer(text):
        if free(*m.span()):
            out.append(("clock", None, m.group(0).strip(), m.start()))
    for m in DAY_REF.finditer(text):
        out.append(("day_ref", None, m.group(0), m.start()))
    return sorted(out, key=lambda x: x[3])


def extract(events):
    msgs = events[events.type == "message"]
    rows = []
    for m in msgs.itertuples():
        if not any(ch.isdigit() for ch in m.text) and "day" not in m.text.lower() and "week" not in m.text.lower():
            continue
        for kind, seconds, matched, pos in find(m.text):
            rows.append((m.event_id, m.ts, m.day, m.actor, m.actor_class, m.room, kind, seconds, matched, pos))
    return pd.DataFrame(rows, columns=["event_id", "ts", "day", "actor", "actor_class", "room", "kind", "seconds", "matched", "pos"])
