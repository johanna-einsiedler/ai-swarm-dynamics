"""Layer B1: questions. One row per question sentence, with the sentences around it."""
import re

import pandas as pd

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])[\"'”’)\]*_]*\s+|\n+")
ASK_START = re.compile(
    r"^\W*(can|could|does|has|is|will|would) (anyone|someone|somebody|anybody)\b|^\W*(any volunteers|who can|who has|who is)\b",
    re.IGNORECASE,
)
STRIP = " \t*_>#-•`\"'“”"


def split_sentences(text):
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if s and s.strip(STRIP)]


def is_question(sentence):
    s = sentence.rstrip(STRIP + ")]")
    return s.endswith("?") or bool(ASK_START.match(sentence))


def extract(events):
    msgs = events[events.type == "message"]
    rows = []
    for m in msgs.itertuples():
        if "?" not in m.text and not ASK_START.search(m.text):
            continue
        sents = split_sentences(m.text)
        for i, s in enumerate(sents):
            if is_question(s):
                before = sents[i - 1] if i > 0 else ""
                after = sents[i + 1] if i + 1 < len(sents) else ""
                rows.append((m.event_id, m.ts, m.day, m.actor, m.actor_class, m.room, i, s, before, after))
    return pd.DataFrame(
        rows,
        columns=["event_id", "ts", "day", "actor", "actor_class", "room", "sentence_idx", "question", "context_before", "context_after"],
    )
