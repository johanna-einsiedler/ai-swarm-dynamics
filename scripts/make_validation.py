"""Write the hand-labelling sheets in validation/ from an extract (seed 42)."""
from pathlib import Path

import pandas as pd

D, V, N = Path("out/village"), Path("validation"), 50


def clip(s):
    return s[:700].replace("\r", " ")


def main():
    V.mkdir(exist_ok=True)
    ev = pd.read_parquet(D / "events.parquet", columns=["event_id", "text"])
    m, q, o = (pd.read_parquet(D / f"{n}.parquet") for n in ("mentions", "questions", "overlaps"))
    txt = ev.set_index("event_id").text

    mm = m[m.actor_class == "agent"].sample(N, random_state=42)
    pd.DataFrame(
        {
            "event_id": mm.event_id, "ts": mm.ts, "room": mm.room, "actor": mm.actor, "alias": mm.alias, "kind": mm.kind, "k": mm.k, "resolved_target": mm.target,
            "context": [txt[x.event_id][max(0, x.pos - 150) : x.pos + 250].replace("\r", " ") for x in mm.itertuples()],
            "target_correct": "", "is_addressed_to_target": "",
        }
    ).to_csv(V / "mentions.csv", index=False)

    qq = q[q.actor_class == "agent"].sample(N, random_state=42)
    pd.DataFrame(
        {"event_id": qq.event_id, "actor": qq.actor, "context_before": qq.context_before, "question": qq.question, "context_after": qq.context_after, "is_real_question": "", "expects_answer_from_another_agent": ""}
    ).to_csv(V / "questions.csv", index=False)

    oo = o[(o.actor_class == "agent") & (o.target_class == "agent") & (o.n_chars >= 40)].sample(N, random_state=42)
    pd.DataFrame(
        {
            "event_id": oo.event_id, "source_id": oo.source_id, "latency_min": (oo.latency_s / 60).round(1), "n_chars": oo.n_chars, "shared_text": oo.shared_text,
            "source_actor": oo.target, "source_text": oo.source_id.map(txt).map(clip), "actor": oo.actor, "text": oo.event_id.map(txt).map(clip), "is_copied_from_source": "",
        }
    ).to_csv(V / "overlaps.csv", index=False)
    print(f"wrote {N} rows each to {V}/mentions.csv, questions.csv, overlaps.csv")


if __name__ == "__main__":
    main()
