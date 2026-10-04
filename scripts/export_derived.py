"""Publishable derived tables: every label and edge, keyed by event id, without the message text.

The raw AI Village dataset is gated and is not redistributed. These tables carry
what was extracted from it (who mentioned whom, which messages hold a request, which
answer which, every label and check) so the statistics can be recomputed by anyone,
and re-joined to the text by anyone with access to the dataset. Human participants
are reduced to "human"; only agents are named.

usage: export_derived.py [--with-quotes]   (quotes are verbatim sentences from agent messages)
"""
import sys
from pathlib import Path

import pandas as pd

from swarm_sna.redact import mask

SRC, OUT = Path("out/village"), Path("derived/village")
QUOTES = "--with-quotes" in sys.argv


def anonymise(actor, actor_class):
    return actor.where(actor_class == "agent", actor_class)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = pd.read_parquet(SRC / "events.parquet")
    ev["n_chars"] = ev.text.str.len()
    ev["actor"] = anonymise(ev.actor, ev.actor_class)
    ev["text"] = ""  # the schema keeps the column; the content stays with the dataset
    ev.to_parquet(OUT / "events.parquet", index=False)

    pd.read_parquet(SRC / "agents.parquet").to_parquet(OUT / "agents.parquet", index=False)

    m = pd.read_parquet(SRC / "mentions.parquet")
    m[m.actor_class == "agent"].to_parquet(OUT / "mentions.parquet", index=False)

    for tag in ("", "_wide"):
        req = pd.read_parquet(SRC / f"requests{tag}.parquet").drop(columns=["addressees"])  # names as written may be human
        resp = pd.read_parquet(SRC / f"responses{tag}.parquet")
        resp = resp[resp.actor_class == "agent"]
        if QUOTES:
            req["quote"] = mask(req.quote)
            resp["quote"], resp["evidence"] = mask(resp.quote), mask(resp.evidence)
        else:
            req = req.drop(columns=["quote"])
            resp = resp.drop(columns=["quote", "evidence"])
        req.to_parquet(OUT / f"requests{tag}.parquet", index=False)
        resp.to_parquet(OUT / f"responses{tag}.parquet", index=False)

    for name in ("dyads", "battery_labels"):
        pd.read_parquet(SRC / f"{name}.parquet").to_parquet(OUT / f"{name}.parquet", index=False)

    sizes = {f.name: f.stat().st_size / 1e6 for f in sorted(OUT.glob("*.parquet"))}
    print("\n".join(f"{v:6.1f} MB  {k}" for k, v in sizes.items()))
    print(f"total {sum(sizes.values()):.1f} MB in {OUT}/ ({'with' if QUOTES else 'without'} verbatim quotes)")


if __name__ == "__main__":
    main()
