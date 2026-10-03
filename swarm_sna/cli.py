import argparse
import importlib
import time
from pathlib import Path

ADAPTERS = ["village"]


def cmd_extract(args):
    from .layers import mentions, overlaps, questions, time_mentions

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    adapter = importlib.import_module(f".adapters.{args.adapter}", __package__)
    events, agents = adapter.build(args.data, args.meta)
    events.to_parquet(out / "events.parquet", index=False)
    agents.to_parquet(out / "agents.parquet", index=False)
    print(f"events    {len(events):>8,}  ({time.time() - t0:.0f}s)")

    m = mentions.extract(events, agents)
    m.to_parquet(out / "mentions.parquet", index=False)
    print(f"mentions  {len(m):>8,}  ({time.time() - t0:.0f}s)")

    q = questions.extract(events)
    q.to_parquet(out / "questions.parquet", index=False)
    print(f"questions {len(q):>8,}  ({time.time() - t0:.0f}s)")

    tm = time_mentions.extract(events)
    tm.to_parquet(out / "time_mentions.parquet", index=False)
    print(f"time      {len(tm):>8,}  ({time.time() - t0:.0f}s)")

    o = overlaps.extract(events, window_min=args.overlap_window)
    o.to_parquet(out / "overlaps.parquet", index=False)
    print(f"overlaps  {len(o):>8,}  ({time.time() - t0:.0f}s)")
    print(f"wrote {out}/")


def cmd_report(args):
    from . import report

    report.run(args.dir, n_perm=args.permutations, group=args.by, seed=args.seed, kinds=tuple(args.mention_kinds.split(",")))


def overlaps_default():
    from .layers.overlaps import WINDOW_MIN

    return WINDOW_MIN


def main():
    p = argparse.ArgumentParser(prog="swarm-sna", description="Social network analysis for multi-agent transcripts")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="transcript -> relational event tables")
    e.add_argument("--adapter", choices=ADAPTERS, required=True)
    e.add_argument("--data", default="data/village", help="directory with the raw dataset")
    e.add_argument("--meta", default="meta", help="directory with curated agents/goals/shocks tables")
    e.add_argument("--out", default="out/village", help="output directory")
    e.add_argument("--overlap-window", type=int, default=overlaps_default(), help="minutes to look back for copied text")
    e.set_defaults(func=cmd_extract)

    r = sub.add_parser("report", help="network statistics, each with permutation nulls")
    r.add_argument("dir", nargs="?", default="out/village", help="directory written by extract")
    r.add_argument("--permutations", type=int, default=1000)
    r.add_argument("--by", default="era", choices=["era", "goal_id"], help="compute the battery separately per group")
    r.add_argument("--mention-kinds", default="at,name", help="which mention kinds form edges: at, name, or both")
    r.add_argument("--seed", type=int, default=0)
    r.set_defaults(func=cmd_report)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
