import argparse
import importlib
import time
from pathlib import Path

ADAPTERS = ["village", "jsonl"]


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


def cmd_label(args):
    import pandas as pd

    from .layers import requests

    d = Path(args.dir)
    cache = Path(args.cache)
    tag = "" if args.screen == "base" else f"_{args.screen}"
    events = pd.read_parquet(d / "events.parquet")
    req = requests.label_requests(events, pd.read_parquet(d / "agents.parquet"), pd.read_parquet(d / "questions.parquet"), cache, workers=args.workers, limit=args.limit, seed=args.seed, screen=args.screen)
    req.to_parquet(d / f"requests{tag}.parquet", index=False)
    if not args.requests_only:
        req, resp = requests.judge_responses(events, req, cache, workers=args.workers, named_sample=args.named_sample, broadcast_sample=args.broadcast_sample)
        req.to_parquet(d / f"requests{tag}.parquet", index=False)
        resp.to_parquet(d / f"responses{tag}.parquet", index=False)
    print(f"wrote {d}/requests{tag}.parquet" + ("" if args.requests_only else f", responses{tag}.parquet"))


def cmd_verify(args):
    """Second-opinion pass over the request-response links, with a stronger model."""
    import pandas as pd

    from .layers import requests

    d = Path(args.dir)
    events = pd.read_parquet(d / "events.parquet")
    for f in sorted(d.glob("responses*.parquet")):
        req = pd.read_parquet(d / f.name.replace("responses", "requests"))
        resp = pd.read_parquet(f).drop(columns="link_verified", errors="ignore")
        helpful = resp.type.isin(["answers", "acts"])
        checked = requests.verify_links(events, req, resp[helpful], Path(args.cache), workers=args.workers, model=args.model)
        resp["link_verified"] = checked.link_verified.reindex(resp.index)
        resp.to_parquet(f, index=False)
        print(f"wrote {f}")


def cmd_helping(args):
    from . import helping

    helping.run(args.dir, n_boot=args.bootstraps, seed=args.seed)


def cmd_hierarchy(args):
    from . import hierarchy

    hierarchy.run(args.dir, group=args.by, meta=args.meta, n_perm=args.permutations, seed=args.seed)


def cmd_trends(args):
    from . import trends

    trends.run(args.dir, meta=args.meta)


def cmd_card(args):
    from . import card

    card.run(args.dir, title=args.title, group=args.by, validation=args.validation)


def cmd_run(args):
    """Everything in order. The LLM layers are optional; without them the card shows the network questions only."""
    out = Path(args.out)
    cmd_extract(argparse.Namespace(adapter=args.adapter, data=args.data, meta=args.meta, out=args.out, overlap_window=overlaps_default()))
    if args.llm:
        cache = str(out.parent / "llm_cache")
        cmd_label(argparse.Namespace(dir=args.out, cache=cache, workers=args.workers, limit=0, named_sample=0, broadcast_sample=0, requests_only=False, screen="base", seed=args.seed))
        cmd_verify(argparse.Namespace(dir=args.out, cache=cache, model="sonnet", workers=args.workers))
    cmd_report(argparse.Namespace(dir=args.out, permutations=args.permutations, by="era", mention_kinds="at,name", seed=args.seed))
    if list(out.glob("responses*.parquet")):
        cmd_helping(argparse.Namespace(dir=args.out, bootstraps=300, seed=args.seed))
        cmd_hierarchy(argparse.Namespace(dir=args.out, by="era", meta=args.meta, permutations=args.permutations, seed=args.seed))
        cmd_trends(argparse.Namespace(dir=args.out, meta=args.meta))
    cmd_card(argparse.Namespace(dir=args.out, title=args.title, by="era", validation="validation"))


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

    lb = sub.add_parser("label", help="LLM pass: requests and their responses, each with a verbatim quote")
    lb.add_argument("dir", nargs="?", default="out/village", help="directory written by extract")
    lb.add_argument("--cache", default="out/llm_cache")
    lb.add_argument("--workers", type=int, default=10, help="parallel LLM calls")
    lb.add_argument("--limit", type=int, default=0, help="label a random sample of this many candidate messages (0 = all)")
    lb.add_argument("--named-sample", type=int, default=6000, help="how many named requests to judge for responses (0 = all)")
    lb.add_argument("--broadcast-sample", type=int, default=2000, help="how many broadcast requests to judge for responses (0 = all)")
    lb.add_argument("--requests-only", action="store_true", help="skip the response pass")
    lb.add_argument("--screen", default="base", choices=["base", "wide"], help="which messages to screen in; 'wide' takes only the ones 'base' misses and writes requests_wide.parquet")
    lb.add_argument("--seed", type=int, default=0)
    lb.set_defaults(func=cmd_label)

    v = sub.add_parser("verify", help="LLM pass: re-check every request-response link one pair at a time, with a stronger model")
    v.add_argument("dir", nargs="?", default="out/village")
    v.add_argument("--cache", default="out/llm_cache")
    v.add_argument("--model", default="sonnet")
    v.add_argument("--workers", type=int, default=10)
    v.set_defaults(func=cmd_verify)

    h = sub.add_parser("helping", help="who answers whose requests: dyad table, reciprocity ladder, bystander curve")
    h.add_argument("dir", nargs="?", default="out/village", help="directory written by extract and label")
    h.add_argument("--bootstraps", type=int, default=500)
    h.add_argument("--seed", type=int, default=0)
    h.set_defaults(func=cmd_helping)

    d = sub.add_parser("hierarchy", help="dominance from directives and compliance (run helping first)")
    d.add_argument("dir", nargs="?", default="out/village", help="directory written by helping")
    d.add_argument("--by", default="era", choices=["era", "goal_id"])
    d.add_argument("--meta", default="meta", help="directory with goals.csv, for the imposed-leader episodes")
    d.add_argument("--permutations", type=int, default=1000)
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(func=cmd_hierarchy)

    t = sub.add_parser("trends", help="weekly series and goal-type breakdown behind the pooled numbers (run helping first)")
    t.add_argument("dir", nargs="?", default="out/village")
    t.add_argument("--meta", default="meta", help="directory with goals.csv, for the goal-type breakdown")
    t.set_defaults(func=cmd_trends)

    c = sub.add_parser("card", help="the report card: one HTML page with every question, its verdict, figure and evidence")
    c.add_argument("dir", nargs="?", default="out/village")
    c.add_argument("--title", default=None)
    c.add_argument("--by", default="era", choices=["era", "goal_id"])
    c.add_argument("--validation", default="validation", help="directory with hand-checked samples, if any")
    c.set_defaults(func=cmd_card)

    a = sub.add_parser("run", help="everything in one go: extract, (label), report, analyses, report card")
    a.add_argument("--adapter", choices=ADAPTERS, required=True)
    a.add_argument("--data", required=True, help="the transcript: a file or directory, as the adapter expects")
    a.add_argument("--out", required=True, help="output directory")
    a.add_argument("--meta", default=None, help="optional directory with curated tables (agents, goals)")
    a.add_argument("--llm", action="store_true", help="also run the LLM layers (requests and responses); needs the Claude Code CLI")
    a.add_argument("--workers", type=int, default=10)
    a.add_argument("--permutations", type=int, default=1000)
    a.add_argument("--title", default=None)
    a.add_argument("--seed", type=int, default=0)
    a.set_defaults(func=cmd_run)

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
