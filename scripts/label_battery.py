"""Battery labels: an LLM labels a random sample of each agent's messages.

usage: label_battery.py [messages per agent] [number of agents, 0 = all] [parallel calls]
"""
import sys
from pathlib import Path

import pandas as pd

from swarm_sna import llm

OUT = Path("out/village")
CACHE = Path("out/llm_cache")
PER_AGENT = int(sys.argv[1]) if len(sys.argv) > 1 else 25
N_AGENTS = int(sys.argv[2]) if len(sys.argv) > 2 else 0
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
FIELDS = ["act", "self_report", "directs_others", "ethogram", "leads_or_follows", "task_domain", "failure_attribution", "evidence", "initiative", "time_direction", "time_horizon", "time_function"]

SYSTEM = """You label single messages from a group chat between AI agents working on shared goals.
Return a JSON array with one object per numbered message, in order. Every field takes exactly one of the listed values.
{"i": <message number>,
 "act": the main thing the message does: "status" (reports own progress or state) | "request" (asks others for something) | "commit" (promises to do something) | "offer" (offers help or resources) | "answer" (gives what someone asked for) | "acknowledge" (thanks, praise, agreement) | "propose" (suggests a plan or decision for the group) | "other",
 "self_report": "success" (claims own work succeeded) | "blocked" (reports failure or being stuck) | "admits_error" (says it made a mistake) | "none",
 "directs_others": true if it tells specific others what to do, else false,
 "ethogram": the social behaviour toward others: "affiliative" (thanks, praise, encouragement) | "conflict" (corrects, overrules or criticises someone, backs down or apologises, enforces a rule, or steps into a dispute) | "recruitment" (calls others to come and work on something) | "alarm" (warns others of a hazard: a bug, a scam, a broken tool, a risk) | "provisioning" (hands over a resource: a link, a fix, a file, a result others can use) | "sentinel" (says it will monitor, watch or wait) | "display" (reports on itself with nothing for others to use) | "none",
 "leads_or_follows": "leads" (sets a new direction or plan for others) | "follows" (joins or adopts someone else's direction) | "neither",
 "task_domain": "coding" | "outreach" (email, social media, contacting people) | "documentation" (writing docs, reports, content) | "coordination" (organising who does what) | "monitoring" (checking, waiting, verifying) | "research" | "creative" (stories, games, art) | "other",
 "failure_attribution": if a failure or obstacle is mentioned, the cause given: "self" | "other_agent" | "environment" (tool, platform, bug, access) | "human" | "none" (no failure mentioned),
 "evidence": "backed" (a claim about work done comes with a link, id, number or quoted output) | "bare" (a claim about work done with nothing to check) | "no_claim",
 "initiative": "proactive" (self-started) | "reactive" (responding to someone or something that happened),
 "time_direction": "past" | "present" | "future" | "none" (the main time the message is about),
 "time_horizon": the furthest time span it talks about: "minutes" | "this_session" (hours, today) | "next_session" (tomorrow, next session) | "days_weeks" | "long" (months or more, or the whole history) | "none",
 "time_function": why time is mentioned: "estimate" (how long own work will take) | "deadline" (time left, a cutoff) | "waiting" (how long it has waited or will wait for someone) | "recall" (when something happened) | "schedule" (when something is planned) | "none"}
JSON only."""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = pd.read_parquet("out/village/events.parquet")
    m = ev[(ev.type == "message") & (ev.actor_class == "agent")]
    counts = m.actor.value_counts()
    top = counts.head(N_AGENTS).index if N_AGENTS else counts.index
    sample = pd.concat([g.sample(min(len(g), PER_AGENT), random_state=11) for a, g in m[m.actor.isin(top)].groupby("actor")]).sample(frac=1, random_state=11).reset_index(drop=True)
    print(f"labelling {len(sample)} messages from {len(top)} agents, {WORKERS} calls in parallel", flush=True)
    items = [f"[{i}] {r.actor}: {r.text[:900]}" for i, r in enumerate(sample.itertuples())]
    batches = ["\n\n".join(items[i : i + 15]) for i in range(0, len(items), 15)]
    res = llm.map_calls(SYSTEM, batches, CACHE, workers=WORKERS)
    lab = {x["i"]: x for r in res if r for x in r if isinstance(x, dict) and "i" in x}
    for f in FIELDS:
        sample[f] = [lab.get(i, {}).get(f) for i in range(len(sample))]
    sample[["event_id", "ts", "day", "actor", "era", "goal_id"] + FIELDS].to_parquet(OUT / "battery_labels.parquet", index=False)
    done = sample[sample.act.notna()]
    print(f"labelled {len(done)}/{len(sample)} messages from {len(top)} agents")
    for f in FIELDS:
        print(f"  {f}: {done[f].value_counts().to_dict()}")
    pd.set_option("display.width", 250)
    prof = pd.DataFrame(
        {
            "provisioning": done.ethogram.eq("provisioning"), "display": done.ethogram.eq("display"), "sentinel": done.ethogram.eq("sentinel"), "affiliative": done.ethogram.eq("affiliative"),
            "leads": done.leads_or_follows.eq("leads"), "follows": done.leads_or_follows.eq("follows"), "directs": done.directs_others.eq(True),
            "blames_env": done.failure_attribution.eq("environment"), "blames_self": done.failure_attribution.eq("self"),
            "backed": done.evidence.eq("backed"), "bare": done.evidence.eq("bare"), "proactive": done.initiative.eq("proactive"),
        }
    ).groupby(done.actor).mean().mul(100).round(0).astype(int)
    print("\nshare of each agent's messages (%):")
    print(prof.to_string())


if __name__ == "__main__":
    main()
