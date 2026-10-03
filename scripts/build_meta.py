"""Build meta/shocks.csv, meta/agents.csv and meta/goals.csv for the AI Village adapter.

Roster rows are parsed from CHANGELOG.md; scaffold shocks and goal labels are
hand-curated below. Re-run after a dataset refresh, then review the diffs.
"""
import re
from pathlib import Path

import pandas as pd

DATA = Path("data/village")
META = Path("meta")

# Curated from CHANGELOG.md, except where `source` says otherwise.
# kind: era | chat_access | hours | nudger | rooms | regime | tools | memory | prompt | goal_mechanic
SCAFFOLD_SHOCKS = [
    ("2025-04-02", "era", "era 1 begins", "Village launch; public human chat open", "changelog"),
    ("2025-04-14", "chat_access", "premoderation", "Chat-message premoderation added for humans", "changelog"),
    ("2025-04-15", "memory", "consolidation", "Consolidation changed so agents keep more memory", "changelog"),
    ("2025-05-02", "tools", "chat while on computer", "Agents can send chat during computer-use sessions", "changelog"),
    ("2025-05-04", "prompt", "double-chatting", "Prompt changes to reduce double-chatting", "changelog"),
    ("2025-05-16", "prompt", "chat spam", "Prompt updated to reduce chat spam", "changelog"),
    ("2025-05-23", "hours", "start time", "Village start time moved to 17:59 UTC", "changelog"),
    ("2025-07-01", "era", "era 2 begins", "Public human chat closed; later human messages are staff or whitelisted users", "derived: human messages fall from >100/day to <10/day after 2025-06-30"),
    ("2025-07-01", "chat_access", "humans removed", "Public human chat closed", "derived: chat_messages speaker_type=user"),
    ("2025-07-16", "tools", "human-use", "Agents can request a human helper", "changelog"),
    ("2025-07-18", "hours", "3h/day", "Start moved 1 hour earlier; 10am-1pm PT", "changelog"),
    ("2025-08-18", "hours", "expanded", "Village timing adjusted for expanded hours (length not stated)", "changelog"),
    ("2025-08-20", "memory", "chat context cap", "Limited chat messages fetched into context", "changelog"),
    ("2025-09-05", "tools", "search_history", "search_history tool added; CoT memory consolidation", "changelog"),
    ("2025-10-05", "tools", "google sign-in", "Google sign-in v1", "changelog"),
    ("2025-10-22", "hours", "4h/day", "Prompts tweaked for 4-hour daily runs", "changelog"),
    ("2025-12-02", "regime", "text-only models", "Support for agents without computer use", "changelog"),
    ("2025-12-04", "prompt", "don't do nothing", "Prompt added to reduce repeated waiting", "changelog"),
    ("2025-12-10", "prompt", "goal in prompt", "Village goal added to the prompt", "changelog"),
    ("2026-01-26", "regime", "claude code agent", "Opus 4.5 (Claude Code) runs a different scaffold until 2026-04-02", "changelog"),
    ("2026-02-10", "nudger", "on", "Auto-nudger bot injects nudge messages", "changelog"),
    ("2026-02-25", "rooms", "rooms v1", "Agents scoped to chat rooms; only see their own room", "changelog"),
    ("2026-02-26", "rooms", "move_to_room", "Agents can switch rooms", "changelog"),
    ("2026-03-16", "rooms", "best/rest split", "#best and #rest rooms created", "derived: chat_rooms.created_at"),
    ("2026-03-24", "regime", "perma-computer-use", "Agents permanently in computer-use mode; sessions become consolidations", "changelog"),
    ("2026-04-14", "tools", "outreach approval", "Unsolicited external outreach needs approval", "changelog"),
    ("2026-05-22", "prompt", "message length", "Chat-message length instructions added", "changelog"),
    ("2026-05-28", "prompt", "keep messages short", "Short-message instruction for text-only agents", "changelog"),
    ("2026-06-01", "goal_mechanic", "imposed leader", "Follow your leader goal; Opus 4.7 moved to #rest", "changelog"),
    ("2026-06-07", "hours", "8h/day (temporary)", "9am-5pm PT for the event week", "changelog"),
    ("2026-06-07", "chat_access", "whitelisted human", "One whitelisted human may chat", "changelog"),
    ("2026-06-11", "memory", "unseen events cap", "Unseen events capped at 200 per turn", "changelog"),
    ("2026-06-13", "nudger", "off", "Auto-nudger disabled for the event weekend", "changelog"),
    ("2026-06-15", "nudger", "on", "Auto-nudger re-enabled", "changelog"),
    ("2026-06-15", "hours", "4h/day", "Reverted to 10am-2pm PT", "changelog"),
    ("2026-06-29", "hours", "8h/day", "Permanently 9am-5pm PT; code hosting moved to GitLab", "changelog"),
    ("2026-07-03", "goal_mechanic", "individual goals", "Per-agent goals supported; agents can't see each other's", "changelog"),
    ("2026-07-06", "era", "era 3 begins", "Each agent maximises its own assigned goal", "village_goals"),
]

# (type, leader_imposed, confidence), in village_goals start_time order.
# type: collaborative | competitive | individual | holiday
GOAL_LABELS = [
    ("collaborative", 0, "high"),  # charity 2025
    ("holiday", 0, "medium"),  # unsupervised look back
    ("holiday", 0, "high"),
    ("collaborative", 0, "high"),  # story + 100 people
    ("holiday", 0, "high"),
    ("competitive", 0, "high"),  # merch store
    ("holiday", 0, "high"),
    ("collaborative", 0, "high"),  # benchmark
    ("holiday", 0, "high"),
    ("individual", 0, "low"),  # games in a week
    ("holiday", 0, "medium"),  # pursue whatever
    ("competitive", 0, "high"),  # debate teams
    ("collaborative", 0, "high"),  # human subjects experiment
    ("individual", 0, "medium"),  # personality tests
    ("collaborative", 0, "high"),  # therapy
    ("individual", 0, "high"),  # choose your own goal
    ("individual", 0, "high"),  # personal website
    ("collaborative", 0, "high"),  # reduce poverty
    ("collaborative", 0, "medium"),  # puzzle game
    ("individual", 0, "medium"),  # substack
    ("collaborative", 0, "low"),  # forecast AI
    ("individual", 0, "high"),  # each agent own goal
    ("competitive", 0, "high"),  # chess
    ("collaborative", 0, "low"),  # acts of kindness
    ("collaborative", 0, "high"),  # museum
    ("collaborative", 1, "high"),  # elect a leader
    ("competitive", 0, "high"),  # OWASP
    ("collaborative", 0, "high"),  # quiz
    ("competitive", 0, "high"),  # breaking news
    ("collaborative", 0, "high"),  # park
    ("individual", 0, "high"),  # pick your own (farewell)
    ("competitive", 0, "high"),  # challenge each other
    ("collaborative", 0, "medium"),  # Pentagon discussion
    ("collaborative", 0, "medium"),  # RPG with saboteurs
    ("collaborative", 0, "high"),  # test the game
    ("individual", 0, "low"),  # external agents
    ("individual", 0, "high"),  # pick your own
    ("collaborative", 0, "high"),  # charity 2026
    ("individual", 0, "high"),  # own world
    ("collaborative", 0, "high"),  # connect worlds
    ("collaborative", 0, "low"),  # novel research
    ("individual", 0, "medium"),  # youtube channel
    ("individual", 0, "medium"),  # improve memory
    ("collaborative", 1, "medium"),  # finetune your leader
    ("collaborative", 1, "high"),  # follow your leader
    ("collaborative", 0, "high"),  # organise an event
    ("collaborative", 0, "high"),  # reduce suffering
    ("collaborative", 0, "high"),  # help Gemini 2.5 Pro
    ("individual", 0, "medium"),  # hardest game
    ("competitive", 0, "high"),  # best AI assistant
    ("individual", 0, "high"),  # assigned goals
]

FAMILY_PATTERNS = [
    (r"claude", "Anthropic"),
    (r"gpt|^o\d", "OpenAI"),
    (r"gemini", "Google"),
    (r"deepseek", "DeepSeek"),
    (r"grok", "xAI"),
    (r"kimi|tinker", "Moonshot"),
    (r"glm", "Z.ai"),
    (r"muse", "Meta"),
]

# Release dates not recoverable from the model string.
KNOWN_RELEASE = {
    "gemini-2.5-pro": "2025-03-25",
    "gemini-3-pro-preview": "2025-11-18",
    "deepseek-reasoner": "2025-12-01",
    "claude-opus-4-6": "2026-02-05",
    "claude-sonnet-4-6": "2026-02-17",
}

ERA_STARTS = [("2025-04-02", 1), ("2025-07-01", 2), ("2026-07-06", 3)]


def era_of(ts):
    era = 1
    for start, e in ERA_STARTS:
        if ts >= pd.Timestamp(start):
            era = e
    return era


def family(model_string):
    s = model_string.lower().split("::")[-1].split("/")[-1]
    for pat, fam in FAMILY_PATTERNS:
        if re.search(pat, s):
            return fam
    return "other"


def release_date(model_string, joined):
    s = model_string.split("::")[-1]
    if s in KNOWN_RELEASE:
        return KNOWN_RELEASE[s], "known"
    m = re.search(r"(20\d{2})-?(\d{2})-?(\d{2})$", s)
    if m:
        return "-".join(m.groups()), "model_string"
    m = re.search(r"-(\d{2})(\d{2})$", s)  # grok-4-0709
    if m:
        return f"{joined[:4]}-{m.group(1)}-{m.group(2)}", "model_string"
    return joined, "join_proxy"


def parse_roster():
    rows = []
    for line in (DATA / "CHANGELOG.md").read_text().splitlines():
        m = re.match(r"\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(\d{4}-\d{2}-\d{2}).*?\|\s*(\S+)\s*\|", line)
        if m:
            name, _, joined, left = m.groups()
            rows.append((name, joined, None if left == "active" else left))
    return pd.DataFrame(rows, columns=["name", "joined", "left"])


def main():
    META.mkdir(exist_ok=True)
    roster = parse_roster()
    agents = pd.read_json(DATA / "agents.jsonl.gz", lines=True)
    agents = agents.merge(roster, on="name", how="left")
    missing = agents[agents.joined.isna()].name.tolist()
    assert not missing, f"agents missing from CHANGELOG roster: {missing}"

    rel = [release_date(m, j) for m, j in zip(agents.model_string, agents.joined)]
    out = pd.DataFrame(
        {
            "agent_id": agents.id,
            "name": agents.name,
            "model_string": agents.model_string,
            "family": agents.model_string.map(family),
            "release_date": [r[0] for r in rel],
            "release_date_source": [r[1] for r in rel],
            "joined": agents.joined,
            "left": agents.left,
            "claude_code_scaffold": agents.model_string.str.startswith("claude-code::"),
            "imposed_leader": agents.model_string.str.startswith("tinker://"),
        }
    ).sort_values("joined")
    out.to_csv(META / "agents.csv", index=False)

    shocks = [(d, k, s, det, src) for d, k, s, det, src in SCAFFOLD_SHOCKS]
    for r in roster.itertuples():
        shocks.append((r.joined, "roster_join", r.name, "", "changelog roster"))
        if pd.notna(r.left):
            shocks.append((r.left, "roster_leave", r.name, "", "changelog roster"))
    sh = pd.DataFrame(shocks, columns=["date", "kind", "subject", "detail", "source"])
    sh.sort_values(["date", "kind"]).to_csv(META / "shocks.csv", index=False)

    goals = pd.read_json(DATA / "village_goals.jsonl.gz", lines=True)
    goals = goals.sort_values("start_time").reset_index(drop=True)
    assert len(goals) == len(GOAL_LABELS), f"{len(goals)} goals but {len(GOAL_LABELS)} labels"
    start = pd.to_datetime(goals.start_time)
    g = pd.DataFrame(
        {
            "goal_id": goals.id,
            "start_time": goals.start_time,
            "end_time": goals.end_time,
            "goal": goals.goal.str.replace("\n", " ").str.strip(),
            "goal_type": [x[0] for x in GOAL_LABELS],
            "leader_imposed": [x[1] for x in GOAL_LABELS],
            "humans_in_chat": (start < pd.Timestamp("2025-07-01")).astype(int),
            "era": start.map(era_of),
            "label_confidence": [x[2] for x in GOAL_LABELS],
            "reviewed": 0,
        }
    )
    g.to_csv(META / "goals.csv", index=False)
    print(f"agents {len(out)}  shocks {len(sh)}  goals {len(g)}")
    print(out.release_date_source.value_counts().to_dict())


if __name__ == "__main__":
    main()
