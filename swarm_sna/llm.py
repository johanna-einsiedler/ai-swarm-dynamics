"""LLM labelling with a disk cache.

Backend: the Claude Code CLI in print mode (uses whatever login the CLI has).
Set SWARM_SNA_CLAUDE_BIN to point at the binary if it is not on PATH.
"""
import glob
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL = "haiku"
BACKOFF_BASE = 20  # seconds; doubles each retry
BACKOFF_CAP = 300
# Nested-session variables that must not leak into the child CLI.
_STRIP_ENV = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_CHILD_SESSION", "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_MESSAGING_TOKEN")


def _binary():
    b = os.environ.get("SWARM_SNA_CLAUDE_BIN") or shutil.which("claude")
    if not b:
        hits = sorted(glob.glob(os.path.expanduser("~/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude")))
        b = hits[-1] if hits else None
    if not b:
        raise RuntimeError("Claude Code CLI not found; set SWARM_SNA_CLAUDE_BIN")
    return b


def _parse_json(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start = min([i for i in (text.find("["), text.find("{")) if i >= 0], default=0)
    return json.JSONDecoder().raw_decode(text[start:])[0]  # ignore anything the model adds after the JSON


def _salvage(text):
    """One bad object should not cost the whole batch: pull out every complete JSON object."""
    out, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            esc = ch == "\\" and not esc
            in_str = not (ch == '"' and not esc)
            continue
        if ch == '"':
            in_str, esc = True, False
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    out.append(json.loads(text[start : i + 1]))
                except json.JSONDecodeError:
                    pass
                start = None
    return out


def call(system, prompt, cache_dir, model=MODEL, retries=5):
    """One call -> parsed JSON. Cached on (model, system, prompt)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{model}\n{system}\n{prompt}".encode()).hexdigest()[:24]
    path = cache_dir / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    env = {k: v for k, v in os.environ.items() if k not in _STRIP_ENV}
    env["MAX_THINKING_TOKENS"] = "0"  # labelling needs no reasoning; with it on a call takes ten times as long
    err = None
    for attempt in range(retries + 1):
        if attempt:  # a rate limit returns empty output, so back off rather than burning the queue
            time.sleep(min(BACKOFF_CAP, BACKOFF_BASE * 2 ** (attempt - 1)) * (0.5 + random.random()))
        try:
            p = subprocess.run(
                [_binary(), "-p", "--model", model, "--output-format", "json", "--system-prompt", system, "--tools", "", "--no-session-persistence", "--strict-mcp-config"],
                input=prompt, capture_output=True, text=True, env=env, timeout=180, cwd=cache_dir,
            )
            result = json.loads(p.stdout)["result"]
            try:
                out = _parse_json(result)
            except json.JSONDecodeError:
                out = _salvage(result)  # keep the objects the model did get right
                if not out:
                    raise
            path.write_text(json.dumps(out))
            return out
        except Exception as e:  # malformed JSON or a CLI failure: retry, then give up on this call
            err = e
    print(f"llm call failed: {err!r}")
    return None


def map_calls(system, prompts, cache_dir, workers=4, model=MODEL):
    done = [0]

    def one(p):
        out = call(system, p, cache_dir, model)
        done[0] += 1
        if done[0] % 10 == 0 or done[0] == len(prompts):
            print(f"  {done[0]}/{len(prompts)} calls", flush=True)
        return out

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, prompts))
