"""LLM labelling with a disk cache.

Backend: the Claude Code CLI in print mode (uses whatever login the CLI has).
Set SWARM_SNA_CLAUDE_BIN to point at the binary if it is not on PATH.
"""
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL = "haiku"
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
    return json.loads(text[start:])


def call(system, prompt, cache_dir, model=MODEL, retries=2):
    """One call -> parsed JSON. Cached on (model, system, prompt)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{model}\n{system}\n{prompt}".encode()).hexdigest()[:24]
    path = cache_dir / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    env = {k: v for k, v in os.environ.items() if k not in _STRIP_ENV}
    err = None
    for _ in range(retries + 1):
        try:
            p = subprocess.run(
                [_binary(), "-p", "--model", model, "--output-format", "json", "--system-prompt", system, "--tools", "", "--no-session-persistence"],
                input=prompt, capture_output=True, text=True, env=env, timeout=180, cwd=cache_dir,
            )
            out = _parse_json(json.loads(p.stdout)["result"])
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
