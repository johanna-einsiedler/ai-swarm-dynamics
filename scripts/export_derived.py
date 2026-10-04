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

SRC, OUT = Path("out/village"), Path("derived/village")
QUOTES = "--with-quotes" in sys.argv


EMAIL = r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
PHONE = r"(?<![\w/.\-])\+?\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}(?![\w/])"


# Agents pasted live credentials into the chat and the dataset's own redaction is best effort.
SECRETS = [
    r"\d{6,}-[a-z0-9]{20,}\.apps\.googleusercontent\.com",  # Google OAuth client id
    r"GOCSPX-[A-Za-z0-9_\-]{20,}",  # Google OAuth client secret
    r"AIza[0-9A-Za-z_\-]{30,}",  # Google API key
    r"ya29\.[0-9A-Za-z_\-]{20,}",  # Google access token
    r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}",  # GitHub
    r"glpat-[A-Za-z0-9_\-]{18,}|gl[a-z]{2,5}-[A-Za-z0-9_\-]{20,}",  # GitLab
    r"sk-[A-Za-z0-9_\-]{20,}|sk_(?:live|test)_[A-Za-z0-9]{20,}|pk_(?:live|test)_[A-Za-z0-9]{20,}|rk_(?:live|test)_[A-Za-z0-9]{20,}",  # OpenAI, Anthropic, Stripe
    r"xox[abprs]-[A-Za-z0-9\-]{10,}",  # Slack
    r"AKIA[0-9A-Z]{16}",  # AWS
    r"nfp_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{30,}|npm_[A-Za-z0-9]{30,}|dop_v1_[a-f0-9]{40,}",  # Netlify, Hugging Face, npm, DigitalOcean
    r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}",  # JWT
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
]
# anything handed over as a credential by name: "password: hunter2", "API_KEY=abc123"
NAMED = r"(?i)\b((?:pass(?:word|wd)?|pwd|secret|token|api[_\- ]?key|client[_\- ]?secret|auth(?:orization)?|bearer|credential)s?[\"'`]?\s*(?:[:=]|is)\s*[\"'`]?)([^\s\"'`,;)]{6,})"


def mask(col):
    """Quotes are published verbatim except for credentials and the contact details of real people."""
    col = col.astype(str)
    for pattern in SECRETS:
        col = col.str.replace(pattern, "[credential]", regex=True)
    col = col.str.replace(NAMED, lambda m: m.group(1) + "[credential]", regex=True)
    return col.str.replace(EMAIL, "[email]", regex=True).str.replace(PHONE, "[phone]", regex=True)


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
