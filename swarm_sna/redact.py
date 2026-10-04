"""What must not leave with a published table: credentials and the contact details of real people.

Agents paste live credentials into chat, and a dataset's own redaction is best effort.
The patterns cover the common key formats plus anything introduced as a credential by name.
"""
import hashlib
import re

import pandas as pd

EMAIL = r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
PHONE = r"(?<![\w/.\-])\+?\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}(?![\w/])"
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


# A free-standing run of 24+ characters mixing lower case, upper case and digits: keys, tokens, base64.
# Not inside a URL or path (those carry document ids, which are evidence), and not a lower-case commit hash.
_RUN = r"[A-Za-z0-9+=_]"
HIGH_ENTROPY = rf"(?<![\w/.\-])(?={_RUN}*[a-z])(?={_RUN}*[A-Z])(?={_RUN}*\d){_RUN}{{24,}}(?![\w/.\-])"
WALLET = r"\b0x[a-fA-F0-9]{40}\b"
PASSWORD_LIKE = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)(?=.*[!@#$%^&*?])[^\s/.]{16,}$")  # for short identifiers only


def mask(col):
    """A text column with credentials, wallet addresses, email addresses and phone numbers replaced by placeholders."""
    col = col.astype(str)
    for pattern in SECRETS:
        col = col.str.replace(pattern, "[credential]", regex=True)
    col = col.str.replace(NAMED, lambda m: m.group(1) + "[credential]", regex=True)
    col = col.str.replace(WALLET, "[wallet]", regex=True).str.replace(HIGH_ENTROPY, "[credential]", regex=True)
    return col.str.replace(EMAIL, "[email]", regex=True).str.replace(PHONE, "[phone]", regex=True)


def label(names):
    """Short strings used as identifiers (item names): masked like text, with a hash suffix where
    masking changed them so that distinct names stay distinct."""
    names = pd.Series(names).astype(str)
    masked = mask(names)
    masked = masked.where(~names.map(lambda n: bool(PASSWORD_LIKE.match(n))), "[credential]")
    tag = names.map(lambda s: hashlib.sha1(s.encode()).hexdigest()[:6])
    return masked.where(masked == names, masked + "#" + tag).to_numpy()
