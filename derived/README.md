# Derived tables (AI Village)

Everything `swarm-sna` extracted from the AI Village chat record, keyed by the
dataset's own event ids. The full messages are **not** here: the raw dataset
is gated (`aidigestorg/ai-village` on Hugging Face) and anyone with access can
join these tables back to it on `event_id`. What is here from the text is the
evidence behind each label: the sentence an agent wrote that makes a request,
the part of a reply that answers it, and the link, id or output that backs
it. Only agents' own messages are quoted. Human chat participants are reduced
to the actor `human`. Inside quotes, email addresses and phone numbers are
masked as `[email]` and `[phone]`, and anything that looks like a credential
(API keys, OAuth secrets, tokens, passwords) as `[credential]`: agents pasted
live credentials into the chat. If you find one that slipped through, report
it to AI Digest and do not use it.

| file | one row per | columns of note |
| --- | --- | --- |
| `events.parquet` | chat message or work-session start | `ts`, `actor`, `room`, `type`, `goal_id`, `era`, `active_n`, `n_chars`; `text` is empty |
| `agents.parquet` | agent | `family`, `release_date`, `joined`, `left`, `aliases` |
| `mentions.parquet` | (message, agent mentioned) | `actor`, `target`, `weight` (1/k when an alias is ambiguous), `kind` (`at` or `name`) |
| `requests.parquet`, `requests_wide.parquet` | request found in a message | `quote` (the request, verbatim), `kind`, `addressed`, `directive`, `beneficiary`, `targets`, `quote_verified`, `judged` |
| `responses.parquet`, `responses_wide.parquet` | (request, message that responds) | `quote` (the part that responds), `evidence` (what makes it checkable), `type`, `latency_s`, `quote_verified`, `backed`, `link_verified` |
| `dyads.parquet` | (request, agent present) | `responded`, `addressed`, `broadcast`, `prior_help_from_requester`, `requester_reputation`, `responder_busy`, `active_n` |
| `battery_labels.parquet` | sampled message | twelve behaviour labels (act, ethogram, leads or follows, ...) |

`_wide` holds what the wider request screen added. A response counts in the
statistics only if `quote_verified` and `link_verified` are both true.

## Recomputing the statistics from these tables

    swarm-sna report    derived/village                 # network statistics with nulls
    swarm-sna helping   derived/village --reuse-dyads   # reciprocity ladder, bystander curve
    swarm-sna hierarchy derived/village --meta meta
    swarm-sna trends    derived/village --meta meta
    swarm-sna card      derived/village

The diffusion analysis needs the full message text and so needs the dataset
itself.

Source: AI Digest, "AI Village dataset", 2026, https://theaidigest.org/village,
used under its research terms. Please cite it in any work that uses these
tables.
