# Derived tables (AI Village)

Everything `swarm-sna` extracted from the AI Village chat record, keyed by the
dataset's own event ids, **without the message text**. The raw dataset is
gated (`aidigestorg/ai-village` on Hugging Face) and is not redistributed
here; anyone with access can join these tables back to the text on
`event_id`. Human chat participants are reduced to the actor `human`; only
agents are named.

| file | one row per | columns of note |
| --- | --- | --- |
| `events.parquet` | chat message or work-session start | `ts`, `actor`, `room`, `type`, `goal_id`, `era`, `active_n`, `n_chars`; `text` is empty |
| `agents.parquet` | agent | `family`, `release_date`, `joined`, `left`, `aliases` |
| `mentions.parquet` | (message, agent mentioned) | `actor`, `target`, `weight` (1/k when an alias is ambiguous), `kind` (`at` or `name`) |
| `requests.parquet`, `requests_wide.parquet` | request found in a message | `kind`, `addressed`, `directive`, `beneficiary`, `targets`, `quote_verified`, `judged` |
| `responses.parquet`, `responses_wide.parquet` | (request, message that responds) | `type`, `latency_s`, `quote_verified`, `backed`, `link_verified` |
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

The diffusion analysis and the evidence drawers need the message text and so
need the dataset itself.

Source: AI Digest, "AI Village dataset", 2026, https://theaidigest.org/village,
used under its research terms. Please cite it in any work that uses these
tables.
