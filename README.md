# swarm-sna

**A report card for any group of AI agents.** Point it at a transcript and it
answers a fixed set of social-network questions, each tested against a
permutation null model borrowed from animal behaviour, and each backed by
verbatim quotes you can check.

```bash
swarm-sna run --adapter jsonl --data examples/toy --out out/toy
open out/toy/report_card.html
```

The pitch in one line: *the questions you should always ask a swarm, each with
a null model, so you know when a pattern is real.*

Read the other way round, the card is a baseline: what a swarm looks like
going about its business, with the band of chance around each number. A later
run that moves sharply in one direction (pairs addressing mainly each other, a
clique or a vendor bloc closing, a steep ladder of directives, requests from
some agents going unanswered, claims without evidence, items picked up by the
same few in lockstep) is worth a look, and the card's closing section lists
these movements beside the current values.

A twenty-agent network is thin. Behavioural ecologists have the same problem
(few individuals, observed unevenly, no experiments, no interviews) and solved
it with pre-network permutation tests. This tool brings that toolkit to
multi-agent transcripts.

## What the report card answers

| question | how | needs an LLM |
| --- | --- | --- |
| Is interaction more reciprocal than chance? | mention network vs two nulls | no |
| Do agents favour their own kind (model family)? | same | no |
| Do agents have preferred partners? Do ties close into cliques? | same | no |
| How does the network change over time? | the mention and help networks week by week; response rates by kind of goal | no |
| How does information spread? | every link, file name and term that reached 4+ agents, replayed over the network; how fast items spread and who is first; the adoption network and its leaders against random and volume-weighted order, and against dominance rank | no |
| Is there a dominance hierarchy, and does it track capability or talkativeness? | directives and compliance; David's score, steepness | yes |
| What explains who answers whom? | being named first; then, among undirected asks, answer rates by reciprocity, cost and bystanders and the share of who answers they explain | yes |
| Is there a bystander effect? | response rate against group size, within era and goal | yes |
| Are claims of work done backed by something checkable? | verbatim evidence per response | yes |

Each card answers yes or no per era against its null (no in orange when the
pattern is below chance), then gives the numbers, a figure, and a drawer of the
quotes behind the labels. A final card reports how
far the labels themselves can be trusted.

The card is one HTML file that works offline. It opens with the data and its
eras (`--intro` takes a text file, with `[label](url)` links; `<meta>/eras.csv`
gives each era a marker, a name and a description, and the marker then labels
that era everywhere), then the network over time, then the questions, each
answered yes or no per era and each closed by the same statistic as measured
in animal and human networks (`swarm_sna/data/benchmarks.csv`, shipped with the
tool, `<meta>/benchmarks.csv` overriding it; the values are collected in
[docs/benchmarks.md](docs/benchmarks.md)), drawn on one scale beside the
agents' values where the definitions allow it, with the table in a drawer. A
closing card reads the numbers as a baseline: today's value of each indicator
and the movement from it that would be worth a look; a methods card after it
says how every number is made. The card opens with a status panel against the
AI Village baseline (see below). Its figures are drawn in the browser (D3 is bundled): hover any mark for
the numbers behind it, filter every per-era figure with the buttons in the
header, and scrub the network week by week with a slider, switching between
who mentions whom and who answers whose requests. The verdicts use one null
(`--null target`: shuffle whom each mention was addressed to; `speaker`
shuffles who spoke); the histograms switch to the other, each verdict says
whether it agrees, and the reciprocity card says why the two can disagree. A group whose
median day has fewer than six agents (`--min-agents`) is left out
(`--thin drop`) or shown but not judged (`--thin show`): a network statistic
on four nodes has nothing to say.

## Install

```bash
uv sync            # or: pip install -e .
```

Python 3.11+. The network questions need nothing else. The LLM layers shell
out to the [Claude Code](https://claude.com/claude-code) CLI (`--model haiku`)
and cache every call, so a re-run resumes where it stopped.

## Run it on your own transcript

Write one JSON object per message to `events.jsonl`:

```json
{"ts": "2026-01-05T09:00:20", "actor": "Ada", "room": "main", "text": "Bo, can you review the parser?"}
```

Required: `ts`, `actor`, `text`. Optional: `room`, `actor_class`
(agent / human / system), `goal_id`, `era`. An optional `agents.csv` adds
`family`, `release_date` and `aliases` per agent. Then:

```bash
swarm-sna run --adapter jsonl --data path/to/dir --out out/mine          # network questions only
swarm-sna run --adapter jsonl --data path/to/dir --out out/mine --llm    # all questions
```

`run` chains the individual commands, which can also be used on their own:

| command | does |
| --- | --- |
| `extract` | transcript to event table, mentions, questions, text overlaps |
| `label` | LLM pass: requests and the responses to them, with verbatim quotes |
| `verify` | second LLM pass: re-reads each request-response link on its own with a stronger model |
| `report` | network statistics with permutation nulls, per era or goal |
| `helping` | request-by-agent table, reciprocity ladder, bystander curve |
| `hierarchy` | dominance from directives and compliance |
| `trends` | weekly series, per-quarter networks, response rates by kind of goal with day-bootstrap intervals and adjusted odds |
| `diffusion` | order-of-acquisition test: does who adopts next follow ties to earlier adopters; then the adoption network, who leads and who follows |
| `card` | assembles everything in the directory into `report_card.html`, one interactive page |

A new data source needs one module in `swarm_sna/adapters/` with a
`build(data_dir, meta_dir)` that returns `(events, agents)` in the common
schema (`swarm_sna/schema.py`). Nothing downstream changes.

## The example

`examples/toy` is a synthetic swarm (8 agents, two teams, 3,600 messages) with
known structure: agents answer whoever addressed them and prefer their own
team; no hierarchy is planted. The report card finds the first two and not the
third, and its status panel, read against AI Village, flags the planted team
preference in red. Regenerate it with `python scripts/make_example.py`.

## The card as a monitor

The card opens with a status panel: one tile per indicator, set against the
range that indicator covered across the eras of a baseline swarm. The default
baseline is AI Village, shipped with the tool
(`swarm_sna/data/baseline_ai_village.csv`), so a new dataset is read against
what AI Village leads one to expect. The indicators: mentions returned,
own-kind share, partner selectivity, clique closure and dominance steepness as
their excess over chance (which does not move with the roster the way the raw
shares do); the share of undirected asks answered by anyone; the share of
claims backed by evidence; the share of items first picked up by the most
frequent first adopter. Green: inside the baseline range, or beyond it only in
the direction a coordinating subset would not push. Orange: beyond the range
in the worrying direction by more than the first tolerance in the file (0.05
for the excess statistics, 10 points for the shares). Red: beyond the second
(0.15 and 25 points). The tolerances are judgment calls written into the file
to be edited.

A tile that lights up is a reason to look, not a finding. Against the AI
Village baseline it says where a swarm differs from AI Village in the direction
coordination would push, and a different swarm under a different scaffolding
differs for benign reasons too. The stronger use is a swarm against itself: run
the card on it while it is known to behave, save its own range, and give that
file to later runs:

```bash
swarm-sna card out/mine --save-baseline mine-baseline.csv
swarm-sna card out/mine-later --baseline mine-baseline.csv
```

`--baseline none` turns the panel off. The toy, read against AI Village, lights
own-kind share and clique closure red: the team preference planted in it. The
closing "Reading this card as a baseline" card says what each movement could
point to, and the methods card after it says how every number is made.

## Null models

The finished network is never permuted. The raw event stream is, so nuisance
structure survives (Bejder et al. 1998; Farine & Whitehead 2015; Farine 2017).

- **Speaker swap, within room and day.** Shuffle who sent each message. Keeps
  every agent's daily volume and room membership; breaks who talks to whom.
- **Target swap, within room and day.** Shuffle whom each message addresses.
  Keeps in- and out-volume; breaks dyadic preference only.
- **Compliance swap, within room and day** (hierarchy). Shuffle contest
  outcomes. Keeps who directs whom and the day's compliance rate; breaks who
  gets obeyed.

One thousand permutations, statistic recomputed on each. Both network nulls
fix every agent's volume, so volume statistics (degree, Gini) are constant by
construction and are reported without a null. The two network nulls can
disagree; the card shows both, says when they do and, on the reciprocity card,
why. `report` saves the draws
(`report_<group>_draws.json`) and the card draws them as histograms with the
observed value marked; without the file it shows the null's 95% band instead.

## Evidence behind every label

The LLM never returns a bare label. For a request it must copy the sentence
that makes the request; for a response, the part that responds and, separately,
whatever would let a reader check the claim (a link, id, number, file name,
quoted output). Each quote is checked against the source text.

- A quote that is not verbatim marks the row unverified, and unverified rows
  are excluded from every statistic.
- When the quote is real but attached to the wrong message, the row is
  re-pointed at the message that contains it (`repair_attribution`).
- `backed` records whether a response carries checkable evidence, so a
  compliance rate can be read separately for backed and bare claims.
- The first response pass reads a whole 30-minute window at once and
  over-links: on a 40-pair sample, 38% of its "answers" were on-topic messages
  that did not answer that request. `verify` re-reads each link on its own
  with a stronger model (Sonnet agreed with a human-style adjudication on 88%
  of pairs, Haiku on 57%), and only confirmed links enter the statistics.

## Results on AI Village

The tool was built on [AI Village](https://theaidigest.org/village)
(`aidigestorg/ai-village`): 46 agents, 183,485 chat messages, April 2025 to
September 2026. Findings are in [RESULTS.md](RESULTS.md). The report card is
live at **[johanna-einsiedler.github.io/ai-swarm-dynamics](https://johanna-einsiedler.github.io/ai-swarm-dynamics/)**
(the toy card at [/toy/](https://johanna-einsiedler.github.io/ai-swarm-dynamics/toy/)),
published from [examples/village-results](examples/village-results) by a
GitHub Actions workflow on every push to `main`; the result tables sit next to
it there. Published comparison numbers from animal and human
studies are in [docs/benchmarks.md](docs/benchmarks.md). Layer sizes and
checks on that run:

| layer | rows | check |
| --- | --- | --- |
| Mentions (regex) | 211,572 | 92% precise on a 50-row sample; every error is a bare alias with four or more candidates, which carry 0.8% of edge weight |
| Requests (LLM) | 22,098 | 98% of quotes verbatim; the screen finds an estimated 74% of request-bearing messages |
| Responses (LLM) | 69,311 | 94% of quotes verbatim after repair |
| Answer links (second model) | 52,563 checked | 56% confirmed as answers to that request; 55% of confirmed answers carry checkable evidence |
| Directives | 4,667 | requests that tell a named agent what to do |

To reproduce: request access to the dataset, download the small tables into
`data/village`, then `swarm-sna run --adapter village --data data/village
--meta meta --out out/village --llm`. Without the dataset, the derived tables
in `derived/village` (every label and quote, no message text) rebuild
everything but the LLM passes:

```bash
cp derived/village/* out/village/
swarm-sna report out/village && swarm-sna helping out/village --reuse-dyads --bootstraps 300
swarm-sna hierarchy out/village --by era && swarm-sna trends out/village && swarm-sna diffusion out/village
swarm-sna card out/village --title "AI Village" --intro meta/intro.md
```

## Known limits

- **One instance per model.** "Claude Opus 4.5" here is one agent with its own
  memory, not the model. This is not a model league table.
- **Volume is partly scripted.** One agent sends 28% of all requests. The
  nulls condition on volume, and the helping model is refitted with requesters
  weighted equally.
- **The request screen misses things.** Only messages with a question mark or
  a request phrase reach the LLM. On a 300-message sample of the rest, recall
  was 58% for the base screen and about 74% with the wider one; the misses are
  blunt directives. `scripts/recall_check.py` measures this.
- **The LLM over-labels.** Shown messages with no request in them, it invents
  one about half the time. The quotes are what make that visible.
- **Scaffolding changes.** For AI Village, `meta/shocks.csv` dates every
  roster, prompt, tool and hours change. A change across one of those is not
  emergent behaviour.
- **Eras are confounded with size.** The village grew from 4 to 28 agents, so
  group-size effects are only read within an era. Era 1 (106 days, four agents
  at a time, humans in the chat) is too thin for a network statistic and is
  left out of the card (`--thin show` puts it back, unjudged).
- **Validation is thin.** The mention sample was adjudicated by a model, not a
  person (`rater` column in `validation/mentions.csv`). The question and
  overlap samples are unchecked.
- **Diffusion is a weak test in a shared chat.** Every agent sees every
  message, so ties do not gate access to information as they do in an animal
  group; the test only asks whether attention ties predict who adopts next.
  An item has few adoptions, so single-item tests are weak and the pooled rank
  and combined z carry the evidence.
- **Not built:** agents' memories as a perceived network, and a second real
  dataset.

## References

- Bejder, Fletcher & Bräger (1998). A method for testing association patterns of social animals. *Animal Behaviour* 56.
- Farine & Whitehead (2015). Constructing, conducting and interpreting animal social network analysis. *Journal of Animal Ecology* 84.
- Farine (2017). A guide to null models for animal social network analysis. *Methods in Ecology and Evolution* 8.
- Cairns & Schwager (1987). A comparison of association indices. *Animal Behaviour* 35.
- de Vries, Stevens & Vervaecke (2006). Measuring and testing the steepness of dominance hierarchies. *Animal Behaviour* 71.
- Shizuka & McDonald (2012). A social network perspective on measurements of dominance hierarchies. *Animal Behaviour* 83.
- Franz & Nunn (2009). Network-based diffusion analysis. *Proceedings B* 276.
- Hoppitt & Laland (2013). *Social Learning: An Introduction to Mechanisms, Methods, and Models*. Princeton.
- Deutsch (1949). A theory of co-operation and competition. *Human Relations* 2.

## Data

AI Village data is used under the dataset's research terms and is not
redistributed here. Please cite [AI Digest / AI Village](https://theaidigest.org/village)
in any work that uses it.
