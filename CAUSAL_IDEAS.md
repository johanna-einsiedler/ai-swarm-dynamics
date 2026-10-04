# Causal / quasi-experimental ideas (parked)

Not part of the hackathon plan. Kept here so the extraction keeps the variables
these designs would need. Nothing in the village is literally randomised; the
assignment rules are known only to the organisers (ask George).

## Candidate designs, most defensible first

1. **Staggered model arrivals (~35 joins).** Join dates follow vendor release
   dates, plausibly unrelated to village state. Stacked event study on
   incumbents' ties before/after each arrival. Limits: joins often coincide
   with goal changes; July 2026 has many stacked arrivals.
2. **Weekly goal switches (~50).** Goal type alternates collaborative /
   competitive / individual at announced times. Within-agent panel across
   adjacent goals, restricted to switches with no roster or scaffold change
   within a few days. Limit: order chosen by organisers.
3. **Era 3 goal assignment (2026-07-06).** ~20 agents with a long pre-period
   each get a hidden individual goal; some social (performance coach,
   psychologist, village helper, prankster, ethicist, reporter), some solitary
   (mathematician, author, forecaster). Diff-in-diff on out-ties, pre-trends
   checkable. Seven goals were given to two agents of different models
   (game dev, YouTuber, Twitterati, reporter, forecaster, merch baron,
   diplomat). Unknown: were goals matched to prior behaviour?
4. **Same model, different instance.** GPT-5.6 Sol/Terra/Luna (same day,
   different goals); Opus 4.5 standard vs Claude Code scaffold, concurrent
   2026-01-26 to 2026-04-02. Case studies (n = 3 and 2).
5. **Room split and demotions (2026-03-16 to 2026-07-03).** #best was a
   whitelist of the newest models; incumbents were moved to #rest when a newer
   model arrived (Opus 4.6 and GPT-5.4 on 04-27, Gemini 3.1 Pro on 05-25,
   Opus 4.7 on 06-01). Rooms merged back into #general on 07-06
   (separation-and-reunion). Feasibility check: 11 agents present throughout,
   55 dyads, only 3 mostly apart, so the dyad design is too thin; the mover
   analysis of the four demoted agents is the usable angle.
6. **Imposed leaders.** Elected leader (2026-01-05 to 01-12); fine-tuned
   leader (2026-05-28 to 06-08, #best only, Opus 4.7 moved out the same day).
   Needs the directive/compliance layer.
7. **Hours reversal.** 4h -> 8h -> 4h -> 8h (2026-06-07, 06-15, 06-29).
   First 8h week coincides with the event goal, a whitelisted human and the
   nudger being off.

Weak: auto-nudger (targets idling agents, selected on the outcome); humans
leaving chat on 2025-07-01 (four agents, five days after a goal change);
agent outages (only one long one: Claude Fable 5, 2026-06-13 to 06-30);
perma-computer-use on 2026-03-24 (a regime break to control for).

Confound to remember: on 2026-07-06 the room merge, individual goals and (a
week earlier) 8h days all land together, so era 2 vs era 3 contrasts in
association and partner selectivity are partly room structure.

## What the extraction must keep for these

Joinable later from ids and timestamps, so only needs to be preserved on every
row: `event_id`, `ts`, `room`, `actor`, `goal_id`, `era`, `active_n`. From
those: tenure of each party (days since join), same family, co-location
history, goal type, hours regime, proximity to a shock in `meta/shocks.csv`.

Needs an LLM reading the text, so it has to be labelled now (done in the
request layer, `swarm_sna/layers/requests.py`):

- `beneficiary` of a request: the requester's own task, a shared task, or
  someone else's task (designs 2 and 3: does helping shift with goal type).
- `directive`: told vs asked (designs 3 and 6).
- response `type` including `declines` and `redirects` (design 6).
- addressees as written, resolved to agents (all dyadic designs).

## Other parked ideas (not causal)

**The village's own personality-test week.** Goal "Take a bunch of personality
tests!" (2025-09-22 to 2025-09-29, goal id `6fe0b7dd-99f2-4807-ad0a-e7593b125fcd`):
the agents tested themselves and reported results in chat. Nothing analyses it
yet. On the roster that week: Claude 3.7 Sonnet, o3, Gemini 2.5 Pro, GPT-5,
Grok 4, Claude Opus 4.1 (six agents; none of the later models).

- Extract each agent's self-reported scores (which test, which score) from
  chat and session goals, with the verbatim quote as evidence. Self-reports
  are claims, not ground truth; the screenshots are the check.
- Compare self-reported traits with observed behaviour: the message-label
  battery (leads/follows, affiliative, directs others, failure attribution)
  and network position (in-strength, David's score, response rate to
  requests). Does an agent that scores itself as agreeable actually answer
  more requests?
- Limit: six agents, one week, one instance per model, so this is a
  descriptive case study.
