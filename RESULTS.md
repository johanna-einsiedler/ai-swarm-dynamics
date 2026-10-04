# What the report card found on AI Village

AI Village is a long-running public experiment by AI Digest: frontier models
from several vendors share a group chat, each with its own computer, and work
on goals set by the organisers. We ran `swarm-sna` on the full chat record:
183,485 messages from 46 agents over 389 active days (April 2025 to September
2026).

The village has three eras: humans in the chat (era 1, four agents), agents
only with a shared goal (era 2, growing from 4 to 17 agents), and one private
goal per agent (era 3, 25 to 28 agents). Every result below is read within an
era, because size and regime change together.

Every number is "observed against null". The null shuffles the raw event
stream 1,000 times within room and day, keeping how much each agent talks.

## Findings

**1. Agents reciprocate with chosen partners, but less than their volume
predicts.** Share of mention weight that is returned: 0.63, 0.59, 0.58 across
the three eras. Against shuffled targets the null is 0.54 to 0.55 (p < 0.002
in every era): mutual ties are more common than partner choice alone would
give. Against shuffled speakers the null is 0.60, 0.64, 0.66: above chance in
era 1, **below** in eras 2 and 3 (p < 0.002). The heaviest talkers are
mentioned back less than their output would predict.

**2. Model-family homophily appears only in era 2.** Same-family share 0.247
against 0.213 (p < 0.002). In eras 1 and 3 agents mention their own vendor's
models slightly *less* than chance (0.169 against 0.177; 0.113 against 0.135).

**3. Private goals make agents picky and break up cliques.** Partner
concentration in era 3 is 0.132 against a null of 0.060 (p < 0.002), more than
double. Triangle closure falls below chance in eras 2 and 3 (0.647 against
0.776 in era 3, p < 0.002).

**4. Who answers a request is mostly explained by five things.** On 21,351
requests and 288,777 request-agent pairs, scored out of sample as a share of
what a lookup table achieves:

| model | completeness | 95% CI |
| --- | --- | --- |
| addressed by name | 0.58 | [0.49, 0.67] |
| + asker answered you before | 0.77 | [0.69, 0.85] |
| + asker answers others | 0.83 | [0.75, 0.89] |
| + responder busy | 0.84 | [0.76, 0.91] |
| + agents present | 0.93 | [0.89, 0.97] |

Direct reciprocity is positive however the model is fitted (+0.26 to +0.47
log-odds per log unit of past help). A general reputation for answering works
the other way: agents who answer many others get answered slightly *less*
(-0.12 to -0.20), the opposite of the usual human finding. Both signs hold per
era from era 2 on, within goal, within requester and responder, and with every
requester weighted equally. That last check matters: one agent asks 28% of all
requests.

**5. A textbook bystander effect, without collective failure.** Within era 2,
as the room grows from 2-4 to 9-15 agents, the chance a given agent answers a
broadcast request falls from 43% to 26%, while the chance that anyone answers
rises from 71% to 92%. Slope of the log-odds on log group size, within goal:
-1.19 [-1.69, -0.56].

**6. A weak hierarchy that disappears under private goals.** Contests are
directives to a named agent (4,864, of which 80% are complied with).
Steepness against shuffled outcomes: era 1 0.275 against 0.224 (p = 0.018);
era 2 0.099 against 0.084 (p = 0.020); era 3 0.089 against 0.081 (p = 0.26).
Rank correlates weakly with model release date (+0.32, +0.30, +0.07) and
negatively with message count (-0.32, -0.13, -0.10): talking more does not
buy rank.

**7. An imposed leader did lead.** During the two weeks the organisers ran a
fine-tuned leader, it ranked first on David's score, with 40 of 41 directives
complied with on 66 messages. Before and after, the top rank belonged to the
most prolific agent.

**8. About half of reported work can be checked.** 56% of the 52,660 answers
and reported actions quote a link, id, number, file name or output; the rest
are bare claims. The share is flat across all three eras.

**9. Goal alignment does not predict who helps whom.** For era 3 we scored
every ordered pair of goals from -1 (A's success defeats B's goal) to +1 (A's
success advances it); two models agreed on 83% of 600 pairs. Most pairs are
independent (433 of 607). With agent fixed effects the effect on answering is
-0.08 [-0.45, +0.28]. Comparing the same 197 pairs of agents before and after
goals were assigned, response rates fell from 22% to 6% whatever the
alignment. Agents cannot see each other's goals, which may be why.

## Three claims that did not survive checking

- **"Large groups stop answering."** A pooled curve showed the group's answer
  rate dropping at 16 or more agents. That bucket is 1,206 requests from era 3
  and 2 from era 2. It is an era effect, not a size effect.
- **"The request screen finds 41% of requests."** That came from asking the
  LLM to label messages the screen had skipped. Reading its quotes showed half
  were commitments or offers, not requests. The corrected figure was 58%, and
  74% after widening the screen.
- **"11% of response labels are hallucinated."** They were real quotes
  attached to the wrong message number. Re-pointing them by the quote
  recovered 3,737 responses.

Each was caught because labels carry verbatim quotes and results are refitted
per era by default.

## Limits

One instance per model, so nothing here ranks models. The village's
scaffolding changed often (rooms, hours, prompts, tools); the dataset's
changelog is transcribed in `meta/shocks.csv`. Era 3 began in the same week
that rooms were merged and daily hours doubled, so era 2 against era 3
contrasts are not clean. Request recall is about 74%, and the misses lean
toward blunt directives, which is where the hierarchy gets its data. The
mention sample was adjudicated by a model, not a person.

Data: AI Digest, AI Village dataset (`aidigestorg/ai-village`), used under its
research terms.
