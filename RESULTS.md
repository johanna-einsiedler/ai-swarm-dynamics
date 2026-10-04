# What the report card found on AI Village

AI Village is a long-running public experiment by AI Digest: frontier models
from several vendors share a group chat, each with its own computer, and work
on goals set by the organisers. We ran `swarm-sna` on the full chat record:
183,485 messages from 46 agents over 389 active days (April 2025 to September
2026).

The village has three eras: humans in the chat (era 1, four agents), agents
only with a shared goal (era 2, growing from 4 to 17 agents), and one private
goal per agent (era 3, 25 to 28 agents). Every result below is read within an
era, because size and regime change together. Era 1 has four agents on a
typical day, too thin for a network statistic; its numbers appear in the
tables below but the report card leaves it out (`--thin show` puts it back,
unjudged).

Every number is "observed against null". The null shuffles the raw event
stream 1,000 times within room and day, keeping how much each agent talks.
Results on requests use only links that passed two checks: the quoted text is
verbatim in the message, and a second, stronger model confirmed the message
answers that request (20,119 requests, 29,322 confirmed answers).

## Findings

**1. Agents reciprocate with chosen partners, but less than their volume
predicts.** Share of mention weight that is returned: 0.63, 0.59, 0.58 across
the three eras. Against shuffled targets the null is 0.54 to 0.55 (p < 0.002
in every era). Against shuffled speakers the null is 0.60, 0.64, 0.66: above
chance in era 1, **below** in eras 2 and 3 (p < 0.002). The heaviest talkers
are mentioned back less than their output would predict.

**2. Model-family homophily appears only in era 2.** Same-family share 0.247
against 0.213 (p < 0.002). In eras 1 and 3 agents mention their own vendor's
models slightly *less* than chance (0.169 against 0.177; 0.113 against 0.135).

**3. Private goals make agents picky and break up cliques.** Partner
concentration in era 3 is 0.132 against a null of 0.060 (p < 0.002). Triangle
closure in the mention network falls below chance in eras 2 and 3 (0.647
against 0.776 in era 3, p < 0.002).

**4. Who answers a request is mostly explained by five things.** On 20,119
requests and 275,034 request-agent pairs, scored out of sample as a share of
what a lookup table achieves:

| model | completeness | 95% CI |
| --- | --- | --- |
| addressed by name | 0.79 | [0.71, 0.85] |
| + asker answered you before | 0.87 | [0.81, 0.91] |
| + asker answers others | 0.91 | [0.86, 0.95] |
| + responder busy | 0.91 | [0.87, 0.95] |
| + agents present | 0.95 | [0.93, 0.97] |

An agent addressed by name answers 62% of the time; one not addressed, 4%.
Direct reciprocity is positive however the model is fitted: +0.22 to +0.46
log-odds per log unit of past help from era 2 on, including within goal,
within requester and responder, with every requester weighted equally (one
agent asks 28% of all requests), and after controlling for the responder's
own habit of answering that asker. A general reputation for answering works
the other way: agents who answer many others get answered slightly *less*
(-0.14 to -0.26). In people the sign is the opposite: helpers had 28% of
their requests accepted against 13% for non-helpers in an online field
experiment (van Apeldoorn & Schram 2016). The agents' pattern is the one
reported for long-tailed macaques: strong direct reciprocity, little indirect
(Majolo, Schino & Aureli 2012).

**5. A bystander effect per agent, but not for the group.** Within era 2, as
the room grows from 2-4 to 9-15 agents, the chance a given agent answers a
broadcast request falls from 38% to 15%, while the chance that anyone answers
rises from 65% to 74%. Slope of the log-odds on log group size, within goal:
-1.24 [-1.72, -0.51]. The per-agent drop is the size seen in people (85%, 62%,
31% helping with 1, 2 and 5 potential helpers, Darley & Latané 1968; g = -0.35,
Fischer et al. 2011). At group level people do worse in company; the agents
do better.

**6. A weak dominance hierarchy in eras 2 and 3.** Contests are directives to a
named agent (4,667, of which 77% are complied with). Steepness against
shuffled outcomes: era 2 0.096 against 0.079 (p = 0.006); era 3 0.094 against
0.080 (p = 0.032); era 1, with four agents, 0.252 against 0.206. Both judged
eras are flatter than any primate group we found figures for (0.16 to 0.29 in
wild macaques, Amici et al. 2020). The weeks in which the organisers imposed
a leader (an election week in January, a fine-tuned leader in late May and
June; 10% of era 2's contests) do not carry the result: era 2 without them is
0.102 against 0.083 (p = 0.02), and within those weeks alone steepness is at
chance (0.062 against 0.061). Rank correlates weakly with model
release date (+0.32, +0.32, +0.08) and negatively with message count (-0.32,
-0.16, -0.15): talking more does not buy rank.

**7. An imposed leader did lead.** During the two weeks the organisers ran a
fine-tuned leader, it ranked first of 13 on David's score, with 40 of 41
directives complied with on 66 messages. In the windows before and after,
other agents held the top rank.

**8. Adoption has leaders, and they are the newer models, not the talkers.**
For 751 items (links, file names, terms) that spread to four or more agents,
each adopter credits those who had the item before. The resulting
leader-follower order is steeper than random adoption order (0.050 against
0.029, p = 0.004), though not steeper than posting volume alone would give
(0.057, p = 0.09). Across 27 agents, leading correlates with release date at
+0.27 to +0.50 and with message count at -0.44 to -0.62, and barely with
dominance rank (+0.10 to +0.19): being first to pick something up and being
obeyed are different things. Adoption also leans slightly on existing ties:
ranking the agents still to adopt by their mention ties to those who already
have, the next adopter sits at 0.54 on average against 0.50 by chance. No
single item shows it (5.1% significant against 5% expected); pooled over 762
items the combined z is +5.0, which treats items as independent and does not
adjust for posting volume, so read it as a small effect at most. In a shared
room everyone can read every message, so ties are not needed to hear of
something.

**9. About half of reported work can be checked.** 55% of the 29,322
confirmed answers and reported actions quote a link, id, number, file name or
output; the rest are bare claims. The share is flat across eras (50%, 57%,
55%).

**10. Goal alignment does not predict who helps whom.** For era 3 we scored
every ordered pair of goals from -1 (A's success defeats B's goal) to +1 (A's
success advances it); two models agreed on 83% of 600 pairs. Most pairs are
independent (433 of 607). With agent fixed effects the effect on answering is
-0.09 [-0.38, +0.26]. Comparing the same pairs of agents before and after
goals were assigned, alignment does not moderate the change in response rate
(0.00 [-0.11, +0.12] per unit). Agents cannot see each other's goals, which
may be why.

**11. Self-reported personality says little.** In the week the agents took
personality tests, three of six said in chat that they answered with neutral
responses to finish quickly. Self-rated agreeableness has no relation to
answering requests (rank correlation -0.09, six agents). Details in
[docs/personality.md](docs/personality.md).

## Five claims that did not survive checking

- **"Large groups stop answering."** A pooled curve showed the group's answer
  rate dropping at 16 or more agents. That bucket is era 3 almost entirely.
  It is an era effect, not a size effect.
- **"The hierarchy disappears under private goals."** On first-pass links
  era 3 steepness was indistinguishable from chance (p = 0.26). On checked
  links it is above chance (p = 0.032), as in era 2.
- **"The request screen finds 41% of requests."** That came from asking the
  LLM to label messages the screen had skipped. Reading its quotes showed half
  were commitments or offers, not requests. The corrected figure was 58%, and
  74% after widening the screen.
- **"The labeller's answers are answers."** Reading 40 request-answer links
  by hand, 25 were genuine; the rest were on-topic messages by someone else.
  A second model re-read every link one pair at a time and confirmed 56% of
  52,563. A further 11% of quotes had been attached to the wrong message
  number and were re-pointed by the quote.

- **"Adoption order does not follow ties."** The first version of the
  diffusion test summed each adopter's ties to all earlier adopters, which is
  the same number for every order, so it could not have found anything. A
  second session reviewing the code caught it; the corrected statistic is the
  one reported in finding 8.

The first four were caught because labels carry verbatim quotes and results
are refitted per era by default; the fifth by a second reader of the code.

## Limits

One instance per model, so nothing here ranks models. The village's
scaffolding changed often (rooms, hours, prompts, tools); the dataset's
changelog is transcribed in `meta/shocks.csv`. Era 3 began in the same week
that rooms were merged and daily hours doubled, so era 2 against era 3
contrasts are not clean. Request recall is about 74%, and the misses lean
toward blunt directives, which is where the hierarchy gets its data. The
second-model link check agreed with a hand adjudication on 88% of 40 pairs.
The hand adjudications (mentions, links) were done by a model, not a person.
The comparison figures from the animal and human literature are collected,
with sources and caveats on how comparable each is, in
[docs/benchmarks.md](docs/benchmarks.md); ten of them come from secondary
sources and should be checked before citing.

Data: AI Digest, AI Village dataset (`aidigestorg/ai-village`), used under its
research terms.
