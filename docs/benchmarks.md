# Benchmarks from animal and human studies

Published comparison values for the statistics in [RESULTS.md](../RESULTS.md),
collected by web search on 2026-10-04. Each value is the number given in the
source, not a paraphrase.

**How to read the source column.** No mark: the number was read in the paper's
own text or abstract. A dagger (†) means the number was taken from a secondary
summary (textbook, review or search snippet) and should be checked against the
paper before it goes into print. "Not found" means no figure could be
retrieved; nothing here is estimated.

---

## 1. Reciprocity

| statistic | setting / species | value | source (authors, year, journal) |
| --- | --- | --- | --- |
| Correlation across dyads between grooming given and grooming received (weighted mean r) | females in 48 groups, 22 primate species | r = 0.583 (95% CI 0.472 to 0.676) | Schino & Aureli 2008, *Biology Letters* 4:9-11 |
| Same, controlling for maternal kinship | 22 groups | r = 0.468 (95% CI 0.287 to 0.616) | Schino & Aureli 2008, *Biology Letters* 4:9-11 |
| Share of grooming bouts in which the recipient groomed back | golden snub-nosed monkeys, 855 bouts, 6 one-male units | 29.7% | Wei et al. 2012, *PLoS ONE* 7 (PMC3348896) |
| Share of events on a link that are reciprocations; share of links with at least one reciprocation | email, one European research institution | 0.45; 0.90 | Chowdhary et al. 2023, *EPJ Data Science* (arXiv:2207.03910) |
| Same two measures | Twitter mentions | 0.11; 0.40 | Chowdhary et al. 2023, *EPJ Data Science* (arXiv:2207.03910) |
| Share of follow links that are mutual | Twitter follower graph | 22.1% | Kwak et al. 2010, *Proc. WWW* † |

Other channels in Chowdhary et al.: SMS 0.74 and 0.99, phone calls 0.44 and
0.95, private messaging 0.67 and 0.87. For exchange of grooming for a
different service (support in fights), Schino 2007 (*Behavioral Ecology*
18:115-120) gives a much weaker r = 0.154 over 36 studies of 14 species.

**Comparability.** Ours is the share of mention weight that is returned, a
weighted share on a directed network. The event-based share in Chowdhary et
al. is the closest definition; the primate figure is a correlation across
dyads, so it shares a scale with ours only by coincidence. None of these
sources tests against a permutation null that holds volume fixed. Not found:
the share of mutual dyads in a primate grooming network, and reciprocity in
question-and-answer sites or gift exchange.

## 2. Homophily and assortativity

| statistic | setting / species | value | source (authors, year, journal) |
| --- | --- | --- | --- |
| Adults naming any confidant of another race | US national sample, networks of two or more | 8%, "less than one seventh" of the mixing expected under random choice | Marsden 1987, reported in McPherson, Smith-Lovin & Cook 2001, *Annual Review of Sociology* 27:415-444 |
| Cross-race friendships observed, relative to chance | US schoolchildren | two thirds of expected in third grade; 10% of expected by middle school | Shrum et al. 1988, reported in McPherson, Smith-Lovin & Cook 2001, *Annual Review of Sociology* 27:415-444 |
| Sex mixing of confidant networks, relative to the population | US national sample | about 70%; 22% of people have no cross-sex confidant | Marsden 1987, reported in McPherson, Smith-Lovin & Cook 2001, *Annual Review of Sociology* 27:415-444 |
| Assortativity coefficient by sex (Newman's r) | bottlenose dolphins, 62 individuals | r = 0.346 ± 0.053 | Lusseau & Newman 2004, *Proc. R. Soc. B* 271 (Suppl.):S477-S481 |
| Assortativity coefficient by age class | same dolphin network | r = 0.148 ± 0.044 | Lusseau & Newman 2004, *Proc. R. Soc. B* 271 (Suppl.):S477-S481 |

**Comparability.** We report the same-family share of mention weight against
a null. The human rows are also observed-over-expected ratios, so they
compare directly once ours is written the same way. The dolphin value is
Newman's coefficient, which we did not compute. For kinship, the only number
retrieved is indirect: controlling for kinship lowers grooming reciprocity
from r = 0.583 to r = 0.468 (section 1). The share of grooming directed to
kin was not found.

## 3. Dominance hierarchy: steepness, linearity, transitivity

| statistic | setting / species | value | source (authors, year, journal) |
| --- | --- | --- | --- |
| Steepness (slope of normalised David's scores on rank; de Vries et al. 2006) | four wild macaque groups: Japanese, long-tailed, Barbary, moor; 20 to 55 individuals | 0.280, 0.290, 0.244, 0.163 | Amici et al. 2020, *Scientific Reports* 10 (PMC7744554) |
| Steepness (same method) | five wild lemur groups, 6 to 13 individuals | 0.258 to 0.776 | Norscia & Palagi 2015, *PeerJ* 3:e729 |
| Linearity, Landau's h' | same five lemur groups | 0.509 to 0.988 | Norscia & Palagi 2015, *PeerJ* 3:e729 |
| Triangle transitivity, ttri | same five lemur groups | 0.360 to 1.000 | Norscia & Palagi 2015, *PeerJ* 3:e729 |
| Triangle transitivity, ttri, mean ± SE | 101 published dominance matrices across taxa | 0.88 ± 0.02 (0 is the random expectation, where 75% of triangles are transitive; 1 is no cycles) | Shizuka & McDonald 2012, *Animal Behaviour* 83:925-934 |
| Datasets with no cyclic triads (ttri = 1); datasets with Landau's h = 1 | same 101 matrices | 33 of 101; 4 of 101 | Shizuka & McDonald 2012, *Animal Behaviour* 83:925-934 |

**Comparability.** Same statistic, different contest: ours are directives and
compliance, theirs are agonistic wins and losses. Steepness falls when many
pairs are never observed; the macaque values come from matrices with 62% of
relationships unknown, and 26% of dyads were empty on average in the 101
matrices. Our own steepness should therefore be read against its permutation
null, as RESULTS.md does. Not found: per-species macaque values in
Balasubramaniam et al. 2012 (*Am. J. Primatol.* 74:915-925, paywalled); a
summary value in Huang et al. 2024 (*Behavioral Ecology* 35:arae066; 153
data points, 27 species, none reported in the text); a summary ttri in
Shizuka & McDonald 2015 (*J. R. Soc. Interface* 12:20150080; 172 groups, 85
species, "cycles were very rare"); and the share of the 101 matrices with
ttri above 0.8.

## 4. Bystander effect

| statistic | setting / species | value | source (authors, year, journal) |
| --- | --- | --- | --- |
| Overall effect of bystanders on helping, Hedges' g | meta-analysis, 105 effect sizes, over 7,700 participants | g = -0.35 | Fischer et al. 2011, *Psychological Bulletin* 137:517-537 |
| Share who reported a staged seizure, by number of people believed able to help (1, 2, 5) | laboratory, students on an intercom | 85%, 62%, 31% | Darley & Latané 1968, *J. Personality and Social Psychology* 8:377-383 † |
| Share who reported smoke: alone; three naive participants; with two passive confederates | laboratory | 75%; 38%; 10% | Latané & Darley 1968, *J. Personality and Social Psychology* 10:215-221 † |
| Help rate alone against in groups | review of 56 comparisons, over 6,000 participants | about 75% against under 53%; alone higher in 48 of 56 | Latané & Nida 1981, *Psychological Bulletin* 89:308-324 † |
| Time to first reply to a help request, no name given against one participant named | about 400 online chat groups | 51 s against 37 s; delay grows with the number present only when no name is given (coefficient not found) | Markey 2000, *Computers in Human Behavior* 16:183-188 † |
| Share replying to an emailed request, by number of other recipients shown (0, 1, 14, 49) | university students, 100 per condition | 38%, 33%, 14%, 14% | Blair, Thompson & Wuensch 2005, *Basic and Applied Social Psychology* 27:171-178 |
| Replies to an emailed request, one recipient against five | email | more, longer and more helpful replies to a single recipient; rates not found | Barron & Yechiam 2002, *Computers in Human Behavior* 18:507-520 |

**Comparability.** The laboratory rows are per-person response probabilities
by group size, which is our per-agent rate. Two differences matter: those
are emergencies seen once, ours are routine requests that stay on screen;
and their group size is assigned, ours varies with the day. The effect is
not universal online: a corporate email study found no group-size effect
(77.1% replied overall; LaMastro & McFarland 2011, *Issues in Information
Systems* 12(1):189-193). Abbate et al. 2022 (*Frontiers in Psychology*
13:945630) report "78.3%" for a sole recipient against "21.7%" with 14
others, but the two figures sum to 100 and look like shares of all helpers
rather than helping rates, so they are left out of the table.

## 5. Response rates to requests for help online

| statistic | setting / species | value | source (authors, year, journal) |
| --- | --- | --- | --- |
| Questions with at least one answer; median time to first answer | Stack Overflow, data to 2010 | 92.6%; 11 minutes | Mamykina et al. 2011, *Proc. CHI* |
| Questions answered; median time; answered within one hour; answered after more than a day | Stack Overflow | 91.3%; 16 minutes; 63.5%; 9.98% | Bhat et al. 2014, *Proc. ASONAM* |
| Emailed requests that got a reply | one company, internal email | 77.1% | LaMastro & McFarland 2011, *Issues in Information Systems* 12(1):189-193 |
| Service requests to strangers that got any response | online service-exchange community, 189 requests | 47% | van Apeldoorn & Schram 2016, *PLoS ONE* 11:e0152076 |

Mamykina et al. also cite 88.2% for Yahoo! Answers and about 66% for Naver
KiN. Weaker figures: nearly 80% of newcomers' posts to four open-source
mailing lists got a reply (Jensen, King & Kuechler 2011, *Proc. HICSS* †);
10.16% of 118,216 questions in Reddit celebrity interviews got an answer
(Danish, Dahiya & Talukdar 2015, arXiv:1512.01768 †), which is one answerer
facing thousands of askers and not a peer community.

**Comparability.** These are group-level rates (did anyone answer), the
counterpart of our "anyone answers" rate, not our per-agent rate. Stack
Overflow's 92% has no time limit, while our responses are linked within a
30-minute window, so the one-hour figure (63.5%) is the fairer match. Not
found: reply shares for Usenet (Arguello et al. 2006) and for developer chat
rooms.

## 6. Direct and indirect reciprocity

| statistic | setting / species | value | source (authors, year, journal) |
| --- | --- | --- | --- |
| Donations by receiver's record of giving to others | laboratory giving game, 79 students, direct reciprocity ruled out | donations "more frequent to receivers who had been generous to others"; shares not found | Wedekind & Milinski 2000, *Science* 288:850-852; replicated with 60 participants by Russell, Stoilova & Dosoftei 2020, *Games* 11:58 |
| Helping rate when the donor's own record is public against private | laboratory helping game, 80 subjects, 80 rounds | 74% against 37% (recipient's record public); 32% when neither record is public; help rises with the recipient's record in both cases | Engelmann & Fischbacher 2009, *Games and Economic Behavior* 67:399-407 |
| Requests accepted outright, requester with a history of serving others against a neutral profile | field experiment, online service-exchange community | 28.1% against 12.9% | van Apeldoorn & Schram 2016, *PLoS ONE* 11:e0152076 |
| Direct, indirect and generalised reciprocity in grooming | long-tailed macaques | "strong evidence for direct reciprocity, limited support for indirect reciprocity and no evidence for generalized reciprocity" | Majolo, Schino & Aureli 2012, *Animal Behaviour* 83:763-771 |

**Comparability.** In people, a record of helping others raises the help
received, in the laboratory and in the field. Our reputation term is the
effect of "asker answers others" on being answered, estimated with direct
reciprocity already in the model, so it is closest to the field experiment,
where the requester had never helped the person asked. There the record is a
visible profile; agents have to infer it from the chat.

## 7. Group size and contribution per member

| statistic | setting / species | value | source (authors, year, journal) |
| --- | --- | --- | --- |
| Pull per person: alone, groups of 3, groups of 8 | rope pulling, men | 63 kg, 53 kg, 31 kg (85% and 49% of solo) | Ringelmann 1913, reported in Kravitz & Martin 1986, *J. Personality and Social Psychology* 50:936-941 † |
| Sound per person as a share of solo output: real groups of 2, 4, 6; then alone but believing 1 or 5 others are shouting | shouting and clapping, students | 71%, 51%, 40%; then 82%, 74% | Latané, Williams & Harkins 1979, *J. Personality and Social Psychology* 37:822-832 † |
| Mean effect of working collectively against working individually | meta-analysis, 78 studies | d = 0.44 | Karau & Williams 1993, *J. Personality and Social Psychology* 65:681-706 † |

Ingham et al. 1974 (*J. Experimental Social Psychology* 10:371-384 †) found
about 90% of solo pull in believed pairs and 85% in believed groups of six.
For LLM agents, Bertalanič & Fortuna 2026 (arXiv:2606.02646) report that
"thirty dense debating agents produce no more answer diversity than one."

**Comparability.** Only loosely comparable. These tasks are additive: every
unit of effort adds to the group's output. Answering a broadcast request is
not: one answer is enough, so a falling per-agent rate need not mean lost
output. The believed-group figures (82%, 74%) isolate motivation; the
real-group figures also include coordination loss.

---

## How the agents compare

| question | agents (RESULTS.md) | closest benchmark | caveat |
| --- | --- | --- | --- |
| Reciprocity | 0.58 to 0.63 of mention weight returned; null 0.54 to 0.55 (targets shuffled) | email 0.45, private messaging 0.67, Twitter mentions 0.11 (Chowdhary et al. 2023) | Event share against weight share; no human or primate source uses a volume-preserving null, and ours shows only 0.03 to 0.09 of the agents' value is beyond chance |
| Homophily | era 2: same-family share 0.247 against 0.213, so cross-family mentions run at 96% of chance; eras 1 and 3 slightly below chance | cross-race confidants at under one seventh of chance; cross-race school friendships at 10% to 67% of chance; dolphins by sex r = 0.346 | Ratios are comparable; the dolphin coefficient is a different statistic. Model family is not visible to agents the way race or sex is to people |
| Steepness | 0.275, 0.099, 0.089 by era; nulls 0.224, 0.084, 0.081 | wild macaques 0.16 to 0.29; lemurs 0.26 to 0.78 | Era 1 (four agents) sits in the macaque range, eras 2 and 3 below every primate value found. Contests are directives, not fights, and sparse matrices lower steepness in both literatures |
| Transitivity | not computed for the hierarchy | ttri 0.88 ± 0.02 over 101 animal matrices | Our "triangle closure" (0.647 against 0.776) is clustering in the mention network, a different quantity; it must not be set beside ttri |
| Bystander effect, per individual | 43% to 26% as the room grows from 2-4 to 9-15 agents (60% retained); slope -1.19 [-1.69, -0.56] log-odds per log group size | seizure study 85% to 31% from 1 to 5 helpers (36% retained); email 38% to 14% from 1 to 15 recipients (37% retained); g = -0.35 overall | Same direction, similar size. Slopes computed here from the published percentages, on the same scale as ours, are about -1.6 (seizure) and -0.5 (email); these are our arithmetic, not figures from the papers |
| Bystander effect, whole group | anyone answers: 71% to 92% as the room grows | smoke study: 38% of three-person groups reported against 75% of people alone | The agents' group does better as it grows; the classic emergency result is that the group does worse. Requests in a chat persist and cost little to answer, unlike a one-off emergency |
| Response rate | 71% to 92% of broadcast requests answered by someone (era 2, 30-minute window) | Stack Overflow 63.5% within an hour and 91 to 93% eventually; corporate email 77% | Our denominator is LLM-labelled requests with about 74% recall, and the misses are blunt directives |
| Reputation | being known to answer others lowers the chance of being answered: -0.12 to -0.20 log-odds; direct reciprocity +0.26 to +0.47 | people: 28.1% against 12.9% accepted for requesters with a serving record, an odds ratio of about 2.6 by our arithmetic; macaques: direct reciprocity only | The agents' sign is opposite to the human one and their pattern matches the macaque one. Our coefficient is per log unit of past answers and conditional on direct reciprocity, so magnitudes are not comparable, only signs |
| Contribution per member | per-agent answer rate at 9-15 agents is 60% of the rate at 2-4 | rope pulling 49% of solo at 8; shouting 40% at 6 (real groups), 74% at 6 (believed groups) | Additive effort tasks against a task where one answer suffices; the comparison is of shape, not of mechanism |

## Not found, or to verify before citing

- Exact donation shares in Wedekind & Milinski 2000 and helping rates in
  Seinen & Schram 2006 (*European Economic Review* 50:581-602).
- Response rates in Barron & Yechiam 2002 and the group-size correlation in
  Markey 2000.
- Share of mutual dyads in primate grooming networks; share of grooming
  directed to kin; reciprocity on question-and-answer sites.
- Worked-example steepness values in de Vries, Stevens & Vervaecke 2006
  (*Animal Behaviour* 71:585-592) and typical Landau's h for baboons or
  chimpanzees.
- Whether higher-reputation askers on Stack Overflow get answered more:
  Calefato, Lanubile & Novielli 2018 (*Information and Software Technology*
  94:186-207) report asker reputation as the strongest predictor of getting
  an accepted answer, but the effect size was not retrieved.
- Every row marked † above: the number is the one usually quoted for that
  paper, taken here from a secondary source.
- Publication details not confirmed against the journal page: the venue of
  Chowdhary et al. (taken as *EPJ Data Science* 2023) and the journal of
  Blair, Thompson & Wuensch 2005.
