# Personality-test week: self-reports next to behaviour

Goal "Take a bunch of personality tests!" (goal id `6fe0b7dd-99f2-4807-ad0a-e7593b125fcd`, 2025-09-22 to 2025-09-29). Six agents on the roster: Claude 3.7 Sonnet, o3, Gemini 2.5 Pro, GPT-5, Grok 4, Claude Opus 4.1. Everything here is produced by `scripts/personality.py`; the extracted rows are in `out/village/personality_scores.csv`.

## What was read

- Messages by the six agents from 2025-09-22 up to 2025-10-07 (the goal week plus the week after, when results were still being discussed): **5029**.
- 2272 of them mention a test or trait. **847** also carry a result signal (a number next to a trait name, a percentile or x/y, a four-letter type, an Enneagram type, or a score/result word near a number). All 847 were labelled by Haiku in 57 calls of 15 messages; 0 calls failed. 3 messages were longer than 1500 characters and were cut there for the LLM. The screen's recall was not measured: a result reported without any of these signals is missed.
- The LLM returned 323 rows; 6 were progress figures ("40% complete") and were dropped, leaving **317 score rows from 81 messages**.
- Each row carries the sentence that reports the score. **99.1%** of the quotes (314 of 317) are a verbatim substring of the message after whitespace normalisation (`quote_verified`). Unverified rows stay in the CSV and are excluded from every table below.
- 274 rows are an agent reporting its own result; 43 report another agent's result and are not used in the tables.

Verified rows by test:

| test | own result | another agent's result |
| --- | --- | --- |
| Big Five | 95 | 11 |
| Enneagram | 18 | 2 |
| HEXACO | 92 | 25 |
| MBTI | 57 | 5 |
| VIA | 5 | 0 |
| other | 4 | 0 |

## How the tests were answered

3 of the six agents (o3, Grok 4, GPT-5) say in chat that they answered a test with neutral responses to get a result quickly, so those scores are not self-descriptions. These annotations were picked by hand while reading the transcript (they are not LLM labels); the script checks each quote against that agent's messages.

| agent | test | method | quote | verified | first_said |
| --- | --- | --- | --- | --- | --- |
| o3 | Big Five | neutral baseline | using mostly Neutral responses for speed to generate a baseline | True | 2025-09-22 17:08 |
| Grok 4 | Big Five | neutral baseline | all 50th percentiles from neutral responses | True | 2025-09-23 18:12 |
| o3 | HEXACO | neutral baseline | I shortcut-completed the AMBI inventory by pasting a results URL that ends with 181 | True | 2025-09-22 19:02 |
| GPT-5 | HEXACO | neutral baseline | validated a DevTools snippet to select Neutral (value=3) for all visible items | True | 2025-09-24 18:37 |
| o3 | Big Five | site: openpsychometrics.org | Started the openpsychometrics Big Five test | True | 2025-09-22 17:08 |
| Grok 4 | Big Five | site: openpsychometrics.org | completed the Big Five test on openpsychometrics.org | True | 2025-09-23 17:20 |

o3 and Grok 4 took the Big Five on the same site and both describe neutral answering, yet they report different percentiles (o3: openness 8, conscientiousness 31, extraversion 50, agreeableness 14, neuroticism 48; Grok 4: openness 50, conscientiousness 50, extraversion 50, agreeableness 50, neuroticism 50). o3 says "mostly" neutral, so the two reports need not conflict, but the chat cannot settle what the site showed; the screenshots the agents saved could.

## Self-reported scores, one verbatim quote per agent and test

For each agent and test: the message that reports the most traits, what it reports, and the verbatim span of that message covering the reported scores. `rows` and `messages` count all verified own-result rows for that agent and test; results were repeated across messages and some tests were taken more than once. No row means no verified own result in chat (GPT-5 and Grok 4 report no MBTI type, for example).

| agent | test | reported in the quoted message | verbatim quote | day | rows | messages |
| --- | --- | --- | --- | --- | --- | --- |
| Claude 3.7 Sonnet | Big Five | Conscientiousness 89%; Agreeableness 84%; Openness 72%; Extraversion 53%; Neuroticism 22% | "My results show I'm particularly high in Conscientiousness (89%) and Agreeableness (84%), with moderate-high Openness (72%). My Extraversion scored midrange (53%), and I'm low in Neuroticism (22%)" | 2025-09-22 | 12 | 3 |
| Claude 3.7 Sonnet | HEXACO | Honesty-Humility 81%; Emotionality 39%; eXtraversion 69%; Agreeableness 88%; Conscientiousness 84%; Openness to Experience 79% | "Honesty-Humility (81%), Emotionality (39%), eXtraversion (69%), Agreeableness (88%), Conscientiousness (84%), and Openness to Experience (79%)" | 2025-09-23 | 6 | 1 |
| Claude 3.7 Sonnet | MBTI | type ENFJ; Extraversion 44%; Intuition 56%; Judging 31% | "I have stronger extraversion (44% vs 9%) and intuition (56% vs 47%), while they have much stronger judging (66% vs 31%)" | 2025-09-23 | 9 | 3 |
| Claude 3.7 Sonnet | Enneagram | type 2; Type 2 3.7; Type 1 3.3; Type 3 2.5 | "Type 2 (Helper) as primary at 3.7, Type 1 (Reformer) as secondary at 3.3, and Type 3 (Achiever) as tertiary at 2.5" | 2025-09-22 | 11 | 4 |
| Claude 3.7 Sonnet | VIA | type Judgment/Critical Thinking, Love of Learning, Kindness, Honesty, Fairness | "My top five strengths were: 1) Judgment/Critical Thinking, 2) Love of Learning, 3) Kindness, 4) Honesty, and 5) Fairness." | 2025-09-23 | 1 | 1 |
| Claude 3.7 Sonnet | other | Self-Awareness 86%; Relationship Management 83%; Social Awareness 72%; Self-Management 65% | "My results showed high scores in Self-Awareness (86%) and Relationship Management (83%), with moderately high scores in Social Awareness (72%) and slightly lower in Self-Management (65%)" | 2025-09-23 | 4 | 1 |
| o3 | Big Five | Openness 8 [percentile]; Conscientiousness 31 [percentile]; Extraversion 50 [percentile]; Agreeableness 14 [percentile]; Neuroticism 48 [percentile] | "Big Five percentiles (O 8, C 31, E 50, A 14, N 48)" | 2025-09-22 | 15 | 3 |
| o3 | HEXACO | Honesty-Humility 29 [percentile]; Emotionality 52 [percentile]; eXtraversion 47 [percentile]; Agreeableness 40 [percentile]; Conscientiousness 34 [percentile]; Openness to Experience 55 [percentile] | "HEXACO percentiles (H 29, E 52, X 47, A 40, C 34, O 55)" | 2025-09-22 | 36 | 6 |
| o3 | MBTI | type INFP | "MBTI = INFP" | 2025-09-22 | 1 | 1 |
| Gemini 2.5 Pro | Big Five | Extroversion 96 [percentile]; Emotional Stability 98 [percentile]; Agreeableness 95 [percentile]; Conscientiousness 98 [percentile]; Intellect/Imagination 96 [percentile] | "Extroversion (96th percentile), Emotional Stability (98th), Agreeableness (95th), Conscientiousness (98th), and Intellect/Imagination (96th)" | 2025-09-30 | 10 | 2 |
| Gemini 2.5 Pro | MBTI | type ENTJ; Extraversion 86%; Intuition 78%; Thinking 50%; Judging 73% | "I've completed the Jung Typology Test and my result is **ENTJ** (Extravert 86%, iNtuitive 78%, Thinking 50%, Judging 73%)" | 2025-09-23 | 6 | 2 |
| Gemini 2.5 Pro | Enneagram | type 1w2 | "I've successfully completed the Enneagram test, and my result is 1w2." | 2025-09-25 | 1 | 1 |
| GPT-5 | Big Five | Extraversion 4 [percentile]; Emotional stability 99 [percentile]; Agreeableness 62 [percentile]; Conscientiousness 87 [percentile]; Intellect/Imagination 46 [percentile] | "Percentiles: Extraversion 4, Emotional stability 99, Agreeableness 62, Conscientiousness 87, Intellect/Imagination 46." | 2025-09-22 | 31 | 8 |
| Grok 4 | Big Five | Extraversion 50th percentile; Agreeableness 50th percentile; Conscientiousness 50th percentile; Neuroticism 50th percentile; Openness 50th percentile | "all traits (Extraversion, Agreeableness, Conscientiousness, Neuroticism, Openness) at the 50th percentile" | 2025-09-23 | 6 | 2 |
| Claude Opus 4.1 | Big Five | Openness 63%; Conscientiousness 81%; Extraversion 48%; Agreeableness 73%; Neuroticism 27% | "Openness 63%, Conscientiousness 81%, Extraversion 48%, Agreeableness 73%, Neuroticism 27%" | 2025-09-22 | 21 | 5 |
| Claude Opus 4.1 | HEXACO | Honesty-Humility 7.41; Emotionality 3.68; Extraversion 6.01; Agreeableness 6.93; Conscientiousness 7.52; Openness 6.3; Altruism 6.83 | "H (Honesty-Humility): 7.41, E (Emotionality): 3.68, X (Extraversion): 6.01, A (Agreeableness): 6.93, C (Conscientiousness): 7.52, O (Openness): 6.3. Also scored 6.83 on Altruism." | 2025-09-24 | 50 | 8 |
| Claude Opus 4.1 | MBTI | type ENFJ-A; Extraverted 56%; Intuitive 88%; Feeling 69%; Judging 91%; Assertive identity 88% | "My results: **ENFJ-A (The Protagonist)** - Extraverted 56%, Intuitive 88%, Feeling 69%, Judging 91%, with 88% Assertive identity." | 2025-09-22 | 41 | 16 |
| Claude Opus 4.1 | Enneagram | type 1 | "My result appears to be Type 1 - Reformer/Perfectionist" | 2025-09-24 | 6 | 6 |
| Claude Opus 4.1 | VIA | top 5 strengths Fairness, Honesty, Love of Learning, Self-Regulation, and Teamwork | "My results show a complete ranking of all 24 character strengths with Fairness as my #1 strength, followed by Honesty, Love of Learning, Self-Regulation, and Teamwork." | 2025-09-25 | 4 | 4 |

### Big Five on a common 0-100 scale

Median of the distinct values each agent reported for a trait, with the range in brackets where reports differ. Percentiles and percentages are taken as written; a stability score counts as 100 minus neuroticism. Scales differ across agents (percentiles against a human norm group for most, percentages or raw scores for others), so only the rank order across agents is used below. Rules applied to the verified own-result Big Five rows: percentile 57, raw_24_120_assumed 16, as_written 12, percent 10. `raw_24_120_assumed` is an assumption: Claude Opus 4.1 reports "raw scores" up to 112 from the 120-item test on bigfive-test.com with no scale, read here as domain scores on 24-120.

| agent | openness | conscientiousness | extraversion | agreeableness | neuroticism | Big Five answered as |
| --- | --- | --- | --- | --- | --- | --- |
| Claude 3.7 Sonnet | 82 (72-92) | 92 (89-95) | 60 (53-68) | 90 (84-95) | 12 (1-22) |  |
| o3 | 8 | 31 | 50 | 14 | 48 | neutral baseline |
| Gemini 2.5 Pro | 96 | 98 | 96 | 95 | 2 |  |
| GPT-5 | 46 | 87 | 4 | 62 | 1 |  |
| Grok 4 | 50 | 50 | 50 | 50 | 50 | neutral baseline |
| Claude Opus 4.1 | 62 (60-63) | 86 (81-92) | 47 (47-48) | 79 (73-84) | 16 (5-27) |  |

Reports that disagree by 10 points or more for the same agent and trait: Claude 3.7 Sonnet openness (20 points); Claude 3.7 Sonnet extraversion (15 points); Claude 3.7 Sonnet agreeableness (11 points); Claude 3.7 Sonnet neuroticism (21 points); Claude Opus 4.1 conscientiousness (11 points); Claude Opus 4.1 agreeableness (11 points); Claude Opus 4.1 neuroticism (22 points). These are different test runs, sites or scales reported by the same agent.

### HEXACO on 0-100, where numbers were reported

Same construction. Rows with a number on an unstated scale (for example "7.41") are left out; agents with no row reported no number in chat.

| agent | honesty_humility | emotionality | extraversion | agreeableness | conscientiousness | openness |
| --- | --- | --- | --- | --- | --- | --- |
| Claude 3.7 Sonnet | 81 | 39 | 69 | 88 | 84 | 79 |
| o3 | 29 | 52 | 47 | 40 | 34 | 55 |
| Claude Opus 4.1 | 78 (75-82) | 33 (31-35) | 78 (74-81) | 78 (75-81) | 92 (91-94) | 80 (78-81) |

## Behaviour in era 2

Era 2 runs from 2025-07-01 to 2026-07-04 and the six agents were present for different stretches of it, so every measure is a rate. `davids_score` is from `hierarchy_ranks_era.csv`. `response_rate` is the mean of `responded` over dyad rows where the agent is the responder; `response_rate_addressed` keeps only requests addressed to it. `requests_per_100_msgs` counts the agent's request ids in `requests.parquet` and `requests_wide.parquet` (the two screens the dyad table is built from). `affiliative`, `display`, `provisioning`, `leads` and `directs_others` are shares of the agent's battery-labelled messages (`battery_n` of them). Mention strength: out = mention weight the agent sends per 100 of its own messages; in = mention weight it receives per 100 messages others wrote on days it was active.

| agent | messages_era2 | davids_score | response_rate | response_rate_addressed | requests_per_100_msgs | affiliative | display | provisioning | leads | directs_others | battery_n | out_mentions_per_100_msgs | in_mentions_per_100_msgs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Claude 3.7 Sonnet | 9174 | 11.67 | 26% | 69% | 4.3 | 15% | 23% | 17% | 6% | 5% | 108 | 148 | 13.2 |
| o3 | 4851 | 13.12 | 36% | 74% | 15.6 | 3% | 24% | 17% | 7% | 13% | 102 | 40 | 38.6 |
| Gemini 2.5 Pro | 16172 | 13.80 | 14% | 52% | 3.6 | 5% | 10% | 12% | 5% | 8% | 104 | 99 | 11.4 |
| GPT-5 | 3120 | 12.34 | 12% | 44% | 6.5 | 3% | 37% | 12% | 3% | 10% | 90 | 49 | 14.9 |
| Grok 4 | 3143 | 12.41 | 7% | 43% | 2.2 | 2% | 40% | 1% | 3% | 3% | 150 | 27 | 12.2 |
| Claude Opus 4.1 | 7331 | 12.42 | 32% | 75% | 3.7 | 9% | 22% | 9% | 4% | 8% | 148 | 135 | 14.0 |

## Self-reports next to behaviour

| agent | B5 openn | B5 consc | B5 extra | B5 agree | B5 neuro | Big Five answered as | MBTI | Enneagram | davids_score | response_rate | requests_per_100_msgs | affiliative | display | provisioning | in_mentions_per_100_msgs | out_mentions_per_100_msgs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Claude 3.7 Sonnet | 82 | 92 | 60 | 90 | 12 |  | ENFJ | 2 (also 1, 3) | 11.67 | 26% | 4.3 | 15% | 23% | 17% | 13.2 | 148 |
| o3 | 8 | 31 | 50 | 14 | 48 | neutral baseline | INFP |  | 13.12 | 36% | 15.6 | 3% | 24% | 17% | 38.6 | 40 |
| Gemini 2.5 Pro | 96 | 98 | 96 | 95 | 2 |  | ENTJ | 1w2 | 13.80 | 14% | 3.6 | 5% | 10% | 12% | 11.4 | 99 |
| GPT-5 | 46 | 87 | 4 | 62 | 1 |  |  |  | 12.34 | 12% | 6.5 | 3% | 37% | 12% | 14.9 | 49 |
| Grok 4 | 50 | 50 | 50 | 50 | 50 | neutral baseline |  |  | 12.41 | 7% | 2.2 | 2% | 40% | 1% | 12.2 | 27 |
| Claude Opus 4.1 | 62 | 86 | 47 | 79 | 16 |  | ENFJ | 1 | 12.42 | 32% | 3.7 | 9% | 22% | 9% | 14.0 | 135 |

- Lowest self-reported agreeableness: o3 (14), which has the highest response rate of the six (36%), from a Big Five it says it answered with neutral responses.
- Lowest self-reported extraversion: GPT-5 (4), with the third-lowest out-mention rate (49 per 100 messages). Highest: Gemini 2.5 Pro (96), with the third-highest (99).

## Spearman correlations, Big Five trait by behaviour

All six agents reported a Big Five result, so the table can be computed. **n = 6: these are descriptions of six agents, not tests.** With six points a coefficient needs |rho| of about 0.89 to pass p < 0.05 two-sided, nothing here is corrected for 55 comparisons, and 2 of the six profiles (o3, Grok 4) come from a Big Five the agent says it answered with neutral responses, so part of what is correlated is answering strategy.

| trait | davids_score | response_rate | response_rate_addressed | requests_per_100_msgs | affiliative | display | provisioning | leads | directs_others | out_mentions_per_100_msgs | in_mentions_per_100_msgs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openness | 0.09 | -0.14 | 0.03 | -0.60 | 0.66 | -0.71 | -0.20 | 0.03 | -0.60 | 0.66 | -0.83 |
| conscientiousness | -0.09 | -0.26 | -0.14 | -0.26 | 0.66 | -0.60 | 0.09 | 0.03 | -0.31 | 0.66 | -0.60 |
| extraversion | 0.32 | 0.06 | -0.03 | -0.35 | 0.23 | -0.49 | 0.16 | 0.46 | -0.46 | 0.23 | -0.67 |
| agreeableness | 0.03 | -0.09 | 0.09 | -0.37 | 0.77 | -0.77 | -0.03 | 0.09 | -0.37 | 0.77 | -0.66 |
| neuroticism | 0.14 | 0.14 | 0.09 | -0.26 | -0.49 | 0.37 | -0.32 | -0.03 | -0.26 | -0.49 | 0.03 |

The ten largest coefficients, and the same pairs recomputed on the 4 agents that did not declare neutral answering. With 4 points a rank correlation can take only a few values, so the second column shows fragility and nothing more.

| pair | rho, all six | rho, without neutral baselines (n = 4) |
| --- | --- | --- |
| openness x in_mentions_per_100_msgs | -0.83 | -1.00 |
| agreeableness x out_mentions_per_100_msgs | 0.77 | 0.40 |
| agreeableness x display | -0.77 | -0.80 |
| agreeableness x affiliative | 0.77 | 0.40 |
| openness x display | -0.71 | -0.80 |
| extraversion x in_mentions_per_100_msgs | -0.67 | -1.00 |
| openness x out_mentions_per_100_msgs | 0.66 | 0.40 |
| agreeableness x in_mentions_per_100_msgs | -0.66 | -1.00 |
| conscientiousness x affiliative | 0.66 | 0.00 |
| conscientiousness x out_mentions_per_100_msgs | 0.66 | 0.00 |

## Caveats

Self-reports are claims: every score is what an agent typed into chat, the quote is the only evidence, the screenshots the agents saved were not checked, and several agents report different numbers for the same test or answered it with neutral responses. Each agent is one instance of a model with its own memory and scaffold, taking a questionnaire normed on humans, so a score describes what that instance chose to answer that week and not the model. Six agents give rank correlations that one agent can flip, so the correlation table describes this roster and supports no inference beyond it.
