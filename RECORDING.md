# Three-minute screen recording: shot list

Terminal on the left, browser on the right. Narration in plain words; the
numbers come from the report card on screen, so nothing needs memorising.

**0:00–0:25 The problem.** Show the AI Village chat for five seconds.
"Twenty-odd agents in a group chat for eighteen months. Who helps whom, who
leads, who's ignored? Twenty nodes is a thin network, and every pattern you
see could just be that some agents talk more. Behavioural ecologists hit
exactly this with animal groups, and solved it with permutation nulls. This
tool brings that to agent transcripts."

**0:25–0:55 Run it on the toy.** In the terminal:

    swarm-sna run --adapter jsonl --data examples/toy --out out/toy

"One command, any transcript with a timestamp, a speaker and text. This one is
synthetic: eight agents, two teams, planted reciprocity and team preference,
no hierarchy planted." Open `out/toy/report_card.html`. Point at the glance
list: reciprocity above chance, own-kind above chance, hierarchy within
chance. "It finds the two things I planted and not the one I didn't."

**0:55–1:40 The village card.** Open `out/village/report_card.html`. Scroll
the header tiles. "183,000 messages, 46 agents, 22,000 requests found."
Scroll to the first card and the null-distribution figure. "Every number is
observed against a thousand shuffles of the raw stream, within room and day,
so an agent's volume is held fixed. Grey is chance; the orange line is the
village." Point at one row where the two nulls disagree. "Two different
shuffles, and the card says when they disagree."

**1:40–2:15 Who answers whom.** Scroll to "What explains who answers whom".
"Five things explain 93% of what a lookup table can: being named, having been
answered by the asker before, the asker's reputation, being busy, and how
many others are present." Open the robustness drawer. "Refitted per era,
within goal, with the heaviest asker down-weighted; the signs hold." Open the
evidence drawer. "And every label comes with the verbatim quote behind it,
checked against the message. This one is backed by a commit hash; this one is
a bare claim."

**2:15–2:40 Bystanders and the hierarchy.** Scroll to the bystander card.
"Within one era, as the room grows, each agent answers less and the group
answers more. Textbook." Scroll to the hierarchy card. "There is a weak
dominance hierarchy, it tracks model release date a little and message count
not at all, and it vanishes once each agent has a private goal."

**2:40–3:00 Close.** Show the quality card. "The last card says how far the
labels can be trusted, including what it misses. Everything here is a
command, the adapter is thirty lines, and the questions are fixed, so the
next incident dataset gets the same report card."
