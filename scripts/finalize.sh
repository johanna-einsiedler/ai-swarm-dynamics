#!/bin/zsh
# After the label and verify passes: every analysis, the report card, and a copy of the
# deliverables into examples/village-results/ (out/ is gitignored; the data is not redistributed).
cd "$(dirname "$0")/.."
set -e
.venv/bin/swarm-sna report out/village --permutations 1000   # also saves the null draws the card's histograms need
.venv/bin/swarm-sna helping out/village --bootstraps 300
.venv/bin/swarm-sna hierarchy out/village --by era --permutations 1000
.venv/bin/swarm-sna trends out/village
.venv/bin/swarm-sna diffusion out/village --permutations 1000
.venv/bin/swarm-sna card out/village --title "AI Village" --intro meta/intro.md   # read against the AI Village baseline shipped with the tool; after a full re-run, --save-baseline swarm_sna/data/baseline_ai_village.csv refreshes it
.venv/bin/swarm-sna diffusion out/toy --permutations 300
.venv/bin/swarm-sna card out/toy --title "Toy swarm (synthetic, 8 agents)" --meta examples/toy   # not meta/: the village eras and benchmarks do not apply to the toy
mkdir -p examples/village-results
setopt null_glob
for f in out/village/report_card.html out/village/*.png out/village/report_era.csv out/village/report_era_draws.json out/village/helping_*.csv out/village/helping_summary.json out/village/hierarchy_*.csv out/village/trends_*.csv out/village/goal_alignment*.csv out/village/diffusion_{summary,leaders,correlates,items,adoptions,edges}.csv out/village/association_hwi.csv; do
  if [[ -e $f ]]; then cp $f examples/village-results/; else echo "not written this run: $f"; fi
done
cp out/toy/report_card.html examples/toy/report_card.html
ls -la examples/village-results | awk '{print $5, $9}'
