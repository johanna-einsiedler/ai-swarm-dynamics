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
.venv/bin/swarm-sna card out/village --title "AI Village"
.venv/bin/swarm-sna diffusion out/toy --permutations 300
.venv/bin/swarm-sna card out/toy --title "Toy swarm (synthetic, 8 agents)"
mkdir -p examples/village-results
cp out/village/report_card.html out/village/*.png out/village/report_era.csv out/village/helping_*.csv out/village/hierarchy_*.csv out/village/trends_*.csv out/village/goal_alignment*.csv out/village/diffusion_summary.csv out/village/diffusion_leaders.csv out/village/diffusion_correlates.csv out/village/diffusion_items.csv out/village/diffusion_adoptions.csv out/village/association_hwi.csv examples/village-results/
cp out/toy/report_card.html examples/toy/report_card.html
ls -la examples/village-results | awk '{print $5, $9}'
