#!/bin/zsh
# Three stages, run unattended. Every LLM call is cached, so a re-run resumes.
#   1. responses to the requests the base screen found (full reply text, 30-min window)
#   2. the requests the base screen misses (imperatives, hand-offs)
#   3. responses to those, same schema
cd "$(dirname "$0")/.."
LOG=out/village/overnight.log
echo "=== stage 1: responses, base screen  $(date) ===" >> $LOG
.venv/bin/swarm-sna label out/village --workers 10 --named-sample 0 --broadcast-sample 0 >> $LOG 2>&1
echo "=== stages 2+3: wide screen requests and their responses  $(date) ===" >> $LOG
.venv/bin/swarm-sna label out/village --screen wide --workers 10 --named-sample 0 --broadcast-sample 0 >> $LOG 2>&1
echo "=== all stages done  $(date) ===" >> $LOG
