#!/usr/bin/env bash
# Evaluates the 9 checkpoints on the `train` split.
# Assumes models are available on HuggingFace under the shashaank0707 namespace.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RESULTS_ROOT="${RESULTS_ROOT:-./results/diagnostics}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"
HF_NAMESPACE="shashaank0707"

mkdir -p "$RESULTS_ROOT"

log() { printf '\n\033[1m[run_p1_evals]\033[0m %s\n' "$1"; }

# The 9 primary RL arms
ARMS=("E1" "E3" "E4")
SEEDS=("42" "123" "456")

log "Evaluating checkpoints on the TRAIN split to check for memorization..."

for arm in "${ARMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    tag="${arm}_s${seed}"
    # NOTE: Adjust the HuggingFace repository name if it differs!
    adapter="${HF_NAMESPACE}/AgentDebugger-${tag}"
    
    out="$RESULTS_ROOT/${tag}_train_eval.json"
    
    # We use format 'structured' for all of these as per the design
    format="structured"
    
    # Evaluate on Train
    if [[ -f "$out" ]]; then
      log "$tag already evaluated on train -> $out (skipping)"
    else
      log "Evaluating $tag on the train split..."
      python -m agentdebugger.cli evaluate-curriculum \
        --base-model "$BASE_MODEL" \
        --adapter "$adapter" \
        --split train \
        --format "$format" \
        --output "$out"
    fi
    
    # Evaluate on QuixBugs
    quixbugs_out="$RESULTS_ROOT/${tag}_quixbugs_eval.json"
    if [[ -f "$quixbugs_out" ]]; then
      log "$tag already evaluated on quixbugs -> $quixbugs_out (skipping)"
    else
      log "Evaluating $tag on the QuixBugs transfer set..."
      ./scripts/eval_quixbugs.py \
        --base-model "$BASE_MODEL" \
        --adapter "$adapter" \
        --format "$format" \
        --output "$quixbugs_out"
    fi
  done
done

log "P1 Evaluations complete. Results saved to $RESULTS_ROOT."
