#!/usr/bin/env bash
# Runs the full Publication-Strategy §0 experiment matrix: B0, B1, then the 9 RL
# runs (E1, E3, E4 x seeds {42,123,456}), training then evaluating each on the
# held-out split. Safe to re-run after an interruption (spot/preemptible GPU) --
# anything already finished (checkpoint or report present) is skipped.
#
# Run the calibration run FIRST and separately (see docs/CONTEXT.md "Quick
# reference"); this script is for the real matrix only, after you've picked
# --reward-workers based on what the calibration run showed.
#
# Usage:
#   ./scripts/run_matrix.sh                 # everything
#   ONLY_ARMS="E1 E3" ./scripts/run_matrix.sh   # just some arms (still all seeds)
#   SEEDS="42" ./scripts/run_matrix.sh          # just one seed, for a quick check
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CKPT_ROOT="${CKPT_ROOT:-./ckpt}"
RESULTS_ROOT="${RESULTS_ROOT:-./results}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"
MAX_STEPS="${MAX_STEPS:-500}"
REWARD_WORKERS="${REWARD_WORKERS:-1}"
SEEDS="${SEEDS:-42 123 456}"
ONLY_ARMS="${ONLY_ARMS:-B0 B1 E1 E3 E4}"
PY="${PY:-./.venv/bin/python}"

mkdir -p "$RESULTS_ROOT"

log() { printf '\n\033[1m[run_matrix]\033[0m %s\n' "$1"; }

# arm -> "reward_config format" ("-" for zero-shot baselines, no reward config).
# A case statement rather than an associative array: bash 3.2 (macOS's default,
# if you're dry-running this locally) doesn't support `declare -A`.
arm_spec() {
  case "$1" in
    B0) echo "- structured" ;;
    B1) echo "- free_form" ;;
    E1) echo "R0 structured" ;;
    E2) echo "R1 free_form" ;;
    E3) echo "R1 structured" ;;
    E4) echo "R2 structured" ;;
    *) echo "" ;;
  esac
}

run_baseline() {
  local arm="$1" format="$2"
  local out="$RESULTS_ROOT/${arm}.json"
  if [[ -f "$out" ]]; then
    log "$arm already evaluated -> $out (skipping)"
    return
  fi
  log "$arm: zero-shot eval, format=$format"
  "$PY" -m agentdebugger.cli evaluate-curriculum \
    --base-model "$BASE_MODEL" --split heldout --format "$format" --output "$out"
}

run_rl_arm() {
  local arm="$1" reward_config="$2" format="$3" seed="$4"
  local tag="${arm}_s${seed}"
  local ckpt="$CKPT_ROOT/$tag"
  local report="$RESULTS_ROOT/${tag}.json"

  if [[ -f "$ckpt/adapter_config.json" ]]; then
    log "$tag already trained -> $ckpt (skipping training)"
  else
    log "$tag: training (reward=$reward_config format=$format seed=$seed max_steps=$MAX_STEPS)"
    WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-agentdebugger-matrix}" \
    WANDB_NAME="$tag" \
    "$PY" -m agentdebugger.cli train \
      --model "$BASE_MODEL" --max-steps "$MAX_STEPS" --seed "$seed" \
      --reward-config "$reward_config" --format "$format" \
      --reward-workers "$REWARD_WORKERS" --output-dir "$ckpt"
  fi

  if [[ -f "$report" ]]; then
    log "$tag already evaluated -> $report (skipping eval)"
    return
  fi
  log "$tag: evaluating on held-out split"
  "$PY" -m agentdebugger.cli evaluate-curriculum \
    --base-model "$BASE_MODEL" --adapter "$ckpt" --split heldout \
    --format "$format" --output "$report"
}

for arm in $ONLY_ARMS; do
  spec="$(arm_spec "$arm")"
  if [[ -z "$spec" ]]; then
    echo "unknown arm '$arm' (expected one of: B0 B1 E1 E2 E3 E4)" >&2
    exit 1
  fi
  read -r reward_config format <<< "$spec"

  if [[ "$reward_config" == "-" ]]; then
    run_baseline "$arm" "$format"
    continue
  fi
  for seed in $SEEDS; do
    run_rl_arm "$arm" "$reward_config" "$format" "$seed"
  done
done

log "matrix run finished. Reports in $RESULTS_ROOT/, checkpoints in $CKPT_ROOT/"
