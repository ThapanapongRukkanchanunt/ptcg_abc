#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-search-selfplay
#SBATCH --output=experiments/rl/slurm-%A-%a-phase5-search-selfplay.out
#SBATCH --error=experiments/rl/slurm-%A-%a-phase5-search-selfplay.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

set -euo pipefail

module purge
module load miniconda3/3.12

cd "$SLURM_SUBMIT_DIR"
GAME_DATA_ROOT="${GAME_DATA_ROOT:-/project/SIGGI/thapanapong.r@cmu.ac.th}"
mkdir -p experiments/rl/phase5_search_selfplay "$GAME_DATA_ROOT" models/rl reports

export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
PY="$SLURM_SUBMIT_DIR/.conda_ptcg/bin/python"

"$PY" --version
"$PY" -c "import sys; print(sys.executable)"

MODEL="${MODEL:-models/rl/phase5_symbolic_policy_10shards.pt}"
GAMES_PER_SHARD="${GAMES_PER_SHARD:-36}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_COUNT="${SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}"
GAME_OFFSET_BASE="${GAME_OFFSET:-0}"
ABS_GAME_OFFSET=$((GAME_OFFSET_BASE + SHARD_INDEX * GAMES_PER_SHARD))
DECK_POOL="${DECK_POOL:-tournament-9}"

DATASET_DIR="${DATASET_DIR:-${GAME_DATA_ROOT}/phase5_search_selfplay/shards}"
SUMMARY_DIR="${SUMMARY_DIR:-experiments/rl/phase5_search_selfplay/summaries}"
TRACE_DIR="${TRACE_DIR:-experiments/rl/phase5_search_selfplay/traces}"
mkdir -p "$DATASET_DIR" "$SUMMARY_DIR" "$TRACE_DIR"

OUT="${DATASET_DIR}/phase5_search_selfplay_shard-${SHARD_INDEX}.jsonl"
SUMMARY="${SUMMARY_DIR}/phase5_search_selfplay_summary_shard-${SHARD_INDEX}.json"
TRACE="${SEARCH_TRACE_OUTPUT:-${TRACE_DIR}/phase5_search_selfplay_traces_shard-${SHARD_INDEX}.jsonl}"

DECK_ARGS=()
if [[ -n "${SELFPLAY_DECK_INDICES:-}" ]]; then
  for deck_index in $SELFPLAY_DECK_INDICES; do
    DECK_ARGS+=(--deck-index "$deck_index")
  done
fi
if [[ -n "${DECK_A_INDEX:-}" ]]; then
  DECK_ARGS+=(--deck-a-index "$DECK_A_INDEX")
fi
if [[ -n "${DECK_B_INDEX:-}" ]]; then
  DECK_ARGS+=(--deck-b-index "$DECK_B_INDEX")
fi

SEARCH_ARGS=()
if [[ -n "${SEARCH_TOP_K:-}" ]]; then
  SEARCH_ARGS+=(--search-top-k "$SEARCH_TOP_K")
fi
if [[ -n "${SEARCH_ROLLOUT_STEPS:-}" ]]; then
  SEARCH_ARGS+=(--search-rollout-steps "$SEARCH_ROLLOUT_STEPS")
fi
if [[ -n "${POLICY_PRIOR_WEIGHT:-}" ]]; then
  SEARCH_ARGS+=(--policy-prior-weight "$POLICY_PRIOR_WEIGHT")
fi
if [[ -n "${NEURAL_ACTION_VALUE_WEIGHT:-}" ]]; then
  SEARCH_ARGS+=(--neural-action-value-weight "$NEURAL_ACTION_VALUE_WEIGHT")
fi
if [[ -n "${NEURAL_TACTICAL_WEIGHT:-}" ]]; then
  SEARCH_ARGS+=(--neural-tactical-weight "$NEURAL_TACTICAL_WEIGHT")
fi

POOL_ARGS=()
if [[ -n "${POLICY_POOL_MODELS:-}" ]]; then
  for path in $POLICY_POOL_MODELS; do
    POOL_ARGS+=(--policy-pool-model "$path")
  done
fi

TRACE_ARGS=()
if [[ -n "${SEARCH_TRACE_GAMES:-}" ]]; then
  TRACE_ARGS+=(--search-trace-output "$TRACE" --search-trace-games "$SEARCH_TRACE_GAMES")
fi

echo "MODEL=$MODEL"
echo "DECK_POOL=$DECK_POOL"
echo "GAMES_PER_SHARD=$GAMES_PER_SHARD"
echo "ABS_GAME_OFFSET=$ABS_GAME_OFFSET"
echo "DATASET_DIR=$DATASET_DIR"
echo "SUMMARY_DIR=$SUMMARY_DIR"
echo "TRACE_DIR=$TRACE_DIR"
echo "POLICY_POOL_MODELS=${POLICY_POOL_MODELS:-<unset>}"

"$PY" -m ptcg_abc rl-generate-phase5-search-selfplay \
  --model "$MODEL" \
  --games "$GAMES_PER_SHARD" \
  --game-offset "$ABS_GAME_OFFSET" \
  --max-steps "${MAX_STEPS:-600}" \
  --deck-pool "$DECK_POOL" \
  --output "$OUT" \
  --report-json "$SUMMARY" \
  "${DECK_ARGS[@]}" \
  "${SEARCH_ARGS[@]}" \
  "${POOL_ARGS[@]}" \
  "${TRACE_ARGS[@]}"
