#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-search-selfplay-10k
#SBATCH --output=experiments/rl/slurm-%A-%a-phase5-search-selfplay-10k.out
#SBATCH --error=experiments/rl/slurm-%A-%a-phase5-search-selfplay-10k.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --array=0-1

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

GAME_DATA_ROOT="${GAME_DATA_ROOT:-/project/SIGGI/thapanapong.r@cmu.ac.th}"
RUN_NAME="${PHASE5_SELFPLAY_RUN_NAME:-phase5_search_selfplay_10k}"
TOTAL_GAMES="${TOTAL_GAMES:-10000}"
SHARD_COUNT=2
GAMES_PER_SHARD="${GAMES_PER_SHARD:-$(((TOTAL_GAMES + SHARD_COUNT - 1) / SHARD_COUNT))}"

if [[ "${SEARCH_TRACE_GAMES:-}" == "0" ]]; then
  echo "SEARCH_TRACE_GAMES=0 traces all games; leave it unset or use a small positive value." >&2
  exit 2
fi

export GAME_DATA_ROOT
export MODEL="${MODEL:-models/rl/phase5_symbolic_policy_10shards.pt}"
export SHARD_COUNT
export GAMES_PER_SHARD
export MAX_STEPS="${MAX_STEPS:-600}"
export DECK_POOL="${DECK_POOL:-tournament-9}"
export SELFPLAY_DECK_INDICES="${SELFPLAY_DECK_INDICES:-1 2 3 4 5 6 7 8 9}"
export DATASET_DIR="${DATASET_DIR:-${GAME_DATA_ROOT}/${RUN_NAME}/shards}"
export SUMMARY_DIR="${SUMMARY_DIR:-experiments/rl/${RUN_NAME}/summaries}"
export TRACE_DIR="${TRACE_DIR:-experiments/rl/${RUN_NAME}/traces}"

mkdir -p "experiments/rl/${RUN_NAME}" "$DATASET_DIR" "$SUMMARY_DIR" "$TRACE_DIR"

echo "Phase 5 search self-play large two-shard run"
echo "RUN_NAME=$RUN_NAME"
echo "TOTAL_GAMES=$TOTAL_GAMES"
echo "SHARD_COUNT=$SHARD_COUNT"
echo "GAMES_PER_SHARD=$GAMES_PER_SHARD"
echo "MODEL=$MODEL"
echo "DECK_POOL=$DECK_POOL"
echo "DATASET_DIR=$DATASET_DIR"
echo "SUMMARY_DIR=$SUMMARY_DIR"
echo "TRACE_DIR=$TRACE_DIR"
echo "SELFPLAY_DECK_INDICES=$SELFPLAY_DECK_INDICES"
echo "SEARCH_TRACE_GAMES=${SEARCH_TRACE_GAMES:-<unset>}"
echo "POLICY_POOL_MODELS=${POLICY_POOL_MODELS:-<unset>}"

exec bash scripts/slurm/phase5_search_selfplay_conda.sbatch
