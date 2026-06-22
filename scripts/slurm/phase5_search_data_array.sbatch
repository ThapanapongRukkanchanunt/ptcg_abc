#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-search
#SBATCH --output=experiments/rl/slurm-%A-%a-phase5-search.out
#SBATCH --error=experiments/rl/slurm-%A-%a-phase5-search.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-31

set -euo pipefail

module purge
module load miniconda3/3.12

cd "$SLURM_SUBMIT_DIR"
mkdir -p experiments/rl data/datasets/rl reports

export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
PY="$SLURM_SUBMIT_DIR/.conda_ptcg/bin/python"

"$PY" --version
"$PY" -c "import sys; print(sys.executable)"

SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_COUNT="${SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:-32}}"
DATASET_DIR="${DATASET_DIR:-data/datasets/rl/phase5_search/shards}"
TRACE_DIR="${TRACE_DIR:-experiments/rl/phase5_search/traces}"
SUMMARY_DIR="${SUMMARY_DIR:-experiments/rl/phase5_search/summaries}"

mkdir -p "$DATASET_DIR" "$TRACE_DIR" "$SUMMARY_DIR"

OUT="${DATASET_DIR}/phase5_search_decisions_shard-${SHARD_INDEX}.jsonl"
TRACE="${TRACE_DIR}/phase5_search_traces_shard-${SHARD_INDEX}.jsonl"
SUMMARY="${SUMMARY_DIR}/phase5_search_summary_shard-${SHARD_INDEX}.json"

"$PY" -m ptcg_abc rl-generate-search-data \
  --games "${GAMES_PER_SHARD:-1000}" \
  --game-offset "${GAME_OFFSET:-0}" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT" \
  --max-steps "${MAX_STEPS:-600}" \
  --top-k "${TOP_K:-4}" \
  --rollout-steps "${ROLLOUT_STEPS:-18}" \
  --min-candidates "${MIN_CANDIDATES:-2}" \
  --output "$OUT" \
  --trace-output "$TRACE" \
  ${REQUIRE_CHANGED:+--require-changed} \
  | tee "$SUMMARY"
