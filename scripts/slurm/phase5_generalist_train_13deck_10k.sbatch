#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-generalist-13deck
#SBATCH --output=experiments/rl/slurm-%j-phase5-generalist-13deck-train.out
#SBATCH --error=experiments/rl/slurm-%j-phase5-generalist-13deck-train.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

set -euo pipefail

GAME_DATA_ROOT="${GAME_DATA_ROOT:-/project/SIGGI/thapanapong.r@cmu.ac.th}"

export GAME_DATA_ROOT
export DECISION_DATASET="${DECISION_DATASET:-${GAME_DATA_ROOT}/phase5_search_decisions_10shards.jsonl}"
if [[ -z "${SELFPLAY_DATASETS:-}" ]]; then
  SELFPLAY_DATASETS="${GAME_DATA_ROOT}/phase5_search_selfplay_10k/shards/phase5_search_selfplay_shard-0.jsonl"
  SELFPLAY_DATASETS="${SELFPLAY_DATASETS} ${GAME_DATA_ROOT}/phase5_search_selfplay_10k/shards/phase5_search_selfplay_shard-1.jsonl"
  SELFPLAY_DATASETS="${SELFPLAY_DATASETS} ${GAME_DATA_ROOT}/phase5_search_selfplay_13deck_10k/shards/phase5_search_selfplay_shard-0.jsonl"
  SELFPLAY_DATASETS="${SELFPLAY_DATASETS} ${GAME_DATA_ROOT}/phase5_search_selfplay_13deck_10k/shards/phase5_search_selfplay_shard-1.jsonl"
fi
export SELFPLAY_DATASETS
export CHECKPOINT="${CHECKPOINT:-models/rl/phase5_generalist_policy_13deck_10k.pt}"
export REPORT_JSON="${REPORT_JSON:-experiments/rl/phase5_generalist_train_report_13deck_10k.json}"
export DECISION_LIMIT="${DECISION_LIMIT:-0}"
export SELFPLAY_LIMIT="${SELFPLAY_LIMIT:-0}"

exec bash scripts/slurm/phase5_generalist_train_conda.sbatch
