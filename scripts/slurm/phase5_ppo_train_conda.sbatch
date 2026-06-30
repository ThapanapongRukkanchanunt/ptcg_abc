#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-ppo
#SBATCH --output=experiments/rl/slurm-%j-phase5-ppo-train.out
#SBATCH --error=experiments/rl/slurm-%j-phase5-ppo-train.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

set -euo pipefail

module purge
module load miniconda3/3.12

cd "$SLURM_SUBMIT_DIR"
GAME_DATA_ROOT="${GAME_DATA_ROOT:-/project/SIGGI/thapanapong.r@cmu.ac.th}"
mkdir -p experiments/rl "$GAME_DATA_ROOT" models/rl

export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
PY="$SLURM_SUBMIT_DIR/.conda_ptcg/bin/python"

TRAJECTORY_DATASETS="${TRAJECTORY_DATASETS:-${GAME_DATA_ROOT}/phase5_search_selfplay_13deck_10k/shards/phase5_search_selfplay_shard-0.jsonl ${GAME_DATA_ROOT}/phase5_search_selfplay_13deck_10k/shards/phase5_search_selfplay_shard-1.jsonl}"
CHECKPOINT="${CHECKPOINT:-models/rl/phase5_generalist_policy_13deck_10k.pt}"
OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-models/rl/phase5_generalist_policy_13deck_ppo.pt}"
REPORT_JSON="${REPORT_JSON:-experiments/rl/phase5_ppo_train_report_13deck.json}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.00005}"
CLIP_EPSILON="${CLIP_EPSILON:-0.2}"
POLICY_LOSS_WEIGHT="${POLICY_LOSS_WEIGHT:-1.0}"
VALUE_LOSS_WEIGHT="${VALUE_LOSS_WEIGHT:-0.5}"
ENTROPY_WEIGHT="${ENTROPY_WEIGHT:-0.01}"
SELFPLAY_LIMIT="${SELFPLAY_LIMIT:-0}"

DATASET_ARGS=()
for path in $TRAJECTORY_DATASETS; do
  DATASET_ARGS+=(--trajectory-dataset "$path")
done

"$PY" -m ptcg_abc rl-train-phase5-ppo \
  "${DATASET_ARGS[@]}" \
  --checkpoint "$CHECKPOINT" \
  --output-checkpoint "$OUTPUT_CHECKPOINT" \
  --report-json "$REPORT_JSON" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --clip-epsilon "$CLIP_EPSILON" \
  --policy-loss-weight "$POLICY_LOSS_WEIGHT" \
  --value-loss-weight "$VALUE_LOSS_WEIGHT" \
  --entropy-weight "$ENTROPY_WEIGHT" \
  --selfplay-limit "$SELFPLAY_LIMIT"
