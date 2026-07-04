#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-alpha-ppo
#SBATCH --output=experiments/rl/slurm-%j-phase5-alpha-ppo-specialists.out
#SBATCH --error=experiments/rl/slurm-%j-phase5-alpha-ppo-specialists.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

set -euo pipefail

module purge
module load miniconda3/3.12

cd "$SLURM_SUBMIT_DIR"

GAME_DATA_ROOT="${GAME_DATA_ROOT:-/project/SIGGI/thapanapong.r@cmu.ac.th}"
LEAGUE_ROOT="${LEAGUE_ROOT:-${GAME_DATA_ROOT}/phase5_league_alpha}"
ITERATION="${ITERATION:-2}"
if [[ -z "${SOURCE_ITERATION:-}" ]]; then
  if (( ITERATION > 0 )); then
    SOURCE_ITERATION="$((ITERATION - 1))"
  else
    SOURCE_ITERATION="0"
  fi
fi
ITERATION_PADDED="$(printf "%04d" "$ITERATION")"
SOURCE_ITERATION_PADDED="$(printf "%04d" "$SOURCE_ITERATION")"
ITERATION_DIR="${ITERATION_DIR:-${LEAGUE_ROOT}/iterations/iter-${ITERATION_PADDED}}"
TRAJECTORY_DATASET="${TRAJECTORY_DATASET:-${ITERATION_DIR}/raw_train/phase5_alpha_league_selfplay.jsonl}"
SOURCE_CHECKPOINT_DIR="${SOURCE_CHECKPOINT_DIR:-models/rl/phase5_league_alpha/iter-${SOURCE_ITERATION_PADDED}/specialists}"
OUTPUT_CHECKPOINT_DIR="${OUTPUT_CHECKPOINT_DIR:-models/rl/phase5_league_alpha/iter-${ITERATION_PADDED}/specialists}"
REPORT_DIR="${REPORT_DIR:-experiments/rl/phase5_league_alpha/iter-${ITERATION_PADDED}/ppo_specialists}"
REPORT_JSON="${REPORT_JSON:-experiments/rl/phase5_league_alpha/iter-${ITERATION_PADDED}_ppo_specialists_report.json}"
DECK_INDICES="${DECK_INDICES:-1 2 3 4 5 6 7 8 9 10 11 12 13}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.00005}"
CLIP_EPSILON="${CLIP_EPSILON:-0.2}"
POLICY_LOSS_WEIGHT="${POLICY_LOSS_WEIGHT:-1.0}"
VALUE_LOSS_WEIGHT="${VALUE_LOSS_WEIGHT:-0.5}"
ENTROPY_WEIGHT="${ENTROPY_WEIGHT:-0.01}"
SELFPLAY_LIMIT="${SELFPLAY_LIMIT:-0}"

mkdir -p experiments/rl/phase5_league_alpha "$OUTPUT_CHECKPOINT_DIR" "$REPORT_DIR"

export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
PY="$SLURM_SUBMIT_DIR/.conda_ptcg/bin/python"

echo "Phase 5 Alpha on-policy PPO specialist update"
echo "ITERATION=$ITERATION"
echo "SOURCE_ITERATION=$SOURCE_ITERATION"
echo "TRAJECTORY_DATASET=$TRAJECTORY_DATASET"
echo "SOURCE_CHECKPOINT_DIR=$SOURCE_CHECKPOINT_DIR"
echo "OUTPUT_CHECKPOINT_DIR=$OUTPUT_CHECKPOINT_DIR"
echo "REPORT_DIR=$REPORT_DIR"
echo "REPORT_JSON=$REPORT_JSON"
echo "DECK_INDICES=$DECK_INDICES"

DECK_ARGS=()
for index in $DECK_INDICES; do
  DECK_ARGS+=(--deck-index "$index")
done

"$PY" -m ptcg_abc rl-train-phase5-alpha-ppo-specialists \
  --trajectory-dataset "$TRAJECTORY_DATASET" \
  --source-checkpoint-dir "$SOURCE_CHECKPOINT_DIR" \
  --output-checkpoint-dir "$OUTPUT_CHECKPOINT_DIR" \
  --report-dir "$REPORT_DIR" \
  --report-json "$REPORT_JSON" \
  --iteration "$ITERATION" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --clip-epsilon "$CLIP_EPSILON" \
  --policy-loss-weight "$POLICY_LOSS_WEIGHT" \
  --value-loss-weight "$VALUE_LOSS_WEIGHT" \
  --entropy-weight "$ENTROPY_WEIGHT" \
  --selfplay-limit "$SELFPLAY_LIMIT" \
  "${DECK_ARGS[@]}"
