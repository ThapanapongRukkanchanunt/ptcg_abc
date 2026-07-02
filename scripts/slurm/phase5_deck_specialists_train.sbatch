#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-specialists
#SBATCH --output=experiments/rl/slurm-%j-phase5-deck-specialists.out
#SBATCH --error=experiments/rl/slurm-%j-phase5-deck-specialists.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

set -euo pipefail

module purge
module load miniconda3/3.12

cd "$SLURM_SUBMIT_DIR"

GAME_DATA_ROOT="${GAME_DATA_ROOT:-/project/SIGGI/thapanapong.r@cmu.ac.th}"
LEAGUE_ROOT="${LEAGUE_ROOT:-${GAME_DATA_ROOT}/phase5_league_alpha}"
ITERATION="${ITERATION:-0}"
ITERATION_PADDED="$(printf "%04d" "$ITERATION")"
ITERATION_DIR="${ITERATION_DIR:-${LEAGUE_ROOT}/iterations/iter-${ITERATION_PADDED}}"
BOOTSTRAP_DATASET="${BOOTSTRAP_DATASET:-${ITERATION_DIR}/raw_train/phase5_alpha_rule_bootstrap.jsonl}"
DECISION_DATASET="${DECISION_DATASET:-${GAME_DATA_ROOT}/phase5_search_decisions_10shards.jsonl}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-models/rl/phase5_league_alpha/iter-${ITERATION_PADDED}/specialists}"
REPORT_DIR="${REPORT_DIR:-experiments/rl/phase5_league_alpha/iter-${ITERATION_PADDED}/specialists}"
REPORT_JSON="${REPORT_JSON:-experiments/rl/phase5_league_alpha/iter-${ITERATION_PADDED}_deck_specialists_report.json}"
DECK_INDICES="${DECK_INDICES:-1 2 3 4 5 6 7 8 9 10 11 12 13}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
D_MODEL="${D_MODEL:-128}"
MAX_ENTITIES="${MAX_ENTITIES:-96}"
MAX_ACTIONS="${MAX_ACTIONS:-128}"
MAX_PREVIOUS_ACTIONS="${MAX_PREVIOUS_ACTIONS:-16}"
DECISION_LIMIT="${DECISION_LIMIT:-0}"
SELFPLAY_LIMIT="${SELFPLAY_LIMIT:-0}"

mkdir -p experiments/rl/phase5_league_alpha "$CHECKPOINT_DIR" "$REPORT_DIR"

export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
PY="$SLURM_SUBMIT_DIR/.conda_ptcg/bin/python"

echo "Phase 5 Alpha deck-specialist train"
echo "ITERATION=$ITERATION"
echo "BOOTSTRAP_DATASET=$BOOTSTRAP_DATASET"
echo "DECISION_DATASET=$DECISION_DATASET"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "REPORT_DIR=$REPORT_DIR"
echo "REPORT_JSON=$REPORT_JSON"
echo "DECK_INDICES=$DECK_INDICES"

DECK_ARGS=()
for index in $DECK_INDICES; do
  DECK_ARGS+=(--deck-index "$index")
done

ARGS=(
  rl-train-phase5-deck-specialists
  --decision-dataset "$DECISION_DATASET"
  --selfplay-dataset "$BOOTSTRAP_DATASET"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --report-dir "$REPORT_DIR"
  --report-json "$REPORT_JSON"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --d-model "$D_MODEL"
  --max-entities "$MAX_ENTITIES"
  --max-actions "$MAX_ACTIONS"
  --max-previous-actions "$MAX_PREVIOUS_ACTIONS"
  --decision-limit "$DECISION_LIMIT"
  --selfplay-limit "$SELFPLAY_LIMIT"
  "${DECK_ARGS[@]}"
)

if [[ "${PAIRWISE_CHANGED:-1}" == "1" || "${PAIRWISE_CHANGED:-1}" == "true" || "${PAIRWISE_CHANGED:-1}" == "TRUE" ]]; then
  ARGS+=(--pairwise-changed)
fi

"$PY" -m ptcg_abc "${ARGS[@]}"
