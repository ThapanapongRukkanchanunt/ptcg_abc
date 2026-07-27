#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-search-distill
#SBATCH --output=experiments/rl/slurm-%j-phase5-search-distill.out
#SBATCH --error=experiments/rl/slurm-%j-phase5-search-distill.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

set -euo pipefail

module purge
module load miniconda3/3.12

cd "$SLURM_SUBMIT_DIR"
mkdir -p experiments/rl reports models/rl

export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
PY="$SLURM_SUBMIT_DIR/.conda_ptcg/bin/python"

GAME_DATA_ROOT="${GAME_DATA_ROOT:-/project/SIGGI/thapanapong.r@cmu.ac.th}"
PUBLIC_AGENT_ROOTS="${PUBLIC_AGENT_ROOTS:-${PUBLIC_AGENT_ROOT:-data/public_agents}}"
RUN_NAME="${RUN_NAME:-phase5_dragapult_search_distill}"
COLLECT_AGENT="${COLLECT_AGENT:-phase5-search}"
CONTROLLED_PUBLIC_AGENT_KEY="${CONTROLLED_PUBLIC_AGENT_KEY:-sample_dragapult}"
OPPONENT_PUBLIC_AGENT_KEY="${OPPONENT_PUBLIC_AGENT_KEY:-sample_lucario}"
CONTROLLED_DECK_INDEX="${CONTROLLED_DECK_INDEX:-101}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-models/rl/phase5_one_deck_public_ppo_dominant/phase5_dragapult_vs_lucario_gae_game_shuffled/gen-0001/specialists}"
TRAIN_GAMES="${TRAIN_GAMES:-1000}"
EVAL_GAMES="${EVAL_GAMES:-1000}"
MAX_STEPS="${MAX_STEPS:-600}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.00005}"
SELFPLAY_LIMIT="${SELFPLAY_LIMIT:-0}"
CHANGED_WEIGHT="${CHANGED_WEIGHT:-1.0}"
UNCHANGED_WEIGHT="${UNCHANGED_WEIGHT:-1.0}"
PAIRWISE_CHANGED="${PAIRWISE_CHANGED:-0}"
PAIRWISE_WEIGHT="${PAIRWISE_WEIGHT:-0.25}"
COLLECT_DATASET="${COLLECT_DATASET:-1}"
CLEAN_RAW_AFTER_TRAIN="${CLEAN_RAW_AFTER_TRAIN:-1}"
EVAL_SEED="${EVAL_SEED:-20260727}"

MODEL_DIR="${MODEL_DIR:-models/rl/phase5_one_deck_search_distill/${RUN_NAME}/specialists}"
REPORT_ROOT="${REPORT_ROOT:-experiments/rl/phase5_one_deck_search_distill/${RUN_NAME}}"
RAW_ROOT="${RAW_ROOT:-${GAME_DATA_ROOT}/phase5_one_deck_search_distill/${RUN_NAME}/raw_train}"
TRAIN_DATASET="${TRAIN_DATASET:-${RAW_ROOT}/${COLLECT_AGENT}_trajectories_${TRAIN_GAMES}g.jsonl}"
COLLECT_REPORT="${COLLECT_REPORT:-${REPORT_ROOT}/collection_report.json}"
TRAIN_REPORT="${TRAIN_REPORT:-${REPORT_ROOT}/train_report.json}"
EVAL_REPORT_JSON="${EVAL_REPORT_JSON:-reports/${RUN_NAME}_direct_eval_${EVAL_GAMES}g.json}"
EVAL_REPORT_MD="${EVAL_REPORT_MD:-reports/${RUN_NAME}_direct_eval_${EVAL_GAMES}g.md}"
STATUS_JSON="${STATUS_JSON:-reports/${RUN_NAME}_direct_eval_status.json}"
REPLAY_OUTPUT_DIR="${REPLAY_OUTPUT_DIR:-${REPORT_ROOT}/replays}"
SOURCE_CHECKPOINT="${BASE_MODEL_DIR}/$(printf "deck-%02d.pt" "$CONTROLLED_DECK_INDEX")"
OUTPUT_CHECKPOINT="${MODEL_DIR}/$(printf "deck-%02d.pt" "$CONTROLLED_DECK_INDEX")"

case "$COLLECT_AGENT" in
  phase5-search|phase5-symbolic)
    ;;
  *)
    echo "COLLECT_AGENT must be phase5-search or phase5-symbolic, got: $COLLECT_AGENT" >&2
    exit 2
    ;;
esac

if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
  echo "Missing source checkpoint: $SOURCE_CHECKPOINT" >&2
  exit 2
fi

mkdir -p "$MODEL_DIR" "$REPORT_ROOT" "$RAW_ROOT" "$REPLAY_OUTPUT_DIR"

ROOT_ARGS=()
for root in $PUBLIC_AGENT_ROOTS; do
  ROOT_ARGS+=(--public-agent-root "$root")
done

echo "Phase 5 one-deck search distillation arm"
echo "RUN_NAME=$RUN_NAME"
echo "COLLECT_AGENT=$COLLECT_AGENT"
echo "CONTROLLED_PUBLIC_AGENT_KEY=$CONTROLLED_PUBLIC_AGENT_KEY"
echo "OPPONENT_PUBLIC_AGENT_KEY=$OPPONENT_PUBLIC_AGENT_KEY"
echo "CONTROLLED_DECK_INDEX=$CONTROLLED_DECK_INDEX"
echo "BASE_MODEL_DIR=$BASE_MODEL_DIR"
echo "TRAIN_GAMES=$TRAIN_GAMES"
echo "EVAL_GAMES=$EVAL_GAMES"
echo "TRAIN_EPOCHS=$TRAIN_EPOCHS"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "LEARNING_RATE=$LEARNING_RATE"
echo "CHANGED_WEIGHT=$CHANGED_WEIGHT"
echo "UNCHANGED_WEIGHT=$UNCHANGED_WEIGHT"
echo "PAIRWISE_CHANGED=$PAIRWISE_CHANGED"
echo "PAIRWISE_WEIGHT=$PAIRWISE_WEIGHT"
echo "COLLECT_DATASET=$COLLECT_DATASET"
echo "CLEAN_RAW_AFTER_TRAIN=$CLEAN_RAW_AFTER_TRAIN"
echo "EVAL_SEED=$EVAL_SEED"
echo "TRAIN_DATASET=$TRAIN_DATASET"
echo "OUTPUT_CHECKPOINT=$OUTPUT_CHECKPOINT"

case "$COLLECT_DATASET" in
  0|1)
    ;;
  *)
    echo "COLLECT_DATASET must be 0 or 1, got: $COLLECT_DATASET" >&2
    exit 2
    ;;
esac

if [[ "$COLLECT_DATASET" == "1" ]]; then
  "$PY" -m ptcg_abc rl-generate-phase5-public-agent-trajectories \
    "${ROOT_ARGS[@]}" \
    --public-agent-key "$OPPONENT_PUBLIC_AGENT_KEY" \
    --controlled-public-agent-key "$CONTROLLED_PUBLIC_AGENT_KEY" \
    --controlled-deck-index "$CONTROLLED_DECK_INDEX" \
    --agent "$COLLECT_AGENT" \
    --specialist-model-dir "$BASE_MODEL_DIR" \
    --games-per-matchup "$TRAIN_GAMES" \
    --max-steps "$MAX_STEPS" \
    --require-min-opponents 1 \
    --outcome-reward-scale 0.0 \
    --outcome-reward-assignment terminal \
    --tactical-reward-mode none \
    --output "$TRAIN_DATASET" \
    --report-json "$COLLECT_REPORT" \
    --overwrite
else
  if [[ ! -s "$TRAIN_DATASET" ]]; then
    echo "Shared trajectory dataset is missing or empty: $TRAIN_DATASET" >&2
    exit 2
  fi
  if [[ ! -s "$COLLECT_REPORT" ]]; then
    echo "Shared collection report is missing or empty: $COLLECT_REPORT" >&2
    exit 2
  fi
  echo "Reusing shared trajectory dataset: $TRAIN_DATASET"
  echo "Reusing shared collection report: $COLLECT_REPORT"
fi

PAIRWISE_ARGS=()
if [[ "$PAIRWISE_CHANGED" != "0" ]]; then
  PAIRWISE_ARGS+=(--pairwise-changed --pairwise-weight "$PAIRWISE_WEIGHT")
fi

"$PY" -m ptcg_abc rl-train-phase5-generalist \
  --no-decision-dataset \
  --selfplay-dataset "$TRAIN_DATASET" \
  --initial-checkpoint "$SOURCE_CHECKPOINT" \
  --checkpoint "$OUTPUT_CHECKPOINT" \
  --report-json "$TRAIN_REPORT" \
  --epochs "$TRAIN_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --changed-weight "$CHANGED_WEIGHT" \
  --unchanged-weight "$UNCHANGED_WEIGHT" \
  --selfplay-limit "$SELFPLAY_LIMIT" \
  --selfplay-weight 1.0 \
  --value-loss-weight 0.0 \
  --action-value-loss-weight 0.0 \
  --tactical-loss-weight 0.0 \
  --deck-index-filter "$CONTROLLED_DECK_INDEX" \
  "${PAIRWISE_ARGS[@]}"

if [[ "$CLEAN_RAW_AFTER_TRAIN" != "0" ]]; then
  rm -f -- "$TRAIN_DATASET"
  if [[ -e "$TRAIN_DATASET" ]]; then
    echo "Failed to remove consumed trajectory dataset: $TRAIN_DATASET" >&2
    exit 2
  fi
  echo "Deleted consumed trajectory dataset: $TRAIN_DATASET"
else
  echo "Retained shared trajectory dataset for dependent arm: $TRAIN_DATASET"
fi

"$PY" -m ptcg_abc rl-evaluate-phase5-public-agents \
  "${ROOT_ARGS[@]}" \
  --public-agent-key "$OPPONENT_PUBLIC_AGENT_KEY" \
  --controlled-public-agent-key "$CONTROLLED_PUBLIC_AGENT_KEY" \
  --controlled-deck-index "$CONTROLLED_DECK_INDEX" \
  --agent phase5-symbolic \
  --specialist-model-dir "$MODEL_DIR" \
  --games-per-matchup "$EVAL_GAMES" \
  --max-steps "$MAX_STEPS" \
  --game-seed "$EVAL_SEED" \
  --require-min-opponents 1 \
  --report-json "$EVAL_REPORT_JSON" \
  --report-md "$EVAL_REPORT_MD" \
  --status-json "$STATUS_JSON" \
  --replay-output-dir "$REPLAY_OUTPUT_DIR" \
  --saved-win-replays 1 \
  --saved-loss-replays 1

echo "Completed Phase 5 search distillation arm: $RUN_NAME"
