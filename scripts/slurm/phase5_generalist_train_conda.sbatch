#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-generalist
#SBATCH --output=experiments/rl/slurm-%j-phase5-generalist-train.out
#SBATCH --error=experiments/rl/slurm-%j-phase5-generalist-train.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

set -euo pipefail

module purge
module load miniconda3/3.12

cd "$SLURM_SUBMIT_DIR"
GAME_DATA_ROOT="${GAME_DATA_ROOT:-/project/SIGGI/thapanapong.r@cmu.ac.th}"
mkdir -p experiments/rl "$GAME_DATA_ROOT" models/rl reports

export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
PY="$SLURM_SUBMIT_DIR/.conda_ptcg/bin/python"

"$PY" --version
"$PY" -c "import sys; print(sys.executable)"
"$PY" - <<'PY'
import torch
from ptcg_abc.rl.phase5_policy import PHASE5_POLICY_CHECKPOINT_FORMAT

print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("checkpoint_format", PHASE5_POLICY_CHECKPOINT_FORMAT)
PY

DECISION_DATASET="${DECISION_DATASET:-${GAME_DATA_ROOT}/phase5_search_decisions_10shards.jsonl}"
SELFPLAY_DATASETS="${SELFPLAY_DATASETS:-${GAME_DATA_ROOT}/phase5_search_selfplay_10k/shards/phase5_search_selfplay_shard-0.jsonl ${GAME_DATA_ROOT}/phase5_search_selfplay_10k/shards/phase5_search_selfplay_shard-1.jsonl}"
CHECKPOINT="${CHECKPOINT:-models/rl/phase5_generalist_policy_10k.pt}"
REPORT_JSON="${REPORT_JSON:-experiments/rl/phase5_generalist_train_report_10k.json}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
D_MODEL="${D_MODEL:-128}"
MAX_ENTITIES="${MAX_ENTITIES:-96}"
MAX_ACTIONS="${MAX_ACTIONS:-128}"
MAX_PREVIOUS_ACTIONS="${MAX_PREVIOUS_ACTIONS:-16}"
TARGET_SOURCE="${TARGET_SOURCE:-search}"
RULE_DEMO_TARGET_SOURCE="${RULE_DEMO_TARGET_SOURCE:-baseline}"
CHANGED_WEIGHT="${CHANGED_WEIGHT:-2.0}"
UNCHANGED_WEIGHT="${UNCHANGED_WEIGHT:-0.5}"
SEARCH_DECISION_WEIGHT="${SEARCH_DECISION_WEIGHT:-1.0}"
RULE_DEMO_WEIGHT="${RULE_DEMO_WEIGHT:-0.25}"
SELFPLAY_WEIGHT="${SELFPLAY_WEIGHT:-1.0}"
PAIRWISE_CHANGED="${PAIRWISE_CHANGED:-1}"
PAIRWISE_WEIGHT="${PAIRWISE_WEIGHT:-0.25}"
PAIRWISE_MARGIN="${PAIRWISE_MARGIN:-1.0}"
PAIRWISE_NEGATIVES="${PAIRWISE_NEGATIVES:-baseline}"
VALUE_LOSS_WEIGHT="${VALUE_LOSS_WEIGHT:-0.25}"
ACTION_VALUE_LOSS_WEIGHT="${ACTION_VALUE_LOSS_WEIGHT:-0.25}"
TACTICAL_LOSS_WEIGHT="${TACTICAL_LOSS_WEIGHT:-0.05}"
DECISION_LIMIT="${DECISION_LIMIT:-0}"
SELFPLAY_LIMIT="${SELFPLAY_LIMIT:-0}"
NO_DECISION_DATASET="${NO_DECISION_DATASET:-0}"

SELFPLAY_ARGS=()
for path in $SELFPLAY_DATASETS; do
  SELFPLAY_ARGS+=(--selfplay-dataset "$path")
done

ARGS=(
  rl-train-phase5-generalist
  --checkpoint "$CHECKPOINT"
  --report-json "$REPORT_JSON"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --d-model "$D_MODEL"
  --max-entities "$MAX_ENTITIES"
  --max-actions "$MAX_ACTIONS"
  --max-previous-actions "$MAX_PREVIOUS_ACTIONS"
  --target-source "$TARGET_SOURCE"
  --rule-demo-target-source "$RULE_DEMO_TARGET_SOURCE"
  --changed-weight "$CHANGED_WEIGHT"
  --unchanged-weight "$UNCHANGED_WEIGHT"
  --search-decision-weight "$SEARCH_DECISION_WEIGHT"
  --rule-demo-weight "$RULE_DEMO_WEIGHT"
  --selfplay-weight "$SELFPLAY_WEIGHT"
  --pairwise-weight "$PAIRWISE_WEIGHT"
  --pairwise-margin "$PAIRWISE_MARGIN"
  --pairwise-negatives "$PAIRWISE_NEGATIVES"
  --value-loss-weight "$VALUE_LOSS_WEIGHT"
  --action-value-loss-weight "$ACTION_VALUE_LOSS_WEIGHT"
  --tactical-loss-weight "$TACTICAL_LOSS_WEIGHT"
  --decision-limit "$DECISION_LIMIT"
  --selfplay-limit "$SELFPLAY_LIMIT"
  "${SELFPLAY_ARGS[@]}"
)

if [[ "$NO_DECISION_DATASET" == "1" || "$NO_DECISION_DATASET" == "true" || "$NO_DECISION_DATASET" == "TRUE" ]]; then
  ARGS+=(--no-decision-dataset)
else
  ARGS+=(--decision-dataset "$DECISION_DATASET")
fi

if [[ "$PAIRWISE_CHANGED" == "1" || "$PAIRWISE_CHANGED" == "true" || "$PAIRWISE_CHANGED" == "TRUE" ]]; then
  ARGS+=(--pairwise-changed)
fi

"$PY" -m ptcg_abc "${ARGS[@]}"
