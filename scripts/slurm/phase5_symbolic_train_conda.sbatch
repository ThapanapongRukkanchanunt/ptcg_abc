#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-symbolic
#SBATCH --output=experiments/rl/slurm-%j-phase5-symbolic-train.out
#SBATCH --error=experiments/rl/slurm-%j-phase5-symbolic-train.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

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

DATASET="${DATASET:-${GAME_DATA_ROOT}/phase5_search_decisions_merged.jsonl}"
CHECKPOINT="${CHECKPOINT:-models/rl/phase5_symbolic_policy_10shards.pt}"
REPORT_JSON="${REPORT_JSON:-experiments/rl/phase5_symbolic_train_report_10shards.json}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
D_MODEL="${D_MODEL:-128}"
MAX_ENTITIES="${MAX_ENTITIES:-96}"
MAX_ACTIONS="${MAX_ACTIONS:-128}"
MAX_PREVIOUS_ACTIONS="${MAX_PREVIOUS_ACTIONS:-16}"
TARGET_SOURCE="${TARGET_SOURCE:-search}"
CHANGED_WEIGHT="${CHANGED_WEIGHT:-4.0}"
UNCHANGED_WEIGHT="${UNCHANGED_WEIGHT:-0.5}"
PAIRWISE_CHANGED="${PAIRWISE_CHANGED:-0}"
PAIRWISE_WEIGHT="${PAIRWISE_WEIGHT:-1.0}"
PAIRWISE_MARGIN="${PAIRWISE_MARGIN:-1.0}"
PAIRWISE_NEGATIVES="${PAIRWISE_NEGATIVES:-all}"
VALUE_LOSS_WEIGHT="${VALUE_LOSS_WEIGHT:-0.0}"
LIMIT="${LIMIT:-0}"
CHANGED_ONLY="${CHANGED_ONLY:-0}"

ARGS=(
  rl-train-phase5-symbolic
  --dataset "$DATASET"
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
  --changed-weight "$CHANGED_WEIGHT"
  --unchanged-weight "$UNCHANGED_WEIGHT"
  --pairwise-weight "$PAIRWISE_WEIGHT"
  --pairwise-margin "$PAIRWISE_MARGIN"
  --pairwise-negatives "$PAIRWISE_NEGATIVES"
  --value-loss-weight "$VALUE_LOSS_WEIGHT"
  --limit "$LIMIT"
)

if [[ "$PAIRWISE_CHANGED" == "1" || "$PAIRWISE_CHANGED" == "true" || "$PAIRWISE_CHANGED" == "TRUE" ]]; then
  ARGS+=(--pairwise-changed)
fi

if [[ "$CHANGED_ONLY" == "1" || "$CHANGED_ONLY" == "true" || "$CHANGED_ONLY" == "TRUE" ]]; then
  ARGS+=(--changed-only)
fi

"$PY" -m ptcg_abc "${ARGS[@]}"
