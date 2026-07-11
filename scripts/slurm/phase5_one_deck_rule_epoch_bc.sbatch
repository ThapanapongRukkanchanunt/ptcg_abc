#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-rule-epoch-bc
#SBATCH --output=experiments/rl/slurm-%j-phase5-rule-epoch-bc.out
#SBATCH --error=experiments/rl/slurm-%j-phase5-rule-epoch-bc.err
#SBATCH --time=120:00:00
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
RUN_NAME="${RUN_NAME:-phase5_dragapult_lucario_rule_epoch_bc}"
DRAGAPULT_PUBLIC_AGENT_KEY="${DRAGAPULT_PUBLIC_AGENT_KEY:-sample_dragapult}"
LUCARIO_PUBLIC_AGENT_KEY="${LUCARIO_PUBLIC_AGENT_KEY:-sample_lucario}"
DRAGAPULT_DECK_INDEX="${DRAGAPULT_DECK_INDEX:-101}"
LUCARIO_DECK_INDEX="${LUCARIO_DECK_INDEX:-102}"
RULE_GAMES_PER_MATCHUP="${RULE_GAMES_PER_MATCHUP:-1000}"
EVAL_GAMES_PER_EPOCH="${EVAL_GAMES_PER_EPOCH:-100}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
MIN_EPOCHS="${MIN_EPOCHS:-10}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
MAX_STEPS="${MAX_STEPS:-600}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
D_MODEL="${D_MODEL:-128}"
MAX_ENTITIES="${MAX_ENTITIES:-96}"
MAX_ACTIONS="${MAX_ACTIONS:-128}"
MAX_PREVIOUS_ACTIONS="${MAX_PREVIOUS_ACTIONS:-16}"
SELFPLAY_LIMIT="${SELFPLAY_LIMIT:-0}"
SELFPLAY_WEIGHT="${SELFPLAY_WEIGHT:-1.0}"
VALUE_LOSS_WEIGHT="${VALUE_LOSS_WEIGHT:-0.25}"
ACTION_VALUE_LOSS_WEIGHT="${ACTION_VALUE_LOSS_WEIGHT:-0.25}"
TACTICAL_LOSS_WEIGHT="${TACTICAL_LOSS_WEIGHT:-0.05}"
OVERWRITE_SCRATCH="${OVERWRITE_SCRATCH:-0}"
REGENERATE_RULE_DATA="${REGENERATE_RULE_DATA:-0}"

MODEL_ROOT="${MODEL_ROOT:-models/rl/phase5_one_deck_rule_epoch_bc/${RUN_NAME}}"
REPORT_ROOT="${REPORT_ROOT:-experiments/rl/phase5_one_deck_rule_epoch_bc/${RUN_NAME}}"
RAW_ROOT="${RAW_ROOT:-${GAME_DATA_ROOT}/phase5_one_deck_rule_epoch_bc/${RUN_NAME}}"
DEFAULT_DRAGAPULT_RULE_DATASET="${GAME_DATA_ROOT}/phase5_one_deck_public_mixed/phase5_dragapult_vs_lucario_mixed/rule_bootstrap/phase5_public_rule_bootstrap_gen-0000.jsonl"
DRAGAPULT_RULE_DATASET="${DRAGAPULT_RULE_DATASET:-$DEFAULT_DRAGAPULT_RULE_DATASET}"
LUCARIO_RULE_DATASET="${LUCARIO_RULE_DATASET:-${RAW_ROOT}/rule_bootstrap/phase5_lucario_vs_dragapult_rule_bootstrap.jsonl}"

echo "Phase 5 one-deck rule epoch BC diagnostic"
echo "GAME_DATA_ROOT=$GAME_DATA_ROOT"
echo "PUBLIC_AGENT_ROOTS=$PUBLIC_AGENT_ROOTS"
echo "RUN_NAME=$RUN_NAME"
echo "DRAGAPULT_PUBLIC_AGENT_KEY=$DRAGAPULT_PUBLIC_AGENT_KEY"
echo "LUCARIO_PUBLIC_AGENT_KEY=$LUCARIO_PUBLIC_AGENT_KEY"
echo "DRAGAPULT_DECK_INDEX=$DRAGAPULT_DECK_INDEX"
echo "LUCARIO_DECK_INDEX=$LUCARIO_DECK_INDEX"
echo "RULE_GAMES_PER_MATCHUP=$RULE_GAMES_PER_MATCHUP"
echo "EVAL_GAMES_PER_EPOCH=$EVAL_GAMES_PER_EPOCH"
echo "MAX_EPOCHS=$MAX_EPOCHS"
echo "MIN_EPOCHS=$MIN_EPOCHS"
echo "EARLY_STOP_PATIENCE=$EARLY_STOP_PATIENCE"
echo "MAX_STEPS=$MAX_STEPS"
echo "MODEL_ROOT=$MODEL_ROOT"
echo "REPORT_ROOT=$REPORT_ROOT"
echo "RAW_ROOT=$RAW_ROOT"
echo "DRAGAPULT_RULE_DATASET=$DRAGAPULT_RULE_DATASET"
echo "LUCARIO_RULE_DATASET=$LUCARIO_RULE_DATASET"
echo "REGENERATE_RULE_DATA=$REGENERATE_RULE_DATA"

ROOT_ARGS=()
for root in $PUBLIC_AGENT_ROOTS; do
  ROOT_ARGS+=(--public-agent-root "$root")
done

epoch_model_dir() {
  local epoch="$1"
  local padded
  padded="$(printf "%04d" "$epoch")"
  echo "${MODEL_ROOT}/epoch-${padded}/specialists"
}

epoch_report_dir() {
  local epoch="$1"
  local padded
  padded="$(printf "%04d" "$epoch")"
  echo "${REPORT_ROOT}/epoch-${padded}"
}

checkpoint_name() {
  local deck_index="$1"
  printf "deck-%02d.pt" "$deck_index"
}

ensure_rule_dataset() {
  local controlled_key="$1"
  local controlled_index="$2"
  local opponent_key="$3"
  local output_path="$4"
  local report_path="$5"
  mkdir -p "$(dirname "$output_path")" "$(dirname "$report_path")"
  if [[ "$REGENERATE_RULE_DATA" == "0" && -s "$output_path" ]]; then
    echo "Keeping existing rule dataset: $output_path"
    return
  fi
  "$PY" -m ptcg_abc rl-generate-phase5-public-agent-trajectories \
    "${ROOT_ARGS[@]}" \
    --public-agent-key "$opponent_key" \
    --controlled-public-agent-key "$controlled_key" \
    --controlled-deck-index "$controlled_index" \
    --agent rule \
    --games-per-matchup "$RULE_GAMES_PER_MATCHUP" \
    --max-steps "$MAX_STEPS" \
    --require-min-opponents 1 \
    --output "$output_path" \
    --report-json "$report_path" \
    --overwrite
}

train_one_epoch() {
  local deck_label="$1"
  local deck_index="$2"
  local dataset_path="$3"
  local previous_dir="$4"
  local current_dir="$5"
  local report_json="$6"
  "$PY" -m ptcg_abc rl-train-phase5-generalist \
    --no-decision-dataset \
    --selfplay-dataset "$dataset_path" \
    --initial-checkpoint "${previous_dir}/$(checkpoint_name "$deck_index")" \
    --checkpoint "${current_dir}/$(checkpoint_name "$deck_index")" \
    --report-json "$report_json" \
    --epochs 1 \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --d-model "$D_MODEL" \
    --max-entities "$MAX_ENTITIES" \
    --max-actions "$MAX_ACTIONS" \
    --max-previous-actions "$MAX_PREVIOUS_ACTIONS" \
    --selfplay-limit "$SELFPLAY_LIMIT" \
    --selfplay-weight "$SELFPLAY_WEIGHT" \
    --value-loss-weight "$VALUE_LOSS_WEIGHT" \
    --action-value-loss-weight "$ACTION_VALUE_LOSS_WEIGHT" \
    --tactical-loss-weight "$TACTICAL_LOSS_WEIGHT" \
    --deck-index-filter "$deck_index"
  echo "Trained $deck_label deck index $deck_index for one epoch."
}

eval_matchup() {
  local controlled_key="$1"
  local controlled_index="$2"
  local opponent_key="$3"
  local model_dir="$4"
  local report_json="$5"
  local report_md="$6"
  local status_json="$7"
  "$PY" -m ptcg_abc rl-evaluate-phase5-public-agents \
    "${ROOT_ARGS[@]}" \
    --public-agent-key "$opponent_key" \
    --controlled-public-agent-key "$controlled_key" \
    --controlled-deck-index "$controlled_index" \
    --agent phase5-symbolic \
    --specialist-model-dir "$model_dir" \
    --games-per-matchup "$EVAL_GAMES_PER_EPOCH" \
    --max-steps "$MAX_STEPS" \
    --require-min-opponents 1 \
    --report-json "$report_json" \
    --report-md "$report_md" \
    --status-json "$status_json"
}

GEN0_DIR="$(epoch_model_dir 0)"
GEN0_REPORT_DIR="$(epoch_report_dir 0)"
mkdir -p "$GEN0_DIR" "$GEN0_REPORT_DIR"

INIT_ARGS=()
if [[ "$OVERWRITE_SCRATCH" != "0" ]]; then
  INIT_ARGS+=(--overwrite)
fi

DRAGAPULT_SCRATCH="${GEN0_DIR}/$(checkpoint_name "$DRAGAPULT_DECK_INDEX")"
LUCARIO_SCRATCH="${GEN0_DIR}/$(checkpoint_name "$LUCARIO_DECK_INDEX")"
if [[ ! -f "$DRAGAPULT_SCRATCH" || "$OVERWRITE_SCRATCH" != "0" ]]; then
  "$PY" -m ptcg_abc rl-init-phase5-policy-checkpoint \
    --checkpoint "$DRAGAPULT_SCRATCH" \
    --report-json "${GEN0_REPORT_DIR}/dragapult_scratch_init_report.json" \
    --deck-index "$DRAGAPULT_DECK_INDEX" \
    --controlled-public-agent-key "$DRAGAPULT_PUBLIC_AGENT_KEY" \
    --experiment "$RUN_NAME" \
    "${INIT_ARGS[@]}"
fi
if [[ ! -f "$LUCARIO_SCRATCH" || "$OVERWRITE_SCRATCH" != "0" ]]; then
  "$PY" -m ptcg_abc rl-init-phase5-policy-checkpoint \
    --checkpoint "$LUCARIO_SCRATCH" \
    --report-json "${GEN0_REPORT_DIR}/lucario_scratch_init_report.json" \
    --deck-index "$LUCARIO_DECK_INDEX" \
    --controlled-public-agent-key "$LUCARIO_PUBLIC_AGENT_KEY" \
    --experiment "$RUN_NAME" \
    "${INIT_ARGS[@]}"
fi

ensure_rule_dataset \
  "$DRAGAPULT_PUBLIC_AGENT_KEY" \
  "$DRAGAPULT_DECK_INDEX" \
  "$LUCARIO_PUBLIC_AGENT_KEY" \
  "$DRAGAPULT_RULE_DATASET" \
  "${GEN0_REPORT_DIR}/dragapult_rule_bootstrap_report.json"
ensure_rule_dataset \
  "$LUCARIO_PUBLIC_AGENT_KEY" \
  "$LUCARIO_DECK_INDEX" \
  "$DRAGAPULT_PUBLIC_AGENT_KEY" \
  "$LUCARIO_RULE_DATASET" \
  "${GEN0_REPORT_DIR}/lucario_rule_bootstrap_report.json"

BEST_COMBINED_WINS=-1
BEST_EPOCH=0
NO_IMPROVE_EPOCHS=0

for epoch in $(seq 1 "$MAX_EPOCHS"); do
  padded="$(printf "%04d" "$epoch")"
  previous_epoch="$((epoch - 1))"
  PREVIOUS_DIR="$(epoch_model_dir "$previous_epoch")"
  CURRENT_DIR="$(epoch_model_dir "$epoch")"
  EPOCH_REPORT_DIR="$(epoch_report_dir "$epoch")"
  mkdir -p "$CURRENT_DIR" "$EPOCH_REPORT_DIR"

  echo "Epoch $epoch / $MAX_EPOCHS"
  echo "PREVIOUS_DIR=$PREVIOUS_DIR"
  echo "CURRENT_DIR=$CURRENT_DIR"

  train_one_epoch \
    "Dragapult" \
    "$DRAGAPULT_DECK_INDEX" \
    "$DRAGAPULT_RULE_DATASET" \
    "$PREVIOUS_DIR" \
    "$CURRENT_DIR" \
    "${EPOCH_REPORT_DIR}/dragapult_train_report.json"

  train_one_epoch \
    "Lucario" \
    "$LUCARIO_DECK_INDEX" \
    "$LUCARIO_RULE_DATASET" \
    "$PREVIOUS_DIR" \
    "$CURRENT_DIR" \
    "${EPOCH_REPORT_DIR}/lucario_train_report.json"

  DRAGAPULT_EVAL_JSON="reports/${RUN_NAME}_epoch-${padded}_dragapult_vs_lucario_${EVAL_GAMES_PER_EPOCH}g.json"
  DRAGAPULT_EVAL_MD="reports/${RUN_NAME}_epoch-${padded}_dragapult_vs_lucario_${EVAL_GAMES_PER_EPOCH}g.md"
  DRAGAPULT_STATUS_JSON="reports/${RUN_NAME}_epoch-${padded}_dragapult_vs_lucario_status.json"
  LUCARIO_EVAL_JSON="reports/${RUN_NAME}_epoch-${padded}_lucario_vs_dragapult_${EVAL_GAMES_PER_EPOCH}g.json"
  LUCARIO_EVAL_MD="reports/${RUN_NAME}_epoch-${padded}_lucario_vs_dragapult_${EVAL_GAMES_PER_EPOCH}g.md"
  LUCARIO_STATUS_JSON="reports/${RUN_NAME}_epoch-${padded}_lucario_vs_dragapult_status.json"

  eval_matchup \
    "$DRAGAPULT_PUBLIC_AGENT_KEY" \
    "$DRAGAPULT_DECK_INDEX" \
    "$LUCARIO_PUBLIC_AGENT_KEY" \
    "$CURRENT_DIR" \
    "$DRAGAPULT_EVAL_JSON" \
    "$DRAGAPULT_EVAL_MD" \
    "$DRAGAPULT_STATUS_JSON"

  eval_matchup \
    "$LUCARIO_PUBLIC_AGENT_KEY" \
    "$LUCARIO_DECK_INDEX" \
    "$DRAGAPULT_PUBLIC_AGENT_KEY" \
    "$CURRENT_DIR" \
    "$LUCARIO_EVAL_JSON" \
    "$LUCARIO_EVAL_MD" \
    "$LUCARIO_STATUS_JSON"

  SUMMARY_JSON="${EPOCH_REPORT_DIR}/epoch_summary.json"
  "$PY" - "$SUMMARY_JSON" "$epoch" "$DRAGAPULT_EVAL_JSON" "$LUCARIO_EVAL_JSON" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
epoch = int(sys.argv[2])
reports = [Path(path) for path in sys.argv[3:]]
matchups = []
totals = {"games": 0, "wins": 0, "losses": 0, "draws": 0, "timeouts": 0, "errors": 0}
for path in reports:
    data = json.loads(path.read_text())
    rows = data.get("rows", [])
    wins = sum(int(row.get("wins", 0)) for row in rows)
    games = sum(int(row.get("games", 0)) for row in rows)
    losses = sum(int(row.get("losses", 0)) for row in rows)
    draws = sum(int(row.get("draws", 0)) for row in rows)
    timeouts = sum(int(row.get("timeouts", 0)) for row in rows)
    errors = sum(int(row.get("errors", 0)) for row in rows)
    payload = {
        "report": path.as_posix(),
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "timeouts": timeouts,
        "errors": errors,
        "win_rate": wins / games if games else 0.0,
    }
    matchups.append(payload)
    for key in totals:
        totals[key] += int(payload[key])
totals["win_rate"] = totals["wins"] / totals["games"] if totals["games"] else 0.0
out.write_text(
    json.dumps({"epoch": epoch, "matchups": matchups, "combined": totals}, indent=2, sort_keys=True),
    encoding="utf-8",
)
print(totals["wins"])
PY
  COMBINED_WINS="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['combined']['wins'])" "$SUMMARY_JSON")"
  echo "Epoch $epoch combined eval wins: $COMBINED_WINS"

  if (( COMBINED_WINS > BEST_COMBINED_WINS )); then
    BEST_COMBINED_WINS="$COMBINED_WINS"
    BEST_EPOCH="$epoch"
    NO_IMPROVE_EPOCHS=0
  else
    NO_IMPROVE_EPOCHS="$((NO_IMPROVE_EPOCHS + 1))"
  fi
  echo "Best epoch so far: $BEST_EPOCH with combined wins $BEST_COMBINED_WINS"
  echo "No-improve epochs: $NO_IMPROVE_EPOCHS"

  if (( EARLY_STOP_PATIENCE > 0 && epoch >= MIN_EPOCHS && NO_IMPROVE_EPOCHS >= EARLY_STOP_PATIENCE )); then
    echo "Early stopping after epoch $epoch: no combined eval improvement for $NO_IMPROVE_EPOCHS epochs."
    break
  fi
done
