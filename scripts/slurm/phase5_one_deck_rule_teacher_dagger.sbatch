#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-rule-dagger
#SBATCH --output=experiments/rl/slurm-%j-phase5-rule-dagger.out
#SBATCH --error=experiments/rl/slurm-%j-phase5-rule-dagger.err
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
RUN_NAME="${RUN_NAME:-phase5_dragapult_lucario_rule_teacher_dagger}"
DRAGAPULT_PUBLIC_AGENT_KEY="${DRAGAPULT_PUBLIC_AGENT_KEY:-sample_dragapult}"
LUCARIO_PUBLIC_AGENT_KEY="${LUCARIO_PUBLIC_AGENT_KEY:-sample_lucario}"
DRAGAPULT_DECK_INDEX="${DRAGAPULT_DECK_INDEX:-101}"
LUCARIO_DECK_INDEX="${LUCARIO_DECK_INDEX:-102}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-models/rl/phase5_one_deck_rule_epoch_bc/phase5_dragapult_lucario_rule_epoch_bc/epoch-0008/specialists}"
DAGGER_ITERATIONS="${DAGGER_ITERATIONS:-10}"
TEACHER_GAMES_PER_ITERATION="${TEACHER_GAMES_PER_ITERATION:-500}"
RULE_GAMES_PER_MATCHUP="${RULE_GAMES_PER_MATCHUP:-1000}"
EVAL_GAMES_PER_ITERATION="${EVAL_GAMES_PER_ITERATION:-100}"
TRAIN_EPOCHS_PER_ITERATION="${TRAIN_EPOCHS_PER_ITERATION:-1}"
MIN_ITERATIONS="${MIN_ITERATIONS:-3}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
MAX_STEPS="${MAX_STEPS:-600}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.00005}"
D_MODEL="${D_MODEL:-128}"
MAX_ENTITIES="${MAX_ENTITIES:-96}"
MAX_ACTIONS="${MAX_ACTIONS:-128}"
MAX_PREVIOUS_ACTIONS="${MAX_PREVIOUS_ACTIONS:-16}"
SELFPLAY_LIMIT="${SELFPLAY_LIMIT:-0}"
SELFPLAY_WEIGHT="${SELFPLAY_WEIGHT:-1.0}"
VALUE_LOSS_WEIGHT="${VALUE_LOSS_WEIGHT:-0.0}"
ACTION_VALUE_LOSS_WEIGHT="${ACTION_VALUE_LOSS_WEIGHT:-0.0}"
TACTICAL_LOSS_WEIGHT="${TACTICAL_LOSS_WEIGHT:-0.0}"
AGENT="${AGENT:-phase5-symbolic}"
TEACHER_AGENT="${TEACHER_AGENT:-rule}"
OUTCOME_REWARD_SCALE="${OUTCOME_REWARD_SCALE:-0.0}"
TACTICAL_REWARD_MODE="${TACTICAL_REWARD_MODE:-none}"
TACTICAL_FRACTIONAL_PRIZE_WEIGHT="${TACTICAL_FRACTIONAL_PRIZE_WEIGHT:-1.0}"
TACTICAL_FRACTIONAL_OPPONENT_WEIGHT="${TACTICAL_FRACTIONAL_OPPONENT_WEIGHT:-1.0}"
EVAL_BASE="${EVAL_BASE:-1}"
REGENERATE_RULE_DATA="${REGENERATE_RULE_DATA:-0}"
REUSE_EXISTING_TRAJECTORIES="${REUSE_EXISTING_TRAJECTORIES:-0}"

MODEL_ROOT="${MODEL_ROOT:-models/rl/phase5_one_deck_rule_teacher_dagger/${RUN_NAME}}"
REPORT_ROOT="${REPORT_ROOT:-experiments/rl/phase5_one_deck_rule_teacher_dagger/${RUN_NAME}}"
RAW_ROOT="${RAW_ROOT:-${GAME_DATA_ROOT}/phase5_one_deck_rule_teacher_dagger/${RUN_NAME}}"
DEFAULT_DRAGAPULT_RULE_DATASET="${GAME_DATA_ROOT}/phase5_one_deck_public_mixed/phase5_dragapult_vs_lucario_mixed/rule_bootstrap/phase5_public_rule_bootstrap_gen-0000.jsonl"
DEFAULT_LUCARIO_RULE_DATASET="${GAME_DATA_ROOT}/phase5_one_deck_rule_epoch_bc/phase5_dragapult_lucario_rule_epoch_bc/rule_bootstrap/phase5_lucario_vs_dragapult_rule_bootstrap.jsonl"
DRAGAPULT_RULE_DATASET="${DRAGAPULT_RULE_DATASET:-$DEFAULT_DRAGAPULT_RULE_DATASET}"
LUCARIO_RULE_DATASET="${LUCARIO_RULE_DATASET:-$DEFAULT_LUCARIO_RULE_DATASET}"

echo "Phase 5 one-deck rule-teacher DAgger diagnostic"
echo "GAME_DATA_ROOT=$GAME_DATA_ROOT"
echo "PUBLIC_AGENT_ROOTS=$PUBLIC_AGENT_ROOTS"
echo "RUN_NAME=$RUN_NAME"
echo "DRAGAPULT_PUBLIC_AGENT_KEY=$DRAGAPULT_PUBLIC_AGENT_KEY"
echo "LUCARIO_PUBLIC_AGENT_KEY=$LUCARIO_PUBLIC_AGENT_KEY"
echo "DRAGAPULT_DECK_INDEX=$DRAGAPULT_DECK_INDEX"
echo "LUCARIO_DECK_INDEX=$LUCARIO_DECK_INDEX"
echo "BASE_MODEL_DIR=$BASE_MODEL_DIR"
echo "DAGGER_ITERATIONS=$DAGGER_ITERATIONS"
echo "TEACHER_GAMES_PER_ITERATION=$TEACHER_GAMES_PER_ITERATION"
echo "RULE_GAMES_PER_MATCHUP=$RULE_GAMES_PER_MATCHUP"
echo "EVAL_GAMES_PER_ITERATION=$EVAL_GAMES_PER_ITERATION"
echo "TRAIN_EPOCHS_PER_ITERATION=$TRAIN_EPOCHS_PER_ITERATION"
echo "MIN_ITERATIONS=$MIN_ITERATIONS"
echo "EARLY_STOP_PATIENCE=$EARLY_STOP_PATIENCE"
echo "MAX_STEPS=$MAX_STEPS"
echo "AGENT=$AGENT"
echo "TEACHER_AGENT=$TEACHER_AGENT"
echo "OUTCOME_REWARD_SCALE=$OUTCOME_REWARD_SCALE"
echo "TACTICAL_REWARD_MODE=$TACTICAL_REWARD_MODE"
echo "TACTICAL_FRACTIONAL_PRIZE_WEIGHT=$TACTICAL_FRACTIONAL_PRIZE_WEIGHT"
echo "TACTICAL_FRACTIONAL_OPPONENT_WEIGHT=$TACTICAL_FRACTIONAL_OPPONENT_WEIGHT"
echo "MODEL_ROOT=$MODEL_ROOT"
echo "REPORT_ROOT=$REPORT_ROOT"
echo "RAW_ROOT=$RAW_ROOT"
echo "DRAGAPULT_RULE_DATASET=$DRAGAPULT_RULE_DATASET"
echo "LUCARIO_RULE_DATASET=$LUCARIO_RULE_DATASET"
echo "REGENERATE_RULE_DATA=$REGENERATE_RULE_DATA"
echo "REUSE_EXISTING_TRAJECTORIES=$REUSE_EXISTING_TRAJECTORIES"

ROOT_ARGS=()
for root in $PUBLIC_AGENT_ROOTS; do
  ROOT_ARGS+=(--public-agent-root "$root")
done

iteration_model_dir() {
  local iteration="$1"
  local padded
  padded="$(printf "%04d" "$iteration")"
  echo "${MODEL_ROOT}/iter-${padded}/specialists"
}

iteration_report_dir() {
  local iteration="$1"
  local padded
  padded="$(printf "%04d" "$iteration")"
  echo "${REPORT_ROOT}/iter-${padded}"
}

iteration_raw_dir() {
  local iteration="$1"
  local padded
  padded="$(printf "%04d" "$iteration")"
  echo "${RAW_ROOT}/iterations/iter-${padded}/raw_train"
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
    --outcome-reward-scale "$OUTCOME_REWARD_SCALE" \
    --tactical-reward-mode "$TACTICAL_REWARD_MODE" \
    --tactical-fractional-prize-weight "$TACTICAL_FRACTIONAL_PRIZE_WEIGHT" \
    --tactical-fractional-opponent-weight "$TACTICAL_FRACTIONAL_OPPONENT_WEIGHT" \
    --output "$output_path" \
    --report-json "$report_path" \
    --overwrite
}

generate_teacher_dataset() {
  local label="$1"
  local controlled_key="$2"
  local controlled_index="$3"
  local opponent_key="$4"
  local model_dir="$5"
  local output_path="$6"
  local report_path="$7"
  mkdir -p "$(dirname "$output_path")" "$(dirname "$report_path")"
  if [[ "$REUSE_EXISTING_TRAJECTORIES" != "0" && -s "$output_path" ]]; then
    echo "Reusing existing teacher trajectory dataset for $label: $output_path"
    return
  fi
  "$PY" -m ptcg_abc rl-generate-phase5-public-agent-trajectories \
    "${ROOT_ARGS[@]}" \
    --public-agent-key "$opponent_key" \
    --controlled-public-agent-key "$controlled_key" \
    --controlled-deck-index "$controlled_index" \
    --agent "$AGENT" \
    --specialist-model-dir "$model_dir" \
    --teacher-agent "$TEACHER_AGENT" \
    --games-per-matchup "$TEACHER_GAMES_PER_ITERATION" \
    --max-steps "$MAX_STEPS" \
    --require-min-opponents 1 \
    --outcome-reward-scale "$OUTCOME_REWARD_SCALE" \
    --tactical-reward-mode "$TACTICAL_REWARD_MODE" \
    --tactical-fractional-prize-weight "$TACTICAL_FRACTIONAL_PRIZE_WEIGHT" \
    --tactical-fractional-opponent-weight "$TACTICAL_FRACTIONAL_OPPONENT_WEIGHT" \
    --output "$output_path" \
    --report-json "$report_path" \
    --overwrite
}

train_from_datasets() {
  local label="$1"
  local deck_index="$2"
  local previous_dir="$3"
  local current_dir="$4"
  local report_json="$5"
  local dataset_array_name="$6"
  local -n datasets="$dataset_array_name"
  local dataset_args=()
  local dataset
  for dataset in "${datasets[@]}"; do
    dataset_args+=(--selfplay-dataset "$dataset")
  done
  "$PY" -m ptcg_abc rl-train-phase5-generalist \
    --no-decision-dataset \
    "${dataset_args[@]}" \
    --initial-checkpoint "${previous_dir}/$(checkpoint_name "$deck_index")" \
    --checkpoint "${current_dir}/$(checkpoint_name "$deck_index")" \
    --report-json "$report_json" \
    --epochs "$TRAIN_EPOCHS_PER_ITERATION" \
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
  echo "Trained $label deck index $deck_index from ${#datasets[@]} aggregated dataset(s)."
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
    --games-per-matchup "$EVAL_GAMES_PER_ITERATION" \
    --max-steps "$MAX_STEPS" \
    --require-min-opponents 1 \
    --report-json "$report_json" \
    --report-md "$report_md" \
    --status-json "$status_json"
}

write_iteration_summary() {
  local summary_json="$1"
  local iteration="$2"
  local model_dir="$3"
  local dragapult_train_json="$4"
  local lucario_train_json="$5"
  local dragapult_teacher_json="$6"
  local lucario_teacher_json="$7"
  local dragapult_eval_json="$8"
  local lucario_eval_json="$9"
  "$PY" - "$summary_json" "$iteration" "$model_dir" \
    "$dragapult_train_json" "$lucario_train_json" \
    "$dragapult_teacher_json" "$lucario_teacher_json" \
    "$dragapult_eval_json" "$lucario_eval_json" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
iteration = int(sys.argv[2])
model_dir = sys.argv[3]

def maybe_path(value):
    return Path(value) if value else None

train_paths = [maybe_path(sys.argv[4]), maybe_path(sys.argv[5])]
teacher_paths = [maybe_path(sys.argv[6]), maybe_path(sys.argv[7])]
eval_paths = [maybe_path(sys.argv[8]), maybe_path(sys.argv[9])]

def maybe_json(path):
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())

eval_matchups = []
totals = {"games": 0, "wins": 0, "losses": 0, "draws": 0, "timeouts": 0, "errors": 0}
for path in eval_paths:
    data = maybe_json(path)
    if data is None:
        continue
    rows = data.get("rows", [])
    payload = {
        "report": path.as_posix(),
        "games": sum(int(row.get("games", 0)) for row in rows),
        "wins": sum(int(row.get("wins", 0)) for row in rows),
        "losses": sum(int(row.get("losses", 0)) for row in rows),
        "draws": sum(int(row.get("draws", 0)) for row in rows),
        "timeouts": sum(int(row.get("timeouts", 0)) for row in rows),
        "errors": sum(int(row.get("errors", 0)) for row in rows),
    }
    payload["win_rate"] = payload["wins"] / payload["games"] if payload["games"] else 0.0
    eval_matchups.append(payload)
    for key in totals:
        totals[key] += int(payload[key])
totals["win_rate"] = totals["wins"] / totals["games"] if totals["games"] else 0.0

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(
    json.dumps(
        {
            "iteration": iteration,
            "model_dir": model_dir,
            "train_reports": [
                path.as_posix() for path in train_paths if path is not None and path.exists()
            ],
            "teacher_reports": [
                path.as_posix() for path in teacher_paths if path is not None and path.exists()
            ],
            "eval_matchups": eval_matchups,
            "combined": totals,
        },
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)
print(totals["wins"])
PY
}

for checkpoint in \
  "${BASE_MODEL_DIR}/$(checkpoint_name "$DRAGAPULT_DECK_INDEX")" \
  "${BASE_MODEL_DIR}/$(checkpoint_name "$LUCARIO_DECK_INDEX")"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing base checkpoint: $checkpoint" >&2
    exit 2
  fi
done

GEN0_REPORT_DIR="$(iteration_report_dir 0)"
mkdir -p "$GEN0_REPORT_DIR"

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

DRAGAPULT_DATASETS=("$DRAGAPULT_RULE_DATASET")
LUCARIO_DATASETS=("$LUCARIO_RULE_DATASET")

if [[ "$EVAL_BASE" != "0" ]]; then
  padded="0000"
  DRAGAPULT_BASE_EVAL_JSON="reports/${RUN_NAME}_iter-${padded}_dragapult_vs_lucario_${EVAL_GAMES_PER_ITERATION}g.json"
  DRAGAPULT_BASE_EVAL_MD="reports/${RUN_NAME}_iter-${padded}_dragapult_vs_lucario_${EVAL_GAMES_PER_ITERATION}g.md"
  DRAGAPULT_BASE_STATUS_JSON="reports/${RUN_NAME}_iter-${padded}_dragapult_vs_lucario_status.json"
  LUCARIO_BASE_EVAL_JSON="reports/${RUN_NAME}_iter-${padded}_lucario_vs_dragapult_${EVAL_GAMES_PER_ITERATION}g.json"
  LUCARIO_BASE_EVAL_MD="reports/${RUN_NAME}_iter-${padded}_lucario_vs_dragapult_${EVAL_GAMES_PER_ITERATION}g.md"
  LUCARIO_BASE_STATUS_JSON="reports/${RUN_NAME}_iter-${padded}_lucario_vs_dragapult_status.json"
  eval_matchup \
    "$DRAGAPULT_PUBLIC_AGENT_KEY" \
    "$DRAGAPULT_DECK_INDEX" \
    "$LUCARIO_PUBLIC_AGENT_KEY" \
    "$BASE_MODEL_DIR" \
    "$DRAGAPULT_BASE_EVAL_JSON" \
    "$DRAGAPULT_BASE_EVAL_MD" \
    "$DRAGAPULT_BASE_STATUS_JSON"
  eval_matchup \
    "$LUCARIO_PUBLIC_AGENT_KEY" \
    "$LUCARIO_DECK_INDEX" \
    "$DRAGAPULT_PUBLIC_AGENT_KEY" \
    "$BASE_MODEL_DIR" \
    "$LUCARIO_BASE_EVAL_JSON" \
    "$LUCARIO_BASE_EVAL_MD" \
    "$LUCARIO_BASE_STATUS_JSON"
  write_iteration_summary \
    "${GEN0_REPORT_DIR}/iteration_summary.json" \
    0 \
    "$BASE_MODEL_DIR" \
    "" \
    "" \
    "" \
    "" \
    "$DRAGAPULT_BASE_EVAL_JSON" \
    "$LUCARIO_BASE_EVAL_JSON"
fi

BEST_COMBINED_WINS=-1
BEST_ITERATION=0
NO_IMPROVE_ITERATIONS=0
PREVIOUS_DIR="$BASE_MODEL_DIR"

for iteration in $(seq 1 "$DAGGER_ITERATIONS"); do
  padded="$(printf "%04d" "$iteration")"
  CURRENT_DIR="$(iteration_model_dir "$iteration")"
  ITER_REPORT_DIR="$(iteration_report_dir "$iteration")"
  ITER_RAW_DIR="$(iteration_raw_dir "$iteration")"
  mkdir -p "$CURRENT_DIR" "$ITER_REPORT_DIR" "$ITER_RAW_DIR"

  DRAGAPULT_TEACHER_DATASET="${ITER_RAW_DIR}/dragapult_teacher_iter-${padded}.jsonl"
  LUCARIO_TEACHER_DATASET="${ITER_RAW_DIR}/lucario_teacher_iter-${padded}.jsonl"
  DRAGAPULT_TEACHER_REPORT="${ITER_REPORT_DIR}/dragapult_teacher_report.json"
  LUCARIO_TEACHER_REPORT="${ITER_REPORT_DIR}/lucario_teacher_report.json"
  DRAGAPULT_TRAIN_REPORT="${ITER_REPORT_DIR}/dragapult_train_report.json"
  LUCARIO_TRAIN_REPORT="${ITER_REPORT_DIR}/lucario_train_report.json"
  DRAGAPULT_EVAL_JSON="reports/${RUN_NAME}_iter-${padded}_dragapult_vs_lucario_${EVAL_GAMES_PER_ITERATION}g.json"
  DRAGAPULT_EVAL_MD="reports/${RUN_NAME}_iter-${padded}_dragapult_vs_lucario_${EVAL_GAMES_PER_ITERATION}g.md"
  DRAGAPULT_STATUS_JSON="reports/${RUN_NAME}_iter-${padded}_dragapult_vs_lucario_status.json"
  LUCARIO_EVAL_JSON="reports/${RUN_NAME}_iter-${padded}_lucario_vs_dragapult_${EVAL_GAMES_PER_ITERATION}g.json"
  LUCARIO_EVAL_MD="reports/${RUN_NAME}_iter-${padded}_lucario_vs_dragapult_${EVAL_GAMES_PER_ITERATION}g.md"
  LUCARIO_STATUS_JSON="reports/${RUN_NAME}_iter-${padded}_lucario_vs_dragapult_status.json"
  SUMMARY_JSON="${ITER_REPORT_DIR}/iteration_summary.json"

  echo "DAgger iteration $iteration / $DAGGER_ITERATIONS"
  echo "PREVIOUS_DIR=$PREVIOUS_DIR"
  echo "CURRENT_DIR=$CURRENT_DIR"
  echo "DRAGAPULT_DATASETS=${#DRAGAPULT_DATASETS[@]}"
  echo "LUCARIO_DATASETS=${#LUCARIO_DATASETS[@]}"

  generate_teacher_dataset \
    "Dragapult" \
    "$DRAGAPULT_PUBLIC_AGENT_KEY" \
    "$DRAGAPULT_DECK_INDEX" \
    "$LUCARIO_PUBLIC_AGENT_KEY" \
    "$PREVIOUS_DIR" \
    "$DRAGAPULT_TEACHER_DATASET" \
    "$DRAGAPULT_TEACHER_REPORT"
  generate_teacher_dataset \
    "Lucario" \
    "$LUCARIO_PUBLIC_AGENT_KEY" \
    "$LUCARIO_DECK_INDEX" \
    "$DRAGAPULT_PUBLIC_AGENT_KEY" \
    "$PREVIOUS_DIR" \
    "$LUCARIO_TEACHER_DATASET" \
    "$LUCARIO_TEACHER_REPORT"

  DRAGAPULT_DATASETS+=("$DRAGAPULT_TEACHER_DATASET")
  LUCARIO_DATASETS+=("$LUCARIO_TEACHER_DATASET")

  train_from_datasets \
    "Dragapult" \
    "$DRAGAPULT_DECK_INDEX" \
    "$PREVIOUS_DIR" \
    "$CURRENT_DIR" \
    "$DRAGAPULT_TRAIN_REPORT" \
    DRAGAPULT_DATASETS
  train_from_datasets \
    "Lucario" \
    "$LUCARIO_DECK_INDEX" \
    "$PREVIOUS_DIR" \
    "$CURRENT_DIR" \
    "$LUCARIO_TRAIN_REPORT" \
    LUCARIO_DATASETS

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

  write_iteration_summary \
    "$SUMMARY_JSON" \
    "$iteration" \
    "$CURRENT_DIR" \
    "$DRAGAPULT_TRAIN_REPORT" \
    "$LUCARIO_TRAIN_REPORT" \
    "$DRAGAPULT_TEACHER_REPORT" \
    "$LUCARIO_TEACHER_REPORT" \
    "$DRAGAPULT_EVAL_JSON" \
    "$LUCARIO_EVAL_JSON"

  COMBINED_WINS="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['combined']['wins'])" "$SUMMARY_JSON")"
  echo "DAgger iteration $iteration combined eval wins: $COMBINED_WINS"

  if (( COMBINED_WINS > BEST_COMBINED_WINS )); then
    BEST_COMBINED_WINS="$COMBINED_WINS"
    BEST_ITERATION="$iteration"
    NO_IMPROVE_ITERATIONS=0
  else
    NO_IMPROVE_ITERATIONS="$((NO_IMPROVE_ITERATIONS + 1))"
  fi
  echo "Best DAgger iteration so far: $BEST_ITERATION with combined wins $BEST_COMBINED_WINS"
  echo "No-improve iterations: $NO_IMPROVE_ITERATIONS"

  PREVIOUS_DIR="$CURRENT_DIR"

  if (( EARLY_STOP_PATIENCE > 0 && iteration >= MIN_ITERATIONS && NO_IMPROVE_ITERATIONS >= EARLY_STOP_PATIENCE )); then
    echo "Early stopping after iteration $iteration: no combined eval improvement for $NO_IMPROVE_ITERATIONS iterations."
    break
  fi
done
