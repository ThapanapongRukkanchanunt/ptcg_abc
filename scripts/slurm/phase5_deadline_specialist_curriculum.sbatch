#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-deadline
#SBATCH --output=experiments/rl/slurm-%j-phase5-deadline-specialist.out
#SBATCH --error=experiments/rl/slurm-%j-phase5-deadline-specialist.err
#SBATCH --time=48:00:00
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
PUBLIC_AGENT_ROOTS="${PUBLIC_AGENT_ROOTS:-${GAME_DATA_ROOT}/phase5_public_agents}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME}"
CONTROLLED_DECK_INDEX="${CONTROLLED_DECK_INDEX:?Set CONTROLLED_DECK_INDEX}"
CONTROLLED_PUBLIC_AGENT_KEY="${CONTROLLED_PUBLIC_AGENT_KEY:-}"
OPPONENT_POOL="${OPPONENT_POOL:-league-rule}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-}"
START_GENERATION="${START_GENERATION:-0}"
GENERATIONS_PER_JOB="${GENERATIONS_PER_JOB:-5}"
TRAIN_GAMES_PER_GENERATION="${TRAIN_GAMES_PER_GENERATION:-1000}"
EVAL_GAMES_PER_MATCHUP="${EVAL_GAMES_PER_MATCHUP:-8}"
EPSILON_START="${EPSILON_START:-0.10}"
EPSILON_END="${EPSILON_END:-0.10}"
POLICY_SEED="${POLICY_SEED:-20260820}"
MAX_STEPS="${MAX_STEPS:-600}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.00001}"
CLIP_EPSILON="${CLIP_EPSILON:-0.2}"
POLICY_LOSS_WEIGHT="${POLICY_LOSS_WEIGHT:-1.0}"
VALUE_LOSS_WEIGHT="${VALUE_LOSS_WEIGHT:-0.5}"
VALUE_BACKPROP_SCOPE="${VALUE_BACKPROP_SCOPE:-head-only}"
ENTROPY_WEIGHT="${ENTROPY_WEIGHT:-0.001}"
REWARD_OBJECTIVE="${REWARD_OBJECTIVE:-discounted-turn-prizes}"
TURN_PRIZE_DISCOUNT_GAMMA="${TURN_PRIZE_DISCOUNT_GAMMA:-0.97}"
MODEL_ROOT="${MODEL_ROOT:-models/rl/phase5_deadline_specialists/${RUN_NAME}}"
REPORT_ROOT="${REPORT_ROOT:-experiments/rl/phase5_deadline_specialists/${RUN_NAME}}"
RAW_ROOT="${RAW_ROOT:-${GAME_DATA_ROOT}/phase5_deadline_specialists/${RUN_NAME}}"
AUTO_CONTINUE="${AUTO_CONTINUE:-1}"
MAX_GENERATION="${MAX_GENERATION:-20}"
EVAL_PROFILE="${EVAL_PROFILE:-}"
if [[ -z "$EVAL_PROFILE" && -f "${MODEL_ROOT}/evaluation-profile" ]]; then
  IFS= read -r EVAL_PROFILE < "${MODEL_ROOT}/evaluation-profile"
fi
DEFAULT_EVAL_TACTICAL_SCORE_WEIGHT="1.0"
DEFAULT_EVAL_RULE_PRIOR_WEIGHT="0.08"
DEFAULT_EVAL_LEAF_STATE_VALUE_WEIGHT="0.0"
DEFAULT_EVAL_HANDCRAFTED_REWARD_WEIGHT="0.0"
DEFAULT_EVAL_HANDCRAFTED_REWARD_DECK_INDEX="0"
DEFAULT_EVAL_TERMINAL_OUTCOME_GUARD="0"
DEFAULT_EVAL_SEARCH_SETUP_ACTIVE="0"
DEFAULT_EVAL_REPORT_SUFFIX=""
if [[ "$EVAL_PROFILE" == "dragapult-reward50-v1" ]]; then
  DEFAULT_EVAL_TACTICAL_SCORE_WEIGHT="0.0"
  DEFAULT_EVAL_RULE_PRIOR_WEIGHT="0.0"
  DEFAULT_EVAL_LEAF_STATE_VALUE_WEIGHT="0.5"
  DEFAULT_EVAL_HANDCRAFTED_REWARD_WEIGHT="0.5"
  DEFAULT_EVAL_HANDCRAFTED_REWARD_DECK_INDEX="3"
  DEFAULT_EVAL_TERMINAL_OUTCOME_GUARD="1"
  DEFAULT_EVAL_SEARCH_SETUP_ACTIVE="1"
  DEFAULT_EVAL_REPORT_SUFFIX="_reward50"
fi
EVAL_TACTICAL_SCORE_WEIGHT="${EVAL_TACTICAL_SCORE_WEIGHT:-$DEFAULT_EVAL_TACTICAL_SCORE_WEIGHT}"
EVAL_RULE_PRIOR_WEIGHT="${EVAL_RULE_PRIOR_WEIGHT:-$DEFAULT_EVAL_RULE_PRIOR_WEIGHT}"
EVAL_LEAF_STATE_VALUE_WEIGHT="${EVAL_LEAF_STATE_VALUE_WEIGHT:-$DEFAULT_EVAL_LEAF_STATE_VALUE_WEIGHT}"
EVAL_HANDCRAFTED_REWARD_WEIGHT="${EVAL_HANDCRAFTED_REWARD_WEIGHT:-$DEFAULT_EVAL_HANDCRAFTED_REWARD_WEIGHT}"
EVAL_HANDCRAFTED_REWARD_DECK_INDEX="${EVAL_HANDCRAFTED_REWARD_DECK_INDEX:-$DEFAULT_EVAL_HANDCRAFTED_REWARD_DECK_INDEX}"
EVAL_VALUE_NORMALIZATION_EPSILON="${EVAL_VALUE_NORMALIZATION_EPSILON:-0.000001}"
EVAL_TERMINAL_OUTCOME_GUARD="${EVAL_TERMINAL_OUTCOME_GUARD:-$DEFAULT_EVAL_TERMINAL_OUTCOME_GUARD}"
EVAL_SEARCH_SETUP_ACTIVE="${EVAL_SEARCH_SETUP_ACTIVE:-$DEFAULT_EVAL_SEARCH_SETUP_ACTIVE}"
EVAL_REPORT_SUFFIX="${EVAL_REPORT_SUFFIX:-$DEFAULT_EVAL_REPORT_SUFFIX}"

if (( START_GENERATION < 0 )); then
  echo "START_GENERATION must be non-negative" >&2
  exit 2
fi
if (( GENERATIONS_PER_JOB <= 0 )); then
  echo "GENERATIONS_PER_JOB must be positive" >&2
  exit 2
fi
if (( TRAIN_GAMES_PER_GENERATION != 1000 )); then
  echo "This deadline curriculum requires exactly 1000 games per update" >&2
  exit 2
fi

ROOT_ARGS=()
for root in $PUBLIC_AGENT_ROOTS; do
  ROOT_ARGS+=(--public-agent-root "$root")
done

CONTROLLED_ARGS=()
if [[ -n "$CONTROLLED_PUBLIC_AGENT_KEY" ]]; then
  CONTROLLED_ARGS+=(--controlled-public-agent-key "$CONTROLLED_PUBLIC_AGENT_KEY")
  CONTROLLED_ARGS+=(--controlled-deck-index "$CONTROLLED_DECK_INDEX")
else
  CONTROLLED_ARGS+=(--deck-index "$CONTROLLED_DECK_INDEX")
fi

generation_dir() {
  local padded
  padded="$(printf "%04d" "$1")"
  echo "${MODEL_ROOT}/gen-${padded}/specialists"
}

report_generation_dir() {
  local padded
  padded="$(printf "%04d" "$1")"
  echo "${REPORT_ROOT}/gen-${padded}"
}

raw_generation_dir() {
  local padded
  padded="$(printf "%04d" "$1")"
  echo "${RAW_ROOT}/generations/gen-${padded}/raw_train"
}

checkpoint_path() {
  echo "$(generation_dir "$1")/deck-$(printf "%02d" "$CONTROLLED_DECK_INDEX").pt"
}

epsilon_for_local_generation() {
  "$PY" -c "start=float('$EPSILON_START'); end=float('$EPSILON_END'); count=max(1,int('$GENERATIONS_PER_JOB')); local_gen=int('$1'); print(end if count == 1 else start + (end - start) * ((local_gen - 1) / (count - 1)))"
}

echo "Phase 5 deadline specialist curriculum"
echo "RUN_NAME=$RUN_NAME"
echo "CONTROLLED_DECK_INDEX=$CONTROLLED_DECK_INDEX"
echo "CONTROLLED_PUBLIC_AGENT_KEY=$CONTROLLED_PUBLIC_AGENT_KEY"
echo "OPPONENT_POOL=$OPPONENT_POOL"
echo "INITIAL_CHECKPOINT=$INITIAL_CHECKPOINT"
echo "START_GENERATION=$START_GENERATION"
echo "GENERATIONS_PER_JOB=$GENERATIONS_PER_JOB"
echo "TRAIN_GAMES_PER_GENERATION=$TRAIN_GAMES_PER_GENERATION"
echo "EPSILON_START=$EPSILON_START"
echo "EPSILON_END=$EPSILON_END"
echo "LEARNING_RATE=$LEARNING_RATE"
echo "VALUE_BACKPROP_SCOPE=$VALUE_BACKPROP_SCOPE"
echo "REWARD_OBJECTIVE=$REWARD_OBJECTIVE"
echo "TURN_PRIZE_DISCOUNT_GAMMA=$TURN_PRIZE_DISCOUNT_GAMMA"
echo "MODEL_ROOT=$MODEL_ROOT"
echo "REPORT_ROOT=$REPORT_ROOT"
echo "RAW_ROOT=$RAW_ROOT"
echo "AUTO_CONTINUE=$AUTO_CONTINUE"
echo "MAX_GENERATION=$MAX_GENERATION"
echo "EVAL_PROFILE=$EVAL_PROFILE"
echo "EVAL_TACTICAL_SCORE_WEIGHT=$EVAL_TACTICAL_SCORE_WEIGHT"
echo "EVAL_RULE_PRIOR_WEIGHT=$EVAL_RULE_PRIOR_WEIGHT"
echo "EVAL_LEAF_STATE_VALUE_WEIGHT=$EVAL_LEAF_STATE_VALUE_WEIGHT"
echo "EVAL_HANDCRAFTED_REWARD_WEIGHT=$EVAL_HANDCRAFTED_REWARD_WEIGHT"
echo "EVAL_HANDCRAFTED_REWARD_DECK_INDEX=$EVAL_HANDCRAFTED_REWARD_DECK_INDEX"
echo "EVAL_VALUE_NORMALIZATION_EPSILON=$EVAL_VALUE_NORMALIZATION_EPSILON"
echo "EVAL_TERMINAL_OUTCOME_GUARD=$EVAL_TERMINAL_OUTCOME_GUARD"
echo "EVAL_SEARCH_SETUP_ACTIVE=$EVAL_SEARCH_SETUP_ACTIVE"
echo "EVAL_REPORT_SUFFIX=$EVAL_REPORT_SUFFIX"

EVAL_SEARCH_ARGS=(
  --tactical-score-weight "$EVAL_TACTICAL_SCORE_WEIGHT"
  --rule-prior-weight "$EVAL_RULE_PRIOR_WEIGHT"
  --leaf-state-value-weight "$EVAL_LEAF_STATE_VALUE_WEIGHT"
  --handcrafted-reward-weight "$EVAL_HANDCRAFTED_REWARD_WEIGHT"
  --handcrafted-reward-deck-index "$EVAL_HANDCRAFTED_REWARD_DECK_INDEX"
  --handcrafted-reward-gamma "$TURN_PRIZE_DISCOUNT_GAMMA"
  --value-normalization-epsilon "$EVAL_VALUE_NORMALIZATION_EPSILON"
)
if [[ "$EVAL_TERMINAL_OUTCOME_GUARD" != "0" ]]; then
  EVAL_SEARCH_ARGS+=(--terminal-outcome-guard)
fi
if [[ "$EVAL_SEARCH_SETUP_ACTIVE" != "0" ]]; then
  EVAL_SEARCH_ARGS+=(--search-setup-active)
fi

START_CHECKPOINT="$(checkpoint_path "$START_GENERATION")"
mkdir -p "$(dirname "$START_CHECKPOINT")" "$(report_generation_dir "$START_GENERATION")"
if (( START_GENERATION == 0 )); then
  if [[ -z "$INITIAL_CHECKPOINT" || ! -f "$INITIAL_CHECKPOINT" ]]; then
    echo "INITIAL_CHECKPOINT must exist for generation 0: $INITIAL_CHECKPOINT" >&2
    exit 2
  fi
  if [[ ! -f "$START_CHECKPOINT" ]]; then
    cp "$INITIAL_CHECKPOINT" "$START_CHECKPOINT"
  fi
elif [[ ! -f "$START_CHECKPOINT" ]]; then
  echo "Previous chained checkpoint is missing: $START_CHECKPOINT" >&2
  exit 2
fi

for local_generation in $(seq 1 "$GENERATIONS_PER_JOB"); do
  generation="$((START_GENERATION + local_generation))"
  previous_generation="$((generation - 1))"
  padded="$(printf "%04d" "$generation")"
  PREVIOUS_DIR="$(generation_dir "$previous_generation")"
  CURRENT_DIR="$(generation_dir "$generation")"
  GEN_REPORT_DIR="$(report_generation_dir "$generation")"
  GEN_RAW_DIR="$(raw_generation_dir "$generation")"
  mkdir -p "$CURRENT_DIR" "$GEN_REPORT_DIR" "$GEN_RAW_DIR"

  EPSILON="$(epsilon_for_local_generation "$local_generation")"
  TRAJECTORY_DATASET="${GEN_RAW_DIR}/phase5_deadline_gen-${padded}.jsonl"
  TRAJECTORY_REPORT="${GEN_REPORT_DIR}/trajectory_report.json"
  PPO_REPORT="${GEN_REPORT_DIR}/ppo_report.json"
  EVAL_TOTAL_GAMES="$((EVAL_GAMES_PER_MATCHUP * 13))"
  EVAL_JSON="reports/${RUN_NAME}${EVAL_REPORT_SUFFIX}_gen-${padded}_eval_${EVAL_TOTAL_GAMES}g.json"
  EVAL_MD="reports/${RUN_NAME}${EVAL_REPORT_SUFFIX}_gen-${padded}_eval_${EVAL_TOTAL_GAMES}g.md"
  STATUS_JSON="reports/${RUN_NAME}${EVAL_REPORT_SUFFIX}_gen-${padded}_status.json"

  echo "Generation $generation; local_generation=$local_generation; epsilon=$EPSILON"
  "$PY" -m ptcg_abc rl-generate-phase5-public-agent-trajectories \
    "${ROOT_ARGS[@]}" \
    --opponent-pool "$OPPONENT_POOL" \
    "${CONTROLLED_ARGS[@]}" \
    --agent phase5-epsilon-mixture \
    --policy-epsilon "$EPSILON" \
    --policy-seed "$((POLICY_SEED + generation * 1000000))" \
    --specialist-model-dir "$PREVIOUS_DIR" \
    --games-total "$TRAIN_GAMES_PER_GENERATION" \
    --max-steps "$MAX_STEPS" \
    --reward-objective "$REWARD_OBJECTIVE" \
    --turn-prize-discount-gamma "$TURN_PRIZE_DISCOUNT_GAMMA" \
    --require-min-opponents 1 \
    --output "$TRAJECTORY_DATASET" \
    --report-json "$TRAJECTORY_REPORT" \
    --overwrite

  "$PY" -m ptcg_abc rl-train-phase5-ppo \
    --trajectory-dataset "$TRAJECTORY_DATASET" \
    --checkpoint "$(checkpoint_path "$previous_generation")" \
    --output-checkpoint "$(checkpoint_path "$generation")" \
    --report-json "$PPO_REPORT" \
    --deck-index-filter "$CONTROLLED_DECK_INDEX" \
    --require-on-policy \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --clip-epsilon "$CLIP_EPSILON" \
    --policy-loss-weight "$POLICY_LOSS_WEIGHT" \
    --value-loss-weight "$VALUE_LOSS_WEIGHT" \
    --value-backprop-scope "$VALUE_BACKPROP_SCOPE" \
    --entropy-weight "$ENTROPY_WEIGHT"

  rm -f "$TRAJECTORY_DATASET"
  echo "Deleted consumed raw trajectory: $TRAJECTORY_DATASET"

  if (( EVAL_GAMES_PER_MATCHUP > 0 )); then
    "$PY" -m ptcg_abc rl-evaluate-phase5-public-agents \
      "${ROOT_ARGS[@]}" \
      --opponent-pool "$OPPONENT_POOL" \
      "${CONTROLLED_ARGS[@]}" \
      --agent phase5-search \
      --specialist-model-dir "$CURRENT_DIR" \
      --search-top-k 4 \
      "${EVAL_SEARCH_ARGS[@]}" \
      --games-per-matchup "$EVAL_GAMES_PER_MATCHUP" \
      --max-steps "$MAX_STEPS" \
      --prize-discount-gamma "$TURN_PRIZE_DISCOUNT_GAMMA" \
      --require-min-opponents 1 \
      --report-json "$EVAL_JSON" \
      --report-md "$EVAL_MD" \
      --status-json "$STATUS_JSON"
  fi
done

end_generation="$((START_GENERATION + GENERATIONS_PER_JOB))"
if [[ "$AUTO_CONTINUE" != "0" ]] && (( end_generation < MAX_GENERATION )); then
  next_job_name="p5-$(echo "$RUN_NAME" | tr -cd '[:alnum:]-' | cut -c1-20)-${end_generation}"
  export_values="RUN_NAME=${RUN_NAME},CONTROLLED_DECK_INDEX=${CONTROLLED_DECK_INDEX},OPPONENT_POOL=${OPPONENT_POOL},START_GENERATION=${end_generation},GENERATIONS_PER_JOB=${GENERATIONS_PER_JOB},TRAIN_GAMES_PER_GENERATION=${TRAIN_GAMES_PER_GENERATION},EVAL_GAMES_PER_MATCHUP=${EVAL_GAMES_PER_MATCHUP},EPSILON_START=0.10,EPSILON_END=0.10,POLICY_SEED=${POLICY_SEED},MAX_STEPS=${MAX_STEPS},EPOCHS=${EPOCHS},BATCH_SIZE=${BATCH_SIZE},LEARNING_RATE=${LEARNING_RATE},CLIP_EPSILON=${CLIP_EPSILON},POLICY_LOSS_WEIGHT=${POLICY_LOSS_WEIGHT},VALUE_LOSS_WEIGHT=${VALUE_LOSS_WEIGHT},VALUE_BACKPROP_SCOPE=${VALUE_BACKPROP_SCOPE},ENTROPY_WEIGHT=${ENTROPY_WEIGHT},REWARD_OBJECTIVE=${REWARD_OBJECTIVE},TURN_PRIZE_DISCOUNT_GAMMA=${TURN_PRIZE_DISCOUNT_GAMMA},MODEL_ROOT=${MODEL_ROOT},REPORT_ROOT=${REPORT_ROOT},RAW_ROOT=${RAW_ROOT},AUTO_CONTINUE=${AUTO_CONTINUE},MAX_GENERATION=${MAX_GENERATION},EVAL_PROFILE=${EVAL_PROFILE},EVAL_TACTICAL_SCORE_WEIGHT=${EVAL_TACTICAL_SCORE_WEIGHT},EVAL_RULE_PRIOR_WEIGHT=${EVAL_RULE_PRIOR_WEIGHT},EVAL_LEAF_STATE_VALUE_WEIGHT=${EVAL_LEAF_STATE_VALUE_WEIGHT},EVAL_HANDCRAFTED_REWARD_WEIGHT=${EVAL_HANDCRAFTED_REWARD_WEIGHT},EVAL_HANDCRAFTED_REWARD_DECK_INDEX=${EVAL_HANDCRAFTED_REWARD_DECK_INDEX},EVAL_VALUE_NORMALIZATION_EPSILON=${EVAL_VALUE_NORMALIZATION_EPSILON},EVAL_TERMINAL_OUTCOME_GUARD=${EVAL_TERMINAL_OUTCOME_GUARD},EVAL_SEARCH_SETUP_ACTIVE=${EVAL_SEARCH_SETUP_ACTIVE},EVAL_REPORT_SUFFIX=${EVAL_REPORT_SUFFIX}"
  if [[ -n "$CONTROLLED_PUBLIC_AGENT_KEY" ]]; then
    export_values="${export_values},CONTROLLED_PUBLIC_AGENT_KEY=${CONTROLLED_PUBLIC_AGENT_KEY}"
  fi
  next_job=""
  for attempt in $(seq 1 20); do
    if next_job="$(sbatch --parsable --job-name="$next_job_name" --export="ALL,${export_values}" scripts/slurm/phase5_deadline_specialist_curriculum.sbatch)"; then
      break
    fi
    echo "Continuation submission attempt $attempt failed; retrying in 30 seconds"
    sleep 30
  done
  if [[ -z "$next_job" ]]; then
    echo "Unable to submit continuation after 20 attempts" >&2
    exit 3
  fi
  echo "Submitted continuation job $next_job at generation $end_generation"
fi
