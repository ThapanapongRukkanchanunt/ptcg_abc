#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-1deck-ppo-led
#SBATCH --output=experiments/rl/slurm-%j-phase5-one-deck-ppo-led.out
#SBATCH --error=experiments/rl/slurm-%j-phase5-one-deck-ppo-led.err
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
RUN_NAME="${RUN_NAME:-phase5_dragapult_vs_lucario_ppo_dominant}"
CONTROLLED_PUBLIC_AGENT_KEY="${CONTROLLED_PUBLIC_AGENT_KEY:-sample_dragapult}"
OPPONENT_PUBLIC_AGENT_KEYS="${OPPONENT_PUBLIC_AGENT_KEYS:-sample_lucario}"
CONTROLLED_DECK_INDEX="${CONTROLLED_DECK_INDEX:-101}"
GENERATIONS="${GENERATIONS:-3}"
TRAIN_GAMES_PER_GENERATION="${TRAIN_GAMES_PER_GENERATION:-1000}"
EVAL_GAMES_PER_GENERATION="${EVAL_GAMES_PER_GENERATION:-200}"
EPSILON_START="${EPSILON_START:-0.90}"
EPSILON_END="${EPSILON_END:-0.10}"
MAX_STEPS="${MAX_STEPS:-600}"
BC_EPOCHS="${BC_EPOCHS:-1}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.00005}"
CLIP_EPSILON="${CLIP_EPSILON:-0.2}"
RULE_ANCHOR_FRACTION="${RULE_ANCHOR_FRACTION:-0.0}"
BC_LOSS_WEIGHT="${BC_LOSS_WEIGHT:-$RULE_ANCHOR_FRACTION}"
PPO_POLICY_LOSS_WEIGHT="${PPO_POLICY_LOSS_WEIGHT:-1.0}"
PPO_VALUE_LOSS_WEIGHT="${PPO_VALUE_LOSS_WEIGHT:-0.5}"
ENTROPY_WEIGHT="${ENTROPY_WEIGHT:-0.01}"
GRADIENT_DIAGNOSTIC_BATCHES="${GRADIENT_DIAGNOSTIC_BATCHES:-16}"
ADVANTAGE_NORMALIZATION="${ADVANTAGE_NORMALIZATION:-batch}"
VALUE_BACKPROP_SCOPE="${VALUE_BACKPROP_SCOPE:-shared}"
OUTCOME_REWARD_SCALE="${OUTCOME_REWARD_SCALE:-1.0}"
TACTICAL_REWARD_MODE="${TACTICAL_REWARD_MODE:-basic-fractional-prize}"
TACTICAL_FRACTIONAL_PRIZE_WEIGHT="${TACTICAL_FRACTIONAL_PRIZE_WEIGHT:-0.25}"
TACTICAL_FRACTIONAL_OPPONENT_WEIGHT="${TACTICAL_FRACTIONAL_OPPONENT_WEIGHT:-1.0}"
INIT_SEED="${INIT_SEED:-20260722}"
POLICY_SEED="${POLICY_SEED:-20260722}"
BC_RULE_STEP_LIMIT="${BC_RULE_STEP_LIMIT:-0}"
ONLINE_RULE_STEP_LIMIT="${ONLINE_RULE_STEP_LIMIT:-0}"
ON_POLICY_STEP_LIMIT="${ON_POLICY_STEP_LIMIT:-0}"
MODEL_ROOT="${MODEL_ROOT:-models/rl/phase5_one_deck_public_ppo_dominant/${RUN_NAME}}"
REPORT_ROOT="${REPORT_ROOT:-experiments/rl/phase5_one_deck_public_ppo_dominant/${RUN_NAME}}"
RAW_ROOT="${RAW_ROOT:-${GAME_DATA_ROOT}/phase5_one_deck_public_ppo_dominant/${RUN_NAME}}"
DELETE_ON_POLICY_DATA="${DELETE_ON_POLICY_DATA:-1}"
OVERWRITE_BOOTSTRAP="${OVERWRITE_BOOTSTRAP:-0}"
REUSE_EXISTING_TRAJECTORIES="${REUSE_EXISTING_TRAJECTORIES:-0}"

ROOT_ARGS=()
for root in $PUBLIC_AGENT_ROOTS; do
  ROOT_ARGS+=(--public-agent-root "$root")
done

OPPONENT_ARGS=()
for key in $OPPONENT_PUBLIC_AGENT_KEYS; do
  OPPONENT_ARGS+=(--public-agent-key "$key")
done

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

epsilon_for_generation() {
  "$PY" -c "start=float('$EPSILON_START'); end=float('$EPSILON_END'); generations=max(1,int('$GENERATIONS')); generation=int('$1'); print(end if generations == 1 else start + (end - start) * ((generation - 1) / (generations - 1)))"
}

evaluate_generation() {
  local generation="$1"
  local model_dir="$2"
  local padded
  padded="$(printf "%04d" "$generation")"
  local generation_report_dir
  generation_report_dir="$(report_generation_dir "$generation")"
  mkdir -p "$generation_report_dir/replays"
  "$PY" -m ptcg_abc rl-evaluate-phase5-public-agents \
    "${ROOT_ARGS[@]}" \
    "${OPPONENT_ARGS[@]}" \
    --controlled-public-agent-key "$CONTROLLED_PUBLIC_AGENT_KEY" \
    --controlled-deck-index "$CONTROLLED_DECK_INDEX" \
    --agent phase5-symbolic \
    --specialist-model-dir "$model_dir" \
    --games-per-matchup "$EVAL_GAMES_PER_GENERATION" \
    --max-steps "$MAX_STEPS" \
    --require-min-opponents 1 \
    --replay-output-dir "$generation_report_dir/replays" \
    --saved-win-replays 1 \
    --saved-loss-replays 1 \
    --replay-trace-limit 160 \
    --report-json "reports/${RUN_NAME}_gen-${padded}_eval_${EVAL_GAMES_PER_GENERATION}g.json" \
    --report-md "reports/${RUN_NAME}_gen-${padded}_eval_${EVAL_GAMES_PER_GENERATION}g.md" \
    --status-json "reports/${RUN_NAME}_gen-${padded}_status.json"
}

echo "Phase 5 one-deck one-time BC then PPO-dominant curriculum"
echo "RUN_NAME=$RUN_NAME"
echo "CONTROLLED_PUBLIC_AGENT_KEY=$CONTROLLED_PUBLIC_AGENT_KEY"
echo "OPPONENT_PUBLIC_AGENT_KEYS=$OPPONENT_PUBLIC_AGENT_KEYS"
echo "GENERATIONS=$GENERATIONS"
echo "TRAIN_GAMES_PER_GENERATION=$TRAIN_GAMES_PER_GENERATION"
echo "EVAL_GAMES_PER_GENERATION=$EVAL_GAMES_PER_GENERATION"
echo "EPSILON_START=$EPSILON_START"
echo "EPSILON_END=$EPSILON_END"
echo "RULE_ANCHOR_FRACTION=$RULE_ANCHOR_FRACTION"
echo "BC_LOSS_WEIGHT=$BC_LOSS_WEIGHT"
echo "GRADIENT_DIAGNOSTIC_BATCHES=$GRADIENT_DIAGNOSTIC_BATCHES"
echo "ADVANTAGE_NORMALIZATION=$ADVANTAGE_NORMALIZATION"
echo "VALUE_BACKPROP_SCOPE=$VALUE_BACKPROP_SCOPE"
echo "MODEL_ROOT=$MODEL_ROOT"
echo "REPORT_ROOT=$REPORT_ROOT"
echo "RAW_ROOT=$RAW_ROOT"

RULE_BOOTSTRAP_DATASET="${RULE_BOOTSTRAP_DATASET:-${RAW_ROOT}/rule_bootstrap/phase5_public_rule_bootstrap_gen-0000.jsonl}"
if [[ ! -s "$RULE_BOOTSTRAP_DATASET" ]]; then
  echo "Rule bootstrap dataset not found or empty: $RULE_BOOTSTRAP_DATASET" >&2
  exit 2
fi
echo "RULE_BOOTSTRAP_DATASET=$RULE_BOOTSTRAP_DATASET"

GEN0_DIR="$(generation_dir 0)"
GEN0_REPORT_DIR="$(report_generation_dir 0)"
SCRATCH_DIR="${MODEL_ROOT}/scratch"
SCRATCH_CHECKPOINT="${SCRATCH_DIR}/deck-$(printf "%02d" "$CONTROLLED_DECK_INDEX").pt"
BOOTSTRAP_CHECKPOINT="${GEN0_DIR}/deck-$(printf "%02d" "$CONTROLLED_DECK_INDEX").pt"
mkdir -p "$GEN0_DIR" "$GEN0_REPORT_DIR" "$SCRATCH_DIR"

if [[ "$OVERWRITE_BOOTSTRAP" != "0" ]]; then
  rm -f "$SCRATCH_CHECKPOINT" "$BOOTSTRAP_CHECKPOINT"
fi
if [[ ! -f "$SCRATCH_CHECKPOINT" ]]; then
  "$PY" -m ptcg_abc rl-init-phase5-policy-checkpoint \
    --checkpoint "$SCRATCH_CHECKPOINT" \
    --report-json "${GEN0_REPORT_DIR}/scratch_init_report.json" \
    --deck-index "$CONTROLLED_DECK_INDEX" \
    --controlled-public-agent-key "$CONTROLLED_PUBLIC_AGENT_KEY" \
    --experiment "$RUN_NAME" \
    --seed "$INIT_SEED"
fi

if [[ ! -f "$BOOTSTRAP_CHECKPOINT" ]]; then
  BC_LIMIT_ARGS=()
  if [[ "$BC_RULE_STEP_LIMIT" != "0" ]]; then
    BC_LIMIT_ARGS+=(--rule-step-limit "$BC_RULE_STEP_LIMIT")
  fi
  "$PY" -m ptcg_abc rl-train-phase5-trajectory-bc \
    --rule-trajectory-dataset "$RULE_BOOTSTRAP_DATASET" \
    --checkpoint "$SCRATCH_CHECKPOINT" \
    --output-checkpoint "$BOOTSTRAP_CHECKPOINT" \
    --report-json "${GEN0_REPORT_DIR}/trajectory_bc_report.json" \
    --deck-index-filter "$CONTROLLED_DECK_INDEX" \
    --epochs "$BC_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    "${BC_LIMIT_ARGS[@]}"
fi

evaluate_generation 0 "$GEN0_DIR"

USE_RULE_ANCHOR="$("$PY" -c "print(1 if float('$RULE_ANCHOR_FRACTION') > 0.0 else 0)")"
for generation in $(seq 1 "$GENERATIONS"); do
  padded="$(printf "%04d" "$generation")"
  previous_generation="$((generation - 1))"
  PREVIOUS_DIR="$(generation_dir "$previous_generation")"
  CURRENT_DIR="$(generation_dir "$generation")"
  GEN_REPORT_DIR="$(report_generation_dir "$generation")"
  GEN_RAW_DIR="$(raw_generation_dir "$generation")"
  mkdir -p "$CURRENT_DIR" "$GEN_REPORT_DIR" "$GEN_RAW_DIR"

  EPSILON="$(epsilon_for_generation "$generation")"
  ON_POLICY_DATASET="${GEN_RAW_DIR}/phase5_public_epsilon_mixture_gen-${padded}.jsonl"
  ON_POLICY_REPORT="${GEN_REPORT_DIR}/epsilon_mixture_trajectory_report.json"
  UPDATE_REPORT="${GEN_REPORT_DIR}/ppo_dominant_report.json"
  echo "Generation $generation / $GENERATIONS; epsilon=$EPSILON"

  if [[ "$REUSE_EXISTING_TRAJECTORIES" != "0" && -s "$ON_POLICY_DATASET" ]]; then
    echo "Reusing existing on-policy trajectory dataset: $ON_POLICY_DATASET"
  else
    "$PY" -m ptcg_abc rl-generate-phase5-public-agent-trajectories \
      "${ROOT_ARGS[@]}" \
      "${OPPONENT_ARGS[@]}" \
      --controlled-public-agent-key "$CONTROLLED_PUBLIC_AGENT_KEY" \
      --controlled-deck-index "$CONTROLLED_DECK_INDEX" \
      --agent phase5-epsilon-mixture \
      --policy-epsilon "$EPSILON" \
      --policy-seed "$((POLICY_SEED + generation * 1000000))" \
      --specialist-model-dir "$PREVIOUS_DIR" \
      --games-per-matchup "$TRAIN_GAMES_PER_GENERATION" \
      --max-steps "$MAX_STEPS" \
      --require-min-opponents 1 \
      --outcome-reward-scale "$OUTCOME_REWARD_SCALE" \
      --tactical-reward-mode "$TACTICAL_REWARD_MODE" \
      --tactical-fractional-prize-weight "$TACTICAL_FRACTIONAL_PRIZE_WEIGHT" \
      --tactical-fractional-opponent-weight "$TACTICAL_FRACTIONAL_OPPONENT_WEIGHT" \
      --output "$ON_POLICY_DATASET" \
      --report-json "$ON_POLICY_REPORT" \
      --overwrite
  fi

  RULE_ANCHOR_ARGS=()
  if [[ "$USE_RULE_ANCHOR" != "0" ]]; then
    RULE_ANCHOR_ARGS+=(--rule-trajectory-dataset "$RULE_BOOTSTRAP_DATASET")
    if [[ "$ONLINE_RULE_STEP_LIMIT" != "0" ]]; then
      RULE_ANCHOR_ARGS+=(--rule-step-limit "$ONLINE_RULE_STEP_LIMIT")
    fi
  fi
  ON_POLICY_LIMIT_ARGS=()
  if [[ "$ON_POLICY_STEP_LIMIT" != "0" ]]; then
    ON_POLICY_LIMIT_ARGS+=(--on-policy-step-limit "$ON_POLICY_STEP_LIMIT")
  fi

  "$PY" -m ptcg_abc rl-train-phase5-bc-ppo \
    "${RULE_ANCHOR_ARGS[@]}" \
    --on-policy-trajectory-dataset "$ON_POLICY_DATASET" \
    --checkpoint "${PREVIOUS_DIR}/deck-$(printf "%02d" "$CONTROLLED_DECK_INDEX").pt" \
    --output-checkpoint "${CURRENT_DIR}/deck-$(printf "%02d" "$CONTROLLED_DECK_INDEX").pt" \
    --report-json "$UPDATE_REPORT" \
    --deck-index-filter "$CONTROLLED_DECK_INDEX" \
    --epochs "$PPO_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --clip-epsilon "$CLIP_EPSILON" \
    --bc-loss-weight "$BC_LOSS_WEIGHT" \
    --ppo-policy-loss-weight "$PPO_POLICY_LOSS_WEIGHT" \
    --ppo-value-loss-weight "$PPO_VALUE_LOSS_WEIGHT" \
    --entropy-weight "$ENTROPY_WEIGHT" \
    --update-schedule ppo-epoch \
    --rule-anchor-fraction "$RULE_ANCHOR_FRACTION" \
    --gradient-diagnostic-batches "$GRADIENT_DIAGNOSTIC_BATCHES" \
    --advantage-normalization "$ADVANTAGE_NORMALIZATION" \
    --value-backprop-scope "$VALUE_BACKPROP_SCOPE" \
    "${ON_POLICY_LIMIT_ARGS[@]}"

  if [[ "$DELETE_ON_POLICY_DATA" != "0" ]]; then
    rm -f "$ON_POLICY_DATASET"
    echo "Deleted consumed on-policy trajectory dataset: $ON_POLICY_DATASET"
  fi

  evaluate_generation "$generation" "$CURRENT_DIR"
done
