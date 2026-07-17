#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-1deck-mix
#SBATCH --output=experiments/rl/slurm-%j-phase5-one-deck-mixed.out
#SBATCH --error=experiments/rl/slurm-%j-phase5-one-deck-mixed.err
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
RUN_NAME="${RUN_NAME:-phase5_dragapult_vs_lucario_mixed}"
CONTROLLED_PUBLIC_AGENT_KEY="${CONTROLLED_PUBLIC_AGENT_KEY:-sample_dragapult}"
OPPONENT_PUBLIC_AGENT_KEYS="${OPPONENT_PUBLIC_AGENT_KEYS:-sample_lucario}"
CONTROLLED_DECK_INDEX="${CONTROLLED_DECK_INDEX:-101}"
GENERATIONS="${GENERATIONS:-10}"
TRAIN_GAMES_PER_GENERATION="${TRAIN_GAMES_PER_GENERATION:-1000}"
RULE_BOOTSTRAP_GAMES="${RULE_BOOTSTRAP_GAMES:-$TRAIN_GAMES_PER_GENERATION}"
EVAL_GAMES_PER_GENERATION="${EVAL_GAMES_PER_GENERATION:-100}"
EPSILON_START="${EPSILON_START:-1.0}"
EPSILON_END="${EPSILON_END:-0.10}"
MAX_STEPS="${MAX_STEPS:-600}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.00005}"
CLIP_EPSILON="${CLIP_EPSILON:-0.2}"
POLICY_LOSS_WEIGHT="${POLICY_LOSS_WEIGHT:-1.0}"
VALUE_LOSS_WEIGHT="${VALUE_LOSS_WEIGHT:-0.5}"
ENTROPY_WEIGHT="${ENTROPY_WEIGHT:-0.01}"
OUTCOME_REWARD_SCALE="${OUTCOME_REWARD_SCALE:-1.0}"
TACTICAL_REWARD_MODE="${TACTICAL_REWARD_MODE:-none}"
TACTICAL_FRACTIONAL_PRIZE_WEIGHT="${TACTICAL_FRACTIONAL_PRIZE_WEIGHT:-1.0}"
TACTICAL_FRACTIONAL_OPPONENT_WEIGHT="${TACTICAL_FRACTIONAL_OPPONENT_WEIGHT:-1.0}"
MODEL_ROOT="${MODEL_ROOT:-models/rl/phase5_one_deck_public_mixed/${RUN_NAME}}"
REPORT_ROOT="${REPORT_ROOT:-experiments/rl/phase5_one_deck_public_mixed/${RUN_NAME}}"
RAW_ROOT="${RAW_ROOT:-${GAME_DATA_ROOT}/phase5_one_deck_public_mixed/${RUN_NAME}}"
OVERWRITE_SCRATCH="${OVERWRITE_SCRATCH:-0}"
REGENERATE_RULE_BOOTSTRAP="${REGENERATE_RULE_BOOTSTRAP:-0}"
REUSE_EXISTING_TRAJECTORIES="${REUSE_EXISTING_TRAJECTORIES:-0}"
REQUIRE_ON_POLICY="${REQUIRE_ON_POLICY:-0}"

echo "Phase 5 one-deck public mixed rule+epsilon curriculum"
echo "GAME_DATA_ROOT=$GAME_DATA_ROOT"
echo "PUBLIC_AGENT_ROOTS=$PUBLIC_AGENT_ROOTS"
echo "RUN_NAME=$RUN_NAME"
echo "CONTROLLED_PUBLIC_AGENT_KEY=$CONTROLLED_PUBLIC_AGENT_KEY"
echo "OPPONENT_PUBLIC_AGENT_KEYS=$OPPONENT_PUBLIC_AGENT_KEYS"
echo "CONTROLLED_DECK_INDEX=$CONTROLLED_DECK_INDEX"
echo "GENERATIONS=$GENERATIONS"
echo "RULE_BOOTSTRAP_GAMES=$RULE_BOOTSTRAP_GAMES"
echo "TRAIN_GAMES_PER_GENERATION=$TRAIN_GAMES_PER_GENERATION"
echo "EVAL_GAMES_PER_GENERATION=$EVAL_GAMES_PER_GENERATION"
echo "EPSILON_START=$EPSILON_START"
echo "EPSILON_END=$EPSILON_END"
echo "MAX_STEPS=$MAX_STEPS"
echo "MODEL_ROOT=$MODEL_ROOT"
echo "REPORT_ROOT=$REPORT_ROOT"
echo "RAW_ROOT=$RAW_ROOT"
echo "OUTCOME_REWARD_SCALE=$OUTCOME_REWARD_SCALE"
echo "TACTICAL_REWARD_MODE=$TACTICAL_REWARD_MODE"
echo "TACTICAL_FRACTIONAL_PRIZE_WEIGHT=$TACTICAL_FRACTIONAL_PRIZE_WEIGHT"
echo "TACTICAL_FRACTIONAL_OPPONENT_WEIGHT=$TACTICAL_FRACTIONAL_OPPONENT_WEIGHT"
echo "REGENERATE_RULE_BOOTSTRAP=$REGENERATE_RULE_BOOTSTRAP"
echo "REUSE_EXISTING_TRAJECTORIES=$REUSE_EXISTING_TRAJECTORIES"
echo "REQUIRE_ON_POLICY=$REQUIRE_ON_POLICY"

ROOT_ARGS=()
for root in $PUBLIC_AGENT_ROOTS; do
  ROOT_ARGS+=(--public-agent-root "$root")
done

OPPONENT_ARGS=()
for key in $OPPONENT_PUBLIC_AGENT_KEYS; do
  OPPONENT_ARGS+=(--public-agent-key "$key")
done

generation_dir() {
  local generation="$1"
  local padded
  padded="$(printf "%04d" "$generation")"
  echo "${MODEL_ROOT}/gen-${padded}/specialists"
}

report_generation_dir() {
  local generation="$1"
  local padded
  padded="$(printf "%04d" "$generation")"
  echo "${REPORT_ROOT}/gen-${padded}"
}

raw_generation_dir() {
  local generation="$1"
  local padded
  padded="$(printf "%04d" "$generation")"
  echo "${RAW_ROOT}/generations/gen-${padded}/raw_train"
}

epsilon_for_generation() {
  local generation="$1"
  "$PY" -c "start=float('$EPSILON_START'); end=float('$EPSILON_END'); generations=max(1,int('$GENERATIONS')); generation=int('$generation'); print(end if generations == 1 else start + (end - start) * ((generation - 1) / (generations - 1)))"
}

GEN0_DIR="$(generation_dir 0)"
GEN0_REPORT_DIR="$(report_generation_dir 0)"
mkdir -p "$GEN0_DIR" "$GEN0_REPORT_DIR"
SCRATCH_CHECKPOINT="${GEN0_DIR}/deck-$(printf "%02d" "$CONTROLLED_DECK_INDEX").pt"
SCRATCH_REPORT="${GEN0_REPORT_DIR}/scratch_init_report.json"
INIT_ARGS=()
if [[ "$OVERWRITE_SCRATCH" != "0" ]]; then
  INIT_ARGS+=(--overwrite)
fi
if [[ ! -f "$SCRATCH_CHECKPOINT" || "$OVERWRITE_SCRATCH" != "0" ]]; then
  "$PY" -m ptcg_abc rl-init-phase5-policy-checkpoint \
    --checkpoint "$SCRATCH_CHECKPOINT" \
    --report-json "$SCRATCH_REPORT" \
    --deck-index "$CONTROLLED_DECK_INDEX" \
    --controlled-public-agent-key "$CONTROLLED_PUBLIC_AGENT_KEY" \
    --experiment "$RUN_NAME" \
    "${INIT_ARGS[@]}"
fi

RULE_BOOTSTRAP_DIR="${RAW_ROOT}/rule_bootstrap"
RULE_BOOTSTRAP_DATASET="${RULE_BOOTSTRAP_DIR}/phase5_public_rule_bootstrap_gen-0000.jsonl"
RULE_BOOTSTRAP_REPORT="${GEN0_REPORT_DIR}/rule_bootstrap_report.json"
mkdir -p "$RULE_BOOTSTRAP_DIR"

echo "Generation 0 rule bootstrap"
echo "RULE_BOOTSTRAP_DATASET=$RULE_BOOTSTRAP_DATASET"
if [[ "$REGENERATE_RULE_BOOTSTRAP" == "0" && -s "$RULE_BOOTSTRAP_DATASET" ]]; then
  echo "Keeping existing rule bootstrap dataset: $RULE_BOOTSTRAP_DATASET"
else
  "$PY" -m ptcg_abc rl-generate-phase5-public-agent-trajectories \
    "${ROOT_ARGS[@]}" \
    "${OPPONENT_ARGS[@]}" \
    --controlled-public-agent-key "$CONTROLLED_PUBLIC_AGENT_KEY" \
    --controlled-deck-index "$CONTROLLED_DECK_INDEX" \
    --agent rule \
    --games-per-matchup "$RULE_BOOTSTRAP_GAMES" \
    --max-steps "$MAX_STEPS" \
    --require-min-opponents 1 \
    --outcome-reward-scale "$OUTCOME_REWARD_SCALE" \
    --tactical-reward-mode "$TACTICAL_REWARD_MODE" \
    --tactical-fractional-prize-weight "$TACTICAL_FRACTIONAL_PRIZE_WEIGHT" \
    --tactical-fractional-opponent-weight "$TACTICAL_FRACTIONAL_OPPONENT_WEIGHT" \
    --output "$RULE_BOOTSTRAP_DATASET" \
    --report-json "$RULE_BOOTSTRAP_REPORT" \
    --overwrite
fi

PPO_ON_POLICY_ARGS=()
if [[ "$REQUIRE_ON_POLICY" != "0" ]]; then
  PPO_ON_POLICY_ARGS+=(--require-on-policy)
fi

for generation in $(seq 1 "$GENERATIONS"); do
  padded="$(printf "%04d" "$generation")"
  previous_generation="$((generation - 1))"
  PREVIOUS_DIR="$(generation_dir "$previous_generation")"
  CURRENT_DIR="$(generation_dir "$generation")"
  GEN_REPORT_DIR="$(report_generation_dir "$generation")"
  GEN_RAW_DIR="$(raw_generation_dir "$generation")"
  mkdir -p "$CURRENT_DIR" "$GEN_REPORT_DIR" "$GEN_RAW_DIR"

  EPSILON="$(epsilon_for_generation "$generation")"
  EPSILON_TRAJECTORY_DATASET="${GEN_RAW_DIR}/phase5_public_epsilon_gen-${padded}.jsonl"
  EPSILON_TRAJECTORY_REPORT="${GEN_REPORT_DIR}/epsilon_trajectory_report.json"
  PPO_REPORT_JSON="${GEN_REPORT_DIR}/ppo_report.json"
  EVAL_JSON="reports/${RUN_NAME}_gen-${padded}_eval_${EVAL_GAMES_PER_GENERATION}g.json"
  EVAL_MD="reports/${RUN_NAME}_gen-${padded}_eval_${EVAL_GAMES_PER_GENERATION}g.md"
  STATUS_JSON="reports/${RUN_NAME}_gen-${padded}_status.json"
  REPLAY_DIR="${GEN_REPORT_DIR}/replays"

  echo "Generation $generation / $GENERATIONS"
  echo "EPSILON=$EPSILON"
  echo "PREVIOUS_DIR=$PREVIOUS_DIR"
  echo "CURRENT_DIR=$CURRENT_DIR"
  echo "RULE_BOOTSTRAP_DATASET=$RULE_BOOTSTRAP_DATASET"
  echo "EPSILON_TRAJECTORY_DATASET=$EPSILON_TRAJECTORY_DATASET"

  if [[ "$REUSE_EXISTING_TRAJECTORIES" != "0" && -s "$EPSILON_TRAJECTORY_DATASET" ]]; then
    echo "Reusing existing epsilon trajectory dataset: $EPSILON_TRAJECTORY_DATASET"
  else
    "$PY" -m ptcg_abc rl-generate-phase5-public-agent-trajectories \
      "${ROOT_ARGS[@]}" \
      "${OPPONENT_ARGS[@]}" \
      --controlled-public-agent-key "$CONTROLLED_PUBLIC_AGENT_KEY" \
      --controlled-deck-index "$CONTROLLED_DECK_INDEX" \
      --agent phase5-epsilon \
      --policy-epsilon "$EPSILON" \
      --specialist-model-dir "$PREVIOUS_DIR" \
      --games-per-matchup "$TRAIN_GAMES_PER_GENERATION" \
      --max-steps "$MAX_STEPS" \
      --require-min-opponents 1 \
      --outcome-reward-scale "$OUTCOME_REWARD_SCALE" \
      --tactical-reward-mode "$TACTICAL_REWARD_MODE" \
      --tactical-fractional-prize-weight "$TACTICAL_FRACTIONAL_PRIZE_WEIGHT" \
      --tactical-fractional-opponent-weight "$TACTICAL_FRACTIONAL_OPPONENT_WEIGHT" \
      --output "$EPSILON_TRAJECTORY_DATASET" \
      --report-json "$EPSILON_TRAJECTORY_REPORT" \
      --overwrite
  fi

  "$PY" -m ptcg_abc rl-train-phase5-ppo \
    --trajectory-dataset "$RULE_BOOTSTRAP_DATASET" \
    --trajectory-dataset "$EPSILON_TRAJECTORY_DATASET" \
    --checkpoint "${PREVIOUS_DIR}/deck-$(printf "%02d" "$CONTROLLED_DECK_INDEX").pt" \
    --output-checkpoint "${CURRENT_DIR}/deck-$(printf "%02d" "$CONTROLLED_DECK_INDEX").pt" \
    --report-json "$PPO_REPORT_JSON" \
    --deck-index-filter "$CONTROLLED_DECK_INDEX" \
    "${PPO_ON_POLICY_ARGS[@]}" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --clip-epsilon "$CLIP_EPSILON" \
    --policy-loss-weight "$POLICY_LOSS_WEIGHT" \
    --value-loss-weight "$VALUE_LOSS_WEIGHT" \
    --entropy-weight "$ENTROPY_WEIGHT"

  rm -f "$EPSILON_TRAJECTORY_DATASET"

  "$PY" -m ptcg_abc rl-evaluate-phase5-public-agents \
    "${ROOT_ARGS[@]}" \
    "${OPPONENT_ARGS[@]}" \
    --controlled-public-agent-key "$CONTROLLED_PUBLIC_AGENT_KEY" \
    --controlled-deck-index "$CONTROLLED_DECK_INDEX" \
    --agent phase5-symbolic \
    --specialist-model-dir "$CURRENT_DIR" \
    --games-per-matchup "$EVAL_GAMES_PER_GENERATION" \
    --max-steps "$MAX_STEPS" \
    --require-min-opponents 1 \
    --replay-output-dir "$REPLAY_DIR" \
    --saved-win-replays 1 \
    --saved-loss-replays 1 \
    --replay-trace-limit 160 \
    --report-json "$EVAL_JSON" \
    --report-md "$EVAL_MD" \
    --status-json "$STATUS_JSON"
done
