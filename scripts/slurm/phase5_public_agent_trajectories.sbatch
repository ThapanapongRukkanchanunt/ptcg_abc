#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-public-traj
#SBATCH --output=experiments/rl/slurm-%j-phase5-public-traj.out
#SBATCH --error=experiments/rl/slurm-%j-phase5-public-traj.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

set -euo pipefail

module purge
module load miniconda3/3.12

cd "$SLURM_SUBMIT_DIR"
mkdir -p experiments/rl reports

GAME_DATA_ROOT="${GAME_DATA_ROOT:-/project/SIGGI/thapanapong.r@cmu.ac.th}"
PUBLIC_AGENT_ROOTS="${PUBLIC_AGENT_ROOTS:-${PUBLIC_AGENT_ROOT:-data/public_agents}}"
PUBLIC_AGENT_KEYS="${PUBLIC_AGENT_KEYS:-}"
CONTROLLED_PUBLIC_AGENT_KEY="${CONTROLLED_PUBLIC_AGENT_KEY:-}"
CONTROLLED_DECK_INDEX="${CONTROLLED_DECK_INDEX:-101}"
ROSTER_NOTEBOOK="${ROSTER_NOTEBOOK:-}"
MODEL="${MODEL:-models/rl/phase5_generalist_policy_13deck_10k.pt}"
SPECIALIST_MODEL_DIR="${SPECIALIST_MODEL_DIR:-models/rl/phase5_league_alpha/iter-0005/specialists}"
AGENT="${AGENT:-phase5-rl}"
POLICY_EPSILON="${POLICY_EPSILON:-0.0}"
DECK_INDICES="${DECK_INDICES:-}"
GAMES_PER_MATCHUP="${GAMES_PER_MATCHUP:-10}"
MAX_STEPS="${MAX_STEPS:-600}"
GAME_OFFSET="${GAME_OFFSET:-0}"
REQUIRE_MIN_OPPONENTS="${REQUIRE_MIN_OPPONENTS:-1}"
OUTCOME_REWARD_SCALE="${OUTCOME_REWARD_SCALE:-1.0}"
TACTICAL_REWARD_MODE="${TACTICAL_REWARD_MODE:-none}"
TACTICAL_ATTACK_BONUS="${TACTICAL_ATTACK_BONUS:-0.10}"
TACTICAL_ATTACH_BONUS="${TACTICAL_ATTACH_BONUS:-0.06}"
TACTICAL_MISSED_ATTACK_PENALTY="${TACTICAL_MISSED_ATTACK_PENALTY:--0.10}"
TACTICAL_MISSED_ATTACH_PENALTY="${TACTICAL_MISSED_ATTACH_PENALTY:--0.06}"
OUTPUT="${OUTPUT:-${GAME_DATA_ROOT}/phase5_public_agent_rule_train/phase5_public_agent_trajectories.jsonl}"
REPORT_JSON="${REPORT_JSON:-experiments/rl/phase5_public_agent_trajectories_report.json}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:-}"

export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
PY="$SLURM_SUBMIT_DIR/.conda_ptcg/bin/python"

echo "Phase 5 public-agent trajectory generation"
echo "GAME_DATA_ROOT=$GAME_DATA_ROOT"
echo "PUBLIC_AGENT_ROOTS=$PUBLIC_AGENT_ROOTS"
echo "PUBLIC_AGENT_KEYS=$PUBLIC_AGENT_KEYS"
echo "CONTROLLED_PUBLIC_AGENT_KEY=$CONTROLLED_PUBLIC_AGENT_KEY"
echo "CONTROLLED_DECK_INDEX=$CONTROLLED_DECK_INDEX"
echo "ROSTER_NOTEBOOK=$ROSTER_NOTEBOOK"
echo "AGENT=$AGENT"
echo "POLICY_EPSILON=$POLICY_EPSILON"
echo "MODEL=$MODEL"
echo "SPECIALIST_MODEL_DIR=$SPECIALIST_MODEL_DIR"
echo "DECK_INDICES=$DECK_INDICES"
echo "GAMES_PER_MATCHUP=$GAMES_PER_MATCHUP"
echo "MAX_STEPS=$MAX_STEPS"
echo "GAME_OFFSET=$GAME_OFFSET"
echo "REQUIRE_MIN_OPPONENTS=$REQUIRE_MIN_OPPONENTS"
echo "OUTCOME_REWARD_SCALE=$OUTCOME_REWARD_SCALE"
echo "TACTICAL_REWARD_MODE=$TACTICAL_REWARD_MODE"
echo "TACTICAL_ATTACK_BONUS=$TACTICAL_ATTACK_BONUS"
echo "TACTICAL_ATTACH_BONUS=$TACTICAL_ATTACH_BONUS"
echo "TACTICAL_MISSED_ATTACK_PENALTY=$TACTICAL_MISSED_ATTACK_PENALTY"
echo "TACTICAL_MISSED_ATTACH_PENALTY=$TACTICAL_MISSED_ATTACH_PENALTY"
echo "OUTPUT=$OUTPUT"
echo "REPORT_JSON=$REPORT_JSON"

EXTRA_ARGS=()
ROOT_ARGS=()
KEY_ARGS=()
DECK_ARGS=()
for root in $PUBLIC_AGENT_ROOTS; do
  ROOT_ARGS+=(--public-agent-root "$root")
done
for key in $PUBLIC_AGENT_KEYS; do
  KEY_ARGS+=(--public-agent-key "$key")
done
for index in $DECK_INDICES; do
  DECK_ARGS+=(--deck-index "$index")
done
if [[ -n "$ROSTER_NOTEBOOK" ]]; then
  EXTRA_ARGS+=(--roster-notebook "$ROSTER_NOTEBOOK")
fi
if [[ -n "$CONTROLLED_PUBLIC_AGENT_KEY" ]]; then
  EXTRA_ARGS+=(--controlled-public-agent-key "$CONTROLLED_PUBLIC_AGENT_KEY")
  EXTRA_ARGS+=(--controlled-deck-index "$CONTROLLED_DECK_INDEX")
fi
if [[ -n "$SPECIALIST_MODEL_DIR" && "$AGENT" != "rule" ]]; then
  EXTRA_ARGS+=(--specialist-model-dir "$SPECIALIST_MODEL_DIR")
fi
if [[ "$AGENT" == "phase5-epsilon" ]]; then
  EXTRA_ARGS+=(--policy-epsilon "$POLICY_EPSILON")
fi
if [[ -n "$OVERWRITE_FLAG" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

"$PY" -m ptcg_abc rl-generate-phase5-public-agent-trajectories \
  "${ROOT_ARGS[@]}" \
  "${KEY_ARGS[@]}" \
  "${DECK_ARGS[@]}" \
  --agent "$AGENT" \
  --model "$MODEL" \
  --games-per-matchup "$GAMES_PER_MATCHUP" \
  --max-steps "$MAX_STEPS" \
  --game-offset "$GAME_OFFSET" \
  --require-min-opponents "$REQUIRE_MIN_OPPONENTS" \
  --outcome-reward-scale "$OUTCOME_REWARD_SCALE" \
  --tactical-reward-mode "$TACTICAL_REWARD_MODE" \
  --tactical-attack-bonus "$TACTICAL_ATTACK_BONUS" \
  --tactical-attach-bonus "$TACTICAL_ATTACH_BONUS" \
  --tactical-missed-attack-penalty "$TACTICAL_MISSED_ATTACK_PENALTY" \
  --tactical-missed-attach-penalty "$TACTICAL_MISSED_ATTACH_PENALTY" \
  --output "$OUTPUT" \
  --report-json "$REPORT_JSON" \
  "${EXTRA_ARGS[@]}"
