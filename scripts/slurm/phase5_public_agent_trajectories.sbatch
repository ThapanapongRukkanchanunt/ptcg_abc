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
ROSTER_NOTEBOOK="${ROSTER_NOTEBOOK:-}"
MODEL="${MODEL:-models/rl/phase5_generalist_policy_13deck_10k.pt}"
SPECIALIST_MODEL_DIR="${SPECIALIST_MODEL_DIR:-models/rl/phase5_league_alpha/iter-0005/specialists}"
AGENT="${AGENT:-phase5-rl}"
DECK_INDICES="${DECK_INDICES:-}"
GAMES_PER_MATCHUP="${GAMES_PER_MATCHUP:-10}"
MAX_STEPS="${MAX_STEPS:-600}"
GAME_OFFSET="${GAME_OFFSET:-0}"
REQUIRE_MIN_OPPONENTS="${REQUIRE_MIN_OPPONENTS:-1}"
OUTPUT="${OUTPUT:-${GAME_DATA_ROOT}/phase5_public_agent_rule_train/phase5_public_agent_trajectories.jsonl}"
REPORT_JSON="${REPORT_JSON:-experiments/rl/phase5_public_agent_trajectories_report.json}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:-}"

export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
PY="$SLURM_SUBMIT_DIR/.conda_ptcg/bin/python"

echo "Phase 5 public-agent trajectory generation"
echo "GAME_DATA_ROOT=$GAME_DATA_ROOT"
echo "PUBLIC_AGENT_ROOTS=$PUBLIC_AGENT_ROOTS"
echo "PUBLIC_AGENT_KEYS=$PUBLIC_AGENT_KEYS"
echo "ROSTER_NOTEBOOK=$ROSTER_NOTEBOOK"
echo "AGENT=$AGENT"
echo "MODEL=$MODEL"
echo "SPECIALIST_MODEL_DIR=$SPECIALIST_MODEL_DIR"
echo "DECK_INDICES=$DECK_INDICES"
echo "GAMES_PER_MATCHUP=$GAMES_PER_MATCHUP"
echo "MAX_STEPS=$MAX_STEPS"
echo "GAME_OFFSET=$GAME_OFFSET"
echo "REQUIRE_MIN_OPPONENTS=$REQUIRE_MIN_OPPONENTS"
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
if [[ -n "$SPECIALIST_MODEL_DIR" ]]; then
  EXTRA_ARGS+=(--specialist-model-dir "$SPECIALIST_MODEL_DIR")
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
  --output "$OUTPUT" \
  --report-json "$REPORT_JSON" \
  "${EXTRA_ARGS[@]}"
