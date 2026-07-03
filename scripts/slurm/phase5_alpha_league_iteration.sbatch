#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-alpha-league
#SBATCH --output=experiments/rl/slurm-%j-phase5-alpha-league.out
#SBATCH --error=experiments/rl/slurm-%j-phase5-alpha-league.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

set -euo pipefail

module purge
module load miniconda3/3.12

cd "$SLURM_SUBMIT_DIR"

GAME_DATA_ROOT="${GAME_DATA_ROOT:-/project/SIGGI/thapanapong.r@cmu.ac.th}"
LEAGUE_ROOT="${LEAGUE_ROOT:-${GAME_DATA_ROOT}/phase5_league_alpha}"
ITERATION="${ITERATION:-1}"
if [[ -z "${SOURCE_ITERATION:-}" ]]; then
  if (( ITERATION > 0 )); then
    SOURCE_ITERATION="$((ITERATION - 1))"
  else
    SOURCE_ITERATION="0"
  fi
fi
ITERATION_PADDED="$(printf "%04d" "$ITERATION")"
SOURCE_ITERATION_PADDED="$(printf "%04d" "$SOURCE_ITERATION")"
ITERATION_DIR="${ITERATION_DIR:-${LEAGUE_ROOT}/iterations/iter-${ITERATION_PADDED}}"
SPECIALIST_MODEL_DIR="${SPECIALIST_MODEL_DIR:-models/rl/phase5_league_alpha/iter-${SOURCE_ITERATION_PADDED}/specialists}"
REPORT_JSON="${REPORT_JSON:-experiments/rl/phase5_league_alpha/iter-${ITERATION_PADDED}_league_iteration_report.json}"
GAMES_PER_DECK="${GAMES_PER_DECK:-100}"
GAME_OFFSET="${GAME_OFFSET:-$((ITERATION * GAMES_PER_DECK * 13))}"
MAX_STEPS="${MAX_STEPS:-600}"
AGENT="${AGENT:-phase5-full}"
DECK_INDICES="${DECK_INDICES:-1 2 3 4 5 6 7 8 9 10 11 12 13}"
SEARCH_TRACE_OUTPUT="${SEARCH_TRACE_OUTPUT:-}"
SEARCH_TRACE_GAMES="${SEARCH_TRACE_GAMES:-3}"
SEARCH_TOP_K="${SEARCH_TOP_K:-}"
SEARCH_ROLLOUT_STEPS="${SEARCH_ROLLOUT_STEPS:-}"
POLICY_PRIOR_WEIGHT="${POLICY_PRIOR_WEIGHT:-}"
NEURAL_ACTION_VALUE_WEIGHT="${NEURAL_ACTION_VALUE_WEIGHT:-}"
NEURAL_TACTICAL_WEIGHT="${NEURAL_TACTICAL_WEIGHT:-}"

mkdir -p experiments/rl/phase5_league_alpha "$LEAGUE_ROOT"

export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
PY="$SLURM_SUBMIT_DIR/.conda_ptcg/bin/python"

echo "Phase 5 Alpha learned-agent league iteration"
echo "LEAGUE_ROOT=$LEAGUE_ROOT"
echo "ITERATION=$ITERATION"
echo "SOURCE_ITERATION=$SOURCE_ITERATION"
echo "ITERATION_DIR=$ITERATION_DIR"
echo "SPECIALIST_MODEL_DIR=$SPECIALIST_MODEL_DIR"
echo "AGENT=$AGENT"
echo "GAMES_PER_DECK=$GAMES_PER_DECK"
echo "GAME_OFFSET=$GAME_OFFSET"
echo "MAX_STEPS=$MAX_STEPS"
echo "DECK_INDICES=$DECK_INDICES"
echo "REPORT_JSON=$REPORT_JSON"

DECK_ARGS=()
for index in $DECK_INDICES; do
  DECK_ARGS+=(--deck-index "$index")
done

EXTRA_ARGS=()
if [[ -n "$SEARCH_TRACE_OUTPUT" ]]; then
  EXTRA_ARGS+=(--search-trace-output "$SEARCH_TRACE_OUTPUT")
  EXTRA_ARGS+=(--search-trace-games "$SEARCH_TRACE_GAMES")
fi
if [[ -n "$SEARCH_TOP_K" ]]; then
  EXTRA_ARGS+=(--search-top-k "$SEARCH_TOP_K")
fi
if [[ -n "$SEARCH_ROLLOUT_STEPS" ]]; then
  EXTRA_ARGS+=(--search-rollout-steps "$SEARCH_ROLLOUT_STEPS")
fi
if [[ -n "$POLICY_PRIOR_WEIGHT" ]]; then
  EXTRA_ARGS+=(--policy-prior-weight "$POLICY_PRIOR_WEIGHT")
fi
if [[ -n "$NEURAL_ACTION_VALUE_WEIGHT" ]]; then
  EXTRA_ARGS+=(--neural-action-value-weight "$NEURAL_ACTION_VALUE_WEIGHT")
fi
if [[ -n "$NEURAL_TACTICAL_WEIGHT" ]]; then
  EXTRA_ARGS+=(--neural-tactical-weight "$NEURAL_TACTICAL_WEIGHT")
fi

"$PY" -m ptcg_abc rl-generate-phase5-alpha-league-iteration \
  --iteration "$ITERATION" \
  --league-root "$LEAGUE_ROOT" \
  --iteration-dir "$ITERATION_DIR" \
  --specialist-model-dir "$SPECIALIST_MODEL_DIR" \
  --agent "$AGENT" \
  --games-per-deck "$GAMES_PER_DECK" \
  --game-offset "$GAME_OFFSET" \
  --max-steps "$MAX_STEPS" \
  --report-json "$REPORT_JSON" \
  "${DECK_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
