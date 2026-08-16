#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-last10-eval
#SBATCH --output=experiments/rl/slurm-%j-phase5-last10-reward50.out
#SBATCH --error=experiments/rl/slurm-%j-phase5-last10-reward50.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

module purge
module load miniconda3/3.12

cd "$SLURM_SUBMIT_DIR"
mkdir -p experiments/rl reports

export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
PY="$SLURM_SUBMIT_DIR/.conda_ptcg/bin/python"

GAME_DATA_ROOT="${GAME_DATA_ROOT:-/project/SIGGI/thapanapong.r@cmu.ac.th}"
PUBLIC_AGENT_ROOTS="${PUBLIC_AGENT_ROOTS:-${GAME_DATA_ROOT}/phase5_public_agents}"
MODEL_ROOT="${MODEL_ROOT:?Set MODEL_ROOT}"
CONTROLLED_DECK_INDEX="${CONTROLLED_DECK_INDEX:?Set CONTROLLED_DECK_INDEX}"
START_GENERATION="${START_GENERATION:?Set START_GENERATION}"
END_GENERATION="${END_GENERATION:?Set END_GENERATION}"
REPORT_PREFIX="${REPORT_PREFIX:?Set REPORT_PREFIX}"
OPPONENT_POOL="${OPPONENT_POOL:-league-rule}"
GAMES_PER_MATCHUP="${GAMES_PER_MATCHUP:-8}"
MAX_STEPS="${MAX_STEPS:-600}"
GAME_SEED="${GAME_SEED:-20260827}"
PROTECT_SETUP_MEOWTH="${PROTECT_SETUP_MEOWTH:-0}"

if (( START_GENERATION < 0 || END_GENERATION < START_GENERATION )); then
  echo "Require 0 <= START_GENERATION <= END_GENERATION" >&2
  exit 2
fi

ROOT_ARGS=()
for root in $PUBLIC_AGENT_ROOTS; do
  ROOT_ARGS+=(--public-agent-root "$root")
done

CONTROLLED_ARGS=(--deck-index "$CONTROLLED_DECK_INDEX")
SEARCH_ARGS=(
  --search-top-k 4
  --tactical-score-weight 0.0
  --rule-prior-weight 0.0
  --leaf-state-value-weight 0.5
  --handcrafted-reward-weight 0.5
  --handcrafted-reward-deck-index "$CONTROLLED_DECK_INDEX"
  --handcrafted-reward-gamma 0.97
  --value-normalization-epsilon 0.000001
  --terminal-outcome-guard
  --search-setup-active
)
if [[ "$PROTECT_SETUP_MEOWTH" != "0" ]]; then
  SEARCH_ARGS+=(--protect-setup-meowth)
fi

echo "Phase 5 last-10 reward50 evaluation"
echo "MODEL_ROOT=$MODEL_ROOT"
echo "CONTROLLED_DECK_INDEX=$CONTROLLED_DECK_INDEX"
echo "START_GENERATION=$START_GENERATION"
echo "END_GENERATION=$END_GENERATION"
echo "REPORT_PREFIX=$REPORT_PREFIX"
echo "GAMES_PER_MATCHUP=$GAMES_PER_MATCHUP"
echo "GAME_SEED=$GAME_SEED"
echo "PROTECT_SETUP_MEOWTH=$PROTECT_SETUP_MEOWTH"

for generation in $(seq "$START_GENERATION" "$END_GENERATION"); do
  padded="$(printf '%04d' "$generation")"
  specialist_dir="${MODEL_ROOT}/gen-${padded}/specialists"
  checkpoint="${specialist_dir}/deck-$(printf '%02d' "$CONTROLLED_DECK_INDEX").pt"
  if [[ ! -s "$checkpoint" ]]; then
    echo "Missing completed checkpoint: $checkpoint" >&2
    exit 3
  fi

  report_json="reports/${REPORT_PREFIX}_gen-${padded}_104g.json"
  report_md="reports/${REPORT_PREFIX}_gen-${padded}_104g.md"
  status_json="reports/${REPORT_PREFIX}_gen-${padded}_status.json"
  echo "Evaluating generation $generation"
  "$PY" -m ptcg_abc rl-evaluate-phase5-public-agents \
    "${ROOT_ARGS[@]}" \
    --opponent-pool "$OPPONENT_POOL" \
    "${CONTROLLED_ARGS[@]}" \
    --agent phase5-search \
    --specialist-model-dir "$specialist_dir" \
    "${SEARCH_ARGS[@]}" \
    --games-per-matchup "$GAMES_PER_MATCHUP" \
    --max-steps "$MAX_STEPS" \
    --game-seed "$GAME_SEED" \
    --prize-discount-gamma 0.97 \
    --require-min-opponents 1 \
    --report-json "$report_json" \
    --report-md "$report_md" \
    --status-json "$status_json"
done

echo "Completed generations $START_GENERATION through $END_GENERATION"
