#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-alpha-bootstrap
#SBATCH --output=experiments/rl/slurm-%j-phase5-alpha-bootstrap.out
#SBATCH --error=experiments/rl/slurm-%j-phase5-alpha-bootstrap.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

set -euo pipefail

module purge
module load miniconda3/3.12

cd "$SLURM_SUBMIT_DIR"

GAME_DATA_ROOT="${GAME_DATA_ROOT:-/project/SIGGI/thapanapong.r@cmu.ac.th}"
LEAGUE_ROOT="${LEAGUE_ROOT:-${GAME_DATA_ROOT}/phase5_league_alpha}"
ITERATION="${ITERATION:-0}"
GAMES_PER_PAIR="${GAMES_PER_PAIR:-2}"
MAX_STEPS="${MAX_STEPS:-600}"
DECK_INDICES="${DECK_INDICES:-1 2 3 4 5 6 7 8 9 10 11 12 13}"
ITERATION_PADDED="$(printf "%04d" "$ITERATION")"
ITERATION_DIR="${ITERATION_DIR:-${LEAGUE_ROOT}/iterations/iter-${ITERATION_PADDED}}"
REPORT_JSON="${REPORT_JSON:-experiments/rl/phase5_league_alpha/iter-${ITERATION_PADDED}_rule_bootstrap_report.json}"

mkdir -p experiments/rl/phase5_league_alpha "$LEAGUE_ROOT"

export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
PY="$SLURM_SUBMIT_DIR/.conda_ptcg/bin/python"

echo "Phase 5 Alpha rule bootstrap"
echo "LEAGUE_ROOT=$LEAGUE_ROOT"
echo "ITERATION=$ITERATION"
echo "ITERATION_DIR=$ITERATION_DIR"
echo "GAMES_PER_PAIR=$GAMES_PER_PAIR"
echo "MAX_STEPS=$MAX_STEPS"
echo "DECK_INDICES=$DECK_INDICES"
echo "REPORT_JSON=$REPORT_JSON"

DECK_ARGS=()
for index in $DECK_INDICES; do
  DECK_ARGS+=(--deck-index "$index")
done

"$PY" -m ptcg_abc rl-generate-phase5-alpha-bootstrap \
  --iteration "$ITERATION" \
  --league-root "$LEAGUE_ROOT" \
  --iteration-dir "$ITERATION_DIR" \
  --games-per-pair "$GAMES_PER_PAIR" \
  --max-steps "$MAX_STEPS" \
  --report-json "$REPORT_JSON" \
  "${DECK_ARGS[@]}"
