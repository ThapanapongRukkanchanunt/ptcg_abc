#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-alpha-clean
#SBATCH --output=experiments/rl/slurm-%j-phase5-alpha-clean.out
#SBATCH --error=experiments/rl/slurm-%j-phase5-alpha-clean.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

module purge
module load miniconda3/3.12

cd "$SLURM_SUBMIT_DIR"

GAME_DATA_ROOT="${GAME_DATA_ROOT:-/project/SIGGI/thapanapong.r@cmu.ac.th}"
LEAGUE_ROOT="${LEAGUE_ROOT:-${GAME_DATA_ROOT}/phase5_league_alpha}"
ITERATION="${ITERATION:-0}"
ITERATION_PADDED="$(printf "%04d" "$ITERATION")"
ITERATION_DIR="${ITERATION_DIR:-${LEAGUE_ROOT}/iterations/iter-${ITERATION_PADDED}}"
UPDATE_REPORT="${UPDATE_REPORT:-experiments/rl/phase5_league_alpha/iter-${ITERATION_PADDED}_deck_specialists_report.json}"
CLEANUP_REPORT="${CLEANUP_REPORT:-experiments/rl/phase5_league_alpha/iter-${ITERATION_PADDED}_cleanup_report.json}"

export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
PY="$SLURM_SUBMIT_DIR/.conda_ptcg/bin/python"

echo "Phase 5 Alpha raw-data cleanup"
echo "ITERATION_DIR=$ITERATION_DIR"
echo "UPDATE_REPORT=$UPDATE_REPORT"
echo "CLEANUP_REPORT=$CLEANUP_REPORT"

"$PY" -m ptcg_abc rl-clean-phase5-alpha-iteration \
  --iteration-dir "$ITERATION_DIR" \
  --update-report "$UPDATE_REPORT" \
  --report-json "$CLEANUP_REPORT"
