#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-league-eval
#SBATCH --output=experiments/rl/slurm-%j-phase5-league-eval.out
#SBATCH --error=experiments/rl/slurm-%j-phase5-league-eval.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

set -euo pipefail

module purge
module load miniconda3/3.12

cd "$SLURM_SUBMIT_DIR"
mkdir -p experiments/rl reports

export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
PY="$SLURM_SUBMIT_DIR/.conda_ptcg/bin/python"

MODEL="${MODEL:-models/rl/phase5_generalist_policy_13deck_10k.pt}"
AGENT="${AGENT:-phase5-search}"
GAMES_PER_MATCHUP="${GAMES_PER_MATCHUP:-2}"
MAX_STEPS="${MAX_STEPS:-600}"
REPORT_JSON="${REPORT_JSON:-reports/phase5_league_benchmark.json}"
REPORT_MD="${REPORT_MD:-reports/phase5_league_benchmark.md}"
SEARCH_TRACE_OUTPUT="${SEARCH_TRACE_OUTPUT:-}"
SEARCH_TOP_K="${SEARCH_TOP_K:-}"
SEARCH_ROLLOUT_STEPS="${SEARCH_ROLLOUT_STEPS:-}"
POLICY_PRIOR_WEIGHT="${POLICY_PRIOR_WEIGHT:-}"
NEURAL_ACTION_VALUE_WEIGHT="${NEURAL_ACTION_VALUE_WEIGHT:-}"
NEURAL_TACTICAL_WEIGHT="${NEURAL_TACTICAL_WEIGHT:-}"

EXTRA_ARGS=()
if [[ -n "$SEARCH_TRACE_OUTPUT" ]]; then
  EXTRA_ARGS+=(--search-trace-output "$SEARCH_TRACE_OUTPUT")
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

"$PY" -m ptcg_abc rl-evaluate-phase5-league \
  --agent "$AGENT" \
  --model "$MODEL" \
  --games-per-matchup "$GAMES_PER_MATCHUP" \
  --max-steps "$MAX_STEPS" \
  --report-json "$REPORT_JSON" \
  --report-md "$REPORT_MD" \
  "${EXTRA_ARGS[@]}"
