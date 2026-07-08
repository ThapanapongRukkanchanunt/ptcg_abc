#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-score-components
#SBATCH --output=experiments/rl/slurm-%j-phase5-score-components.out
#SBATCH --error=experiments/rl/slurm-%j-phase5-score-components.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

module purge
module load miniconda3/3.12

cd "$SLURM_SUBMIT_DIR"
mkdir -p experiments/rl reports

export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
PY="$SLURM_SUBMIT_DIR/.conda_ptcg/bin/python"

TRACE_INPUT="${TRACE_INPUT:-experiments/rl/phase5_public_agent_score_traces.jsonl}"
REPORT_JSON="${REPORT_JSON:-reports/phase5_search_score_components.json}"
REPORT_MD="${REPORT_MD:-reports/phase5_search_score_components.md}"

echo "Phase 5 search score-component diagnostics"
echo "TRACE_INPUT=$TRACE_INPUT"
echo "REPORT_JSON=$REPORT_JSON"
echo "REPORT_MD=$REPORT_MD"

"$PY" -m ptcg_abc rl-diagnose-search-score-components \
  --trace-input "$TRACE_INPUT" \
  --report-json "$REPORT_JSON" \
  --report-md "$REPORT_MD"
