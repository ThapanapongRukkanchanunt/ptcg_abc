#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-seq-equiv
#SBATCH --output=experiments/rl/slurm-%j-phase5-sequence-equivalence.out
#SBATCH --error=experiments/rl/slurm-%j-phase5-sequence-equivalence.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

module purge
module load miniconda3/3.12

cd "$SLURM_SUBMIT_DIR"
mkdir -p experiments/rl reports

export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
PY="$SLURM_SUBMIT_DIR/.conda_ptcg/bin/python"

TRACE_INPUT="${TRACE_INPUT:?Set TRACE_INPUT to a schema-v2 search trace JSONL}"
REPORT_JSON="${REPORT_JSON:-reports/phase5_search_sequence_equivalence.json}"
REPORT_MD="${REPORT_MD:-reports/phase5_search_sequence_equivalence.md}"
EXAMPLES="${EXAMPLES:-20}"

echo "Phase 5 search sequence-equivalence diagnostic"
echo "TRACE_INPUT=$TRACE_INPUT"
echo "REPORT_JSON=$REPORT_JSON"
echo "REPORT_MD=$REPORT_MD"
echo "EXAMPLES=$EXAMPLES"

"$PY" -m ptcg_abc rl-diagnose-search-sequence-equivalence \
  --trace-input "$TRACE_INPUT" \
  --examples "$EXAMPLES" \
  --report-json "$REPORT_JSON" \
  --report-md "$REPORT_MD"
