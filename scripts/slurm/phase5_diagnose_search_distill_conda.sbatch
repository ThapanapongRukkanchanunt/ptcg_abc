#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-diag
#SBATCH --output=experiments/rl/slurm-%j-phase5-diag.out
#SBATCH --error=experiments/rl/slurm-%j-phase5-diag.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

set -euo pipefail

module purge
module load miniconda3/3.12

cd "$SLURM_SUBMIT_DIR"
mkdir -p experiments/rl reports

export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
PY="$SLURM_SUBMIT_DIR/.conda_ptcg/bin/python"

DATASET="${DATASET:?Set DATASET to the Phase 5 decision JSONL path.}"
MODEL="${MODEL:-models/rl/phase5_search_distill.json}"
CHECKPOINT="${CHECKPOINT:-}"
TRACE_INPUT="${TRACE_INPUT:-}"
REPORT_JSON="${REPORT_JSON:-reports/phase5_search_distill_diagnostics.json}"
REPORT_MD="${REPORT_MD:-reports/phase5_search_distill_diagnostics.md}"
EXAMPLES="${EXAMPLES:-20}"
export REPORT_JSON

TRACE_ARGS=()
if [[ -n "$TRACE_INPUT" ]]; then
  TRACE_ARGS+=(--trace-input "$TRACE_INPUT")
fi
MODEL_ARGS=(--model "$MODEL")
if [[ -n "$CHECKPOINT" ]]; then
  MODEL_ARGS+=(--checkpoint "$CHECKPOINT")
fi

"$PY" --version
"$PY" -c "import sys; print(sys.executable)"

"$PY" -m ptcg_abc rl-diagnose-search-distill \
  --dataset "$DATASET" \
  "${MODEL_ARGS[@]}" \
  "${TRACE_ARGS[@]}" \
  --examples "$EXAMPLES" \
  --report-json "$REPORT_JSON" \
  --report-md "$REPORT_MD"

"$PY" - <<'PY'
import json
import os

path = os.environ["REPORT_JSON"]
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)

for section in ["overall", "search_changed", "search_unchanged"]:
    stats = data[section]
    print(section)
    for key in [
        "frames",
        "changed",
        "search_hit_rate",
        "baseline_hit_rate",
        "mean_model_search_minus_baseline_score",
        "mean_rule_search_minus_baseline_score",
        "mean_model_score_range",
        "model_score_flat_rate",
    ]:
        print(" ", key, stats.get(key))

trace = data.get("trace") or {}
print("trace")
for key in [
    "records",
    "changed_records",
    "search_errors",
    "candidate_errors",
    "truncated_candidates",
    "mean_search_minus_baseline_combined_score",
    "mean_search_minus_baseline_tactical_score",
]:
    print(" ", key, trace.get(key))

print("recommendations")
for recommendation in data["recommendations"]:
    print("-", recommendation)
PY
