#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-symbolic-diag
#SBATCH --output=experiments/rl/slurm-%j-phase5-symbolic-diag.out
#SBATCH --error=experiments/rl/slurm-%j-phase5-symbolic-diag.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

set -euo pipefail

module purge
module load miniconda3/3.12

cd "$SLURM_SUBMIT_DIR"
mkdir -p experiments/rl reports

export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
PY="$SLURM_SUBMIT_DIR/.conda_ptcg/bin/python"

"$PY" --version
"$PY" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

DATASET="${DATASET:-data/datasets/rl/phase5_search_decisions_merged.jsonl}"
CHECKPOINT="${CHECKPOINT:-models/rl/phase5_symbolic_policy_10shards.pt}"
LIMIT="${LIMIT:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EXAMPLES="${EXAMPLES:-20}"
REPORT_JSON="${REPORT_JSON:-reports/phase5_symbolic_10shards_diagnostics.json}"
REPORT_MD="${REPORT_MD:-reports/phase5_symbolic_10shards_diagnostics.md}"

"$PY" -m ptcg_abc rl-diagnose-phase5-symbolic \
  --dataset "$DATASET" \
  --checkpoint "$CHECKPOINT" \
  --limit "$LIMIT" \
  --batch-size "$BATCH_SIZE" \
  --examples "$EXAMPLES" \
  --report-json "$REPORT_JSON" \
  --report-md "$REPORT_MD"
