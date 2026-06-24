#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-train
#SBATCH --output=experiments/rl/slurm-%j-phase5-train.out
#SBATCH --error=experiments/rl/slurm-%j-phase5-train.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

set -euo pipefail

module purge
module load miniconda3/3.12

cd "$SLURM_SUBMIT_DIR"
mkdir -p experiments/rl data/datasets/rl models/rl reports

export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
PY="$SLURM_SUBMIT_DIR/.conda_ptcg/bin/python"

"$PY" --version
"$PY" -c "import sys; print(sys.executable)"
"$PY" - <<'PY'
import torch
from ptcg_abc.rl.torch_backend import CHECKPOINT_FORMAT

print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("checkpoint_format", CHECKPOINT_FORMAT)
PY

DECISION_GLOB="${DECISION_GLOB:-data/datasets/rl/phase5_search/shards/phase5_search_decisions_shard-*.jsonl}"
TRACE_GLOB="${TRACE_GLOB:-experiments/rl/phase5_search/traces/phase5_search_traces_shard-*.jsonl}"
MERGED_DATASET="${MERGED_DATASET:-data/datasets/rl/phase5_search_decisions_merged.jsonl}"
MERGED_TRACES="${MERGED_TRACES:-experiments/rl/phase5_search_traces_merged.jsonl}"
MERGE_MANIFEST="${MERGE_MANIFEST:-experiments/rl/phase5_search_merge_manifest.json}"
BC_CHANGED_WEIGHT="${BC_CHANGED_WEIGHT:-1.0}"
BC_UNCHANGED_WEIGHT="${BC_UNCHANGED_WEIGHT:-1.0}"
BC_EXCLUDE_FEATURES="${BC_EXCLUDE_FEATURES:-}"
BC_PAIRWISE_CHANGED="${BC_PAIRWISE_CHANGED:-0}"
BC_PAIRWISE_MARGIN="${BC_PAIRWISE_MARGIN:-0.0}"
BC_PAIRWISE_NEGATIVES="${BC_PAIRWISE_NEGATIVES:-baseline}"

"$PY" -m ptcg_abc rl-merge-search-data \
  --input "$DECISION_GLOB" \
  --trace-input "$TRACE_GLOB" \
  --output "$MERGED_DATASET" \
  --trace-output "$MERGED_TRACES" \
  --manifest "$MERGE_MANIFEST"

EXCLUDE_ARGS=()
if [[ -n "$BC_EXCLUDE_FEATURES" ]]; then
  for feature in $BC_EXCLUDE_FEATURES; do
    EXCLUDE_ARGS+=(--exclude-feature "$feature")
  done
fi
PAIRWISE_ARGS=()
if [[ "$BC_PAIRWISE_CHANGED" == "1" || "$BC_PAIRWISE_CHANGED" == "true" || "$BC_PAIRWISE_CHANGED" == "TRUE" ]]; then
  PAIRWISE_ARGS+=(
    --pairwise-changed
    --pairwise-margin "$BC_PAIRWISE_MARGIN"
    --pairwise-negatives "$BC_PAIRWISE_NEGATIVES"
  )
fi

"$PY" -m ptcg_abc rl-train-bc \
  --backend torch \
  --epochs "${BC_EPOCHS:-2}" \
  --learning-rate "${BC_LEARNING_RATE:-0.02}" \
  --changed-weight "$BC_CHANGED_WEIGHT" \
  --unchanged-weight "$BC_UNCHANGED_WEIGHT" \
  "${EXCLUDE_ARGS[@]}" \
  "${PAIRWISE_ARGS[@]}" \
  --dataset "$MERGED_DATASET" \
  --checkpoint "${BC_CHECKPOINT:-models/rl/phase5_search_distill.pt}" \
  --model "${BC_MODEL:-models/rl/phase5_search_distill.json}" \
  --report-json "${BC_REPORT:-experiments/rl/phase5_search_distill_report.json}"
