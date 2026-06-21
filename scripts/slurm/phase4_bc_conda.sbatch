#!/usr/bin/env bash
#SBATCH --job-name=ptcg-bc
#SBATCH --output=experiments/rl/slurm-%j-bc.out
#SBATCH --error=experiments/rl/slurm-%j-bc.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

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

"$PY" -m ptcg_abc rl-collect-bc \
  --games "${BC_GAMES:-1000}" \
  --max-steps "${MAX_STEPS:-600}" \
  --output "${BC_DATASET:-data/datasets/rl/bc_decisions.jsonl}"

"$PY" -m ptcg_abc rl-train-bc \
  --backend torch \
  --epochs "${BC_EPOCHS:-2}" \
  --learning-rate "${BC_LEARNING_RATE:-0.02}" \
  --dataset "${BC_DATASET:-data/datasets/rl/bc_decisions.jsonl}" \
  --checkpoint "${BC_CHECKPOINT:-models/rl/bc_model.pt}" \
  --model "${BC_MODEL:-models/rl/bc_model.json}" \
  --report-json "${BC_REPORT:-experiments/rl/bc_train_report.json}"
