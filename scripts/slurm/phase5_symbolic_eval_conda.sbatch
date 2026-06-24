#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-symbolic-eval
#SBATCH --output=experiments/rl/slurm-%j-phase5-symbolic-eval.out
#SBATCH --error=experiments/rl/slurm-%j-phase5-symbolic-eval.err
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

MODEL="${MODEL:-models/rl/phase5_symbolic_policy_10shards.pt}"
GAMES_PER_MATCHUP="${GAMES_PER_MATCHUP:-1}"
MAX_STEPS="${MAX_STEPS:-600}"
REPORT_JSON="${REPORT_JSON:-reports/phase5_symbolic_10shards_smoke.json}"
REPORT_MD="${REPORT_MD:-reports/phase5_symbolic_10shards_smoke.md}"

"$PY" -m ptcg_abc rl-evaluate \
  --agent phase5-symbolic \
  --model "$MODEL" \
  --games-per-matchup "$GAMES_PER_MATCHUP" \
  --max-steps "$MAX_STEPS" \
  --report-json "$REPORT_JSON" \
  --report-md "$REPORT_MD"
