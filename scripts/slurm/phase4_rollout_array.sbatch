#!/usr/bin/env bash
#SBATCH --job-name=ptcg-rollout
#SBATCH --output=experiments/rl/slurm-%A-%a-rollout.out
#SBATCH --error=experiments/rl/slurm-%A-%a-rollout.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-9

set -euo pipefail

python -m pip install -e .

CHUNK="${SLURM_ARRAY_TASK_ID:-0}"
OUT="data/datasets/rl/rollout_steps_${CHUNK}.jsonl"
python -m ptcg_abc rl-rollout \
  --agent hybrid \
  --games "${ROLLOUT_GAMES_PER_CHUNK:-1000}" \
  --max-steps "${MAX_STEPS:-600}" \
  --output "${OUT}"
