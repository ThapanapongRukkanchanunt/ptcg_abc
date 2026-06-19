#!/usr/bin/env bash
#SBATCH --job-name=ptcg-ppo
#SBATCH --output=experiments/rl/slurm-%j-ppo.out
#SBATCH --error=experiments/rl/slurm-%j-ppo.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

python -m pip install -e .

ROLLOUT="${ROLLOUT:-data/datasets/rl/rollout_steps.jsonl}"
python -m ptcg_abc rl-train-ppo \
  --rollout "${ROLLOUT}" \
  --model "${BASE_MODEL:-models/rl/bc_model.json}" \
  --output-model "${OUTPUT_MODEL:-models/rl/ppo_model.json}" \
  --epochs "${PPO_EPOCHS:-1}"
