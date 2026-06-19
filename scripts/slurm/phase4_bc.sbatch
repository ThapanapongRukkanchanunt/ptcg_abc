#!/usr/bin/env bash
#SBATCH --job-name=ptcg-bc
#SBATCH --output=experiments/rl/slurm-%j-bc.out
#SBATCH --error=experiments/rl/slurm-%j-bc.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

python -m pip install -e .
python -m ptcg_abc rl-collect-bc --games "${BC_GAMES:-20000}" --max-steps "${MAX_STEPS:-600}"
python -m ptcg_abc rl-train-bc --backend "${BC_BACKEND:-torch}" --epochs "${BC_EPOCHS:-3}"
