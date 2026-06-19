#!/usr/bin/env bash
#SBATCH --job-name=ptcg-eval-package
#SBATCH --output=experiments/rl/slurm-%j-eval-package.out
#SBATCH --error=experiments/rl/slurm-%j-eval-package.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

python -m pip install -e .

MODEL="${MODEL:-models/rl/ppo_model.json}"
python -m ptcg_abc rl-evaluate \
  --agent hybrid \
  --model "${MODEL}" \
  --games-per-matchup "${GAMES_PER_MATCHUP:-10}" \
  --max-steps "${MAX_STEPS:-600}"

python -m ptcg_abc rl-package --model "${MODEL}" --deck-index "${PACKAGE_DECK_INDEX:-1}"
