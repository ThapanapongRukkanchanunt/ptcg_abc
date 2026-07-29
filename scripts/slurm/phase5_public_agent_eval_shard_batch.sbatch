#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-public-shards
#SBATCH --output=experiments/rl/slurm-%j-phase5-public-shards.out
#SBATCH --error=experiments/rl/slurm-%j-phase5-public-shards.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

SHARD_INDICES="${SHARD_INDICES:-0 1 2 3 4}"
for shard_index in ${SHARD_INDICES//,/ }; do
  echo "Starting Phase 5 public-evaluation shard $shard_index"
  SHARD_INDEX="$shard_index" \
    bash scripts/slurm/phase5_public_agent_eval_conda.sbatch
done
