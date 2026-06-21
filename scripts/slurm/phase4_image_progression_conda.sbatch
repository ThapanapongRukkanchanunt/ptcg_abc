#!/usr/bin/env bash
#SBATCH --job-name=ptcg-img-prog
#SBATCH --output=experiments/rl/slurm-%j-image-progression.out
#SBATCH --error=experiments/rl/slurm-%j-image-progression.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

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

print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

IMAGE_ARGS=()
for SIZE in ${IMAGE_SIZES:-256 512 1024}; do
  IMAGE_ARGS+=(--image-size "$SIZE")
done

"$PY" -m ptcg_abc rl-image-progression \
  "${IMAGE_ARGS[@]}" \
  --iterations "${ITERATIONS:-10}" \
  --selfplay-games "${SELFPLAY_GAMES:-1000}" \
  --eval-games-per-matchup "${EVAL_GAMES_PER_MATCHUP:-100}" \
  --max-steps "${MAX_STEPS:-600}" \
  --deck-a-index "${DECK_A_INDEX:-9}" \
  --deck-b-index "${DECK_B_INDEX:-9}" \
  --saved-replays-per-matchup "${SAVED_REPLAYS_PER_MATCHUP:-1}" \
  --replay-trace-limit "${REPLAY_TRACE_LIMIT:-60}" \
  --update-epochs "${UPDATE_EPOCHS:-1}"
