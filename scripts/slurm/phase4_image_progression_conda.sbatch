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
for SIZE in ${IMAGE_SIZES:-1024 512 256}; do
  IMAGE_ARGS+=(--image-size "$SIZE")
done

DECK_ARGS=()
if [[ -n "${SELFPLAY_DECK_INDICES:-}" ]]; then
  for DECK_INDEX in ${SELFPLAY_DECK_INDICES}; do
    DECK_ARGS+=(--selfplay-deck-index "$DECK_INDEX")
  done
elif [[ -n "${DECK_A_INDEX:-}" || -n "${DECK_B_INDEX:-}" ]]; then
  if [[ -z "${DECK_A_INDEX:-}" || -z "${DECK_B_INDEX:-}" ]]; then
    echo "Set both DECK_A_INDEX and DECK_B_INDEX for fixed-pair self-play." >&2
    exit 2
  fi
  DECK_ARGS+=(--deck-a-index "$DECK_A_INDEX" --deck-b-index "$DECK_B_INDEX")
fi

ROOT_ARGS=(
  --dataset-root "${DATASET_ROOT:-data/datasets/rl/image_progression}"
  --model-root "${MODEL_ROOT:-models/rl/image_progression}"
  --report-root "${REPORT_ROOT:-reports/image_progression}"
  --output-root "${OUTPUT_ROOT:-experiments/rl/image_progression}"
)

"$PY" -m ptcg_abc rl-image-progression \
  "${IMAGE_ARGS[@]}" \
  "${DECK_ARGS[@]}" \
  "${ROOT_ARGS[@]}" \
  --iterations "${ITERATIONS:-10}" \
  --selfplay-games "${SELFPLAY_GAMES:-1000}" \
  --eval-games-per-matchup "${EVAL_GAMES_PER_MATCHUP:-100}" \
  --max-steps "${MAX_STEPS:-600}" \
  --saved-replays-per-matchup "${SAVED_REPLAYS_PER_MATCHUP:-1}" \
  --replay-trace-limit "${REPLAY_TRACE_LIMIT:-60}" \
  --update-epochs "${UPDATE_EPOCHS:-1}"
