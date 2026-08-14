#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-attack-audit
#SBATCH --output=experiments/rl/slurm-%j-phase5-attack-audit.out
#SBATCH --error=experiments/rl/slurm-%j-phase5-attack-audit.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

set -euo pipefail

module purge
module load miniconda3/3.12
cd "$SLURM_SUBMIT_DIR"

export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
PY="$SLURM_SUBMIT_DIR/.conda_ptcg/bin/python"
GAME_DATA_ROOT="${GAME_DATA_ROOT:-/project/SIGGI/thapanapong.r@cmu.ac.th}"
SAMPLE_DIR="${SAMPLE_DIR:-data/kaggle/input/sample_submission}"
SPECIALIST_MODEL_DIR="${SPECIALIST_MODEL_DIR:?Set SPECIALIST_MODEL_DIR}"
GAMES_TOTAL="${GAMES_TOTAL:-104}"
OUTPUT="${OUTPUT:-${GAME_DATA_ROOT}/phase5_attack_audit/dragapult_gen0006_rule13.jsonl}"
TRAJECTORY_REPORT="${TRAJECTORY_REPORT:-experiments/rl/phase5_attack_audit/trajectory_report.json}"
REPORT_JSON="${REPORT_JSON:-reports/phase5_dragapult_gen0006_attack_frequency.json}"
REPORT_MD="${REPORT_MD:-reports/phase5_dragapult_gen0006_attack_frequency.md}"

if [[ ! -d "$SAMPLE_DIR/cg" && -d "$SAMPLE_DIR/sample_submission/cg" ]]; then
  SAMPLE_DIR="$SAMPLE_DIR/sample_submission"
fi

echo "Dragapult attack-frequency audit"
echo "SPECIALIST_MODEL_DIR=$SPECIALIST_MODEL_DIR"
echo "GAMES_TOTAL=$GAMES_TOTAL"
echo "OUTPUT=$OUTPUT"

"$PY" -m ptcg_abc rl-generate-phase5-public-agent-trajectories \
  --sample-dir "$SAMPLE_DIR" \
  --controlled-public-agent-key sample_dragapult \
  --controlled-deck-index 101 \
  --opponent-pool league-rule \
  --agent phase5-search \
  --specialist-model-dir "$SPECIALIST_MODEL_DIR" \
  --search-top-k 4 \
  --games-total "$GAMES_TOTAL" \
  --max-steps 600 \
  --reward-objective discounted-turn-prizes \
  --turn-prize-discount-gamma 0.97 \
  --output "$OUTPUT" \
  --report-json "$TRAJECTORY_REPORT" \
  --overwrite

"$PY" scripts/analysis/phase5_attack_frequency.py \
  --trajectory "$OUTPUT" \
  --sample-dir "$SAMPLE_DIR" \
  --report-json "$REPORT_JSON" \
  --report-md "$REPORT_MD"

rm -f "$OUTPUT"
echo "Deleted analyzed raw trajectory: $OUTPUT"
