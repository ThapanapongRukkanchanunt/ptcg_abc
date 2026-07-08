#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-public-roster
#SBATCH --output=experiments/rl/slurm-%j-phase5-public-roster.out
#SBATCH --error=experiments/rl/slurm-%j-phase5-public-roster.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

module purge
module load miniconda3/3.12

cd "$SLURM_SUBMIT_DIR"
mkdir -p experiments/rl reports

export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
PY="$SLURM_SUBMIT_DIR/.conda_ptcg/bin/python"

PUBLIC_AGENT_ROOTS="${PUBLIC_AGENT_ROOTS:-${PUBLIC_AGENT_ROOT:-data/public_agents}}"
PUBLIC_AGENT_KEYS="${PUBLIC_AGENT_KEYS:-}"
ROSTER_NOTEBOOK="${ROSTER_NOTEBOOK:-}"
REPORT_JSON="${REPORT_JSON:-reports/phase5_public_agent_roster.json}"
PUBLIC_ONLY="${PUBLIC_ONLY:-0}"
SAMPLES_ONLY="${SAMPLES_ONLY:-0}"
NO_BUILTIN_SAMPLES="${NO_BUILTIN_SAMPLES:-0}"

echo "Phase 5 public-agent roster discovery"
echo "PUBLIC_AGENT_ROOTS=$PUBLIC_AGENT_ROOTS"
echo "PUBLIC_AGENT_KEYS=$PUBLIC_AGENT_KEYS"
echo "ROSTER_NOTEBOOK=$ROSTER_NOTEBOOK"
echo "REPORT_JSON=$REPORT_JSON"
echo "PUBLIC_ONLY=$PUBLIC_ONLY"
echo "SAMPLES_ONLY=$SAMPLES_ONLY"
echo "NO_BUILTIN_SAMPLES=$NO_BUILTIN_SAMPLES"

ROOT_ARGS=()
for root in $PUBLIC_AGENT_ROOTS; do
  ROOT_ARGS+=(--public-agent-root "$root")
done
KEY_ARGS=()
for key in $PUBLIC_AGENT_KEYS; do
  KEY_ARGS+=(--public-agent-key "$key")
done

EXTRA_ARGS=()
if [[ -n "$ROSTER_NOTEBOOK" ]]; then
  EXTRA_ARGS+=(--roster-notebook "$ROSTER_NOTEBOOK")
fi
if [[ "$PUBLIC_ONLY" != "0" ]]; then
  EXTRA_ARGS+=(--public-only)
fi
if [[ "$SAMPLES_ONLY" != "0" ]]; then
  EXTRA_ARGS+=(--samples-only)
fi
if [[ "$NO_BUILTIN_SAMPLES" != "0" ]]; then
  EXTRA_ARGS+=(--no-builtin-samples)
fi

"$PY" -m ptcg_abc phase5-public-agent-roster \
  "${ROOT_ARGS[@]}" \
  "${KEY_ARGS[@]}" \
  --report-json "$REPORT_JSON" \
  "${EXTRA_ARGS[@]}"
