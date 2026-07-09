#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-public-eval
#SBATCH --output=experiments/rl/slurm-%j-phase5-public-eval.out
#SBATCH --error=experiments/rl/slurm-%j-phase5-public-eval.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

set -euo pipefail

module purge
module load miniconda3/3.12

cd "$SLURM_SUBMIT_DIR"
mkdir -p experiments/rl reports

export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
PY="$SLURM_SUBMIT_DIR/.conda_ptcg/bin/python"

PUBLIC_AGENT_ROOTS="${PUBLIC_AGENT_ROOTS:-${PUBLIC_AGENT_ROOT:-data/public_agents}}"
PUBLIC_AGENT_KEYS="${PUBLIC_AGENT_KEYS:-}"
CONTROLLED_PUBLIC_AGENT_KEY="${CONTROLLED_PUBLIC_AGENT_KEY:-}"
CONTROLLED_DECK_INDEX="${CONTROLLED_DECK_INDEX:-101}"
ROSTER_NOTEBOOK="${ROSTER_NOTEBOOK:-}"
MODEL="${MODEL:-models/rl/phase5_generalist_policy_13deck_10k.pt}"
SPECIALIST_MODEL_DIR="${SPECIALIST_MODEL_DIR:-models/rl/phase5_league_alpha/iter-0005/specialists}"
AGENT="${AGENT:-phase5-full}"
POLICY_EPSILON="${POLICY_EPSILON:-0.0}"
DECK_INDICES="${DECK_INDICES:-}"
GAMES_PER_MATCHUP="${GAMES_PER_MATCHUP:-30}"
MAX_STEPS="${MAX_STEPS:-600}"
REQUIRE_MIN_OPPONENTS="${REQUIRE_MIN_OPPONENTS:-1}"
MIN_OPPONENT_WIN_RATE="${MIN_OPPONENT_WIN_RATE:-0.5}"
FAIL_ON_GATE="${FAIL_ON_GATE:-0}"
REPORT_JSON="${REPORT_JSON:-reports/phase5_public_agent_eval_iter0005_30g.json}"
REPORT_MD="${REPORT_MD:-reports/phase5_public_agent_eval_iter0005_30g.md}"
STATUS_JSON="${STATUS_JSON:-reports/phase5_public_agent_status.json}"
REPLAY_OUTPUT_DIR="${REPLAY_OUTPUT_DIR:-}"
SAVED_WIN_REPLAYS="${SAVED_WIN_REPLAYS:-0}"
SAVED_LOSS_REPLAYS="${SAVED_LOSS_REPLAYS:-0}"
REPLAY_TRACE_LIMIT="${REPLAY_TRACE_LIMIT:-120}"
SEARCH_TRACE_OUTPUT="${SEARCH_TRACE_OUTPUT:-}"
SEARCH_TRACE_GAMES="${SEARCH_TRACE_GAMES:-0}"
SEARCH_TOP_K="${SEARCH_TOP_K:-}"
SEARCH_ROLLOUT_STEPS="${SEARCH_ROLLOUT_STEPS:-}"
TACTICAL_SCORE_WEIGHT="${TACTICAL_SCORE_WEIGHT:-}"
NORMALIZE_TACTICAL_SCORE="${NORMALIZE_TACTICAL_SCORE:-0}"
POLICY_PRIOR_WEIGHT="${POLICY_PRIOR_WEIGHT:-}"
NEURAL_ACTION_VALUE_WEIGHT="${NEURAL_ACTION_VALUE_WEIGHT:-}"
NEURAL_TACTICAL_WEIGHT="${NEURAL_TACTICAL_WEIGHT:-}"
LEAF_STATE_VALUE_WEIGHT="${LEAF_STATE_VALUE_WEIGHT:-}"

echo "Phase 5 public-agent evaluation"
echo "PUBLIC_AGENT_ROOTS=$PUBLIC_AGENT_ROOTS"
echo "PUBLIC_AGENT_KEYS=$PUBLIC_AGENT_KEYS"
echo "CONTROLLED_PUBLIC_AGENT_KEY=$CONTROLLED_PUBLIC_AGENT_KEY"
echo "CONTROLLED_DECK_INDEX=$CONTROLLED_DECK_INDEX"
echo "ROSTER_NOTEBOOK=$ROSTER_NOTEBOOK"
echo "AGENT=$AGENT"
echo "POLICY_EPSILON=$POLICY_EPSILON"
echo "MODEL=$MODEL"
echo "SPECIALIST_MODEL_DIR=$SPECIALIST_MODEL_DIR"
echo "DECK_INDICES=$DECK_INDICES"
echo "GAMES_PER_MATCHUP=$GAMES_PER_MATCHUP"
echo "MAX_STEPS=$MAX_STEPS"
echo "REQUIRE_MIN_OPPONENTS=$REQUIRE_MIN_OPPONENTS"
echo "MIN_OPPONENT_WIN_RATE=$MIN_OPPONENT_WIN_RATE"
echo "FAIL_ON_GATE=$FAIL_ON_GATE"
echo "REPORT_JSON=$REPORT_JSON"
echo "REPORT_MD=$REPORT_MD"
echo "STATUS_JSON=$STATUS_JSON"
echo "REPLAY_OUTPUT_DIR=$REPLAY_OUTPUT_DIR"
echo "SAVED_WIN_REPLAYS=$SAVED_WIN_REPLAYS"
echo "SAVED_LOSS_REPLAYS=$SAVED_LOSS_REPLAYS"
echo "REPLAY_TRACE_LIMIT=$REPLAY_TRACE_LIMIT"
echo "SEARCH_TRACE_OUTPUT=$SEARCH_TRACE_OUTPUT"
echo "SEARCH_TRACE_GAMES=$SEARCH_TRACE_GAMES"
echo "SEARCH_TOP_K=$SEARCH_TOP_K"
echo "SEARCH_ROLLOUT_STEPS=$SEARCH_ROLLOUT_STEPS"
echo "TACTICAL_SCORE_WEIGHT=$TACTICAL_SCORE_WEIGHT"
echo "NORMALIZE_TACTICAL_SCORE=$NORMALIZE_TACTICAL_SCORE"
echo "POLICY_PRIOR_WEIGHT=$POLICY_PRIOR_WEIGHT"
echo "NEURAL_ACTION_VALUE_WEIGHT=$NEURAL_ACTION_VALUE_WEIGHT"
echo "NEURAL_TACTICAL_WEIGHT=$NEURAL_TACTICAL_WEIGHT"
echo "LEAF_STATE_VALUE_WEIGHT=$LEAF_STATE_VALUE_WEIGHT"

EXTRA_ARGS=()
ROOT_ARGS=()
KEY_ARGS=()
DECK_ARGS=()
for root in $PUBLIC_AGENT_ROOTS; do
  ROOT_ARGS+=(--public-agent-root "$root")
done
for key in $PUBLIC_AGENT_KEYS; do
  KEY_ARGS+=(--public-agent-key "$key")
done
for index in $DECK_INDICES; do
  DECK_ARGS+=(--deck-index "$index")
done
if [[ -n "$ROSTER_NOTEBOOK" ]]; then
  EXTRA_ARGS+=(--roster-notebook "$ROSTER_NOTEBOOK")
fi
if [[ -n "$CONTROLLED_PUBLIC_AGENT_KEY" ]]; then
  EXTRA_ARGS+=(--controlled-public-agent-key "$CONTROLLED_PUBLIC_AGENT_KEY")
  EXTRA_ARGS+=(--controlled-deck-index "$CONTROLLED_DECK_INDEX")
fi
if [[ -n "$SPECIALIST_MODEL_DIR" ]]; then
  EXTRA_ARGS+=(--specialist-model-dir "$SPECIALIST_MODEL_DIR")
fi
if [[ "$AGENT" == "phase5-epsilon" ]]; then
  EXTRA_ARGS+=(--policy-epsilon "$POLICY_EPSILON")
fi
if [[ "$FAIL_ON_GATE" != "0" ]]; then
  EXTRA_ARGS+=(--fail-on-gate)
fi
if [[ -n "$REPLAY_OUTPUT_DIR" ]]; then
  EXTRA_ARGS+=(--replay-output-dir "$REPLAY_OUTPUT_DIR")
  EXTRA_ARGS+=(--saved-win-replays "$SAVED_WIN_REPLAYS")
  EXTRA_ARGS+=(--saved-loss-replays "$SAVED_LOSS_REPLAYS")
  EXTRA_ARGS+=(--replay-trace-limit "$REPLAY_TRACE_LIMIT")
fi
if [[ -n "$SEARCH_TRACE_OUTPUT" ]]; then
  EXTRA_ARGS+=(--search-trace-output "$SEARCH_TRACE_OUTPUT")
  EXTRA_ARGS+=(--search-trace-games "$SEARCH_TRACE_GAMES")
fi
if [[ -n "$SEARCH_TOP_K" ]]; then
  EXTRA_ARGS+=(--search-top-k "$SEARCH_TOP_K")
fi
if [[ -n "$SEARCH_ROLLOUT_STEPS" ]]; then
  EXTRA_ARGS+=(--search-rollout-steps "$SEARCH_ROLLOUT_STEPS")
fi
if [[ -n "$TACTICAL_SCORE_WEIGHT" ]]; then
  EXTRA_ARGS+=(--tactical-score-weight "$TACTICAL_SCORE_WEIGHT")
fi
if [[ "$NORMALIZE_TACTICAL_SCORE" != "0" ]]; then
  EXTRA_ARGS+=(--normalize-tactical-score)
fi
if [[ -n "$POLICY_PRIOR_WEIGHT" ]]; then
  EXTRA_ARGS+=(--policy-prior-weight "$POLICY_PRIOR_WEIGHT")
fi
if [[ -n "$NEURAL_ACTION_VALUE_WEIGHT" ]]; then
  EXTRA_ARGS+=(--neural-action-value-weight "$NEURAL_ACTION_VALUE_WEIGHT")
fi
if [[ -n "$NEURAL_TACTICAL_WEIGHT" ]]; then
  EXTRA_ARGS+=(--neural-tactical-weight "$NEURAL_TACTICAL_WEIGHT")
fi
if [[ -n "$LEAF_STATE_VALUE_WEIGHT" ]]; then
  EXTRA_ARGS+=(--leaf-state-value-weight "$LEAF_STATE_VALUE_WEIGHT")
fi

"$PY" -m ptcg_abc rl-evaluate-phase5-public-agents \
  "${ROOT_ARGS[@]}" \
  "${KEY_ARGS[@]}" \
  "${DECK_ARGS[@]}" \
  --agent "$AGENT" \
  --model "$MODEL" \
  --games-per-matchup "$GAMES_PER_MATCHUP" \
  --max-steps "$MAX_STEPS" \
  --require-min-opponents "$REQUIRE_MIN_OPPONENTS" \
  --min-opponent-win-rate "$MIN_OPPONENT_WIN_RATE" \
  --report-json "$REPORT_JSON" \
  --report-md "$REPORT_MD" \
  --status-json "$STATUS_JSON" \
  "${EXTRA_ARGS[@]}"
