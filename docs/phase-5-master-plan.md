# Phase 5 Master Plan: Belief-Aware Search And Search-Distilled Training

This is the umbrella Phase 5 plan for the advanced RL track. It consolidates the
strategy, advanced training, and evaluation documents into one implementation map.

The ERAWAN runbook is an operating procedure for the current vertical slice. It is
not the full Phase 5 design.

## Phase Numbering

Project Phase 5 includes the advanced RL work described across:

- `docs/ptcg_rl_strategy_recommendation.md`
- `docs/ptcg_rl_advanced_training_plan.md`
- `docs/ptcg_rl_evaluation_plan.md`

The internal "Phase 1" through "Phase 8" labels in the advanced training plan are
training substages inside project Phase 5. In this repository, search-improved
data generation, search distillation, PPO, and final inference are all Phase 5
work until a later project phase is explicitly opened.

## Source Map

| Source | Owns | Phase 5 usage |
| --- | --- | --- |
| `docs/ptcg_rl_strategy_recommendation.md` | Target agent architecture and search strategy | StateAdapter, LegalOptionAdapter, GameMemory, BeliefState, hidden-state sampler, Search API wrapper, symbolic state/action encoding, policy/value/action scorer, one-turn root search, direct policy mode, candidate filtering, time management, debug traces, and submission-time agent shape. |
| `docs/ptcg_rl_advanced_training_plan.md` | Training objectives and staged training flow | Main legal-action policy, value model, Q/action-value head, auxiliary tactical heads, neural belief model, search-imitation policy, PPO, entity Transformer plus action scorer, dataset format, search-improved data generation, search distillation, PPO self-play, policy-pool iteration, and league evaluation. |
| `docs/ptcg_rl_evaluation_plan.md` | Evaluation gates and reports | Stage 0 data/logging checks, Stage 1 imitation policy checks, Stage 2 value/Q checks, Stage 3 tactical-head checks, Stage 4 belief checks, Stage 5 one-turn root-search checks, Stage 6 search-distillation checks, Stage 7 PPO checks, Stage 8 final inference checks, ablation ladder, and promotion checklist. |
| `docs/phase-4-rl-plan.md` | Implementation base | Existing DecisionFrame data, option featurization, exported JSON option ranker, HybridRlAgent, rule fallback, RL workflow commands, and Kaggle-compatible packaging. |
| `docs/phase-5-erawan-runbook.md` | Current ERAWAN operating sequence | Smoke one-turn search, generate search-improved shards, merge shards, train the first Torch search-distillation checkpoint, and export a Kaggle-compatible JSON ranker. |

## Target Agent

The Phase 5 target agent follows this flow:

```text
raw cabt / Kaggle observation
  -> StateAdapter -> canonical GameState
  -> LegalOptionAdapter -> simulator-provided legal options
  -> GameMemory / BeliefState updater
  -> symbolic state encoder
  -> legal-action encoder
  -> policy/value/action scorer

At selected root decisions:
  -> pre-score legal options
  -> keep top-K candidates
  -> sample hidden state from belief
  -> apply each candidate in Search API
  -> let policy roll out to turn end or a safety cap
  -> score final state and tactical consequences
  -> select the best legal simulator option

At normal decisions:
  -> direct policy mode, with rule fallback when needed
```

The simulator owns legality and effect resolution. Phase 5 code owns ranking,
memory, belief, search control, model scoring, and evaluation.

## Required Components

The complete Phase 5 implementation should include these components, even though
the current vertical slice implements only a subset:

- `StateAdapter`: canonicalizes raw observations into a stable GameState.
- `LegalOptionAdapter`: preserves simulator legal options and maps selected
  local indices back to simulator selections.
- `GameMemory` and `BeliefState`: tracks observed cards, revealed deck order,
  known prizes after deck search, opponent actions, and hidden-card uncertainty.
- Hidden-state sampler: produces legal hidden-state samples consistent with public
  information and memory.
- Search API wrapper: isolates `search_begin`, `search_step`, and `search_end`
  usage behind safe, bounded helpers.
- Symbolic state encoder: encodes zones, entity features, belief features, and
  global turn/resource context.
- Legal-action encoder: scores only simulator-provided legal options rather than
  a fixed global action space.
- Policy/value/action scorer: policy logits over legal options, state value, and
  action-value/Q scores.
- Auxiliary tactical heads: actual damage, damage prevention, KO probability,
  expected prize gain, turn-end probability, resource loss, and self-risk.
- Direct policy mode: fast inference path for most decisions and Kaggle packaging.
- One-turn root search: heavier search path for start-of-turn or high-impact
  decisions.
- Candidate filtering and time management: top-K pruning, safety caps, fallback
  behavior, and per-decision timing diagnostics.
- Debug traces: JSON and optional board snapshots for search disagreement,
  rollout failures, tactical outcomes, and timeout analysis.

## Training Substages

The advanced training plan's internal phases should be treated as Phase 5
substages:

| Substage | Name | Purpose | Current status |
| --- | --- | --- | --- |
| P5.0 | Data/logging foundation | Durable records, legal-option logging, trace output, reproducible dataset generation. | Partially implemented by Phase 4 records plus Phase 5 search traces. |
| P5.1 | Supervised imitation pretraining | Learn the rule-based and benchmark-agent action distribution. | Phase 4 behavior cloning path exists. |
| P5.2 | Value and Q training | Train state value and action-value estimates for search and PPO. | Planned. |
| P5.3 | Auxiliary tactical training | Train damage, KO, prize, turn-end, resource, and self-risk heads. | Planned. |
| P5.4 | Neural belief model | Predict hidden-card distributions, with exact own-prize deduction after deck search. | Planned; current vertical slice uses a lightweight sampler only. |
| P5.5 | Search-improved data generation | Use bounded one-turn root search to relabel selected decision points and emit traces. | Current vertical slice implemented. |
| P5.6 | Search distillation | Train a direct policy to imitate search-improved choices and soft search labels. | First Torch BC/distillation route implemented using Phase 4 model export. |
| P5.7 | PPO self-play | Improve beyond rule/search teachers using legal-action PPO and opponent pools. | Planned. |
| P5.8 | Policy-pool iteration | Iterate self-play, replay mixture, checkpoints, and league selection. | Planned. |

## Current Vertical Slice

The current code intentionally starts with the smallest useful Phase 5 slice:

- Builds on the Phase 4 package rather than replacing it.
- Uses the existing required 9-deck by 4-benchmark grid for the first large run.
- Generates valid search-improved `DecisionFrame` JSONL records from local smoke
  games and ERAWAN shards.
- Writes trace logs proving one-turn root search can change some decisions.
- Uses bounded rollout caps and probe-error accounting to prove safe termination.
- Trains a Torch behavior-cloning/search-distillation model from merged search
  data.
- Exports the trained model to JSON so the submission-side path can remain
  Kaggle-compatible.

Key files:

- `src/ptcg_abc/rl/phase5_search.py`
- `src/ptcg_abc/cli.py`
- `scripts/slurm/phase5_search_data_array.sbatch`
- `scripts/slurm/phase5_merge_train_conda.sbatch`
- `docs/phase-5-erawan-runbook.md`

As of the 2026-06-23 ERAWAN run, a 10-shard partial dataset trained successfully
with CUDA and exported:

- `models/rl/phase5_search_distill_10shards.pt`
- `models/rl/phase5_search_distill_10shards.json`

That validates the data-generation, merge, Torch training, and JSON export path.
It does not yet validate that the distilled model improves battle win rate.

## What Is Not Complete Yet

The current Phase 5 vertical slice is not the full Phase 5 agent. In particular,
it does not yet provide:

- Full canonical `GameState` and legal-option adapter modules.
- Persistent belief memory with exact prize deduction.
- Neural belief model training.
- Entity Transformer plus action scorer architecture.
- Value, Q, and auxiliary tactical heads.
- Online `Phase5RootSearchAgent` evaluation mode that combines the exported
  policy with one-turn root search during battles.
- Stage-gated reports for the evaluation plan.
- Full 13-deck league integration.
- PPO or policy-pool iteration.

## Evaluation Gates

Every Phase 5 substage should produce a report that maps back to
`docs/ptcg_rl_evaluation_plan.md`.

| Evaluation stage | Gate | Required evidence |
| --- | --- | --- |
| Stage 0 | Data generation and logging | Valid records, legal selected actions, no schema drift, trace coverage, low error rate. |
| Stage 1 | Imitation policy | Offline accuracy/top-k metrics and battle comparison against rule fallback. |
| Stage 2 | Value and Q | Value calibration, action-value ranking checks, and battle diagnostics. |
| Stage 3 | Tactical heads | Damage, KO, prize, prevention, turn-end, resource, and risk prediction metrics. |
| Stage 4 | Belief model | Hidden-card calibration, exact own-prize deduction checks, and legal card-accounting constraints. |
| Stage 5 | One-turn root search | Direct policy vs policy+value vs root search vs root search+belief, search-change rate, timeout rate, and whether search disagreements improve outcomes. |
| Stage 6 | Search distillation | KL/top-1/top-3 agreement with search labels, battle comparison of old direct, new direct, old search, and new search agents. |
| Stage 7 | PPO | Reward curves, collapse checks, opponent-pool battle metrics, and promotion rule. |
| Stage 8 | Final inference | Ablation ladder, full battle matrix, timing, legality, packaging, and final report. |

The most important near-term gate is Stage 6: prove the 10-shard or full-shard
distilled model beats the previous direct policy or hybrid baseline before
spending more compute on PPO.

## Implementation Priorities

1. Add a Phase 5 offline evaluation/report command for search-distillation output.
   It should report dataset counts, search-change rate, model agreement with
   search-selected actions, top-k agreement, loss/accuracy, and changed-decision
   diagnostics.
2. Add battle evaluation commands for the exported
   `models/rl/phase5_search_distill_10shards.json` model against the current
   required benchmark.
3. Add an online `Phase5RootSearchAgent` or `rl-evaluate --agent phase5-search`
   mode that can compare direct policy, hybrid policy, and policy plus one-turn
   root search.
4. Refactor reusable search pieces from `phase5_search.py` into stable adapter
   modules so data generation and online evaluation share the same Search API
   wrapper.
5. Implement `GameMemory`, `BeliefState`, and own-prize deduction after deck
   search.
6. Introduce the entity/action model and auxiliary heads once evaluation shows
   the current linear JSON ranker is the bottleneck.
7. Add stage-gated reports that follow the evaluation-plan format.
8. Expand from the current 9-deck required benchmark to the broader 13-deck
   league when the first slice is measurable and stable.

## ERAWAN Operating Track

Use `docs/phase-5-erawan-runbook.md` for commands. The runbook currently covers:

- Login-node smoke.
- Two-shard SLURM smoke.
- Two-shard large waves when QOS rejects larger arrays.
- Shard merge.
- Partial 10-shard training.
- Full-shard training.
- Artifact locations.

The runbook should stay practical and command-focused. This master plan should
hold the architectural, training, and evaluation context.

## Promotion Criteria

Phase 5 is ready to advance when:

- Search data generation is reproducible and low-error on ERAWAN.
- One-turn root search changes decisions often enough to provide signal.
- Search-changed decisions have positive expected value or improved battle
  outcomes.
- The distilled direct model improves over the previous direct/hybrid baseline.
- Policy plus root search still improves over direct policy within acceptable
  timing limits.
- Evaluation reports cover the relevant stage gates.
- The Kaggle submission path remains legal, deterministic, and packageable.
