# Phase 5 Master Plan: Belief-Aware Search And Search-Distilled Training

This is the umbrella Phase 5 plan for the advanced RL track. It consolidates the
strategy, advanced training, and evaluation documents into one implementation map.

The ERAWAN runbook is an operating procedure for the current vertical slice. It is
not the full Phase 5 design.

For chronological implementation history, ERAWAN results, diagnostics, and
report-ready facts, use `docs/phase-5-changelog.md`.

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
| `docs/phase-5-erawan-runbook.md` | Current ERAWAN operating sequence | Historical search-shard commands plus the active adapter/encoder track for symbolic Phase 5 training. |
| `docs/phase-5-changelog.md` | Report source log | Chronological implementation log, ERAWAN results, diagnostics, conclusions, artifact notes, and next steps. |

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

- `src/ptcg_abc/rl/phase5_adapters.py`
- `src/ptcg_abc/rl/phase5_encoder.py`
- `src/ptcg_abc/rl/phase5_policy.py`
- `src/ptcg_abc/rl/phase5_symbolic_training.py`
- `src/ptcg_abc/rl/phase5_search.py`
- `src/ptcg_abc/rl/phase5_diagnostics.py`
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

The first 10-shard model did not beat the rule baseline in the required
benchmark:

- Rule baseline: 126 wins / 360 games, 0.350 win rate.
- Trained `rl` model only: 79 wins / 360 games, 0.219 win rate.
- Hybrid model plus rule blend: 81 wins / 360 games, 0.225 win rate.

This makes offline search-distillation diagnostics the next required gate before
PPO, packaging, or more large-scale data generation.

The first diagnostic result showed the exact failure mode: changed decisions were
about 9.6% of the dataset, but the model had `0.0` search-hit rate on changed
decisions and `1.0` baseline-hit rate. The next training slice should therefore
reweight changed frames and remove direct rule-score features from the exported
ranker before scaling data.

The first reweighted binary retrain improved changed-decision learning
(`search_changed.search_hit_rate` rose to about 0.242 and baseline-hit fell to
about 0.194), but battle smoke suggested the model often drifted to third
actions. The first pairwise search-over-baseline retrain fixed the average
search-vs-baseline margin on changed decisions (`+0.421`), but diagnostics still
showed third-action drift: `search_changed.search_hit_rate` was about 0.163 and
baseline-hit was about 0.093. The next trainer should use pairwise all-negative
updates for changed decisions: score the search-selected action above every
other legal action, while keeping unchanged decisions as a weak regularizer.

The first pairwise all-negative JSON fallback did not improve the gate:
`search_changed.search_hit_rate` was about 0.150, baseline-hit was about 0.121,
and the mean model search-minus-baseline score was about `-0.153`. This means
the exported linear fallback is not promotable. The next diagnostic slice should
score the actual torch checkpoint with `rl-diagnose-search-distill --checkpoint`
before deciding whether to implement torch-checkpoint battle evaluation or return
to search-label/model-capacity work.

The first pairwise all-negative torch checkpoint was also not promotable. It
assigned effectively identical scores to legal actions, so changed-decision
ranking fell back to rule-score tie-breaking: `search_changed.search_hit_rate`
was `0.0`, baseline-hit was `1.0`, and the model search-minus-baseline margin
was `0.0`. An action-residual torch actor format was added as a short-term
mitigation, and diagnostics now report `mean_model_score_range` and
`model_score_flat_rate` to catch this failure mode.

After reviewing the strategy recommendation, we are stopping further large
training on the Phase 4-style model path. The next implementation slice is the
real Phase 5 adapter/encoder foundation:

- `StateAdapter` and `LegalOptionAdapter` for canonical state and simulator
  legal-option records.
- Minimal `GameMemory` and `BeliefState` for observed-card accounting and hidden
  count/candidate tracking.
- Symbolic global/entity/legal-action encoders.
- An AlphaStar-inspired policy module with a transformer entity/state core and
  an autoregressive previous-action context for turn-level action sequences.
- A symbolic supervised trainer that can consume the existing Phase 5
  `DecisionFrame` search dataset without materializing a full expanded tensor
  file.
- A direct `phase5-symbolic` evaluation agent that loads the symbolic torch
  checkpoint and runs through the existing 9-deck required benchmark path.
- An offline symbolic checkpoint diagnostic that reports search-label agreement,
  baseline/rule agreement, third-action drift, action-type bias, and deck splits.

## What Is Not Complete Yet

The current Phase 5 vertical slice is not the full Phase 5 agent. In particular,
it does not yet provide:

- Complete production hardening of the new `GameState` and legal-option adapter
  modules.
- Persistent belief memory with exact own-prize deduction.
- Neural belief model training.
- Value, Q, and auxiliary tactical heads.
- Production-hardened online root search with stable Search API wrappers,
  telemetry, time budgets, and final packaging checks.
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

The previous near-term gate was Stage 5, centered on a single generalist
checkpoint and required-gate promotion. That path is now superseded by the
league-first replacement plan below.

## Implementation Priorities

Active replacement plan:

1. Implement the full Phase 5 agent runtime before more checkpoint chasing.
   - Single inference surface for full agent, rule fallback, per-deck model
     selection, belief/opponent-prior inference, neural priors, root search,
     telemetry, and Kaggle-safe packaging.
   - Stop adding new ad-hoc direct/generalist/search variants unless they feed
     this runtime.
2. Train every model that does not require fresh learned-agent gameplay.
   - Shared foundation encoder from existing rule, search, and self-play data.
   - Per-deck behavior-cloning heads for all 13 league decks.
   - Value, selected-action Q, tactical, and belief heads from existing
     trajectory/search records where labels already exist.
3. Generate clean rule-based 13-deck bootstrap gameplay.
   - 13 agent decks x 13 opponent decks.
   - Balanced player order.
   - Full trajectories only for the bounded bootstrap train window.
4. Train one model/policy per deck.
   - Initialize from the shared foundation.
   - Keep separate checkpoints and optimizer state for deck 1 through deck 13.
   - Mix mostly same-deck data with a smaller shared/general replay slice.
5. Run an AlphaStar-like league.
   - Population includes 13 main deck specialists, historical snapshots, fixed
     rule agents, and later exploiters if diagnostics justify them.
   - Matchmaking prioritizes current main-vs-main, current-vs-history,
     weakness-targeted opponents, and rule-agent calibration.
6. Update every deck policy after every 100 games played by that deck.
   - A global league iteration is 13 x 100 training games = 1,300 games before
     update.
   - Each iteration writes one checkpoint family and one compact train report.
7. Evaluate every iteration with full agent vs rule-based for all 13 x 13
   matchups.
   - 30 games per matchup, balanced player order.
   - 5,070 evaluation games per iteration.
   - Evaluation writes aggregate reports and small sampled traces only, not
     full raw trajectory shards.
8. Promotion is based on the league trend and required-gate health.
   - The 13x13 rule-agent league is the primary breadth signal.
   - The existing required 9x4 gate remains a regression check.
   - A checkpoint is promotable only with 0 critical errors, acceptable timeout
     rate, no severe deck collapse, and improvement over the prior iteration.

Data-retention rule:

- The project folder has about 400 GB of practical capacity. Previous Phase 5
  search self-play shards were about 30 GB each, so raw league gameplay cannot
  accumulate across iterations.
- Generated league gameplay under
  `/project/SIGGI/thapanapong.r@cmu.ac.th` is ephemeral. Remove each raw league
  iteration after the corresponding model/policy update succeeds and the train
  report, model checkpoints, optimizer states, manifests, and aggregate eval
  reports are written.
- Keep reports, manifests, small sampled traces, model checkpoints, and
  comparison summaries in the repo. Do not keep full raw evaluation trajectories
  unless explicitly running a bounded diagnostic.
- At most one active league-training data window should exist at a time. Never
  launch the next league iteration if the previous raw league data has not been
  cleaned or intentionally retained with a written reason.

Near-term implementation slice:

1. Add the full-agent runtime command and package path.
2. Add a rule-based 13-deck bootstrap trajectory generator.
3. Add a deck-specialist offline trainer that writes 13 checkpoints from the
   bootstrap/existing data.
4. Add a league-iteration runner that collects 100 games per deck, updates all
   13 specialists, writes reports/checkpoints, and then cleans raw iteration
   data.
5. Add an iteration evaluation runner for 13 x 13 x 30 full-agent-vs-rule
   reporting.

## ERAWAN Operating Track

Use `docs/phase-5-erawan-runbook.md` for commands. The runbook currently covers:

- Login-node smoke.
- Two-shard SLURM smoke.
- Two-shard large waves when QOS rejects larger arrays.
- Shard merge.
- Partial 10-shard training.
- Full-shard training.
- SLURM diagnostics jobs for search-distillation reports.
- Artifact locations.
- Symbolic pairwise/all-negative retraining after the first symbolic diagnostic.

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
