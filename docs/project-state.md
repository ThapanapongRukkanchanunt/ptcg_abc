# Project State

This is the resume point for the project. Start here after switching machines, cloning the repo, or returning after a long break.

## Current Status

- Repository: `https://github.com/ThapanapongRukkanchanunt/ptcg_abc.git`
- Active branch: `main`
- Selected Limitless format: `TEF-POR`
- Kaggle legal-card source: `EN_Card_Data.csv`
- Latest completed phase: Phase 4, rule-guided hybrid reinforcement learning workflow
- Current phase: Phase 5, advanced RL one-turn root-search vertical slice
  (`docs/phase-5-master-plan.md`,
  `docs/ptcg_rl_strategy_recommendation.md`,
  `docs/ptcg_rl_advanced_training_plan.md`,
  `docs/ptcg_rl_evaluation_plan.md`)

## Phase Log

| Phase | Status | Output |
| --- | --- | --- |
| Phase 0: Repo setup | Complete | GitHub repo initialized with README, `.gitignore`, and implementation plan. |
| Phase 1: Kaggle legality and format selection | Complete | Kaggle card data extracted, Limitless format set to `TEF-POR`, missing-card report shows 0 missing names. |
| Phase 2: Deck corpus exports | Complete | `collect-corpus` writes JSONL, CSV, TXT decklists, and manifest under `data/processed/<snapshot-date>/`. |
| Phase 3: Generic rule-based agent | Complete | Combined generic scorer, random-agent evaluation, archetype sweep, final deck selection, and Kaggle submission bundle. |
| Phase 4: Reinforcement learning workflow | Initial implementation | Rule-guided hybrid RL package, optional PyTorch actor/value BC backend, exported option ranker, workflow commands, and SLURM templates added. |
| Phase 5: Advanced RL strategy, training, and evaluation | Adapter/encoder slice in progress | Search-improved data generation exists; current work is moving from the Phase 4 option-ranker path to canonical adapters, symbolic encoders, and an AlphaStar-style turn policy. |

## Completed Phase Details

### Phase 0: Repo Setup

- Initialized local git repo on `main`.
- Added GitHub remote.
- Pushed initial project files.
- Added the first implementation plan in `docs/implementation-plan.md`.

Representative commits:

- `aa6269c` Initialize project
- `a279ebe` Add deck corpus implementation plan

### Phase 1: Kaggle Legality And Format Selection

- Added Kaggle setup and legal-card discovery commands.
- Used Kaggle competition data from `pokemon-tcg-ai-battle.zip`.
- Extracted the English legal card list from `EN_Card_Data.csv`.
- Compared Limitless card names against Kaggle legal names.
- Chose `TEF-POR` as the working Limitless format.
- Added normalization for apostrophes, mojibake, basic energy names, and exact card-name aliases.
- Canonical report: `reports/missing_limitless_cards.md`
- Detailed conclusion: `docs/phase-1-conclusion.md`

Final result:

- Format: `TEF-POR`
- Missing Limitless card names from Kaggle legal list: 0

Representative commits:

- `4fb5686` Add Kaggle-first legality workflow
- `a4eff91` Filter Limitless deck collection to TEF-CRI
- `a3a6b18` Normalize Limitless and Kaggle card names
- `6037893` Alias Rocky Fighting Energy for TEF-POR report
- `441d325` Use TEF-POR deck format and close legality phase

### Phase 2: Deck Corpus Exports

- Added `collect-corpus`.
- Generated deterministic corpus exports from cached/live Limitless data.
- Kept generated corpus data out of git.
- Detailed conclusion: `docs/phase-2-conclusion.md`

Local generated output path:

```text
data/processed/2026-06-17/
```

Generated files:

- `deck_corpus.jsonl`
- `deck_corpus.csv`
- `decks/*.txt`
- `manifest.json`

Corpus summary:

- Accepted decks: 27
- Archetypes represented: 10
- Variants represented: 14
- Unique deck fingerprints: 27
- Skips and shortfalls recorded in manifest: 20
- Missing card names against Kaggle legality: 0

Representative commit:

- `407c29a` Add TEF-POR deck corpus exports

### Phase 3: Generic Rule-Based Agent Baseline

- Added Kaggle card-ID lookup from `EN_Card_Data.csv`.
- Added corpus JSONL loading and deck-name to card-ID conversion.
- Added deterministic lowest-ID handling for ambiguous Kaggle card names.
- Added the first deck-agnostic rule selector.
- Added `agent-smoke` to run two corpus decks in the local Kaggle sample simulator.
- Detailed checkpoint: `docs/phase-3-baseline.md`
- Added combined score-based generic rule agent.
- Added v1 prize-map planning with automatic key-attacker identification.
- Tightened v1 prize-map actionability after benchmark regression.
- Added v2 prize-map planning for spread and damage-counter attacks.
- Added required Phase 3 benchmark decks for Crustle, Mega Lucario ex, Mega Abomasnow ex,
  and Iono's Bellibolt ex.
- Added the current Phase 3 required benchmark shape: our agent uses nine Limitless
  Tournament 559 decks from ranks 1, 2, 3, 4, 9, 10, 11, 18, and 22 against the four
  fixed benchmark decks.
- Added random-agent evaluation and archetype matchup sweep.
- Selected final archetype: `Hydrapple ex`.
- Selected final deck index: `20`.
- Built local Kaggle submission bundle: `submissions/phase3/submission.tar.gz`.
- Detailed conclusion: `docs/phase-3-conclusion.md`
- Full closeout report: `reports/phase3_closeout.md`
- Current required benchmark report: `reports/phase3_required_benchmark.md`

Current smoke result:

- Selected decks: first two Dragapult ex corpus decks.
- Ambiguous selected names: `Dunsparce` maps to card IDs 65 and 305; current policy chooses 65.
- Simulator start: successful.
- Match finished in 24 selections after v1 prize-map planning.
- Engine error: none.

Closeout result:

- Random-agent mirror evaluation: 18 wins, 2 losses, 0 draws, 0 errors.
- Archetype sweep winner: `Hydrapple ex` with 199 points and 0.733 win rate.
- Generated submission bundle: `submissions/phase3/submission.tar.gz`.

### Phase 4: Rule-Guided Hybrid RL Planning

- Added the Phase 4 RL plan in `docs/phase-4-rl-plan.md`.
- Planned a new `ptcg_abc.rl` package for featurization, datasets, rewards,
  PyTorch policy/value models, PPO training, guidance rules, and reporting.
- Planned a `HybridRlAgent` that scores legal options with an exported RL model,
  blends or falls back to `RuleBasedAgent`, and never emits illegal selections.
- Planned BC warm-start, PPO rollout/training, guidance-rule evaluation, SLURM
  job templates for CMU ERAWAN, and Kaggle packaging without a PyTorch runtime
  dependency.
- Canonical Phase 4 target remains the current required 9-deck by 4-benchmark
  grid: at least 180 wins out of 360 games, with stability and packaging gates.

Representative commit:

- `ff42ccb` Add Phase 4 RL planning doc

### Phase 4: Initial RL Workflow Implementation

- Added `src/ptcg_abc/rl/` with durable `DecisionFrame`, `ActionFrame`, and
  `TrajectoryStep` records.
- Added deterministic board summaries, a fixed-size board-image renderer, and
  per-option feature vectors for legal-option scoring.
- Added a lightweight exported linear option-ranker backend for behavior cloning
  and reward-weighted rollout updates. This preserves the planned model shape
  while keeping Kaggle inference self-contained.
- Added `HybridRlAgent`, which ranks legal options with the exported model,
  applies rule guidance, and falls back to `RuleBasedAgent`.
- Added Phase 4 CLI commands: `rl-collect-bc`, `rl-train-bc`, `rl-rollout`,
  `rl-train-ppo`, `rl-evaluate`, `rl-evaluate-guidance`, and `rl-package`.
- Added Phase 4 hybrid Kaggle packaging support and ensured submission bundles
  include the new `ptcg_abc.rl` package.
- Added SLURM templates under `scripts/slurm/` for BC, rollout arrays, PPO-style
  updates, evaluation, and packaging.
- Added tests for Phase 4 decision frames, record round-trips, model training,
  hybrid selection, rewards, and CLI command exposure.
- Added optional PyTorch actor/value behavior-cloning backend behind
  `rl-train-bc --backend torch`.
- PyTorch backend trains a dynamic legal-option actor plus value head, then
  exports actor weights to the Kaggle-safe JSON `LinearOptionModel` inference
  format.
- Upgraded the model board input from the placeholder `16x16` numeric summary to
  a deterministic `64x64` symbolic board tensor with fixed Active, Bench, Deck,
  Discard, Prize, Stadium, hand-count, global-state, visible-attachment, HP,
  damage, energy, tool, stage/ex, and special-condition zones.
- Updated the optional PyTorch backend to train the planned
  `CNN(board_image) + MLP(option_features)` actor with a board-value head. The
  checkpoint keeps the CNN actor/value model, while `model.json` remains a
  Kaggle-safe linear fallback export.
- Added a human-inspection board snapshot renderer that uses extracted Kaggle
  card art, compact tabletop layout, face-down hidden hands, visible attachment
  stacks, damage/status markers, and active-player perspective snapshots.
- Accepted the compact first-two-turn snapshot baseline under
  `reports/phase4_board_snapshots_compact_first2_turns/` with 28 generated PNGs
  plus `manifest.json`.
- PyTorch backend now trains on CUDA automatically when available and saves a
  portable CPU checkpoint/export.
- Added an `rl` optional dependency group for training environments:
  `python -m pip install -e .[rl]`.
- Added export-equation parity coverage so the PyTorch actor equation and
  exported JSON ranker stay aligned.
- Added `scripts/slurm/phase4_bc_conda.sbatch` as the ERAWAN-safe BC job path:
  direct `.conda_ptcg` Python, CUDA check inside allocation, torch CNN
  checkpoint output, and Kaggle-safe JSON fallback export.
- Added `rl-image-progression` and
  `scripts/slurm/phase4_image_progression_conda.sbatch` for the next experiment:
  10 rounds of Hybrid-vs-Hybrid self-play, model update, and 9x4 benchmark
  evaluation, with compact replay traces capped to one per matchup.
- Ran the Phase 4 local smoke workflow against the copied Kaggle simulator:
  - `rl-collect-bc --games 36 --max-steps 120`: 36 started, 2,218 decisions,
    0 errors, 24 timeouts.
  - `rl-train-bc --epochs 1`: 2,218 frames, 12,409 actions, exported
    `models/rl/bc_model.json`, teacher top-choice accuracy 1.000.
  - `rl-rollout --games 36 --max-steps 120`: 36 started, 1,817 trajectory
    steps, 11 wins, 23 losses, 2 draws, 0 errors, 20 timeouts.
  - `rl-evaluate --games-per-matchup 1 --max-steps 120`: 36 games, 4 wins,
    29 losses, 3 draws, 18 timeouts, 0 errors, 0.111 win rate.
- Installed local PyTorch GPU wheel into the bundled Python runtime:
  `torch 2.7.1+cu118`; CUDA is available on `NVIDIA GeForce RTX 4060 Laptop GPU`.
- Ran GPU actor/value BC smoke:
  - `rl-train-bc --backend torch --epochs 1`: 2,218 frames, 12,409 actions,
    exported `models/rl/bc_torch_gpu_smoke_export.json`, final loss 0.1043,
    teacher top-choice accuracy 1.000, device `cuda`.
  - `rl-evaluate --agent hybrid --model models/rl/bc_torch_gpu_smoke_export.json
    --games-per-matchup 1 --max-steps 120`: 36 games, 11 wins, 23 losses,
    2 draws, 20 timeouts, 0 errors, 0.306 win rate.

Verification:

- `PYTHONPATH=src <bundled-python> -m unittest discover -s tests`: 52 tests pass,
  1 skipped because the missing-PyTorch fallback test is skipped when PyTorch is
  installed.
- Phase 4 smoke gate completed with 0 simulator errors. Smoke runs used
  `max_steps=120`, so timeout counts are expected and not promotion-gate results.

## Recreate Local State From A Clean Checkout

1. Install the package:

```powershell
python -m pip install -e .
```

2. Set up Kaggle data from an already-downloaded archive:

```powershell
python -m ptcg_abc kaggle-setup --archive pokemon-tcg-ai-battle.zip --legal-source data\kaggle\input\EN_Card_Data.csv
```

3. Verify Limitless names against Kaggle legality:

```powershell
python -m ptcg_abc missing-limitless
```

4. Regenerate the deck corpus:

```powershell
python -m ptcg_abc collect-corpus
```

5. Run tests:

```powershell
python -m unittest discover -s tests
```

6. Run the Phase 3 baseline smoke:

```powershell
python -m ptcg_abc agent-smoke --max-steps 50
```

7. Rebuild the Phase 3 submission bundle:

```powershell
python -m ptcg_abc phase3-closeout
```

If `python` is not on PATH in the Codex desktop workspace, use the bundled Python path from the app's workspace dependencies.

## Important Files

- `README.md`: short project overview and current milestone.
- `docs/project-state.md`: this resume log.
- `docs/implementation-plan.md`: implementation-level deck corpus plan.
- `docs/kaggle-first-workflow.md`: Kaggle setup and collection commands.
- `docs/phase-1-conclusion.md`: legality and format-selection conclusion.
- `docs/phase-2-conclusion.md`: deck-corpus export conclusion.
- `docs/phase-3-baseline.md`: first runnable rule-based baseline checkpoint.
- `docs/phase-3-conclusion.md`: Phase 3 final rule agent, evaluation, and submission conclusion.
- `docs/phase-4-rl-plan.md`: Phase 4 rule-guided hybrid RL workflow plan.
- `docs/phase-5-master-plan.md`: umbrella Phase 5 roadmap consolidating the
  strategy, advanced training, and evaluation docs into the current
  implementation plan.
- `docs/ptcg_rl_strategy_recommendation.md`: Phase 5 strategy for a belief-aware
  one-turn root-search agent built on simulator legal options.
- `docs/ptcg_rl_advanced_training_plan.md`: Phase 5 training roadmap covering
  entity/action models, belief modeling, search distillation, PPO, and policy-pool
  iteration.
- `docs/ptcg_rl_evaluation_plan.md`: Phase 5 stage-by-stage evaluation gates,
  ablations, promotion rules, and reporting format.
- `docs/phase-5-erawan-runbook.md`: command-focused ERAWAN operating runbook for
  the current Phase 5 search-distillation vertical slice.
- `docs/erawan-runbook.md`: tested ERAWAN setup, smoke, medium, large, and package commands.
- `docs/rule-inventory.md`: current implemented rules and candidate rules learned from example agents.
- `docs/damage-prevention-pokemon.md`: Pokemon with ability/attack damage prevention and Tera bench-protection watchlist.
- `reports/phase3_closeout.md`: final Phase 3 evaluation report.
- `reports/sample_dragapult_benchmark.md`: copied sample Dragapult agent benchmark against every corpus deck.
- `reports/sample_dragapult_benchmark_prizemap.md`: v1 prize-map benchmark against the copied sample Dragapult agent.
- `reports/sample_dragapult_benchmark_comparison.md`: previous-vs-prize-map benchmark comparison.
- `reports/sample_dragapult_benchmark_prizemap_analysis.md`: diagnosis and next rule upgrade after the v1 prize-map benchmark.
- `reports/sample_dragapult_benchmark_prizemap_v2.md`: guarded v2 prize-map benchmark against the copied sample Dragapult agent.
- `reports/sample_dragapult_benchmark_prizemap_v2_comparison.md`: original-vs-v2 benchmark comparison.
- `reports/sample_dragapult_benchmark_prizemap_v2_analysis.md`: v2 benchmark diagnosis and next rule upgrade plan.
- `reports/sample_dragapult_benchmark_prizemap_v3_setup.md`: narrowed setup-rule benchmark preserving the v2 27-deck result.
- `reports/sample_dragapult_benchmark_phase3_required.md`: historical 31-deck Dragapult benchmark with required sample decks included.
- `reports/sample_dragapult_benchmark_phase3_required_analysis.md`: historical Phase 3 target gap and next rule upgrade plan.
- `reports/phase3_required_benchmark.md`: current 9-deck by 4-benchmark Phase 3 required benchmark.
- `reports/phase3_required_benchmark.json`: machine-readable version of the current required benchmark.
- `reports/missing_limitless_cards.md`: canonical legality report for `TEF-POR`.
- `src/ptcg_abc/`: project code.
- `src/ptcg_abc/rl/`: Phase 4 RL records, featurization, model export, rewards,
  guidance, and workflow helpers.
- `src/ptcg_abc/rl/phase5_search.py`: Phase 5 one-turn root-search vertical
  slice, hidden-state sampler, search trace schema, and search-improved data
  generator.
- `src/ptcg_abc/rl/torch_backend.py`: optional PyTorch actor/value BC backend and
  JSON export bridge.
- `src/ptcg_abc/rl/board_image.py`: human-inspection board snapshot renderer.
- `src/ptcg_abc/rl/snapshots.py`: one-game rule-vs-benchmark snapshot capture.
- `src/ptcg_abc/agent/hybrid_rl.py`: hybrid RL agent with rule fallback.
- `scripts/slurm/`: Phase 4 ERAWAN/SLURM job templates.
- `scripts/slurm/phase5_search_data_array.sbatch`: Phase 5 one-turn root-search
  data generation array job.
- `scripts/slurm/phase5_merge_train_conda.sbatch`: Phase 5 shard merge plus
  Torch BC/search-distillation training job.
- `scripts/slurm/phase4_bc_conda.sbatch`: tested ERAWAN conda BC training job.
- `scripts/slurm/phase4_image_progression_conda.sbatch`: ERAWAN conda job for
  the 256/512/1024 image-size progression experiments.
- `reports/phase4_board_snapshots_compact_first2_turns/`: accepted compact
  board snapshot sample for the first two turns per player.
- `tests/`: regression tests.

## Next Phase Entry Point

Continue Phase 5 from `docs/phase-5-master-plan.md`. Use
`docs/phase-5-erawan-runbook.md` only for the current ERAWAN search-distillation
operating sequence.

Historical Phase 4 follow-ups, kept as implementation context:

- On ERAWAN, use `scripts/slurm/phase4_bc_conda.sbatch` for the next BC run and
  verify the exported JSON model with `rl-evaluate --agent hybrid`.
- Run the controlled progression sweep with `rl-image-progression` or
  `scripts/slurm/phase4_image_progression_conda.sbatch`, then compare
  `progression_summary.json` across image sizes before changing architecture
  again.
- Scale BC collection toward the planned 20,000 rule-agent games on ERAWAN.
- Use rollout chunks and `rl-train-ppo` as the first resumable reward-weighted
  update loop, then replace the update backend with true PPO.
- After the current ERAWAN progression run finishes, analyze whether the model is
  improving before changing architecture. If the trajectory suggests the current
  linear actor/value shape is too weak, consider the board-image upgrade plan
  below.
- Keep the rank 2 legal stand-in in mind: Limitless `Pokemon Center Lady` is not
  present in Kaggle `EN_Card_Data.csv`, so the current simulator deck uses `Cook`.

Implemented model architecture upgrade:

- Replaced the placeholder `16x16` numeric board summary with a deterministic
  synthetic board renderer inspired by a tabletop Pokemon TCG layout.
- Normalized perspective so our side is always at the bottom and the opponent is
  always at the top.
- Uses fixed slots for Active, Bench 1-5, Deck, Discard, Prize, Stadium, hand
  summary, turn/global state, and visible attached cards.
- Renders only legally visible information. Opponent hand, prizes, and hidden deck
  cards should appear as counts or backs, not card faces.
- Encodes card identity and state symbolically instead of relying on OCR:
  card-ID hash/color, card type, Pokemon type, HP/damage bars, energy pips, tool
  marker, ex/Mega/stage markers, and special conditions.
- Keeps dynamic legal-option scoring: `CNN(board_image) + MLP(option_features) ->
  option score`, with a board-value head for PPO.
- The current Kaggle export remains a lightweight linear fallback; use the torch
  checkpoint for CNN scoring/training-side analysis.

Latest benchmark checkpoint:

- Benchmark: our rule-based agent using nine Limitless Tournament 559 decks
  (ranks 1, 2, 3, 4, 9, 10, 11, 18, and 22) against the four fixed benchmark decks:
  Crustle, Mega Lucario ex, Mega Abomasnow ex, and Iono's Bellibolt ex.
- Current full run: 138 wins, 220 losses, 2 draws, 3 timeouts, 0 errors, 0.383 win
  rate across 360 games.
- Current best deck totals: Crustle 0.550, Hydrapple 0.525, Ogerpon Box 0.525.
- Phase 3 target: approximately 180 wins in 360 games, or 50% average win rate.
- Conclusion: the new benchmark shape is implemented and error-free, but the generic
  agent still needs focused rules to reach the 50% target.

Phase 3 completed:

- Load `deck_corpus.jsonl` records.
- Convert Limitless deck names to Kaggle numeric card IDs.
- Add a Kaggle sample-simulator adapter.
- Implement first deck-agnostic action heuristics.
- Add a local smoke command.
- Add combined score-based rule agent.
- Add v1 prize-map planning with automatic key-attacker identification.
- Add guarded v2 prize-map planning for spread and damage-counter attacks.
- Add required Phase 3 sample decks to the copied Dragapult benchmark.
- Add random-agent evaluation.
- Run 10-game archetype matchup sweep.
- Select final archetype and deck.
- Build Kaggle submission bundle.

Phase 4 should keep the Phase 3 rule agent as a fixed baseline opponent while building
state/action/reward traces and training loops.

Phase 5 is now the advanced RL track. Start with the umbrella plan, then use the
source docs for detailed architecture, training, and evaluation requirements:

- `docs/phase-5-master-plan.md`
- `docs/ptcg_rl_strategy_recommendation.md`
- `docs/ptcg_rl_advanced_training_plan.md`
- `docs/ptcg_rl_evaluation_plan.md`

Implemented Phase 5 vertical-slice command:

```powershell
python -m ptcg_abc rl-generate-search-data --games 1 --max-steps 60 --top-k 4 --rollout-steps 18 --require-changed --output data/datasets/rl/phase5_search_smoke.jsonl --trace-output experiments/rl/phase5_search_smoke_traces.jsonl
```

Latest local smoke result:

- Games started: 1
- Training decisions written: 26
- Root decisions searched: 12
- Search-changed decisions: 2
- Candidate probes: 41
- Probe errors: 0
- Truncated search rollouts: 0

Large-scale ERAWAN readiness additions:

- `rl-generate-search-data` supports deterministic `--shard-index`,
  `--shard-count`, `--game-offset`, and `--append` controls.
- `rl-merge-search-data` merges decision and trace JSONL shards into one training
  dataset and writes a manifest.
- `scripts/slurm/phase5_search_data_array.sbatch` generates search-improved shards.
- `scripts/slurm/phase5_merge_train_conda.sbatch` merges shards and trains the
  Torch search-distillation model.
- Full operating sequence: `docs/phase-5-erawan-runbook.md`.

Latest ERAWAN partial search-distillation training:

- Dataset: 10 completed Phase 5 search shards.
- Frames: 791,974.
- Actions: 9,280,556.
- Positive labels: 1,684,328.
- Negative labels: 7,596,228.
- Epochs: 2.
- Exported checkpoint: `models/rl/phase5_search_distill_10shards.pt`.
- Exported JSON model: `models/rl/phase5_search_distill_10shards.json`.
- Accuracy: 0.9039141457203597.
- Final loss: 0.012990633957087994.
- Device: `cuda`.

Latest Phase 5 10-shard benchmark comparison:

- Rule baseline: 126 wins, 233 losses, 1 draw, 5 timeouts, 0 errors, 0.350 win
  rate across 360 games.
- Trained `rl` model only: 79 wins, 280 losses, 1 draw, 2 timeouts, 0 errors,
  0.219 win rate across 360 games.
- Hybrid model plus rule blend: 81 wins, 278 losses, 1 draw, 4 timeouts,
  0 errors, 0.225 win rate across 360 games.
- Conclusion: the 10-shard checkpoint is valid but not promotable. Next gate is
  `rl-diagnose-search-distill` on changed search decisions before PPO,
  packaging, or additional large-scale data generation.

Latest Phase 5 search-distillation diagnostic:

- Overall frames: 791,974.
- Search-changed frames: 76,068, about 9.6% of the dataset.
- Overall search-hit rate: 0.9035637533555394.
- Search-changed search-hit rate: 0.0.
- Search-changed baseline-hit rate: 1.0.
- Search-changed mean model search-minus-baseline score: -6.159176600904512.
- Trace mean search-minus-baseline combined score: 0.5960886340580955.
- Trace mean search-minus-baseline tactical score: 0.6037441871567197.
- Conclusion: the search traces prefer changed labels, but the exported model
  learned unchanged rule-baseline decisions. Next action is a reweighted retrain
  with changed decisions upweighted and `rule_score` / `rule_rank_inv` excluded.

Latest Phase 5 reweighted binary diagnostic and smoke:

- Search-changed search-hit rate improved from 0.0 to 0.24215175895251617.
- Search-changed baseline-hit rate fell from 1.0 to 0.19394489141294632.
- Search-changed mean model search-minus-baseline score improved from
  -6.159176600904512 to -0.37572557840674975.
- Changed-weight `rl` smoke: 7 wins / 36 games, 0.194 win rate, 1 timeout.
- Changed-weight hybrid smoke: 8 wins / 36 games, 0.222 win rate, 1 timeout.
- Conclusion: reweighting fixed baseline-copying but introduced third-action
  drift.

Latest Phase 5 search-over-baseline pairwise diagnostic:

- Overall search-hit rate: 0.4358463787952635.
- Overall baseline-hit rate: 0.42910373320336276.
- Search-changed search-hit rate: 0.16331440290266602.
- Search-changed baseline-hit rate: 0.09311405584477046.
- Search-changed mean model search-minus-baseline score:
  0.421193684133126.
- Conclusion: pairwise baseline training now scores search above baseline on
  average, but still predicts too many third actions. Next action is pairwise
  all-negative changed-decision training:
  `score(search_action) > score(every other legal action)`.

Latest Phase 5 pairwise-all JSON fallback diagnostic:

- Overall search-hit rate: 0.421.
- Overall baseline-hit rate: 0.418.
- Search-changed search-hit rate: 0.150.
- Search-changed baseline-hit rate: 0.121.
- Search-changed mean model search-minus-baseline score: -0.153218.
- Trace search-minus-baseline combined score remains positive at 0.596089.
- Conclusion: the exported linear JSON fallback is still not promotable. The
  torch-checkpoint diagnostic below confirmed that this Phase 4-style path is
  not the right next training target.

Latest Phase 5 pairwise-all torch checkpoint diagnostic:

- Overall search-hit rate: 0.904.
- Overall baseline-hit rate: 1.0.
- Search-changed search-hit rate: 0.0.
- Search-changed baseline-hit rate: 1.0.
- Search-changed model search-minus-baseline score: 0.0.
- Every shown legal option received the same model score, so ranking fell back
  to rule-score tie-breaking.
- Conclusion: the pairwise-all checkpoint collapsed to flat action scores and
  is not promotable. Although an action-residual torch checkpoint format was
  added as a mitigation, the current plan is to skip more Phase 4-style
  diagnostics/training and move to the real Phase 5 adapter/encoder foundation.

Current Phase 5 adapter/encoder slice:

- Added canonical `StateAdapter`, `LegalOptionAdapter`, `GameMemory`, and
  `BeliefState` scaffolding.
- Added symbolic global/entity/legal-action encoder output with masks and
  simulator legal indices.
- Added an AlphaStar-inspired torch policy module with a transformer
  entity/state core and an autoregressive previous-action context for
  turn-level action sequences.
- Added a symbolic DecisionFrame bridge, bounded symbolic dataset writer, direct
  symbolic supervised trainer, and SLURM job script for the first erawan smoke.
- The first 10-shard symbolic checkpoint trained on erawan:
  `models/rl/phase5_symbolic_policy_10shards.pt`, with training accuracy
  `0.780060`, final loss `0.518950`, and `76,068` changed-search examples.
- Added direct `phase5-symbolic` checkpoint evaluation support for the existing
  required benchmark. Next action: run the symbolic smoke and 10-game benchmark
  jobs, then compare against rule, old `rl`, and old `hybrid`.

Operational rule:

- Phase 5 search-distillation diagnostics should always run as SLURM jobs using
  `scripts/slurm/phase5_diagnose_search_distill_conda.sbatch`; avoid running
  large diagnostics on the login node.
