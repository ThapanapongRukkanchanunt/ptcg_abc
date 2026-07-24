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
- Phase 5 reporting rule: record meaningful future implementation updates,
  ERAWAN results, diagnostics, conclusions, artifact decisions, and next steps
  in `docs/phase-5-changelog.md`.
- Phase 5 ERAWAN storage rule: future generated game datasets should be written
  under `/project/SIGGI/thapanapong.r@cmu.ac.th`; keep `reports/`, `models/`,
  and `experiments/` in the repository.
- Latest Phase 5 benchmark milestone: `phase5-search` using
  `models/rl/phase5_symbolic_policy_10shards.pt` reached 139 / 360 wins,
  0.386 win rate, 1 timeout, and 0 errors on the required 10-game benchmark,
  beating the direct symbolic policy at 0.303 and the rule baseline at 0.350.
- Latest implementation slice: `phase5-search` reports now include search
  telemetry for searched decisions, search changes, errors, candidate probes,
  truncations, and per-decision search timing.
- Latest Phase 5 confirmation: the 30-game required benchmark for
  `phase5-search` reached 408 / 1,080 wins, 0.378 win rate, 1 timeout, and
  0 errors. Telemetry showed 44,114 searched decisions, 9,685 search-changed
  decisions, 0 search errors, 0 candidate errors, 6,395 truncated candidates,
  and 0.0588 average search seconds.
- Latest prior comparison: pairwise-mid as the `phase5-search` prior reached
  138 / 360 wins, 0.383 win rate, 0 timeouts, and 0 errors, which is clean but
  does not beat the plain symbolic prior's 139 / 360, 0.386 result.
- Latest trace slice: `rl-evaluate` now supports `--search-trace-output` and the
  ERAWAN eval script accepts `SEARCH_TRACE_OUTPUT` for inspecting changed and
  truncated one-turn root-search decisions.
- Latest trace-capture result: the 3-game `phase5-search` trace run reached
  39 / 108 wins, 0.361 win rate, 0 timeouts, 0 errors, 4,513 searched decisions,
  1,040 search-changed decisions, 0 search/candidate errors, and 699 truncated
  candidates. Next action is to inspect the JSONL trace examples.
- Latest trace inspection: in the first 10 pasted truncated examples, 5 records
  had all candidates truncated and 0 changed records selected a truncated
  candidate. Need full-trace selected-truncation counts before tuning the
  rollout cap or scorer.
- Latest full-trace truncation summary: `rl-diagnose-search-traces` now reports
  selected-truncation metrics. On the 3-game trace file, 97 / 4,513 records
  selected a truncated candidate and 43 / 1,040 changed records selected a
  truncated candidate. This is meaningful but not dominant.
- Latest cap-experiment support: `rl-evaluate` now accepts `--search-top-k` and
  `--search-rollout-steps`; the ERAWAN eval script accepts `SEARCH_TOP_K` and
  `SEARCH_ROLLOUT_STEPS`. Cap 30 is now the default, while overrides remain
  available for experiments.
- Latest cap-30 trace diagnostic: on the 3-game `phase5-search` trace, cap 30
  reduced all-candidates-truncated records from 72 / 4,513 to 0 / 4,239,
  selected-truncated records from 97 / 4,513 to 1 / 4,239, and changed
  selected-truncated records from 43 / 1,040 to 1 / 927. Next action is a
  10-game required benchmark with `SEARCH_ROLLOUT_STEPS=30`.
- Latest cap-30 10-game trace diagnostic: 14,680 records, 3,218 changed records,
  0 search errors, 0 candidate errors, 353 truncated candidates, 64 selected
  truncated records, and 15 changed selected-truncated records. This is much
  cleaner than the cap-18 10-game truncated-candidate count of 2,196.
- Latest cap-30 benchmark decision: cap 30 reached 148 / 360 wins, 0.411 win
  rate, 1 timeout, 0 errors, 14,680 searched decisions, 3,218 changed decisions,
  0.0631 average search seconds, and 3.4057 max search seconds. This beats the
  cap-18 10-game result of 139 / 360, 0.386 while keeping average timing flat,
  so `RootSearchConfig.max_rollout_steps` is promoted to 30. Watch max latency
  in the next 30-game default-cap confirmation run.
- Latest 30-game default-cap trace diagnostic: 44,811 records, 9,693 changed
  records, 0 search errors, 0 candidate errors, 777 truncated candidates,
  140 selected-truncated records, and 32 changed selected-truncated records.
  The 30-game benchmark win/loss/timing summary still needs to be recorded.
- Active Phase 5 training plan as of July 2, 2026: replace the slow
  single-generalist promotion track with a full-agent, AlphaStar-like 13-deck
  league. Implement the full runtime first, train every model that does not
  require fresh learned-agent gameplay, bootstrap with rule-based 13 x 13
  gameplay, train one specialist policy per deck, then run league iterations.
  Each deck updates after 100 training games, so one global iteration is
  1,300 training games followed by updates for all 13 specialists. Every
  iteration is evaluated as full agent vs rule-based across all 13 x 13 matchups
  at 30 games per matchup, for 5,070 evaluation games per iteration.
- Active Phase 5 data-retention rule: keep league data clean and bounded. The
  project folder has about 400 GB of practical capacity, and prior Phase 5
  search self-play shards were about 30 GB each. New raw league gameplay is
  ephemeral: remove each iteration's raw training data after the model/policy
  update succeeds and reports/checkpoints/manifests are written. Evaluation
  should write aggregate reports and tiny sampled traces only, not full raw
  trajectories by default.
- Latest implementation slice: added `rl-generate-phase5-search-selfplay` and
  `scripts/slurm/phase5_search_selfplay_conda.sbatch`. Future self-play
  trajectory shards default to
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search_selfplay/shards`, while
  reports and optional traces stay under `experiments/rl/phase5_search_selfplay`.
- Latest self-play smoke: 2 / 2 games started, 298 trajectory steps, 0 errors,
  0 timeouts, 153 search decisions, 24 search-changed decisions, 0 search
  errors, 0 candidate errors, 1 truncated candidate, 153 trace records.
- Latest bounded self-play gate: two 25-game shards completed over the current
  9-deck pool with 50 / 50 games started, 8,424 trajectory steps, 0 errors,
  0 timeouts, 4,468 searched decisions, 942 search-changed decisions,
  0 search errors, 0 candidate errors, 63 truncated candidates, 0.0813 average
  search seconds, and 165 sampled trace records.
- Latest large self-play support: added
  `scripts/slurm/phase5_search_selfplay_2shard_10k.sbatch`, which runs a
  two-shard `phase5-search` self-play dataset at about 10,000 total games by
  default. Game data writes to
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search_selfplay_10k/shards`;
  summaries and sampled traces stay under
  `experiments/rl/phase5_search_selfplay_10k`.
- Latest 10,000-game self-play result: 10,000 / 10,000 games started,
  1,597,717 trajectory rows, 0 errors, 50 timeouts, 20 draws, 827,784 searched
  decisions, 176,728 search-changed decisions, 0 search errors,
  0 candidate errors, 12,338 truncated candidates, 0.0808 average search
  seconds, and 4.8117 max search seconds. This completes the 9-deck Phase 5
  self-play data generation gate.
- Latest implementation slice: added the Phase 5 mixed generalist trainer.
  `AlphaStarTurnPolicy` now has policy, state-value, selected-action Q, and
  tactical scalar outputs. `rl-train-phase5-generalist` and
  `scripts/slurm/phase5_generalist_train_conda.sbatch` stream the 10-shard
  search-decision data plus the 10,000-game self-play shards without merging
  them in memory.
- Latest generalist evaluation: direct `phase5-symbolic` with
  `models/rl/phase5_generalist_policy_10k.pt` reached 117 / 360 wins
  (0.325) at 10 games per matchup and 361 / 1,080 wins (0.334) at 30 games per
  matchup. It improves over the first direct symbolic model but remains below
  the rule baseline.
- Latest best inference path: `phase5-search` with
  `models/rl/phase5_generalist_policy_10k.pt` reached 138 / 360 wins (0.383)
  at 10 games per matchup and 414 / 1,080 wins (0.383) at 30 games per matchup,
  with 0 search errors, 0 candidate errors, 677 truncated candidates, 0.0514
  average search seconds, and 2.4492 max search seconds in the 30-game run.
  This slightly beats the prior 30-game `phase5-search` result of 408 / 1,080
  while greatly reducing truncation.
- Latest weakness: Alakazam Dudunsparce remains poor under the generalist search
  prior, at 4 / 120 wins in the 30-game benchmark. Treat it as the main targeted
  data/model issue before claiming robust direct-policy strength.
- Latest implementation slice: added the opt-in Phase 5 `league-13` self-play
  deck pool. It combines the nine Tournament 559 decks with the four required
  sample decks while leaving the existing 9x4 required benchmark unchanged.
  `rl-generate-phase5-search-selfplay` and the self-play SLURM scripts now
  accept `DECK_POOL=league-13` / `--deck-pool league-13`, and self-play records
  include `selfplay_deck_pool` metadata.
- Latest 13-deck smoke: `phase5_search_selfplay_13deck_338` completed 338 /
  338 games with `deck_pool=league-13`, 51,945 trajectory rows, 4 draws,
  3 timeouts, 0 errors, 26,920 searched decisions, 5,286 search-changed
  decisions, 0 search errors, 0 candidate errors, 641 truncated candidates,
  0.0539 average search seconds, and 1.6210 max search seconds.
- Current ERAWAN state as of July 1, 2026: the larger
  `phase5_search_selfplay_13deck_10k` data run was reported complete, but
  ERAWAN still needs a code sync before training. Pulling `origin/main` from
  `1411cb3` to `586cedc` was blocked by four untracked local report artifacts:
  `reports/phase5_generalist_search_10g.json`,
  `reports/phase5_generalist_search_10g.md`,
  `reports/phase5_generalist_search_30g.json`, and
  `reports/phase5_generalist_search_30g.md`.
- Latest 13-deck generalist comparison: the expanded
  `models/rl/phase5_generalist_policy_13deck_10k.pt` checkpoint was clean but
  not promotable on the required 9x4 30-game gate. Baseline
  `phase5-search` with `models/rl/phase5_generalist_policy_10k.pt` reached
  414 / 1,080 wins, 0.383 win rate, 5 timeouts, and 0 errors. Candidate
  `phase5-search` with `models/rl/phase5_generalist_policy_13deck_10k.pt`
  reached 399 / 1,080 wins, 0.369 win rate, 6 timeouts, and 0 errors.
  Overall delta: -15 wins, -0.014 win rate, +1 timeout, +0 errors.
- Current decision: keep `phase5_generalist_policy_10k.pt` as the default
  `phase5-search` prior for existing comparisons. Treat the 13-deck checkpoint
  as a retained training artifact, not a promotion candidate or mainline PPO
  seed.
- The league-first implementation now includes:
  - added `src/ptcg_abc/rl/phase5_alpha_league.py`,
  - added `phase5-full` as the public full-agent alias for the existing Phase 5
    policy-plus-root-search runtime,
  - added `rl-generate-phase5-alpha-bootstrap`,
    `rl-generate-phase5-alpha-league-iteration`,
    `rl-train-phase5-deck-specialists`, and
    `rl-clean-phase5-alpha-iteration`,
  - added SLURM scripts for bootstrap, learned-agent league iteration,
    specialist train, raw-data cleanup, and league evaluation,
  - added deck-index filtering to the mixed Phase 5 trainer so specialists can
    train from shared JSONL inputs without copying huge per-deck datasets,
  - added per-deck specialist dispatch so league eval and learned-agent
    self-play can load `deck-01.pt` through `deck-13.pt`.
- Current next code gap: wait for the true iteration-0 specialist eval report,
  then use the learned-agent league iteration runner to generate `iter-0001`
  training data.

## Phase Log

| Phase | Status | Output |
| --- | --- | --- |
| Phase 0: Repo setup | Complete | GitHub repo initialized with README, `.gitignore`, and implementation plan. |
| Phase 1: Kaggle legality and format selection | Complete | Kaggle card data extracted, Limitless format set to `TEF-POR`, missing-card report shows 0 missing names. |
| Phase 2: Deck corpus exports | Complete | `collect-corpus` writes JSONL, CSV, TXT decklists, and manifest under `data/processed/<snapshot-date>/`. |
| Phase 3: Generic rule-based agent | Complete | Combined generic scorer, random-agent evaluation, archetype sweep, final deck selection, and Kaggle submission bundle. |
| Phase 4: Reinforcement learning workflow | Initial implementation | Rule-guided hybrid RL package, optional PyTorch actor/value BC backend, exported option ranker, workflow commands, and SLURM templates added. |
| Phase 5: Advanced RL strategy, training, and evaluation | League-first replacement plan active | Full-agent runtime, offline pretraining, 13 deck specialists, AlphaStar-like league iteration, and strict raw-data cleanup replace the slow single-generalist promotion track. |

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
- `docs/phase-5-changelog.md`: report-ready chronological log of Phase 5
  implementation, ERAWAN results, diagnostics, conclusions, artifacts, and next
  steps.
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
python -m ptcg_abc rl-generate-search-data --games 1 --max-steps 60 --top-k 4 --rollout-steps 30 --require-changed --output /project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search_smoke.jsonl --trace-output experiments/rl/phase5_search_smoke_traces.jsonl
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
  required benchmark.
- Latest symbolic 10-game benchmark: `109/360` wins, `0.303` win rate,
  `errors: 0`, `timeouts: 3`. This is better than the old 10-shard `rl/hybrid`
  path but below the current rule baseline around `0.35`, so it is not yet
  promotable.
- Next action: run `rl-diagnose-phase5-symbolic` as a SLURM job to identify
  whether losses come from weak search-changed agreement, third-action drift,
  action-type bias, or deck-specific failures.

Current Phase 5 generalist/search state as of June 29, 2026:

- Current best inference path is `phase5-search` with
  `models/rl/phase5_generalist_policy_10k.pt` as prior.
- Required 9x4 benchmark with that path:
  - 10 games per matchup: 138 / 360 wins, 0.383 win rate, 0 errors.
  - 30 games per matchup: 414 / 1,080 wins, 0.383 win rate, 0 errors,
    677 truncated candidates, 0.0514 average search seconds.
- Main weakness remains Alakazam Dudunsparce.
- The 13-deck `league-13` self-play smoke passed:
  `phase5_search_selfplay_13deck_338`, 338 / 338 games, 51,945 trajectory rows,
  4 draws, 3 timeouts, 0 errors, 0 search errors, 0 candidate errors.
- The next self-play data target was
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search_selfplay_13deck_10k`.
  It was reported complete on July 1, 2026, and still needs summary and shard
  line-count verification before training.
- ERAWAN code sync from `1411cb3` to `586cedc` was blocked by four untracked
  report artifacts under `reports/phase5_generalist_search_*`; archive them
  before rerunning `git pull --ff-only origin main`.
- ERAWAN lightweight repo CLI commands such as `phase5-compare-benchmarks` need
  `PYTHONPATH="$PWD/src"` unless the active conda environment has installed the
  package. A `No module named ptcg_abc` error is an import-path issue, not a
  benchmark failure.
- `phase5-compare-benchmarks` should not require scraper dependencies. The CLI
  now imports `ptcg_abc.limitless` lazily only for `missing-limitless` and
  `collect-corpus`, after an ERAWAN compare attempt exposed a missing `lxml`
  import at top-level CLI load time.
- The trainer already supports multiple self-play shards, and the 13-deck
  mixed train used explicit artifacts:
  `models/rl/phase5_generalist_policy_13deck_10k.pt` and
  `experiments/rl/phase5_generalist_train_report_13deck_10k.json`.
- The 13-deck checkpoint comparison against the current 30-game required gate
  regressed from 414 / 1,080 wins (0.383) to 399 / 1,080 wins (0.369), with
  0 errors in both reports. It improved Dragapult Dusknoir (+8 wins) and
  Dragapult Blaziken (+4 wins), left Alakazam Dudunsparce flat at 4 / 120, and
  regressed Ogerpon Box (-7), Hydrapple (-5), Dragapult (-5), Raging Bolt
  Ogerpon (-4), Crustle (-3), and Dragapult Dudunsparce (-3).
- The 13-deck checkpoint is not promoted. Keep
  `models/rl/phase5_generalist_policy_10k.pt` as the default `phase5-search`
  prior, and do not use the 13-deck checkpoint as the mainline PPO seed.
- The 13-deck league data remains useful as training expansion and breadth
  diagnostic material. The existing required 9x4 benchmark remains the first
  promotion gate for follow-up checkpoints.
- The active plan has shifted to a full-agent AlphaStar-like league:
  - implement the full agent as the single inference surface,
  - train non-gameplay models offline first,
  - generate rule-based 13 x 13 bootstrap gameplay,
  - train one specialist policy per deck,
  - run league iterations where each deck plays 100 training games before
    all 13 specialists update,
  - evaluate every iteration as full agent vs rule-based over 13 x 13 x 30
    games.
- League storage must stay bounded. Raw league training data is deleted after a
  successful model/policy update and report/checkpoint write. Keep only
  checkpoints, optimizer states, manifests, row counts, train reports,
  aggregate eval reports, comparison reports, and small sampled traces.
- First league-track commands now exist:
  - `rl-generate-phase5-alpha-bootstrap`,
  - `rl-train-phase5-deck-specialists`,
  - `rl-clean-phase5-alpha-iteration`.
- First league-track SLURM scripts now exist:
  - `scripts/slurm/phase5_alpha_rule_bootstrap.sbatch`,
  - `scripts/slurm/phase5_deck_specialists_train.sbatch`,
  - `scripts/slurm/phase5_alpha_cleanup_iteration.sbatch`.
- First league-track ERAWAN smoke passed on July 2, 2026:
  `experiments/rl/phase5_league_alpha/iter-0000_rule_bootstrap_report.json`
  records `ITERATION=0`, `GAMES_PER_PAIR=1`, `MAX_STEPS=300`, rule-agent
  13 x 13 bootstrap gameplay, 169 / 169 games started, 25,349 trajectory
  steps, 1 draw, 5 timeouts, 0 errors, and balanced deck coverage at 26 games
  per deck. The raw trajectory is
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0000/raw_train/phase5_alpha_rule_bootstrap.jsonl`.
  This raw directory can now be cleaned because the deck-specialist smoke update
  report and checkpoint paths exist.
- First deck-specialist training smoke passed on July 2, 2026:
  `experiments/rl/phase5_league_alpha/iter-0000_deck_specialists_report.json`
  records 13 / 13 specialist summaries and checkpoint paths under
  `models/rl/phase5_league_alpha/iter-0000/specialists/`, CUDA, one epoch,
  1,998 decision examples, 1,998 rule-demo examples, 2,000 self-play examples,
  5,996 value examples, 2,069 action-value examples, 33,923 tactical examples,
  358 changed examples, and 4 skipped no-target records. Decks 10-13 had zero
  search-decision examples because the canonical search-decision dataset covers
  only the original 9 tournament decks; those four sample decks relied on
  bootstrap/self-play signals for this smoke.
- Full iteration-0 rule bootstrap passed on July 2, 2026:
  `experiments/rl/phase5_league_alpha/iter-0000_rule_bootstrap_report.json`
  now records the fuller `GAMES_PER_PAIR=2` bootstrap, replacing the current
  working-tree copy of the earlier smoke report. It produced 338 / 338 games,
  53,443 trajectory steps, 3 draws, 7 timeouts, 0 errors, and balanced deck
  coverage at 52 games per deck. The raw trajectory is still
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0000/raw_train/phase5_alpha_rule_bootstrap.jsonl`.
- No-limit iteration-0 deck-specialist update passed on July 2, 2026:
  `experiments/rl/phase5_league_alpha/iter-0000_deck_specialists_report.json`
  now records job `73248`, `iteration=0`, 13 / 13 specialist summaries and
  checkpoint paths under `models/rl/phase5_league_alpha/iter-0000/specialists/`.
  Each specialist scanned 791,974 decision frames and 53,443 self-play steps.
  Aggregate examples: 791,667 decision, 791,667 rule-demo, 53,421 self-play,
  1,636,755 value, 56,902 action-value, 9,582,634 tactical, 152,136 changed,
  and 636 skipped no-target records. Decks 10-13 still have zero
  search-decision examples because the canonical search-decision data covers
  only the original 9 tournament decks, but they now have full-bootstrap
  self-play coverage.
- `rl-evaluate-phase5-league` now supports `--specialist-model-dir` for
  per-deck dispatch, and `scripts/slurm/phase5_league_eval_conda.sbatch` accepts
  `SPECIALIST_MODEL_DIR`. Use this for iteration-0 full-agent-vs-rule eval so
  deck 1 loads `deck-01.pt`, deck 2 loads `deck-02.pt`, and so on.
- A July 3, 2026 uploaded 13 x 13 x 30 eval report was clean but used the old
  single-model path:
  `models/rl/phase5_generalist_policy_13deck_10k.pt`. It is preserved as
  `reports/phase5_alpha_iter0000_full_vs_rule_30g_single_model.json` and `.md`.
  Result: 2,662 / 5,070 wins, 0.525 win rate, 18 draws, 50 timeouts, 0 errors.
  Treat it as a single-generalist breadth diagnostic, not the iteration-0
  specialist eval.
- The true July 3, 2026 iteration-0 specialist eval is preserved as
  `reports/phase5_alpha_iter0000_specialists_full_vs_rule_30g.json` and `.md`.
  It used
  `Model: models/rl/phase5_league_alpha/iter-0000/specialists`, confirming
  per-deck dispatch. Result: 2,651 / 5,070 wins, 0.5229 win rate, 15 draws,
  48 timeouts, 0 errors, 202,460 searched decisions, 31,122 search-changed
  decisions, 0 search errors, and 0 candidate errors. Against the 4 required
  sample rule-agent opponents only, it scored 659 / 1,560 wins, 0.4224 win
  rate. Compared with the single-model diagnostic, the overall result is down
  11 wins and 0.22 percentage points, but Alakazam Dudunsparce, Dragapult, and
  Dragapult Dudunsparce improved. Treat this as the accepted iteration-0
  AlphaStar-like specialist baseline, not as a promoted final policy.
- Next league-track ERAWAN action: launch `ITERATION=1` learned-agent league
  data generation from
  `models/rl/phase5_league_alpha/iter-0000/specialists`, then train
  iteration-1 specialists from that raw window and evaluate the new checkpoint
  family with the same 13 x 13 x 30 full-agent-vs-rule benchmark.
- Learned-agent league iteration support was implemented on July 3, 2026:
  `rl-generate-phase5-alpha-league-iteration` and
  `scripts/slurm/phase5_alpha_league_iteration.sbatch` generate a new raw
  training window using per-deck specialist checkpoints. The default next window
  is `ITERATION=1`, `SOURCE_ITERATION=0`, `GAMES_PER_DECK=100`, and output
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0001/raw_train/phase5_alpha_league_selfplay.jsonl`.
  The specialist trainer SLURM script now accepts `SELFPLAY_DATASET` and can
  consume either rule-bootstrap or learned-agent league raw data.
- Iteration-0 specialist Kaggle package candidates are deck 11 Mega Lucario ex
  and deck 12 Mega Abomasnow ex. Deck 11 scored 85 / 120 against the four
  required rule-based specialist opponents and 297 / 390 in the full 13 x 13
  rule eval. Deck 12 scored 65 / 120 against the four required specialists and
  257 / 390 in the full eval; it was selected over deck 4 Dragapult despite
  deck 4's 66 / 120 sample-specialist result because deck 12 was much stronger
  overall. `phase5-package` now supports `--deck-pool league-13` and
  `--model-dir`, so ERAWAN can build both packages from
  `models/rl/phase5_league_alpha/iter-0000/specialists` into
  `submissions/phase5_alpha_iter0000_specialists_top2`.
- Iteration-1 deck-specialist update passed on ERAWAN job `73372`. The report
  `experiments/rl/phase5_league_alpha/iter-0001_deck_specialists_report.json`
  records 13 / 13 checkpoint paths under
  `models/rl/phase5_league_alpha/iter-0001/specialists/`, trained from
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0001/raw_train/phase5_alpha_league_selfplay.jsonl`.
  Aggregate examples: 791,667 decision, 791,667 rule-demo, 193,598 self-play,
  1,776,932 value, 206,777 action-value, 10,442,154 tactical, 152,136 changed,
  and 614 skipped no-target records. Decks 10-13 still have zero canonical
  search-decision examples, but now have learned-agent self-play coverage.
  Next action: evaluate
  `models/rl/phase5_league_alpha/iter-0001/specialists` with the 13 x 13 x 30
  full-agent-vs-rule benchmark, then clean `iter-0001/raw_train/` according to
  the raw-data retention policy.
- The Alpha league training plan has pivoted to true online RL for future
  iterations. Iteration 0 remains rule-bootstrap supervised/mixed training and
  iteration 1 was already produced with the earlier mixed specialist update, but
  iteration 2 onward should use `AGENT=phase5-rl` for stochastic neural self-play
  and `rl-train-phase5-alpha-ppo-specialists` /
  `scripts/slurm/phase5_alpha_ppo_specialists_train.sbatch` for per-deck PPO.
  New trajectories record policy log-probability, value, and
  `policy_on_policy=true`; the Alpha PPO specialist trainer requires that flag
  by default so legacy search/imitation data is not treated as on-policy RL.
- Iteration-2 true online RL passed on ERAWAN job `73394`. The `phase5-rl`
  self-play window produced 1,300 / 1,300 games, 215,597 trajectory rows, 8
  draws, 26 timeouts, and 0 errors. PPO then consumed exactly 215,597 on-policy
  examples across all 13 specialists with `skipped_off_policy=0` and
  `skipped_no_target=0`, writing
  `models/rl/phase5_league_alpha/iter-0002/specialists`. Deck 11 had the
  strongest online self-play record at 137 / 204 wins; deck 1 remained the
  weakest at 28 / 204. Next action: evaluate
  `models/rl/phase5_league_alpha/iter-0002/specialists` with the 13 x 13 x 30
  full-agent-vs-rule benchmark before cleaning `iter-0002/raw_train/` or
  starting iteration 3.
- Iteration-2 full-agent-vs-rule evaluation passed on ERAWAN job `73396`:
  2,710 / 5,070 wins, 0.5345 win rate, 20 draws, 55 timeouts, 0 errors.
  This is +59 wins over the recorded iteration-0 specialist eval
  (2,651 / 5,070) and +35 wins against the four required sample rule-agent
  opponents (694 / 1,560 vs 659 / 1,560). Deck 11 Mega Lucario ex remains the
  strongest specialist at 292 / 390 wins; deck 1 Alakazam Dudunsparce remains
  the main weakness at 65 / 390 and declined by four wins versus iteration 0.
  Next action: clean `iter-0002/raw_train/`, then launch iteration 3 online
  self-play from `models/rl/phase5_league_alpha/iter-0002/specialists`.
- Iteration-4 online self-play passed on ERAWAN job `73448` while iteration-3
  eval was running. Source checkpoints:
  `models/rl/phase5_league_alpha/iter-0003/specialists`. The raw window
  produced 1,300 / 1,300 games, 211,968 trajectory rows, 11 draws, 27 timeouts,
  and 0 errors under
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0004/raw_train/`.
  Deck 11 remained strongest in self-play at 142 / 204 wins; deck 1 remained
  weakest at 24 / 204. Next action: run the iteration-4 PPO specialist update
  from source iteration 3 to target iteration 4 while waiting for iteration-3
  eval, then inspect both reports before cleaning `iter-0004/raw_train/`.
- Iteration-4 online PPO update passed on ERAWAN job `73451`. It consumed all
  211,968 iter-4 on-policy trajectory rows, skipped 0 no-target rows and 0
  off-policy rows, and wrote all 13 checkpoints under
  `models/rl/phase5_league_alpha/iter-0004/specialists`. Deck 1 had the most
  PPO examples at 28,241; deck 12 had the fewest at 7,488. Next action: queue
  the iteration-4 13 x 13 x 30 full-agent-vs-rule evaluation in parallel with
  the still-running iteration-3 eval, then inspect both reports before cleaning
  the active raw windows.
- Iteration-3 full-agent-vs-rule evaluation passed on ERAWAN job `73454`:
  2,697 / 5,070 wins, 0.5320 win rate, 20 draws, 38 timeouts, 0 errors. This
  is down 13 wins from iteration 2 but still up 46 wins over the recorded
  iteration-0 specialist baseline. Against the four required sample rule-agent
  opponents, iteration 3 scored 677 / 1,560, down 17 from iteration 2 and up 18
  from iteration 0. Deck 11 Mega Lucario ex was strongest at 301 / 390 overall
  and 87 / 120 against the four required sample opponents; deck 1 Alakazam
  Dudunsparce remained weakest at 55 / 390 overall and 4 / 120 against the
  four required sample opponents. Next action: while iteration-4 eval runs, use
  the open ERAWAN slot for iteration-5 online self-play from
  `models/rl/phase5_league_alpha/iter-0004/specialists`; do not start
  iteration-5 PPO until the iteration-5 self-play report is inspected.
- Iteration-5 online self-play passed on ERAWAN job `73456` while iteration-4
  eval was running. Source checkpoints:
  `models/rl/phase5_league_alpha/iter-0004/specialists`. The raw window
  produced 1,300 / 1,300 games, 214,943 trajectory rows, 14 draws, 27 timeouts,
  and 0 errors under
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0005/raw_train/`.
  Deck 11 remained strongest in self-play at 140 / 204 wins, followed by deck
  9 at 126 / 204 and deck 2 at 124 / 204. Deck 1 remained weakest at 24 / 204;
  deck 3 dropped to 52 / 191. Next action: run the iteration-5 PPO specialist
  update from source iteration 4 to target iteration 5 while waiting for
  iteration-4 eval, then inspect both reports before cleaning active raw
  windows.
- Iteration-5 online PPO update passed on ERAWAN job `73460`. It consumed all
  214,943 iter-5 on-policy trajectory rows, skipped 0 no-target rows and 0
  off-policy rows, and wrote all 13 checkpoints under
  `models/rl/phase5_league_alpha/iter-0005/specialists`. Deck 1 had the most
  PPO examples at 28,938; deck 12 had the fewest at 7,702. Next action: queue
  iteration-6 online self-play from
  `models/rl/phase5_league_alpha/iter-0005/specialists` while iteration-4 eval
  continues to run; do not start iteration-6 PPO until the iteration-6
  self-play report is inspected.
- Iteration-4 full-agent-vs-rule evaluation completed from
  `models/rl/phase5_league_alpha/iter-0004/specialists`: 2,692 / 5,070 wins,
  0.5310 win rate, 14 draws, 39 timeouts, 0 errors. This is down 5 wins from
  iteration 3 and down 18 wins from iteration 2, but up 41 wins over iteration
  0. Against the four required sample rule-agent opponents, iteration 4 scored
  686 / 1,560, up 9 from iteration 3 but down 8 from iteration 2. Deck 11 Mega
  Lucario ex remained strongest at 303 / 390 overall and 81 / 120 against the
  required sample opponents; deck 1 Alakazam Dudunsparce remained weakest at
  53 / 390 overall and 4 / 120 against the required sample opponents. Do not
  promote iteration 4 over iteration 2 on the overall gate.
- Iteration-6 online self-play passed on ERAWAN job `73470` while iteration-4
  eval was finishing. Source checkpoints:
  `models/rl/phase5_league_alpha/iter-0005/specialists`. The raw window
  produced 1,300 / 1,300 games, 216,080 trajectory rows, 9 draws, 30 timeouts,
  and 0 errors under
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0006/raw_train/`.
  Deck 11 remained strongest in self-play at 126 / 204 wins, followed by deck
  10 at 121 / 204 and deck 2 at 120 / 191. Deck 1 remained weakest at
  28 / 191. Next action: run iteration-6 PPO from source iteration 5 to target
  iteration 6, and run iteration-5 full-agent-vs-rule eval in the other ERAWAN
  slot.
- Iteration-6 online PPO update passed on ERAWAN job `73489`. It consumed all
  216,080 iter-6 on-policy trajectory rows, skipped 0 no-target rows and 0
  off-policy rows, and wrote all 13 checkpoints under
  `models/rl/phase5_league_alpha/iter-0006/specialists`. Deck 1 had the most
  PPO examples at 28,825; deck 12 had the fewest at 7,564. Next action: while
  iteration-5 eval continues to run, queue iteration-7 online self-play from
  `models/rl/phase5_league_alpha/iter-0006/specialists`; do not start
  iteration-7 PPO until the iteration-7 self-play report is inspected.
- Iteration-7 online self-play passed on ERAWAN job `73496` while iteration-5
  eval was running. Source checkpoints:
  `models/rl/phase5_league_alpha/iter-0006/specialists`. The raw window
  produced 1,300 / 1,300 games, 211,335 trajectory rows, 6 draws, 21 timeouts,
  and 0 errors under
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0007/raw_train/`.
  Deck 11 remained strongest in self-play at 133 / 191 wins, tied by deck 2 at
  133 / 204 on raw wins; deck 10 followed at 122 / 191. Deck 1 remained weakest
  at 20 / 204. Next action: queue iteration-6 full-agent-vs-rule eval from
  `models/rl/phase5_league_alpha/iter-0006/specialists` so iteration-5 and
  iteration-6 evals can run overnight; after either eval slot frees, run
  iteration-7 PPO from source iteration 6 to target iteration 7.
- Iteration-5 full-agent-vs-rule evaluation passed on ERAWAN job `73490`:
  2,714 / 5,070 wins, 0.5353 win rate, 15 draws, 46 timeouts, 0 errors. This
  is up 4 wins from iteration 2, up 22 wins from iteration 4, and up 63 wins
  from iteration 0. Against the four required sample rule-agent opponents,
  iteration 5 scored 704 / 1,560, up 10 wins from iteration 2 and up 45 wins
  from iteration 0. Treat
  `models/rl/phase5_league_alpha/iter-0005/specialists` as the current best
  promotion/package candidate unless a later evaluation beats it; deck 11 Mega
  Lucario ex and deck 12 Mega Abomasnow ex are the top required-sample slice
  candidates at 86 / 120 and 75 / 120 respectively.
- Iteration-6 full-agent-vs-rule evaluation passed on ERAWAN job `73503`:
  2,654 / 5,070 wins, 0.5235 win rate, 16 draws, 50 timeouts, 0 errors. This
  is down 60 wins from iteration 5 and down 56 wins from iteration 2, and only
  3 wins above the recorded iteration-0 specialist baseline. Against the four
  required sample rule-agent opponents, iteration 6 scored 684 / 1,560, down
  20 wins from iteration 5. Do not promote iteration 6. Next action: run
  iteration-7 PPO from source iteration 6 to target iteration 7, using the
  already inspected iteration-7 raw self-play window.
- Iteration-7 online PPO update passed on ERAWAN job `73508`. It consumed all
  211,335 iter-7 on-policy trajectory rows, skipped 0 no-target rows and 0
  off-policy rows, and wrote all 13 checkpoints under
  `models/rl/phase5_league_alpha/iter-0007/specialists`. Deck 1 had the most
  PPO examples at 26,376; deck 12 had the fewest at 7,616. Iteration 5 remains
  the current best promotion/package candidate until a later evaluation beats
  it. Next action: run iteration-7 full-agent-vs-rule eval and iteration-8
  online self-play in parallel from
  `models/rl/phase5_league_alpha/iter-0007/specialists`.
- Evaluation time-series artifacts were generated for the available
  full-agent-vs-rule reports covering iterations 0, 2, 3, 4, 5, and 6:
  `reports/phase5_alpha_eval_winrate_timeseries_combined.svg`,
  `reports/phase5_alpha_eval_winrate_timeseries_overall.svg`,
  `reports/phase5_alpha_eval_winrate_timeseries_per_deck.svg`,
  `reports/phase5_alpha_eval_winrate_timeseries.html`, and
  `reports/phase5_alpha_eval_winrate_timeseries_summary.csv`. Iteration 1 is
  omitted because no local iteration-1 full-agent-vs-rule JSON report was
  available. The plots preserve iteration 5 as the current best checkpoint at
  2,714 / 5,070 wins.
- Iteration-8 online self-play passed on ERAWAN job `73526` while iteration-7
  eval was running. Source checkpoints:
  `models/rl/phase5_league_alpha/iter-0007/specialists`. The raw window
  produced 1,300 / 1,300 games, 208,449 trajectory rows, 9 draws, 22 timeouts,
  and 0 errors under
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0008/raw_train/`.
  Deck 2 was strongest in self-play at 138 / 204 wins, followed by deck 11 at
  129 / 204 and deck 12 at 120 / 204. Deck 1 remained weakest at 29 / 204.
  Next action: while iteration-7 eval continues, run iteration-8 PPO from
  source iteration 7 to target iteration 8. Iteration 5 remains the current
  best promotion/package candidate until a later evaluation beats it.
- Iteration-8 online PPO update passed on ERAWAN job `73542`. It consumed all
  208,449 iter-8 on-policy trajectory rows, skipped 0 no-target rows and 0
  off-policy rows, and wrote all 13 checkpoints under
  `models/rl/phase5_league_alpha/iter-0008/specialists`. Deck 13 had the most
  PPO examples at 25,342; deck 12 had the fewest at 7,794. Next action: while
  iteration-7 eval continues, queue iteration-9 online self-play from
  `models/rl/phase5_league_alpha/iter-0008/specialists`. Iteration 5 remains
  the current best promotion/package candidate until a later evaluation beats
  it.
- Iteration-9 online self-play passed on ERAWAN job `73550` while iteration-7
  eval was still running. Source checkpoints:
  `models/rl/phase5_league_alpha/iter-0008/specialists`. The raw window
  produced 1,300 / 1,300 games, 208,155 trajectory rows, 11 draws, 28 timeouts,
  and 0 errors under
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0009/raw_train/`.
  Deck 11 was strongest in self-play at 145 / 204 wins, followed by deck 2 at
  131 / 191, deck 12 at 130 / 204, and deck 10 at 125 / 204. Deck 1 remained
  weakest at 24 / 191, with decks 3 and 5 also under 30% online win rate. Next
  action: while iteration-7 eval continues, run iteration-9 PPO from source
  iteration 8 to target iteration 9. Iteration 5 remains the current best
  promotion/package candidate until a later evaluation beats it.
- Iteration-7 full-agent-vs-rule evaluation passed on ERAWAN job `73525`:
  2,641 / 5,070 wins, 0.5209 win rate, 18 draws, 52 timeouts, 0 errors. This
  is down 73 wins from iteration 5 and down 13 wins from iteration 6. Against
  the four required sample rule-agent opponents, iteration 7 scored
  639 / 1,560, down 65 wins from iteration 5 and down 45 wins from iteration 6.
  Do not promote iteration 7; iteration 5 remains the current best
  promotion/package candidate. Deck 11 Mega Lucario ex remained the strongest
  evaluated deck at 298 / 390 overall and 80 / 120 against the required sample
  opponents; deck 1 Alakazam Dudunsparce remained weakest at 60 / 390 overall
  and 0 / 120 against required sample opponents.
- Iteration-9 online PPO update passed on ERAWAN job `73571`. It consumed all
  208,155 iter-9 on-policy trajectory rows, skipped 0 no-target rows and 0
  off-policy rows, and wrote all 13 checkpoints under
  `models/rl/phase5_league_alpha/iter-0009/specialists`. Deck 1 had the most
  PPO examples at 26,948; deck 12 had the fewest at 8,288. The iter-9 raw
  training window can be deleted after confirming the report and checkpoints
  are retained. Next action: run iteration-8 full-agent-vs-rule eval from
  `models/rl/phase5_league_alpha/iter-0008/specialists` and iteration-10
  online self-play from
  `models/rl/phase5_league_alpha/iter-0009/specialists` in parallel.
- Iteration-10 online self-play passed on ERAWAN job `73578` while iteration-8
  eval was still running. Source checkpoints:
  `models/rl/phase5_league_alpha/iter-0009/specialists`. The raw window
  produced 1,300 / 1,300 games, 211,241 trajectory rows, 9 draws, 22 timeouts,
  and 0 errors under
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0010/raw_train/`.
  Deck 2 was strongest in self-play at 134 / 204 wins, followed by deck 12 at
  124 / 191, deck 11 at 120 / 191, and deck 8 at 120 / 204. Deck 1 remained
  weakest at 30 / 204, with deck 3 also below 30% online win rate. Next action:
  while iteration-8 eval continues, run iteration-10 PPO from source iteration
  9 to target iteration 10. Iteration 5 remains the current best
  promotion/package candidate until a later evaluation beats it.
- Iteration-10 online PPO update passed on ERAWAN job `73587`. It consumed all
  211,241 iter-10 on-policy trajectory rows, skipped 0 no-target rows and 0
  off-policy rows, and wrote all 13 checkpoints under
  `models/rl/phase5_league_alpha/iter-0010/specialists`. Deck 1 had the most
  PPO examples at 26,282; deck 12 had the fewest at 7,664. Per user direction,
  stop online RL collection and PPO updates at iteration 10. Do not queue
  iteration-11 self-play or PPO unless the plan changes. The iter-10 raw
  training window can be deleted after confirming the report and checkpoints
  are retained. Next action: finish the evaluation backlog only, covering
  iteration 8, iteration 9, and iteration 10 full-agent-vs-rule evaluations.
  Iteration 5 remains the current best promotion/package candidate until a
  later evaluation beats it.
- Final evaluation backlog completed on ERAWAN jobs `73577`, `73593`, and
  `73597`. Iteration 8 scored 2,699 / 5,070 wins, 0.5323 win rate,
  13 draws, 45 timeouts, and 0 errors; required-sample slice was
  671 / 1,560. Iteration 9 scored 2,631 / 5,070 wins, 0.5189 win rate,
  23 draws, 57 timeouts, and 0 errors; required-sample slice was
  636 / 1,560. Iteration 10 scored 2,646 / 5,070 wins, 0.5219 win rate,
  17 draws, 39 timeouts, and 0 errors; required-sample slice was
  690 / 1,560. None beat iteration 5, so
  `models/rl/phase5_league_alpha/iter-0005/specialists` remains the current
  best promotion/package candidate at 2,714 / 5,070 overall and
  704 / 1,560 against the four required sample rule-agent opponents.
- Final eval dashboard artifacts were generated from the uploaded evaluation
  JSONs using `scripts/analysis/phase5_alpha_eval_dashboard.py`:
  `reports/phase5_alpha_eval_dashboard.html`,
  `reports/phase5_alpha_eval_dashboard_summary.csv`,
  `reports/phase5_alpha_eval_dashboard_per_deck.csv`, and
  `reports/phase5_alpha_eval_dashboard_matchups.csv`. The dashboard covers
  iterations 0, 2, 3, 4, 5, 6, 7, 8, 9, and 10; iteration 1 remains absent
  because no eval report is available. Raw uploaded eval reports were not
  copied into the repo to avoid ERAWAN untracked-file pull conflicts.
- Phase 5 Alpha league conclusion as of July 7, 2026: stop online RL at
  iteration 10, keep iteration 5 as the promotion/package candidate, and treat
  deck 1 Alakazam Dudunsparce as the main persistent weakness for any future
  targeted work. Deck 11 Mega Lucario ex remains the strongest specialist, and
  deck 12 Mega Abomasnow ex is the clearest late-iteration improvement.
- Active Phase 5 training pivot as of July 7, 2026: behavior
  cloning/bootstrap helped, but the generic self-play PPO loop did not improve
  beyond iteration 5 and Kaggle replay inspection showed tactical failures such
  as Mega Abomasnow ex missing available energy attachments and attacks. Do not
  resume generic iteration-11 self-play. The next active path is specialized
  public/sample rule-opponent curriculum training.
- Specialized public-agent support now includes a built-in public-20-plus-sample-4
  roster, local `.py`/`.ipynb` public-agent discovery, availability reports,
  `rl-evaluate-phase5-public-agents` with a 50% public-agent gate summary, and
  `rl-generate-phase5-public-agent-trajectories` for PPO update data against
  fixed specialized opponents. Current starting checkpoint remains
  `models/rl/phase5_league_alpha/iter-0005/specialists`; evaluate it against
  the specialized public-agent roster before generating the next PPO window.
- First specialized public-agent eval for iteration 5 ran on ERAWAN job `73657`.
  Only the built-in `Official sample Dragapult ex` adapter was available during
  eval; the other 23 roster entries need exported local agent files under the
  public-agent root. Iteration 5 scored only 50 / 390 wins, 0.1282 win rate,
  with 0 timeouts and 0 errors, so the specialized public-agent gate failed
  decisively. Next action: generate `phase5-rl` trajectories from iteration 5
  specialists against the available specialized public-agent roster, run a PPO
  update from that public-agent trajectory file into the separate
  `models/rl/phase5_public_agent_curriculum/iter-0006/specialists` checkpoint
  root, then re-evaluate against the specialized public-agent gate. Do not
  overwrite the historical generic Alpha league `iter-0006` artifacts.
- Public-agent curriculum trajectory job `73671` completed from iteration 5
  specialists against built-in sample Dragapult: 130 / 130 games started,
  6,729 trajectory rows, 10 wins, 120 losses, 0 draws, 0 timeouts, and
  0 errors. The raw JSONL is
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_public_agent_rule_train/iter-0006_public_agent_trajectories.jsonl`.
  Next action is the PPO update into
  `models/rl/phase5_public_agent_curriculum/iter-0006/specialists`, followed by
  specialized public-agent eval of that candidate.
- Public-agent curriculum PPO job `73706` completed from the targeted
  sample-Dragapult trajectory window. It wrote 13 / 13 deck checkpoints under
  `models/rl/phase5_public_agent_curriculum/iter-0006/specialists`, consumed
  6,729 on-policy examples, and skipped 0 no-target/off-policy rows. Next
  action: evaluate this checkpoint family with
  `rl-evaluate-phase5-public-agents` / `phase5_public_agent_eval_conda.sbatch`
  against the same specialized public-agent gate.
- Public-agent curriculum iteration-6 eval job `73711` regressed against the
  only available specialized opponent, built-in sample Dragapult: 38 / 390
  wins, 0.0974 win rate, 0 timeouts, and 0 errors, down from the iteration-5
  public-agent baseline of 50 / 390. Do not continue by repeating the same
  loss-heavy public-agent PPO loop. Keep
  `models/rl/phase5_league_alpha/iter-0005/specialists` as the current best
  packaged checkpoint family and treat
  `models/rl/phase5_public_agent_curriculum/iter-0006/specialists` as a failed
  targeted PPO experiment. Next work should change the training pipeline or add
  decision-level diagnostics before spending more curriculum compute.
- The current `phase5-rl` data collector is stochastic neural-policy sampling
  with temperature, not epsilon-greedy random legal-action exploration. The
  next recommended diagnostic is a small one-deck, one-opponent micro
  experiment: deck 12 Mega Abomasnow ex vs built-in `sample_dragapult`, 100
  on-policy games from the iteration-5 specialist checkpoint, update only deck
  12 into
  `models/rl/phase5_public_agent_micro/deck12_vs_sample_dragapult_100/specialists`,
  and compare a 30-game single-matchup eval against the same-deck baseline.
  Public-agent commands and SLURM wrappers now support `PUBLIC_AGENT_KEYS` and
  deck filters so this can run without touching the broad checkpoint families.
- Deck 12 vs built-in `sample_dragapult` micro experiment completed on ERAWAN:
  baseline iteration-5 specialist scored 6 / 30 wins; 100-game `phase5-rl`
  collection produced 8 / 100 wins and 2,527 trajectory rows; deck-12-only PPO
  consumed all 2,527 on-policy examples with mean advantage -1.1553; post-update
  eval scored 8 / 30 wins. The path is operationally valid, but the improvement
  is too small/noisy and still far below the 50% gate. Do not scale this exact
  sparse terminal-reward PPO loop; next work should change the training signal
  with denser tactical rewards/diagnostics, successful-trajectory weighting, or
  supervised rule-specialist targets.
- Public-agent trajectory generation now has opt-in tactical reward shaping.
  Defaults preserve old behavior, but the next deck-12 diagnostic should use
  `OUTCOME_REWARD_SCALE=0.25` and `TACTICAL_REWARD_MODE=basic` so attack and
  energy-attachment decisions receive per-step credit/penalty. The shaped
  trajectory report includes `reward_shaping` and `tactical_reward_summary`;
  inspect those counts before deciding whether to tune PPO, reward weights, or a
  supervised rule-specialist target.
- Shaped deck-12 vs built-in `sample_dragapult` trajectory job `73752`
  completed: 100 / 100 games, 8 wins, 92 losses, 2,626 trajectory rows,
  0 timeouts, and 0 errors. Tactical summary showed 868 attack opportunities
  with 353 attacks taken and 39 conservative missed attacks, plus 976 attach
  opportunities with 449 attachments taken and 134 conservative missed
  attachments. This is a useful shaped PPO input; proceed with the deck-12-only
  PPO update/eval, but scale no further unless the held-out eval clearly beats
  the previous noisy +2 / 30 result.
- Tactical-shaped deck-12 PPO eval job `73757` scored 8 / 30 wins against
  built-in `sample_dragapult`, matching the unshaped PPO result and only +2 wins
  above the 6 / 30 iteration-5 baseline. The shaped PPO update consumed all
  2,626 on-policy examples and moved mean advantage from the unshaped -1.1553
  to -0.5657, but that better training signal did not improve held-out play. Do
  not scale shaped PPO to more decks. Next work should pivot to stronger direct
  supervision or diagnostics: rule/search-selected tactical targets for missed
  attack/attachment states, richer action-level reports, or checking whether
  `phase5-full` search/heuristic selection is overriding the learned policy.
- Pure-neural ablation of the tactical-shaped deck-12 checkpoint scored only
  7 / 100 wins against built-in `sample_dragapult`, with 0 timeouts/errors.
  This is close to the shaped collection result of 8 / 100 and much worse than
  `phase5-full` at 8 / 30. Do not remove search/heuristics from the final agent;
  they are currently helping. Use pure neural eval only as a diagnostic while
  the next implementation targets direct supervision/action-level diagnostics
  for missed attack and attachment decisions.
- Current score-scaling diagnostic: public-agent `phase5-full` evals can now
  write root-search candidate traces with game outcomes using
  `SEARCH_TRACE_OUTPUT` / `--search-trace-output`, and
  `scripts/slurm/phase5_search_score_components_conda.sbatch` summarizes
  tactical, neural, prior, and combined score ranges by win/loss sequence.
  Use this before changing `RootSearchConfig` weights; existing eval
  JSON/markdown reports do not contain discarded candidate component values.
- Deck-12 score-component result on July 8, 2026: traced `phase5-full` eval
  against built-in `sample_dragapult` scored 23 / 100. The selected
  `tactical_score` range was 101.5 versus prior range 0.08, and the active
  config had neural action-Q/tactical weights at 0.0. Winning traces had much
  higher selected-minus-baseline combined margins than losing traces, while
  losing traces showed higher immediate damage delta. Next search-weight work
  should normalize rollout tactical score before combining and test small
  policy-prior weights; do not blindly enable neural Q/tactical weights yet.
- Leaf state-value search ablation is now implemented but not yet run. Root
  search can opt into normalized rollout tactical scoring and normalized
  `state_value` leaf scoring with `--normalize-tactical-score`,
  `--tactical-score-weight`, and `--leaf-state-value-weight`. The next ERAWAN
  experiment is deck 12 vs built-in `sample_dragapult`: collect 500
  rule-vs-rule trajectories, train
  `models/rl/phase5_public_agent_micro/deck12_rule_bootstrap_value/specialists/deck-12.pt`
  from that narrow rule window with value loss enabled, then compare normalized
  tactical-only search against normalized tactical plus leaf-value search.
- Deck-12 leaf state-value eval result on July 8, 2026: same checkpoint
  `deck12_rule_bootstrap_value` scored 20 / 100 with normalized tactical-only
  search and 29 / 100 with normalized tactical 0.5 + policy prior 0.25 + leaf
  state value 0.5. Both had 0 errors/timeouts. This is positive evidence that
  learned leaf value adds useful signal, but it still fails the 50% gate. Next
  inspect score-component diagnostics for both traces and the missing
  rule-vs-rule trajectory/train reports before scaling.
- Clean deck-12 score-mixture ablation on July 9, 2026: with the same
  `deck12_rule_bootstrap_value` checkpoint and built-in `sample_dragapult`
  matchup, policy prior 0.5 + tactical score 0.5 scored 23 / 100, while policy
  prior 0.5 + leaf state value 0.5 scored 28 / 100. Both had 0
  errors/timeouts. This confirms leaf state value is more useful than the
  handcrafted tactical leaf score in this setup, but still well below the 50%
  gate. Next inspect the two 50:50 trace files with the score-component SLURM
  diagnostic before more weight tuning.
- Phase 5 policy size diagnostic on July 9, 2026: the active
  `AlphaStarTurnPolicy` config has 542,596 trainable parameters per specialist,
  or 7,053,748 if all 13 stored deck specialists are counted together.
  AlphaStar's public Nature supplementary architecture is much deeper and wider
  but does not publish a single verified total parameter count; do not treat the
  current deck-12 failure as a pure model-size problem until the trace/value
  target diagnostics are cleaner.
- One-deck epsilon curriculum implementation on July 9, 2026: added an
  opt-in 10-generation Dragapult-vs-Lucario script that starts from a scratch
  `deck-101.pt` checkpoint, trains the model-controlled official sample
  Dragapult ex deck against built-in rule-based sample Mega Lucario ex, decays
  epsilon-greedy exploration from 1.0 to 0.10, collects 1000 games per
  generation, runs a 100-game zero-exploration eval after each PPO update,
  deletes each raw trajectory JSONL after successful update, and keeps one win
  plus one loss compact JSON/HTML replay view per generation. Submit
  `scripts/slurm/phase5_one_deck_public_epsilon_curriculum.sbatch` as the next
  ERAWAN job for this experiment.
- One-deck epsilon curriculum job 73798 on July 9, 2026: generation-1
  trajectory collection completed with 1000 / 1000 games, 41,431 trajectory
  steps, 57 / 943 / 0 wins/losses/draws, and 0 errors/timeouts at epsilon 1.0,
  then failed before PPO update with `Unknown Phase 5 league deck index: 101`.
  Root cause was using the 13-deck league specialist PPO wrapper for synthetic
  public deck index 101. The script now uses single-checkpoint
  `rl-train-phase5-ppo --deck-index-filter 101 --require-on-policy` and supports
  `REUSE_EXISTING_TRAJECTORIES=1` for retrying from the completed gen-0001 raw
  JSONL if it remains on ERAWAN.
- One-deck epsilon curriculum retry 73820 on July 9, 2026: the
  Dragapult-vs-Lucario 10-generation scratch PPO curriculum completed end to
  end with 1000 train games and 100 zero-exploration eval games per generation.
  There were no eval errors/timeouts, but the policy did not improve: best eval
  was generations 6-7 at 5 / 100 wins and final generation 10 was 2 / 100 wins.
  As epsilon decayed, attack/attach rates collapsed while `END` selections rose
  from 0.1403 at generation 1 to 0.5777 at generation 10. Do not scale this same
  sparse-reward curriculum; next try dense tactical rewards, explicit penalties
  for ending while useful attack/attach choices are available, or a
  rule-demonstration warm-start before PPO.
- One-deck mixed rule/epsilon curriculum implementation on July 9, 2026: added
  `scripts/slurm/phase5_one_deck_public_mixed_curriculum.sbatch`. It creates a
  retained generation-0 1000-game rule-vs-rule Dragapult-vs-Lucario bootstrap
  dataset, then each generation trains `deck-101.pt` on that retained rule data
  plus a fresh 1000-game epsilon-model-vs-rule window from the previous
  checkpoint. Only the fresh epsilon JSONL is deleted after PPO update; the rule
  bootstrap JSONL is kept and reused. The script intentionally leaves
  `--require-on-policy` off by default so off-policy rule demonstrations are not
  filtered out.
- One-deck mixed rule/epsilon curriculum job 73877 on July 10, 2026: completed
  all 10 generations with 0 eval errors/timeouts. Retained rule bootstrap was
  424 / 1000 wins with attack/attach/end rates 0.2586 / 0.4127 / 0.0160. The
  mixed curriculum fixed the epsilon-only END-collapse pattern: generation-10
  epsilon-window rates were 0.2388 / 0.4169 / 0.0335, compared with
  epsilon-only generation-10 rates of 0.0084 / 0.0294 / 0.5777. Best eval was
  generation 7 at 52 / 100 wins; generation 5 reached 50 / 100; generation 10
  was 45 / 100. Current candidate checkpoint is
  `models/rl/phase5_one_deck_public_mixed/phase5_dragapult_vs_lucario_mixed/gen-0007/specialists/deck-101.pt`.
  Run a larger zero-exploration confirmation eval for generation 7 before
  packaging or promoting.
- One-deck mixed gen7 confirmation job 73901 on July 10, 2026: 1000-game eval
  of gen7 against rule Lucario scored 462 / 1000 wins, 537 losses, 1 draw, and
  0 timeouts/errors. This beats the retained rule-vs-rule baseline of 424 /
  1000 wins: fixed-baseline one-sided binomial `p ~= 0.0084`, sampled-baseline
  two-proportion one-sided `p ~= 0.0436`. It still fails the 50% target, with a
  Wilson 95% interval around 0.431-0.493. Treat mixed gen7 as a real
  baseline-beating improvement but not a promoted/package-ready policy; continue
  from gen7 with the rule bootstrap anchor retained.
- Rule-only epoch diagnostic implementation on July 11, 2026: added
  `--initial-checkpoint` to `rl-train-phase5-generalist` and added
  `scripts/slurm/phase5_one_deck_rule_epoch_bc.sbatch`. The script trains two
  synthetic public-deck specialists from rule-vs-rule trajectories only:
  Dragapult `deck-101.pt` from rule Dragapult vs rule Lucario, and Lucario
  `deck-102.pt` from rule Lucario vs rule Dragapult. It evaluates trained
  Dragapult vs rule Lucario and trained Lucario vs rule Dragapult after every
  supervised epoch, with defaults of min 10 epochs, max 50 epochs, and early
  stop after 10 non-improving combined-eval epochs.
- Rule-only epoch diagnostic job 73948 on July 11, 2026: completed epochs 1-18
  and early-stopped after 10 non-improving combined-eval epochs. Eval had 36
  reports, 0 errors/timeouts, and 4 draws. Baselines were Dragapult rule vs
  Lucario rule 424 / 1000, Lucario rule vs Dragapult rule 191 / 1000, combined
  615 / 2000. Best epoch was epoch 8: combined 77 / 200, Dragapult 50 / 100,
  Lucario 27 / 100. Combined and Lucario beat baseline statistically, but
  Dragapult did not reach the 52 / 100 threshold over its 0.424 baseline. Both
  train accuracies reached 1.000 by epoch 2, so more rule-only epochs are not
  the main fix. Next diagnostic should use rule-teacher dataset aggregation:
  collect model-visited states, label them with rule-agent choices, append to
  bootstrap data, and retrain.
- One-deck rule-teacher DAgger implementation on July 12, 2026: added optional
  teacher-label recording to public-agent trajectory generation via
  `--teacher-agent rule`, added deterministic `--agent phase5-symbolic` support
  to that generator, and added
  `scripts/slurm/phase5_one_deck_rule_teacher_dagger.sbatch`. The script starts
  from the rule-only epoch-8 Dragapult/Lucario checkpoint family, retains the
  original rule bootstrap data, aggregates model-visited correction windows
  labeled by the rule teacher, retrains `deck-101.pt` and `deck-102.pt`, and
  evaluates both one-deck directions after each iteration. Submit this DAgger
  diagnostic next using the runbook defaults.
- One-deck rule-teacher DAgger job 73988 on July 12, 2026: completed base eval
  plus iterations 1-10 with 2200 eval games, 0 errors/timeouts, and 6 draws.
  The base checkpoint was best at 80 / 200 combined wins: Dragapult 52 / 100
  and Lucario 28 / 100. The best post-update checkpoint was iteration 5 at 71 /
  200, and final iteration 10 fell to 47 / 200. Root cause appears to be a
  trainer-context bug in teacher-forced trajectories: teacher targets were used
  both as labels and as previous-action history, while later states had actually
  followed model behavior actions. The trainer now uses
  `behavior_selected_indices` for previous-action context when
  `teacher_forced_target` is set, while keeping the teacher action as the
  supervised target. Rerun DAgger after pulling this fix; treat job 73988 as a
  pre-fix diagnostic only.
- Fixed one-deck rule-teacher DAgger job 74006 on July 13, 2026: completed
  base eval plus iterations 1-9 with 2000 eval games, 0 errors/timeouts, and 4
  draws. Best combined checkpoint was iteration 4:
  `models/rl/phase5_one_deck_rule_teacher_dagger/phase5_dragapult_lucario_rule_teacher_dagger_fixed/iter-0004/specialists`.
  Iteration 4 scored 78 / 200 combined, Dragapult 53 / 100 vs rule Lucario,
  and Lucario 25 / 100 vs rule Dragapult. This is a small positive signal
  after the teacher-context fix, but not promotion-ready: Lucario remains weak
  and the 100-game eval is noisy. Next run should be a larger zero-exploration
  confirmation eval for iteration 4 in both directions, preferably 1000 games
  per matchup, before running more DAgger iterations.
- Fixed DAgger iteration-4 confirmation jobs 74012 and 74013 on July 17, 2026:
  the 1000-game eval did not confirm promotion. Dragapult vs rule Lucario scored
  449 / 1000 with 3 draws, 0 errors/timeouts; Lucario vs rule Dragapult scored
  202 / 1000 with 0 errors/timeouts. Combined was 651 / 2000, only a small
  lift over the 0.3075 combined rule baseline (`p ~= 0.0432` fixed-baseline
  one-sided binomial). Dragapult did not significantly beat its 0.424
  rule-vs-rule baseline (`p ~= 0.0587`) and was below the earlier mixed gen7
  confirmation of 462 / 1000; Lucario did not meaningfully improve over its
  0.191 baseline. Stop scaling this two-deck DAgger loop; next inspect
  action-quality failures and split follow-up experiments by deck.
- Fractional prize reward shaping was implemented on July 17, 2026 for
  public-agent trajectory generation. New tactical reward modes
  `fractional-prize` and `basic-fractional-prize` reward changes in
  `prizes_taken + partial-KO progress`, with Mega ex / regular ex / normal
  prize values of 3 / 2 / 1. The shaped scalar is
  `fractional_prize_weight * (our_delta - fractional_opponent_weight * opponent_delta)`.
  The next queued diagnostic should be a clean Dragapult-vs-Lucario mixed
  rule/epsilon curriculum using `basic-fractional-prize`,
  `OUTCOME_REWARD_SCALE=0.0`, `TACTICAL_FRACTIONAL_PRIZE_WEIGHT=0.25`, and
  `TACTICAL_FRACTIONAL_OPPONENT_WEIGHT=1.0`, then compare eval generations
  against the retained rule-vs-rule baseline and the previous mixed gen7
  confirmation. The ERAWAN run was submitted as SLURM job `74307` with run name
  `phase5_dragapult_vs_lucario_fractional_prize`; the job id was written to
  `experiments/rl/phase5_one_deck_fractional_prize_job.txt`.
- Fractional prize mixed curriculum job 74307 completed on July 17, 2026 and
  is not promotable. The best zero-exploration eval was generation 9 at
  26 / 100 against rule Mega Lucario ex; generation 10 was 20 / 100. All ten
  generations were far below the retained Dragapult-vs-Lucario rule-vs-rule
  baseline around 424 / 1000 and below the earlier mixed gen7 confirmation of
  462 / 1000. The rule bootstrap window itself was healthy at 432 / 1000, but
  its average fractional-prize reward component was negative (`-0.00460`) while
  the basic attack/attach component was positive (`+0.01082`). Epsilon windows
  also had negative fractional-prize reward in every generation. Conclusion:
  the implementation runs, but comparing one controlled-agent decision board to
  the next controlled-agent decision board is probably not clean enough credit
  assignment for dense fractional prizes. Do not continue this curriculum
  as-is; first record immediate post-action board deltas or move fractional
  prize scoring into an attack/turn-end rollout evaluator, then retest with a
  small one-generation diagnostic.
- Post-action fractional-prize credit assignment was implemented on
  July 22, 2026. `run_battle(...)` now gives acting recorders an immediate
  post-`battle_select(...)` observation; `RecordingPolicyAgent` stores a
  same-perspective `post_action_board`; and public-agent fractional rewards use
  that board before falling back to next-frame/final-prize sources. The next
  ERAWAN diagnostics should run two three-generation Dragapult-vs-Lucario jobs
  in parallel: post-action `basic-fractional-prize` with fractional weight
  `0.25`, and `basic` only as the A/B baseline. Confirm
  `fractional_after_board_sources.post-action` in the trajectory reports before
  scaling again. These diagnostics were submitted as SLURM jobs `74745`
  (`phase5_dragapult_vs_lucario_postaction_frac025_diag`) and `74746`
  (`phase5_dragapult_vs_lucario_basic_only_diag`).
- Post-action A/B jobs `74745` and `74746` completed on July 22, 2026. The
  implementation sanity check passed: the fractional run used
  `fractional_after_board_sources.post-action` for every shaped step, and the
  rule-bootstrap fractional component became positive (`+0.00542` average).
  The training result was still bad because both diagnostics set
  `OUTCOME_REWARD_SCALE=0.0`; gen-3 evals were only 8 / 200 for
  post-action fractional and 12 / 200 for basic-only. By gen 3, both dense-only
  runs over-selected tactical actions, with attack taken rates around 0.89 and
  attach taken rates around 0.67-0.72. Conclusion: post-action fractional credit
  assignment is valid, but dense reward alone does not preserve the game
  objective. The next queued diagnostics restore terminal reward:
  `74766` (`phase5_dragapult_vs_lucario_outcome1_postaction_frac025_diag`) and
  `74767` (`phase5_dragapult_vs_lucario_outcome1_none_diag`).
- Outcome-plus-post-action-fractional job `74766` completed on July 22, 2026.
  Zero-exploration evals were 51 / 200, 44 / 200, and 45 / 200 across
  generations 1-3, versus a fresh rule-vs-rule baseline of 446 / 1000. The
  post-action source check passed for every shaped step and the average
  fractional component became positive by generation 3, but evaluation did
  not improve. The generation-3 epsilon window at 221 / 1000 was essentially
  equal to the generation-2 zero-exploration checkpoint, indicating that its
  apparent gain came from epsilon decay rather than the update. A training
  pipeline defect was also confirmed: `REQUIRE_ON_POLICY=0` sends retained
  rule records with `policy_on_policy=false` and `policy_logprob=0.0` through
  the PPO ratio loss, so the rule half is neither valid PPO nor behavior
  cloning. Wait for outcome-only control job `74767`, then separate supervised
  rule imitation from on-policy PPO before another scaled curriculum.
- Outcome-only control job `74767` completed on July 22, 2026. Its
  zero-exploration evals were 12 / 200, 47 / 200, and 19 / 200 across
  generations 1-3, compared with 51 / 200, 44 / 200, and 45 / 200 for terminal
  outcome plus post-action fractional shaping. Fractional shaping won the
  three-generation aggregate 140 / 600 to 78 / 600 and prevented the control's
  generation-3 attack collapse: outcome-only attack-taken rate fell to
  `0.0757`, versus `0.6330` with shaping. Neither arm approached the comparable
  rule-vs-rule baselines of 431-446 / 1000, so reward tuning is not the main
  blocker. Do not scale this trainer again. The next implementation should use
  supervised cross-entropy for retained rule demonstrations, PPO only for
  valid on-policy model records, and interleaved/mixed batches rather than
  streaming all rule data before all epsilon data. Retain terminal outcome plus
  post-action fractional reward as one corrected A/B arm.
- Official engine source audit on July 9, 2026: Kaggle discussion 717141 and
  the current competition data confirm `ptcg_engine/ptcgProgram 22` is the
  official C++ competition engine source. The repo's Python simulator remains a
  thin wrapper around the Kaggle `cg` package. Current packaged Phase 5 `cg`
  copies match Kaggle's current Python `api.py`, `game.py`, and `utils.py`, but
  the compiled binaries are older: our `libcg.so` hash is
  `75D7D619B56E5AD4C5CAEADC698E61FAECC650678DCEAB52AD687F08D5676BEB` while the
  current Kaggle sample `libcg.so` hash is
  `FFD89BF923525A3E6FEB5E6201E96A866C0F456895499ED5C4A566303CAAE67C`. Refresh
  packaged `cg` directories from the current Kaggle sample before the next
  Kaggle submission package.
- Full-agent scaffolds added on June 30, 2026:
  - reusable Phase 5 opponent-prior inference,
  - direct Kaggle zip packaging and raw-exec-safe generated `main.py`,
  - optional policy/action-Q/tactical neural priors inside root-search scoring,
  - self-play policy-pool checkpoint rotation and metadata,
  - `rl-train-phase5-ppo` plus `scripts/slurm/phase5_ppo_train_conda.sbatch`,
  - `rl-evaluate-phase5-league` plus `scripts/slurm/phase5_league_eval_conda.sbatch`,
  - `phase5-compare-benchmarks` for promotion report deltas.
- The concrete 13x13 league evaluation is agent-vs-rule: each of the 13 league
  decks controlled by the selected agent against each 13-deck rule-agent
  opponent, with player-order balance across games per matchup.
- Corrected one-deck training pipeline implemented on July 22, 2026:
  `phase5-epsilon-mixture` now collects differentiable softmax-plus-uniform
  exploration with exact behavior log probabilities, and
  `rl-train-phase5-bc-ppo` interleaves equal rule-BC and valid on-policy PPO
  examples in every optimizer step. Hard epsilon-greedy and off-policy rule
  records are rejected from the PPO half. The new isolated SLURM workflow is
  `scripts/slurm/phase5_one_deck_public_bc_ppo_curriculum.sbatch`; it retains
  one shared rule bootstrap and deletes only consumed per-generation policy
  JSONL files. Local Python validation passed; run its small ERAWAN smoke, then
  compare matched three-generation outcome-plus-fractional and outcome-only
  arms from identical initialization and exploration seeds.
- Corrected BC/PPO smoke job `74786` passed on ERAWAN on July 22, 2026 in
  `00:13:44`: 82,324 rule examples and 224 / 224 valid mixture-policy PPO
  examples, zero off-policy/mode/nonfinite skips, successful checkpoint and
  cleanup, and a non-scientific 3 / 4 eval. The full matched A/B is now running:
  job `74791` is terminal outcome plus post-action fractional weight `0.25`;
  job `74792` is outcome only. Both use 1,000 train and 200 eval games for
  three generations, epsilon `1.0 -> 0.55 -> 0.10`, and identical retained
  rule data, initialization seed, and exploration seed.
- Corrected A/B jobs `74791` and `74792` completed successfully on July 22,
  2026. Fractional evals were 83/200, 75/200, and 90/200; outcome-only evals
  were 84/200, 88/200, and 82/200. Their aggregates, 0.413 and 0.423, are much
  better than the flawed pipeline's 0.233 and 0.130 but statistically
  indistinguishable from one another and near the historical 0.431-0.446 rule
  baseline. BC reached 100% training accuracy, both arms had nearly identical
  attack/attach/END behavior, and eval did not improve monotonically. The
  corrected trainer therefore preserves rule behavior but has not shown
  online PPO improvement. A 50/50 full rule anchor is too strong for the next
  true-RL phase, and epsilon 1.0 gives zero policy gradient for the
  differentiable uniform mixture. Next use one-time generation-0 BC, then
  PPO-dominant updates with epsilon below 1.0 or high-temperature policy
  sampling, plus objective-gradient diagnostics. Matched 200-game
  rule-vs-rule baseline job `74836` failed before gameplay because the public
  evaluator CLI omitted its already-supported `rule` agent choice. The CLI and
  parser test were fixed, but retry `74837` exposed a second pregame guard: the
  wrapper's default specialist path was validated for the rule agent. Rule
  evaluation now ignores all model/specialist paths before validation; pull
  that fix and resubmit. Retry `74838` then completed successfully: the matched
  200-game rule baseline was 87 / 200 (`0.435`). Fractional generation 3 at
  90 / 200 and outcome-only generation 2 at 88 / 200 are only three and one
  wins above the teacher, respectively, and not statistically distinguishable.
  Final conclusion: corrected BC/PPO reaches teacher level but has not shown
  improvement from online PPO.
- A PPO-dominant successor was implemented on July 22, 2026 without replacing
  the completed balanced workflow. It adds one-time rule-trajectory BC,
  a `ppo-epoch` schedule that uses each online row once, optional 0-10% teacher
  anchoring with an explicit BC coefficient, and weighted gradient-norm/cosine
  diagnostics. The new isolated SLURM entry point is
  `scripts/slurm/phase5_one_deck_public_ppo_dominant_curriculum.sbatch`.
  The next gate is a matched no-anchor versus 10%-anchor ERAWAN smoke, followed
  by three generations at epsilon `0.90 -> 0.50 -> 0.10` only if both smokes
  validate checkpointing, gradients, evaluation, and online-data cleanup.
- PPO-dominant smoke jobs `74844` (no anchor) and `74847` (10% anchor) passed
  on July 22, 2026. Both used every valid online row once, rejected zero rows,
  recorded finite weighted objective gradients, completed clean evals, and
  removed all consumed JSONL files. The no-anchor arm had zero BC rows/gradient;
  the 10% arm used 22 / 511 sampled rule rows and measured BC/PPO-policy
  gradient norms `0.1287 / 0.1181`. Full matched jobs are now running:
  `74848` no anchor and `74849` 10% anchor, each with three generations,
  1,000 online games and 200 eval games per generation, epsilon
  `0.90 -> 0.50 -> 0.10`, terminal outcome plus post-action fractional shaping,
  and identical initialization/exploration seeds.
- PPO-dominant full jobs `74848` and `74849` completed on July 23, 2026. The
  no-anchor arm collapsed from 96 / 200 at BC to 50, 43, and 26 / 200. The 10%
  anchor preserved teacher-level strength at 87, 88, 95, and 88 / 200 but did
  not significantly improve it. Diagnosis found that game-sequential minibatch
  normalization cancels the win/loss component broadcast across each game's
  actions, while generation-1 value gradients exceed policy gradients by
  thousands of times on the shared encoder path. The next implementation adds
  opt-in dataset-global advantage normalization, head-only critic backprop, and
  shared-trunk gradient diagnostics. Next run a 10%-anchor A/B with global
  normalization in both arms and only critic scope (`shared` vs `head-only`)
  changed.
- Global-advantage smoke jobs `74862` (`shared`) and `74863` (`head-only`)
  passed on July 23, 2026. Both used every valid PPO row exactly once, recorded
  finite nontrivial dataset-global advantage statistics, completed clean evals,
  and removed all raw JSONLs. The shared critic produced shared-trunk value
  gradient norm `2.707040`; head-only produced exactly `0.0` while retaining a
  nonzero total critic gradient (`5.357164`), validating the detach boundary.
  Full matched jobs were then queued as `74864` shared critic and `74865`
  head-only critic. Both use a 10% rule anchor, global advantages, identical seed
  `20260723`, three generations, 1,000 online games and 200 deterministic eval
  games per generation, and epsilon `0.90 -> 0.50 -> 0.10`.
- Full global-advantage jobs `74864` and `74865` completed on July 23, 2026.
  Shared and head-only critic arms produced the same post-update aggregate,
  263 / 600 (`0.438`), statistically indistinguishable from the 87 / 200
  (`0.435`) rule baseline. Their generation 1-3 scores were 88/91/84 and
  91/93/79. The best trained checkpoint, head-only generation 2 at 93 / 200,
  remains seven wins short of tying the raw 50% one-matchup gate; no broader
  all-specialist gate is yet validated. Critic detachment works as designed but
  is not sufficient. PPO ratios remained essentially `1.0` and clipping was
  almost exactly zero because every online row was used only once. The next
  controlled test keeps global advantages and head-only critic fixed and
  compares four versus eight PPO epochs. BC+PPO reuse-factor reporting now uses
  actual examples consumed across all epochs.
- Multi-epoch PPO smoke jobs `74882` (4 epochs) and `74883` (8 epochs) passed
  on July 23, 2026. Actual online reuse was exactly `4.0` and `8.0`; mean clip
  fraction increased from effectively zero in one-pass PPO to `0.0598` and
  `0.1660`, while weighted policy-gradient norms increased to `0.0978` and
  `0.1157`. Both retained exactly zero shared critic gradient, clean evals, and
  zero raw JSONLs after update. Full matched jobs are now queued as `74885`
  (4 epochs) and `74886` (8 epochs), each with three generations, 1,000 online
  games and 200 deterministic eval games per generation. Current goal status is
  unchanged pending those full evals: best checkpoint 95 / 200 (`0.475`), no
  one-matchup 50% pass, and no validated broader all-specialist pass.
- Full multi-epoch jobs `74885` and `74886` completed on July 24, 2026. The
  four-epoch arm scored 96/82/85 across online generations and the eight-epoch
  arm 77/88/85, all out of 200; aggregates were `0.438` and `0.417`. Four-epoch
  generation 1 at 96 / 200 (`0.480`) is the new best trained checkpoint but is
  four wins short of tying the raw 50% gate and is not significantly above its
  BC checkpoint or the rule baseline. Full-data clipping remained essentially
  zero despite exact 4/8 reuse, showing that the smoke's movement came from its
  intentionally undertrained BC initialization. Eight epochs and the
  three-generation high-exploration schedule are rejected. Matched jobs
  `74932` (epsilon `0.30`) and `74933` (epsilon `0.10`) now test one online
  generation from the exact same copied four-epoch generation-0 BC checkpoint,
  with four PPO epochs, 1,000 train games, and 200 deterministic eval games.

Operational rule:

- Phase 5 search-distillation diagnostics should always run as SLURM jobs using
  `scripts/slurm/phase5_diagnose_search_distill_conda.sbatch`; avoid running
  large diagnostics on the login node.
- ERAWAN-generated report artifacts should be committed and pushed from ERAWAN
  before local inspection whenever practical. Codex should then pull and analyze
  the tracked report files instead of committing uploaded report copies into
  paths that may still exist as untracked ERAWAN outputs. This avoids future
  `git pull --ff-only` collisions on `reports/` artifacts.
