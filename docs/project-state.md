# Project State

This is the resume point for the project. Start here after switching machines, cloning the repo, or returning after a long break.

## Current Status

- Repository: `https://github.com/ThapanapongRukkanchanunt/ptcg_abc.git`
- Active branch: `main`
- Selected Limitless format: `TEF-POR`
- Kaggle legal-card source: `EN_Card_Data.csv`
- Latest completed phase: Phase 3, generic rule-based agent and Kaggle submission bundle
- Current phase: Phase 4, rule-guided hybrid reinforcement learning workflow
  (`docs/phase-4-rl-plan.md`)

## Phase Log

| Phase | Status | Output |
| --- | --- | --- |
| Phase 0: Repo setup | Complete | GitHub repo initialized with README, `.gitignore`, and implementation plan. |
| Phase 1: Kaggle legality and format selection | Complete | Kaggle card data extracted, Limitless format set to `TEF-POR`, missing-card report shows 0 missing names. |
| Phase 2: Deck corpus exports | Complete | `collect-corpus` writes JSONL, CSV, TXT decklists, and manifest under `data/processed/<snapshot-date>/`. |
| Phase 3: Generic rule-based agent | Complete | Combined generic scorer, random-agent evaluation, archetype sweep, final deck selection, and Kaggle submission bundle. |
| Phase 4: Reinforcement learning workflow | Initial implementation | Rule-guided hybrid RL package, optional PyTorch actor/value BC backend, exported option ranker, workflow commands, and SLURM templates added. |
| Phase 5: Deck-building experiments | Deferred | Wildcard/search-based deck construction ideas stay parked until baseline play exists. |

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
- PyTorch backend now trains on CUDA automatically when available and saves a
  portable CPU checkpoint/export.
- Added an `rl` optional dependency group for training environments:
  `python -m pip install -e .[rl]`.
- Added export-equation parity coverage so the PyTorch actor equation and
  exported JSON ranker stay aligned.
- Updated the BC SLURM template to use the torch backend by default on ERAWAN.
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
- `src/ptcg_abc/rl/torch_backend.py`: optional PyTorch actor/value BC backend and
  JSON export bridge.
- `src/ptcg_abc/agent/hybrid_rl.py`: hybrid RL agent with rule fallback.
- `scripts/slurm/`: Phase 4 ERAWAN/SLURM job templates.
- `tests/`: regression tests.

## Next Phase Entry Point

Continue Phase 4 from the implemented workflow in `src/ptcg_abc/rl/` and
`docs/phase-4-rl-plan.md`.

Immediate Phase 4 next steps:

- On ERAWAN or another environment with PyTorch installed, run
  `rl-train-bc --backend torch` and verify the exported JSON model with
  `rl-evaluate --agent hybrid`.
- Scale BC collection toward the planned 20,000 rule-agent games on ERAWAN.
- Use rollout chunks and `rl-train-ppo` as the first resumable reward-weighted
  update loop, then replace the update backend with true PPO.
- Keep the rank 2 legal stand-in in mind: Limitless `Pokemon Center Lady` is not
  present in Kaggle `EN_Card_Data.csv`, so the current simulator deck uses `Cook`.

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
