# Project State

This is the resume point for the project. Start here after switching machines, cloning the repo, or returning after a long break.

## Current Status

- Repository: `https://github.com/ThapanapongRukkanchanunt/ptcg_abc.git`
- Active branch: `main`
- Selected Limitless format: `TEF-POR`
- Kaggle legal-card source: `EN_Card_Data.csv`
- Latest completed phase: Phase 3, generic rule-based agent and Kaggle submission bundle
- Next phase: Phase 4, reinforcement learning workflow

## Phase Log

| Phase | Status | Output |
| --- | --- | --- |
| Phase 0: Repo setup | Complete | GitHub repo initialized with README, `.gitignore`, and implementation plan. |
| Phase 1: Kaggle legality and format selection | Complete | Kaggle card data extracted, Limitless format set to `TEF-POR`, missing-card report shows 0 missing names. |
| Phase 2: Deck corpus exports | Complete | `collect-corpus` writes JSONL, CSV, TXT decklists, and manifest under `data/processed/<snapshot-date>/`. |
| Phase 3: Generic rule-based agent | Complete | Combined generic scorer, random-agent evaluation, archetype sweep, final deck selection, and Kaggle submission bundle. |
| Phase 4: Reinforcement learning workflow | Deferred | Start only after the rule-based baseline and evaluation harness are stable. |
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
- Added random-agent evaluation and archetype matchup sweep.
- Selected final archetype: `Hydrapple ex`.
- Selected final deck index: `20`.
- Built local Kaggle submission bundle: `submissions/phase3/submission.tar.gz`.
- Detailed conclusion: `docs/phase-3-conclusion.md`
- Full closeout report: `reports/phase3_closeout.md`

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
- `reports/sample_dragapult_benchmark_phase3_required.md`: current 31-deck Phase 3 benchmark with required sample decks included.
- `reports/sample_dragapult_benchmark_phase3_required_analysis.md`: current Phase 3 target gap and next rule upgrade plan.
- `reports/missing_limitless_cards.md`: canonical legality report for `TEF-POR`.
- `src/ptcg_abc/`: project code.
- `tests/`: regression tests.

## Next Phase Entry Point

Start Phase 4 by adding reinforcement learning workflow experiments.

Immediate next step before Phase 4:

- Add focused deck-family profiles derived from the public sample agents.
- Start with Crustle wall setup, Mega Lucario Fighting-energy setup, Mega Abomasnow
  water-energy discard math, and Iono Lightning acceleration.
- Add anti-Dragapult defensive rules for Phantom Dive bench-counter pressure.
- Rerun the 31-deck Phase 3 benchmark after each focused profile change.

Latest benchmark checkpoint:

- Benchmark: copied sample Dragapult agent vs all 27 corpus decks plus the 4 required
  Phase 3 sample decks.
- Original baseline: 22 wins, 247 losses, 1 draw, 4 timeouts, 0.081 win rate.
- V1 prize-map run: 10 wins, 260 losses, 0 draws, 0 timeouts, 0.037 win rate.
- V2 guarded prize-map run: 23 wins, 247 losses, 0 draws, 3 timeouts, 0.085 win rate.
- V3 narrowed setup run on the original 27 decks: 23 wins, 247 losses, 0 draws,
  3 timeouts, 0.085 win rate.
- Current required Phase 3 benchmark: 27 wins, 283 losses, 0 draws, 1 timeout,
  0 errors, 0.087 win rate across 310 games.
- Phase 3 target: approximately 155 wins in 310 games, or 50% average win rate.
- Conclusion: the benchmark is now measuring the required decks, but generic setup
  scoring alone is not enough. Focused deck-family rules are the next step.

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
