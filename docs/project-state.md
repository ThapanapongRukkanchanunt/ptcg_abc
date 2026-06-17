# Project State

This is the resume point for the project. Start here after switching machines, cloning the repo, or returning after a long break.

## Current Status

- Repository: `https://github.com/ThapanapongRukkanchanunt/ptcg_abc.git`
- Active branch: `main`
- Selected Limitless format: `TEF-POR`
- Kaggle legal-card source: `EN_Card_Data.csv`
- Latest completed phase: Phase 2, deck corpus exports
- Next phase: Phase 3, generic rule-based agent baseline

## Phase Log

| Phase | Status | Output |
| --- | --- | --- |
| Phase 0: Repo setup | Complete | GitHub repo initialized with README, `.gitignore`, and implementation plan. |
| Phase 1: Kaggle legality and format selection | Complete | Kaggle card data extracted, Limitless format set to `TEF-POR`, missing-card report shows 0 missing names. |
| Phase 2: Deck corpus exports | Complete | `collect-corpus` writes JSONL, CSV, TXT decklists, and manifest under `data/processed/<snapshot-date>/`. |
| Phase 3: Generic rule-based agent | Not started | Next work: load legal decks, implement basic deck-agnostic action selection, add local evaluation tooling. |
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

If `python` is not on PATH in the Codex desktop workspace, use the bundled Python path from the app's workspace dependencies.

## Important Files

- `README.md`: short project overview and current milestone.
- `docs/project-state.md`: this resume log.
- `docs/implementation-plan.md`: implementation-level deck corpus plan.
- `docs/kaggle-first-workflow.md`: Kaggle setup and collection commands.
- `docs/phase-1-conclusion.md`: legality and format-selection conclusion.
- `docs/phase-2-conclusion.md`: deck-corpus export conclusion.
- `reports/missing_limitless_cards.md`: canonical legality report for `TEF-POR`.
- `src/ptcg_abc/`: project code.
- `tests/`: regression tests.

## Next Phase Entry Point

Phase 3 should start by building the generic rule-based agent baseline.

Minimum useful first slice:

- Add a loader for `deck_corpus.jsonl` records.
- Add a Kaggle/sample-submission adapter boundary without hard-coding one deck.
- Implement the first deck-agnostic action heuristics.
- Add a tiny local evaluation command that can run one selected deck through the available Kaggle simulator interface.
- Keep reinforcement learning out of scope until the rule-based baseline is runnable and measurable.
