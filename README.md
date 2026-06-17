# ptcg_abc

Pokemon TCG AI Battle Challenge workspace.

## Phase Status

Kaggle legality setup and Limitless format selection are complete.

See [docs/implementation-plan.md](docs/implementation-plan.md) for the detailed implementation plan.
The current workflow starts with Kaggle files so the legal card list is known before
Limitless decks are collected. See [docs/kaggle-first-workflow.md](docs/kaggle-first-workflow.md).

The selected Limitless format is `TEF-POR`. The latest missing-card report found no
Limitless card names missing from the Kaggle legal list. See
[docs/phase-1-conclusion.md](docs/phase-1-conclusion.md).

The next data milestone will:

- Collect the `TEF-POR` metagame from Limitless TCG.
- Select the top 10 archetypes by points.
- Enumerate variants for each archetype.
- Save the top 2 unique 60-card decklists per variant when available.
- Preserve source URLs and snapshot metadata for reproducibility.

## Roadmap

1. Deck corpus: scrape, normalize, validate, deduplicate, and export decklists.
2. Generic rule-based agent: start with deck-agnostic rules for basic play.
3. Reinforcement learning workflow: add after the rule-based baseline is working.
4. Deck-building experiments: explore wildcard or search-based deck construction later.

## Setup Notes

Use Python 3.12 for local development. In the Codex workspace, a bundled Python 3.12 runtime is also available through the app-provided workspace dependencies.
