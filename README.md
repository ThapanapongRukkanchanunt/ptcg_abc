# ptcg_abc

Pokemon TCG AI Battle Challenge workspace.

## Phase Status

Kaggle legality setup, Limitless format selection, and the first `TEF-POR` deck corpus are complete.

Resume from [docs/project-state.md](docs/project-state.md) for the full phase log and next steps.

See [docs/implementation-plan.md](docs/implementation-plan.md) for the detailed implementation plan.
The current workflow starts with Kaggle files so the legal card list is known before
Limitless decks are collected. See [docs/kaggle-first-workflow.md](docs/kaggle-first-workflow.md).

The selected Limitless format is `TEF-POR`. The latest missing-card report found no
Limitless card names missing from the Kaggle legal list. See
[docs/phase-1-conclusion.md](docs/phase-1-conclusion.md).

The corpus command writes generated outputs under `data/processed/<snapshot-date>/`:

```powershell
python -m ptcg_abc collect-corpus
```

The next milestone will:

- Use the deck corpus to start a generic rule-based agent.
- Keep rule logic deck-agnostic at first.
- Add evaluation harnesses before moving into reinforcement learning.

## Roadmap

1. Deck corpus: scrape, normalize, validate, deduplicate, and export decklists.
2. Generic rule-based agent: start with deck-agnostic rules for basic play.
3. Reinforcement learning workflow: add after the rule-based baseline is working.
4. Deck-building experiments: explore wildcard or search-based deck construction later.

## Setup Notes

Use Python 3.12 for local development. In the Codex workspace, a bundled Python 3.12 runtime is also available through the app-provided workspace dependencies.
