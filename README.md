# ptcg_abc

Pokemon TCG AI Battle Challenge workspace.

## Phase Status

Kaggle legality setup, Limitless format selection, the first `TEF-POR` deck corpus,
and Phase 3 rule-based agent closeout are complete.

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

The current submission bundle is generated locally at:

```text
submissions/phase3/submission.tar.gz
```

The selected Phase 3 archetype is `Hydrapple ex`, using deck index `20` from the
corpus. See [docs/phase-3-conclusion.md](docs/phase-3-conclusion.md).

The next milestone will:

- Start reinforcement learning experiments.
- Use the Phase 3 rule-based agent as the fixed baseline opponent.
- Keep deck-building experiments parked until the RL loop is measurable.

Run the current baseline smoke check:

```powershell
python -m ptcg_abc agent-smoke --max-steps 50
```

Rebuild the Phase 3 submission bundle:

```powershell
python -m ptcg_abc phase3-closeout
```

## Roadmap

1. Deck corpus: scrape, normalize, validate, deduplicate, and export decklists.
2. Generic rule-based agent: start with deck-agnostic rules for basic play.
3. Reinforcement learning workflow: add after the rule-based baseline is working.
4. Deck-building experiments: explore wildcard or search-based deck construction later.

## Setup Notes

Use Python 3.12 for local development. In the Codex workspace, a bundled Python 3.12 runtime is also available through the app-provided workspace dependencies.
