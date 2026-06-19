# ptcg_abc

Pokemon TCG AI Battle Challenge workspace.

## Phase Status

Kaggle legality setup, Limitless format selection, the first `TEF-POR` deck corpus,
and the Phase 3 rule-based baseline are complete. Active Phase 3 work is now focused
on improving the required benchmark against fixed sample decks.

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

The historical Phase 3 closeout selected `Hydrapple ex`, using deck index `20` from
the corpus. See [docs/phase-3-conclusion.md](docs/phase-3-conclusion.md).

The current required benchmark uses our agent with nine Limitless Tournament 559 decks
from ranks `1, 2, 3, 4, 9, 10, 11, 18, 22` against four fixed benchmark decks:
Crustle, Mega Lucario ex, Mega Abomasnow ex, and Iono's Bellibolt ex. The latest full
run is `138-220-2` over `360` games, with `0` errors and `0.383` win rate. See
[reports/phase3_required_benchmark.md](reports/phase3_required_benchmark.md).

Rank 2 from Limitless includes `Pokemon Center Lady`, which is not in Kaggle
`EN_Card_Data.csv`; the simulator deck currently uses `Cook` as a temporary legal
stand-in.

Run the current required benchmark:

```powershell
python -m ptcg_abc benchmark-phase3-required
```

The next milestone will:

- Improve focused deck-family rules until the required benchmark approaches 50%.
- Start reinforcement learning experiments after the rule benchmark is stable.
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
