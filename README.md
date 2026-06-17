# ptcg_abc

Pokemon TCG AI Battle Challenge workspace.

## Current Milestone

Build a reproducible deck corpus from Limitless TCG before starting agent work.

See [docs/implementation-plan.md](docs/implementation-plan.md) for the detailed implementation plan.

The first data milestone will:

- Collect the current Standard metagame from Limitless TCG.
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
