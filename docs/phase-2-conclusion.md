# Phase 2 Conclusion

## Decision

The first deck corpus uses Limitless `TEF-POR` decks and the Kaggle English legal card list.

## Result

Generated locally under `data/processed/2026-06-17/`:

- `deck_corpus.jsonl`
- `deck_corpus.csv`
- `decks/*.txt`
- `manifest.json`

The generated data directory is intentionally ignored by git. Recreate it from a clean checkout with:

```powershell
python -m ptcg_abc collect-corpus
```

## Corpus Summary

- Accepted decks: 27
- Archetypes represented: 10
- Variants represented: 14
- Unique deck fingerprints: 27
- Skips and shortfalls recorded in the manifest: 20
- Missing card names against Kaggle legality: 0

## Next Phase

Start the generic rule-based agent:

- Load one legal deck from the corpus.
- Implement basic deck-agnostic action selection.
- Add local evaluation tooling before any reinforcement learning work.
