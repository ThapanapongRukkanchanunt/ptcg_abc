# Phase 1 Conclusion

## Decision

Use Limitless `TEF-POR` decks for the first deck corpus.

## Evidence

- Kaggle legal card data was extracted from `EN_Card_Data.csv`.
- Limitless deck collection was filtered with `format=TEF-POR`.
- The comparison accepted 27 unique 60-card decklists across 10 archetypes and 14 variants.
- The normalized Limitless card names all match the Kaggle legal card list.
- `reports/missing_limitless_cards.md` is the canonical report for the selected format.

## Name Normalization Locked In

The comparison handles:

- Curly and straight apostrophes.
- Common mojibake, such as `Biancaâ€™s`.
- Limitless basic energy names mapped to Kaggle names, such as `Fire Energy` to `Basic {R} Energy`.
- Exact wording aliases found during this phase, such as `Rocky Fighting Energy` to `Rock Fighting Energy`.

## Next Phase

Build the reproducible deck corpus outputs for `TEF-POR`:

- Structured JSONL deck records.
- Human-readable CSV review table.
- Plain decklist text exports.
- Manifest with accepted decks, duplicate skips, and variant shortfalls.
