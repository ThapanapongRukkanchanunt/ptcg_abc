# Implementation Plan

## Objective

Build a reproducible deck corpus for the Pokemon TCG AI Battle Challenge. Kaggle competition files define the legal card list first; Limitless TCG is then used as the source of current `TEF-POR` metagame decks to compare against Kaggle legality.

This milestone stops at data collection and validation. Rule-based play, reinforcement learning, and deck-building experiments come after the corpus is reliable.

## Success Criteria

- The project can fetch the current Standard metagame from Limitless TCG.
- Kaggle competition files can be downloaded or extracted from a local archive.
- The English Kaggle card data can be converted into a legal card name list.
- The top 10 archetypes are selected by points.
- Every available variant for those archetypes is attempted.
- Up to 2 unique 60-card lists are saved per variant.
- Duplicate decklists are removed by exact 60-card fingerprint.
- Every accepted deck has source metadata: archetype, variant, player, event, placement, source URL, and fetch timestamp.
- Every skipped or unavailable deck is recorded with a reason.
- Limitless card names missing from Kaggle legality are reported for manual alternatives.
- Limitless tournament-history pages are filtered with `format=TEF-POR`; older formats must not leak into the corpus.
- Outputs are deterministic from cached HTML.

## Proposed Project Shape

```text
ptcg_abc/
  README.md
  docs/
    implementation-plan.md
  pyproject.toml
  src/
    ptcg_abc/
      __init__.py
      cli.py
      config.py
      http_client.py
      models.py
      normalize.py
      kaggle_api.py
      legal_cards.py
      limitless.py
  tests/
    test_legal_cards.py
    test_normalize.py
  data/
    raw/
    processed/
```

`data/raw/` and `data/processed/` should be ignored by git unless a tiny fixture is intentionally added under `tests/fixtures/`.

## Data Model

Use typed Python records for the core entities:

- `MetagameArchetype`: rank, name, Limitless deck id, points, share, source URL.
- `Variant`: archetype id, variant name, variant filter id or URL, source URL.
- `TournamentResult`: event name, event date, event type when available, placement, player, decklist URL, variant name.
- `CardLine`: section, count, card name.
- `LegalCardCandidate`: source file, detected card-name column, optional legality column, score, sample names.
- `Decklist`: archetype, variant, result metadata, cards, total count, fingerprint, source URL.
- `CorpusManifest`: fetch timestamp, Limitless filters, accepted counts, skipped rows, warnings, output file paths.

## Collection Workflow

1. Download Kaggle competition data with credentials, or extract an already-downloaded archive.
2. Discover the English legal card list from `EN_Card_Data.csv`, with an override flag if Kaggle file names change.
3. Fetch the current Limitless metagame page from `https://limitlesstcg.com/decks?format=TEF-POR`.
4. Parse the top 10 archetypes by points from the default metagame table.
5. For each archetype, fetch its overview page and parse variant options.
6. For each variant, fetch the tournament history page with `format=TEF-POR` plus the variant filter when available.
7. Parse candidate tournament results in placement order.
8. Fetch candidate decklist pages until 2 unique valid 60-card lists are accepted or the variant is exhausted.
9. Normalize and fingerprint each deck by sorted `(card_name, count)` pairs.
10. Compare unique Limitless card names against the Kaggle legal list and write the missing-card report.

## Outputs

Write generated outputs under `data/processed/<snapshot-date>/`:

- `deck_corpus.jsonl`: full structured deck records.
- `deck_corpus.csv`: compact table for human review.
- `decks/*.txt`: plain decklists grouped by archetype and variant.
- `manifest.json`: run metadata, selected archetypes, counts, skips, and warnings.
- `reports/missing_limitless_cards.md`: card names from Limitless decks that are absent from Kaggle legality.

Raw pages should be cached under `data/raw/<snapshot-date>/` using a URL hash plus readable slug.

Generated corpus data is ignored by git. Recreate it with `python -m ptcg_abc collect-corpus`.

## CLI Commands

Expose commands through `python -m ptcg_abc`:

```powershell
python -m ptcg_abc kaggle-setup --archive pokemon-tcg-ai-battle.zip
python -m ptcg_abc discover-legal-cards --legal-source data\kaggle\input\EN_Card_Data.csv
python -m ptcg_abc missing-limitless --snapshot-date 2026-06-17 --limitless-format TEF-POR
python -m ptcg_abc collect-corpus --snapshot-date 2026-06-17 --limitless-format TEF-POR
```

Defaults:

- `kaggle-setup` uses Kaggle API credentials unless `--archive` is provided.
- `discover-legal-cards` writes `data/kaggle/legal_cards.txt`.
- `missing-limitless` uses today's date when `--snapshot-date` is omitted.
- `missing-limitless` uses cached HTML unless `--refresh` is passed.
- `collect-corpus` writes JSONL, CSV, TXT decklists, and a manifest.

## Dependencies

Start small:

- Standard library for CLI, JSON, CSV, paths, hashing, and dates.
- Standard library HTTP for Kaggle API and simple downloads.
- `lxml` for HTML parsing.
- `pydantic` for validation models if useful.
- `pytest` for tests.
- `ruff` for linting and formatting once the codebase is non-trivial.

## Testing Strategy

Use fixture-first tests so normal checks do not depend on network access.

- Parser tests for metagame tables, variant sections, tournament history rows, and decklist pages.
- Fingerprint tests for ordering, duplicate detection, casing, and whitespace normalization.
- Validation tests for exactly 60 cards, missing sections, malformed counts, and duplicate skip reasons.
- Export tests for deterministic JSONL, CSV, and TXT output.
- One optional live smoke test behind an explicit flag to confirm Limitless page structure still matches assumptions.

## Implementation Order

1. Add `pyproject.toml`, package skeleton, and basic CLI.
2. Add Kaggle archive/API setup and legal-card discovery.
3. Add models and fingerprinting.
4. Add raw HTML cache and Limitless HTTP client.
5. Add parsers using saved fixtures.
6. Add collector orchestration, manifest writing, and missing-card report output.
7. Add README usage instructions.
8. Run tests, then do one live Kaggle/Limitless smoke test.

## Risks and Mitigations

- Limitless HTML may change: keep parser tests around saved fixtures and record skip reasons clearly.
- Variant filter IDs may not be stable: discover variant URLs from the page instead of hard-coding IDs.
- Some variants may have fewer than 2 public unique lists: record shortfalls in the manifest.
- Kaggle names may differ from Limitless display names: report mismatches and maintain a manual alternative map later.
- Live scraping can be brittle: cache all raw pages and make exports reproducible from cache.

## Done Definition

This milestone is done when:

- `kaggle-setup` and `missing-limitless` run from a clean checkout once Kaggle data is available.
- The manifest shows the top 10 archetypes were attempted.
- The corpus has no duplicate fingerprints.
- Every accepted deck totals exactly 60 cards.
- The missing-card report is generated and has no missing names for the selected `TEF-POR` format.
- The README explains how to reproduce the snapshot.
- Tests pass without requiring network access.

Status: complete for the 2026-06-17 local snapshot. See `docs/phase-2-conclusion.md`.
