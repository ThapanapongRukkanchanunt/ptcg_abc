# Implementation Plan

## Objective

Build a reproducible deck corpus for the Pokemon TCG AI Battle Challenge, using Limitless TCG as the source of current Standard metagame decks.

This milestone stops at data collection and validation. Rule-based play, reinforcement learning, and deck-building experiments come after the corpus is reliable.

## Success Criteria

- The project can fetch the current Standard metagame from Limitless TCG.
- The top 10 archetypes are selected by points.
- Every available variant for those archetypes is attempted.
- Up to 2 unique 60-card lists are saved per variant.
- Duplicate decklists are removed by exact 60-card fingerprint.
- Every accepted deck has source metadata: archetype, variant, player, event, placement, source URL, and fetch timestamp.
- Every skipped or unavailable deck is recorded with a reason.
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
      models.py
      fingerprints.py
      limitless/
        __init__.py
        client.py
        parser.py
        collector.py
      export.py
      validate.py
  tests/
    fixtures/
      limitless/
    test_fingerprints.py
    test_limitless_parser.py
    test_validate.py
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
- `Decklist`: archetype, variant, result metadata, cards, total count, fingerprint, source URL.
- `CorpusManifest`: fetch timestamp, Limitless filters, accepted counts, skipped rows, warnings, output file paths.

## Collection Workflow

1. Fetch the current Standard metagame page from `https://limitlesstcg.com/decks`.
2. Parse the top 10 archetypes by points from the default metagame table.
3. For each archetype, fetch its overview page and parse variant options.
4. For each variant, fetch the tournament history page with the variant filter when available.
5. Parse candidate tournament results in placement order.
6. Fetch candidate decklist pages until 2 unique valid 60-card lists are accepted or the variant is exhausted.
7. Normalize and fingerprint each deck by sorted `(card_name, count)` pairs.
8. Write raw HTML cache, processed corpus files, and manifest.

## Outputs

Write generated outputs under `data/processed/<snapshot-date>/`:

- `deck_corpus.jsonl`: full structured deck records.
- `deck_corpus.csv`: compact table for human review.
- `decks/*.txt`: plain decklists grouped by archetype and variant.
- `manifest.json`: run metadata, selected archetypes, counts, skips, and warnings.

Raw pages should be cached under `data/raw/<snapshot-date>/` using a URL hash plus readable slug.

## CLI Commands

Expose commands through `python -m ptcg_abc`:

```powershell
python -m ptcg_abc fetch --snapshot-date 2026-06-17
python -m ptcg_abc validate --snapshot-date 2026-06-17
python -m ptcg_abc export --snapshot-date 2026-06-17
```

Defaults:

- `fetch` uses today's date when `--snapshot-date` is omitted.
- `fetch` uses cached HTML unless `--refresh` is passed.
- `validate` exits non-zero if any accepted deck is not exactly 60 cards.
- `export` reads processed JSONL and rewrites CSV/TXT outputs deterministically.

## Dependencies

Start small:

- Standard library for CLI, JSON, CSV, paths, hashing, and dates.
- `requests` for HTTP.
- `beautifulsoup4` or `lxml` for HTML parsing.
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
2. Add models and fingerprinting.
3. Add raw HTML cache and Limitless HTTP client.
4. Add parsers using saved fixtures.
5. Add collector orchestration and manifest writing.
6. Add validators and deterministic exports.
7. Add README usage instructions.
8. Run tests, then do one live fetch smoke test.

## Risks and Mitigations

- Limitless HTML may change: keep parser tests around saved fixtures and record skip reasons clearly.
- Variant filter IDs may not be stable: discover variant URLs from the page instead of hard-coding IDs.
- Some variants may have fewer than 2 public unique lists: record shortfalls in the manifest.
- Kaggle legality may differ from current Standard: defer legality filtering until competition files are available locally.
- Live scraping can be brittle: cache all raw pages and make exports reproducible from cache.

## Done Definition

This milestone is done when:

- `fetch`, `validate`, and `export` run from a clean checkout.
- The manifest shows the top 10 archetypes were attempted.
- The corpus has no duplicate fingerprints.
- Every accepted deck totals exactly 60 cards.
- The README explains how to reproduce the snapshot.
- Tests pass without requiring network access.
