# Phase 3 Conclusion

Date: 2026-06-17

## Result

Phase 3 is complete.

The project now has a generic rule-based agent that combines the baseline rules with
reusable ideas learned from the Kaggle sample agents:

- score-based option selection,
- board feature extraction,
- card metadata usage,
- attack planning,
- prize-aware target valuation,
- setup and search heuristics,
- energy attachment heuristics,
- retreat/switch gating,
- discard and recovery heuristics,
- draw conservation when deck count is low.

## Final Deck Selection

The Phase 3 closeout sweep used the TEF-POR corpus archetype representatives.

Configuration:

- Archetypes: 10
- Matchups: every unordered archetype pair
- Games per matchup: 10
- Max selections per game: 600
- Random seed: 20260617

Winner:

- Archetype: `Hydrapple ex`
- Selected deck index: `20`
- Selected deck: `Hydrapple ex / All / Matias Matricardi 1st`

The final deck was chosen by selecting the highest-scoring archetype and then randomly
choosing one deck list within that archetype using the configured seed.

## Evaluation Summary

Random-agent mirror evaluation for the selected deck:

- Games: 20
- Wins: 18
- Losses: 2
- Draws: 0
- Timeouts: 0
- Errors: 0
- Win rate: 0.900

Top archetype sweep results:

| Rank | Archetype | Points | Wins | Losses | Win rate |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Hydrapple ex | 199 | 66 | 23 | 0.733 |
| 2 | Mega Lopunny ex | 189 | 63 | 27 | 0.700 |
| 3 | Cynthia's Garchomp ex | 184 | 61 | 27 | 0.678 |

Full report:

- `reports/phase3_closeout.md`
- `reports/phase3_closeout.json`

## Submission Bundle

Generated local bundle:

```text
submissions/phase3/submission.tar.gz
```

The bundle includes:

- `main.py`
- `deck.csv`
- Kaggle `cg` package
- Minimal local `ptcg_abc` agent package used by `main.py`

The `submissions/` directory is ignored by git because it contains generated Kaggle
submission artifacts and copied Kaggle native files.

## Verification

Unit tests:

```text
Ran 42 tests
OK
```

Packaged-agent smoke:

- Loaded `main.py` from `submissions/phase3/`.
- Returned a 60-card deck.
- Ran 50 native simulator selections from the submission folder.
- Engine error: none.

## Next Phase

Phase 4 can start reinforcement learning experiments once the required benchmark is
stable enough to be useful as a baseline. Recommended first step: keep the Phase 3
rule agent as a fixed opponent and build an evaluation loop that logs
state/action/reward traces from local simulator games.

## Post-Closeout Benchmark Update

Date: 2026-06-19

The active Phase 3 target now uses a stricter required benchmark:

- Our agent uses nine Limitless Tournament 559 decks from ranks 1, 2, 3, 4, 9, 10,
  11, 18, and 22.
- Benchmark decks remain fixed: Crustle, Mega Lucario ex, Mega Abomasnow ex, and
  Iono's Bellibolt ex.
- Command: `python -m ptcg_abc benchmark-phase3-required`
- Latest full run: 138 wins, 220 losses, 2 draws, 3 timeouts, 0 errors, 0.383 win
  rate across 360 games.
- Target: approximately 180 wins in 360 games, or 50% average win rate.

Rank 2 from Limitless includes `Pokemon Center Lady`, which is not present in Kaggle
`EN_Card_Data.csv`. The current simulator deck uses `Cook` as a temporary legal
stand-in until a better replacement is chosen.

Current reports:

- `reports/phase3_required_benchmark.md`
- `reports/phase3_required_benchmark.json`
