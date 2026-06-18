# Prize-Map V2 Benchmark Analysis

Date: 2026-06-18

## Summary

V2 adds two corrections after the v1 regression:

- Key-attacker selection is stricter, so low-damage setup Pokemon are no longer promoted
  as primary attackers.
- Prize-map bonuses are gated behind actionable routes, so chip-damage routes do not
  override the older generic setup rules.
- Spread/damage-counter text parsing was added for common attacks such as Dragapult ex's
  `Phantom Dive`.

The guarded v2 run slightly beats the original copied-Dragapult benchmark baseline and
recovers most of the v1 regression.

| Metric | Original baseline | V1 prize map | V2 guarded prize map |
| --- | ---: | ---: | ---: |
| Games | 270 | 270 | 270 |
| Wins | 22 | 10 | 23 |
| Losses | 247 | 260 | 247 |
| Draws | 1 | 0 | 0 |
| Timeouts | 4 | 0 | 3 |
| Errors | 0 | 0 | 0 |
| Win rate | 0.081 | 0.037 | 0.085 |

Full reports:

- Original baseline: `reports/sample_dragapult_benchmark.md`
- V1 report: `reports/sample_dragapult_benchmark_prizemap.md`
- V2 report: `reports/sample_dragapult_benchmark_prizemap_v2.md`
- V2 comparison: `reports/sample_dragapult_benchmark_prizemap_v2_comparison.md`
- V2 trace data: `reports/sample_dragapult_benchmark_prizemap_v2.json`

## Deck Movement

Largest improvements versus original baseline:

- Deck 5 Dragapult Froslass: +2 wins.
- Deck 6 Dragapult Blaziken: +2 wins.
- Deck 12 Alakazam Dudunsparce: +2 wins.
- Deck 20 Hydrapple: +2 wins.

Largest regressions versus original baseline:

- Deck 1 Dragapult ex: -2 wins.
- Deck 17 Mega Lopunny ex: -2 wins.
- Deck 19 Rocket's Mewtwo ex: -2 wins.
- Deck 27 Cynthia's Garchomp ex: -2 wins.

## Diagnosis

V2 fixes the most harmful v1 behavior. Key attackers in sampled traces are now mostly
real attackers such as Fezandipiti ex, Latias ex, Mega Kangaskhan ex, Team Rocket's
Mewtwo ex, and Dragapult ex, rather than Dreepy/Munkidori/Budew dominating the key list.

However, early losing traces still often involve setup stalls, passive ending, or
non-primary Pokemon taking temporary lines before the real attacker is established.
The spread parser is implemented and covered by tests, but sampled losing traces did
not frequently reach a board state where a spread route was active in the first 30
recorded decisions.

## Next Rule Upgrade

1. Add evolution-chain setup scoring.
   - Key attackers should boost all required pre-evolutions, not just direct ones.
   - Stage 2 attackers should make Rare Candy and Stage 1 search more deliberate.

2. Improve opening Active and early bench rules.
   - Avoid opening or overusing support Pokemon when a deck has a real attack plan.
   - Prefer resilient setup basics that evolve into key attackers.

3. Add spread-route execution bonuses.
   - When a spread route exists, target damage-counter contexts toward the mapped
     combination, not just the highest single target.
   - Add trace counters for how often spread routes are available and executed.

4. Re-run the copied Dragapult benchmark after each change, preserving original, v1,
   and v2 baselines for comparison.
