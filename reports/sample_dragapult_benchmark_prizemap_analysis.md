# Prize-Map Benchmark Analysis

Date: 2026-06-18

## Summary

The v1 prize-map planner was benchmarked against the copied sample Dragapult agent.
It is useful diagnostically, but it regressed the current win rate.

| Metric | Previous | V1 prize map | Delta |
| --- | ---: | ---: | ---: |
| Games | 270 | 270 | 0 |
| Wins | 22 | 10 | -12 |
| Losses | 247 | 260 | +13 |
| Draws | 1 | 0 | -1 |
| Timeouts | 4 | 0 | -4 |
| Win rate | 0.081 | 0.037 | -0.044 |

Full reports:

- Previous baseline: `reports/sample_dragapult_benchmark.md`
- V1 prize-map benchmark: `reports/sample_dragapult_benchmark_prizemap.md`
- Comparison table: `reports/sample_dragapult_benchmark_comparison.md`
- Trace data: `reports/sample_dragapult_benchmark_prizemap.json`

## What Improved

- The benchmark produced no errors.
- Timeouts dropped from 4 to 0.
- A few decks improved by one win: deck 6 Dragapult Blaziken, deck 12 Alakazam,
  and deck 20 Hydrapple.
- The debug trace gives enough information to see why decisions were made.

## What Regressed

- Overall wins dropped from 22 to 10.
- The strongest previous deck, deck 17 Mega Lopunny ex, dropped from 3 wins to 1.
- Dragapult decks 1, 8, and 9 lost wins compared with the previous baseline.

## Diagnosis

The v1 planner is finding routes, but the automatic key-attacker layer is too loose.
In sampled losing traces, the most common key attackers were low-stage or utility
Pokemon rather than true deck finishers.

Top sampled key attackers:

| Pokemon | Trace count |
| --- | ---: |
| Dreepy | 188 |
| Munkidori | 178 |
| Moltres | 137 |
| Mega Kangaskhan ex | 109 |
| Dunsparce | 97 |
| Budew | 82 |
| Dragapult ex | 78 |

Top sampled first route attackers:

| Pokemon | Trace count |
| --- | ---: |
| Budew | 68 |
| Dunsparce | 33 |
| Cynthia's Gible | 21 |
| Dragapult ex | 17 |
| Mega Lopunny ex | 16 |
| Wellspring Mask Ogerpon ex | 13 |

This shows the main problem: the planner sometimes treats setup basics or utility
Pokemon as attackers because they can make small progress on the current board.
Those route signals then boost search, benching, attachment, and discard protection
for the wrong cards.

The other limitation is expected: v1 only models single-target damage from the current
board. It cannot yet value Dragapult-style multi-KO turns, bench damage counters,
future evolutions, or attacks whose printed damage understates their real plan.

## Next Rule Upgrade

1. Tighten key-attacker identification.
   - Require meaningful damage or prize efficiency.
   - Prefer Pokemon ex, Mega ex, evolved attackers, and high-damage basics.
   - Do not mark low-damage pre-evolutions as key attackers.

2. Separate setup pieces from attackers.
   - Let Dreepy/Drakloak be setup cards for Dragapult ex.
   - Do not let them receive attacker-level attachment/search priority unless they
     are the only viable route.

3. Gate prize-map bonuses.
   - Apply strong route bonuses only when the prize map has a positive route score
     and at least one prize-taking step.
   - Fall back to previous generic scoring when the route is only chip damage.

4. Add v2 spread/damage-counter mapping.
   - Model attacks that damage Active plus Bench.
   - Model damage-counter placement and multi-KO combinations.
   - Re-test especially Dragapult, Festival Lead, and Munkidori-style lines.
