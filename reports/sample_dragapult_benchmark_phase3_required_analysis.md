# Phase 3 Required Benchmark Analysis

Date: 2026-06-18

## Summary

The copied sample Dragapult benchmark now includes the 27 TEF-POR corpus decks plus
four required Phase 3 sample decks:

- Crustle
- Mega Lucario ex
- Mega Abomasnow ex
- Iono's Bellibolt ex

Current full result:

| Metric | Value |
| --- | ---: |
| Decks | 31 |
| Games | 310 |
| Wins | 27 |
| Losses | 283 |
| Draws | 0 |
| Timeouts | 1 |
| Errors | 0 |
| Win rate | 0.087 |

The Phase 3 target is 50% average win rate against the copied Dragapult benchmark.
That means the current agent needs to improve from 27 wins to about 155 wins over a
310-game run.

## Required Deck Results

| Deck | Wins | Losses | Win rate |
| --- | ---: | ---: | ---: |
| Crustle | 0 | 10 | 0.000 |
| Mega Lucario ex | 2 | 8 | 0.200 |
| Mega Abomasnow ex | 3 | 7 | 0.300 |
| Iono's Bellibolt ex | 1 | 9 | 0.100 |

## Rule Upgrade Result

This iteration added:

- Required benchmark deck coverage.
- Generic opening setup preference for key evolution-chain basics.
- Repeat energy-attachment support for Iono's Bellibolt ex-style abilities.
- Spread route target bonuses for mapped damage-counter and multi-target plans.

The narrowed rule changes preserve the previous 27-deck V2 result:

| Benchmark | Decks | Games | Wins | Win rate |
| --- | ---: | ---: | ---: | ---: |
| V2 guarded prize map | 27 | 270 | 23 | 0.085 |
| V3 setup narrowed | 27 | 270 | 23 | 0.085 |
| Phase 3 required suite | 31 | 310 | 27 | 0.087 |

## Diagnosis

The generic setup rules alone are not enough to approach 50%. A broader chain-scoring
attempt was tested and rejected because it reduced the 27-deck benchmark below V2.
The useful part of this iteration is that the benchmark suite is now honest: the
missing Crustle, Mega Lucario, Mega Abomasnow, and Iono decks are measured directly.

## Next Rule Upgrade

1. Add deck-family profiles derived from the public sample agents.
   - Keep the generic scorer as the default.
   - Add small profile weights for known deck engines such as Crustle wall, Mega
     Lucario Fighting energy setup, Mega Abomasnow water-energy discard, and Iono
     Lightning acceleration.

2. Add anti-Dragapult defensive rules.
   - Crustle should prioritize establishing Crustle against Pokemon ex attackers.
   - Avoid putting low-HP bench liabilities in range of Phantom Dive counters.
   - Prefer damage-counter-safe targets when benching is optional.

3. Add attack-text support for variable damage attacks.
   - Mega Abomasnow's Hammer-lanche and Iono's Voltaic Chain need damage estimates
     based on energy counts rather than printed damage alone.

4. Re-run the 31-deck benchmark after each focused profile change.
