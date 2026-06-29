# Phase 4 Required Benchmark

Agent: `phase5-search`
Model: `models/rl/phase5_generalist_policy_10k.pt`
Games per matchup: 10
Max selections per game: 600

## Overall

- Games: 360
- Wins: 138
- Losses: 222
- Draws: 0
- Timeouts: 0
- Errors: 0
- Win rate: 0.383

## Search Telemetry

- Searched decisions: 14661
- Search-started decisions: 14661
- Search-changed decisions: 3002
- Search change rate: 0.205
- Search errors: 0
- Search error rate: 0.000
- Candidate probes: 53614
- Candidate errors: 0
- Truncated candidates: 261
- Average search seconds: 0.0536
- Max search seconds: 1.4181

## Search Telemetry By Matchup

| Deck | Opponent | Searched | Changed | Search errors | Candidate probes | Candidate errors | Truncated | Avg seconds |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Crustle | 434 | 75 | 0 | 1588 | 0 | 1 | 0.0398 |
| 1 | Mega Lucario ex | 408 | 71 | 0 | 1522 | 0 | 125 | 0.0638 |
| 1 | Mega Abomasnow ex | 487 | 60 | 0 | 1750 | 0 | 30 | 0.0260 |
| 1 | Iono's Bellibolt ex | 512 | 59 | 0 | 1881 | 0 | 55 | 0.0422 |
| 2 | Crustle | 390 | 105 | 0 | 1440 | 0 | 0 | 0.0125 |
| 2 | Mega Lucario ex | 335 | 72 | 0 | 1197 | 0 | 0 | 0.0123 |
| 2 | Mega Abomasnow ex | 255 | 53 | 0 | 922 | 0 | 0 | 0.0121 |
| 2 | Iono's Bellibolt ex | 357 | 92 | 0 | 1272 | 0 | 0 | 0.0121 |
| 3 | Crustle | 508 | 104 | 0 | 1878 | 0 | 4 | 0.0647 |
| 3 | Mega Lucario ex | 381 | 83 | 0 | 1404 | 0 | 1 | 0.0823 |
| 3 | Mega Abomasnow ex | 393 | 91 | 0 | 1468 | 0 | 3 | 0.0595 |
| 3 | Iono's Bellibolt ex | 456 | 102 | 0 | 1753 | 0 | 22 | 0.1085 |
| 4 | Crustle | 392 | 90 | 0 | 1458 | 0 | 0 | 0.0523 |
| 4 | Mega Lucario ex | 403 | 85 | 0 | 1541 | 0 | 0 | 0.1097 |
| 4 | Mega Abomasnow ex | 446 | 82 | 0 | 1705 | 0 | 0 | 0.0563 |
| 4 | Iono's Bellibolt ex | 408 | 70 | 0 | 1561 | 0 | 0 | 0.0810 |
| 5 | Crustle | 508 | 102 | 0 | 1903 | 0 | 3 | 0.0871 |
| 5 | Mega Lucario ex | 418 | 97 | 0 | 1545 | 0 | 2 | 0.0879 |
| 5 | Mega Abomasnow ex | 498 | 114 | 0 | 1886 | 0 | 0 | 0.0708 |
| 5 | Iono's Bellibolt ex | 536 | 95 | 0 | 1989 | 0 | 10 | 0.0828 |
| 6 | Crustle | 334 | 64 | 0 | 1173 | 0 | 0 | 0.0531 |
| 6 | Mega Lucario ex | 323 | 57 | 0 | 1106 | 0 | 0 | 0.0243 |
| 6 | Mega Abomasnow ex | 330 | 75 | 0 | 1204 | 0 | 0 | 0.0327 |
| 6 | Iono's Bellibolt ex | 383 | 85 | 0 | 1392 | 0 | 0 | 0.0365 |
| 7 | Crustle | 435 | 93 | 0 | 1520 | 0 | 0 | 0.0232 |
| 7 | Mega Lucario ex | 341 | 71 | 0 | 1177 | 0 | 0 | 0.0297 |
| 7 | Mega Abomasnow ex | 282 | 49 | 0 | 1015 | 0 | 2 | 0.0309 |
| 7 | Iono's Bellibolt ex | 395 | 106 | 0 | 1428 | 0 | 0 | 0.0257 |
| 8 | Crustle | 555 | 114 | 0 | 2028 | 0 | 2 | 0.0849 |
| 8 | Mega Lucario ex | 381 | 79 | 0 | 1371 | 0 | 1 | 0.0685 |
| 8 | Mega Abomasnow ex | 429 | 92 | 0 | 1563 | 0 | 0 | 0.0680 |
| 8 | Iono's Bellibolt ex | 468 | 87 | 0 | 1746 | 0 | 0 | 0.0940 |
| 9 | Crustle | 393 | 88 | 0 | 1386 | 0 | 0 | 0.0238 |
| 9 | Mega Lucario ex | 328 | 73 | 0 | 1153 | 0 | 0 | 0.0240 |
| 9 | Mega Abomasnow ex | 369 | 65 | 0 | 1306 | 0 | 0 | 0.0229 |
| 9 | Iono's Bellibolt ex | 390 | 102 | 0 | 1383 | 0 | 0 | 0.0262 |

## Matchups

| Deck | Rank | Archetype | Opponent | Wins | Losses | Draws | Timeouts | Errors | Win rate |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | Alakazam Dudunsparce | Crustle | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 1 | 1 | Alakazam Dudunsparce | Mega Lucario ex | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 1 | 1 | Alakazam Dudunsparce | Mega Abomasnow ex | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 1 | 1 | Alakazam Dudunsparce | Iono's Bellibolt ex | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 2 | 2 | Crustle | Crustle | 6 | 4 | 0 | 0 | 0 | 0.600 |
| 2 | 2 | Crustle | Mega Lucario ex | 8 | 2 | 0 | 0 | 0 | 0.800 |
| 2 | 2 | Crustle | Mega Abomasnow ex | 5 | 5 | 0 | 0 | 0 | 0.500 |
| 2 | 2 | Crustle | Iono's Bellibolt ex | 5 | 5 | 0 | 0 | 0 | 0.500 |
| 3 | 3 | Dragapult Dusknoir | Crustle | 1 | 9 | 0 | 0 | 0 | 0.100 |
| 3 | 3 | Dragapult Dusknoir | Mega Lucario ex | 5 | 5 | 0 | 0 | 0 | 0.500 |
| 3 | 3 | Dragapult Dusknoir | Mega Abomasnow ex | 2 | 8 | 0 | 0 | 0 | 0.200 |
| 3 | 3 | Dragapult Dusknoir | Iono's Bellibolt ex | 7 | 3 | 0 | 0 | 0 | 0.700 |
| 4 | 4 | Dragapult | Crustle | 3 | 7 | 0 | 0 | 0 | 0.300 |
| 4 | 4 | Dragapult | Mega Lucario ex | 3 | 7 | 0 | 0 | 0 | 0.300 |
| 4 | 4 | Dragapult | Mega Abomasnow ex | 3 | 7 | 0 | 0 | 0 | 0.300 |
| 4 | 4 | Dragapult | Iono's Bellibolt ex | 7 | 3 | 0 | 0 | 0 | 0.700 |
| 5 | 9 | Dragapult Dudunsparce | Crustle | 6 | 4 | 0 | 0 | 0 | 0.600 |
| 5 | 9 | Dragapult Dudunsparce | Mega Lucario ex | 3 | 7 | 0 | 0 | 0 | 0.300 |
| 5 | 9 | Dragapult Dudunsparce | Mega Abomasnow ex | 2 | 8 | 0 | 0 | 0 | 0.200 |
| 5 | 9 | Dragapult Dudunsparce | Iono's Bellibolt ex | 4 | 6 | 0 | 0 | 0 | 0.400 |
| 6 | 10 | Hydrapple | Crustle | 2 | 8 | 0 | 0 | 0 | 0.200 |
| 6 | 10 | Hydrapple | Mega Lucario ex | 1 | 9 | 0 | 0 | 0 | 0.100 |
| 6 | 10 | Hydrapple | Mega Abomasnow ex | 8 | 2 | 0 | 0 | 0 | 0.800 |
| 6 | 10 | Hydrapple | Iono's Bellibolt ex | 5 | 5 | 0 | 0 | 0 | 0.500 |
| 7 | 11 | Raging Bolt Ogerpon | Crustle | 3 | 7 | 0 | 0 | 0 | 0.300 |
| 7 | 11 | Raging Bolt Ogerpon | Mega Lucario ex | 4 | 6 | 0 | 0 | 0 | 0.400 |
| 7 | 11 | Raging Bolt Ogerpon | Mega Abomasnow ex | 5 | 5 | 0 | 0 | 0 | 0.500 |
| 7 | 11 | Raging Bolt Ogerpon | Iono's Bellibolt ex | 9 | 1 | 0 | 0 | 0 | 0.900 |
| 8 | 18 | Dragapult Blaziken | Crustle | 2 | 8 | 0 | 0 | 0 | 0.200 |
| 8 | 18 | Dragapult Blaziken | Mega Lucario ex | 5 | 5 | 0 | 0 | 0 | 0.500 |
| 8 | 18 | Dragapult Blaziken | Mega Abomasnow ex | 1 | 9 | 0 | 0 | 0 | 0.100 |
| 8 | 18 | Dragapult Blaziken | Iono's Bellibolt ex | 7 | 3 | 0 | 0 | 0 | 0.700 |
| 9 | 22 | Ogerpon Box | Crustle | 7 | 3 | 0 | 0 | 0 | 0.700 |
| 9 | 22 | Ogerpon Box | Mega Lucario ex | 2 | 8 | 0 | 0 | 0 | 0.200 |
| 9 | 22 | Ogerpon Box | Mega Abomasnow ex | 2 | 8 | 0 | 0 | 0 | 0.200 |
| 9 | 22 | Ogerpon Box | Iono's Bellibolt ex | 5 | 5 | 0 | 0 | 0 | 0.500 |
