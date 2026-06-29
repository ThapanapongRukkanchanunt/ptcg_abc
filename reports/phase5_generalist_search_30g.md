# Phase 4 Required Benchmark

Agent: `phase5-search`
Model: `models/rl/phase5_generalist_policy_10k.pt`
Games per matchup: 30
Max selections per game: 600

## Overall

- Games: 1080
- Wins: 414
- Losses: 665
- Draws: 1
- Timeouts: 5
- Errors: 0
- Win rate: 0.383

## Search Telemetry

- Searched decisions: 44267
- Search-started decisions: 44267
- Search-changed decisions: 8584
- Search change rate: 0.194
- Search errors: 0
- Search error rate: 0.000
- Candidate probes: 161729
- Candidate errors: 0
- Truncated candidates: 677
- Average search seconds: 0.0514
- Max search seconds: 2.4492

## Search Telemetry By Matchup

| Deck | Opponent | Searched | Changed | Search errors | Candidate probes | Candidate errors | Truncated | Avg seconds |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Crustle | 1542 | 247 | 0 | 5714 | 0 | 68 | 0.0413 |
| 1 | Mega Lucario ex | 1338 | 186 | 0 | 4920 | 0 | 202 | 0.0527 |
| 1 | Mega Abomasnow ex | 1417 | 201 | 0 | 5162 | 0 | 148 | 0.0332 |
| 1 | Iono's Bellibolt ex | 1386 | 161 | 0 | 5035 | 0 | 166 | 0.0349 |
| 2 | Crustle | 1188 | 268 | 0 | 4270 | 0 | 0 | 0.0129 |
| 2 | Mega Lucario ex | 919 | 195 | 0 | 3222 | 0 | 0 | 0.0124 |
| 2 | Mega Abomasnow ex | 993 | 213 | 0 | 3535 | 0 | 0 | 0.0110 |
| 2 | Iono's Bellibolt ex | 1103 | 278 | 0 | 3936 | 0 | 0 | 0.0129 |
| 3 | Crustle | 1141 | 299 | 0 | 4277 | 0 | 8 | 0.0729 |
| 3 | Mega Lucario ex | 935 | 201 | 0 | 3408 | 0 | 9 | 0.0853 |
| 3 | Mega Abomasnow ex | 1351 | 323 | 0 | 5048 | 0 | 31 | 0.0750 |
| 3 | Iono's Bellibolt ex | 1216 | 245 | 0 | 4542 | 0 | 9 | 0.0842 |
| 4 | Crustle | 1355 | 273 | 0 | 5082 | 0 | 5 | 0.0585 |
| 4 | Mega Lucario ex | 1303 | 204 | 0 | 4855 | 0 | 1 | 0.0627 |
| 4 | Mega Abomasnow ex | 1480 | 232 | 0 | 5604 | 0 | 0 | 0.0651 |
| 4 | Iono's Bellibolt ex | 1323 | 208 | 0 | 5010 | 0 | 0 | 0.0670 |
| 5 | Crustle | 1418 | 284 | 0 | 5295 | 0 | 2 | 0.0771 |
| 5 | Mega Lucario ex | 1180 | 262 | 0 | 4349 | 0 | 2 | 0.0838 |
| 5 | Mega Abomasnow ex | 1494 | 323 | 0 | 5553 | 0 | 9 | 0.0711 |
| 5 | Iono's Bellibolt ex | 1471 | 296 | 0 | 5484 | 0 | 5 | 0.0803 |
| 6 | Crustle | 1117 | 196 | 0 | 3988 | 0 | 0 | 0.0710 |
| 6 | Mega Lucario ex | 959 | 187 | 0 | 3449 | 0 | 0 | 0.0290 |
| 6 | Mega Abomasnow ex | 1075 | 230 | 0 | 3945 | 0 | 0 | 0.0342 |
| 6 | Iono's Bellibolt ex | 1109 | 211 | 0 | 3891 | 0 | 0 | 0.0345 |
| 7 | Crustle | 1097 | 243 | 0 | 3960 | 0 | 0 | 0.0241 |
| 7 | Mega Lucario ex | 987 | 196 | 0 | 3434 | 0 | 0 | 0.0255 |
| 7 | Mega Abomasnow ex | 1096 | 247 | 0 | 3930 | 0 | 1 | 0.0259 |
| 7 | Iono's Bellibolt ex | 1212 | 274 | 0 | 4355 | 0 | 0 | 0.0278 |
| 8 | Crustle | 1506 | 284 | 0 | 5505 | 0 | 0 | 0.0869 |
| 8 | Mega Lucario ex | 1413 | 264 | 0 | 5110 | 0 | 2 | 0.0677 |
| 8 | Mega Abomasnow ex | 1396 | 255 | 0 | 5165 | 0 | 0 | 0.0678 |
| 8 | Iono's Bellibolt ex | 1493 | 249 | 0 | 5554 | 0 | 6 | 0.0822 |
| 9 | Crustle | 1237 | 260 | 0 | 4481 | 0 | 3 | 0.0295 |
| 9 | Mega Lucario ex | 968 | 181 | 0 | 3431 | 0 | 0 | 0.0246 |
| 9 | Mega Abomasnow ex | 821 | 136 | 0 | 2860 | 0 | 0 | 0.0212 |
| 9 | Iono's Bellibolt ex | 1228 | 272 | 0 | 4370 | 0 | 0 | 0.0268 |

## Matchups

| Deck | Rank | Archetype | Opponent | Wins | Losses | Draws | Timeouts | Errors | Win rate |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | Alakazam Dudunsparce | Crustle | 3 | 27 | 0 | 0 | 0 | 0.100 |
| 1 | 1 | Alakazam Dudunsparce | Mega Lucario ex | 0 | 29 | 1 | 3 | 0 | 0.000 |
| 1 | 1 | Alakazam Dudunsparce | Mega Abomasnow ex | 0 | 30 | 0 | 0 | 0 | 0.000 |
| 1 | 1 | Alakazam Dudunsparce | Iono's Bellibolt ex | 1 | 29 | 0 | 2 | 0 | 0.033 |
| 2 | 2 | Crustle | Crustle | 23 | 7 | 0 | 0 | 0 | 0.767 |
| 2 | 2 | Crustle | Mega Lucario ex | 10 | 20 | 0 | 0 | 0 | 0.333 |
| 2 | 2 | Crustle | Mega Abomasnow ex | 17 | 13 | 0 | 0 | 0 | 0.567 |
| 2 | 2 | Crustle | Iono's Bellibolt ex | 18 | 12 | 0 | 0 | 0 | 0.600 |
| 3 | 3 | Dragapult Dusknoir | Crustle | 10 | 20 | 0 | 0 | 0 | 0.333 |
| 3 | 3 | Dragapult Dusknoir | Mega Lucario ex | 5 | 25 | 0 | 0 | 0 | 0.167 |
| 3 | 3 | Dragapult Dusknoir | Mega Abomasnow ex | 3 | 27 | 0 | 0 | 0 | 0.100 |
| 3 | 3 | Dragapult Dusknoir | Iono's Bellibolt ex | 15 | 15 | 0 | 0 | 0 | 0.500 |
| 4 | 4 | Dragapult | Crustle | 14 | 16 | 0 | 0 | 0 | 0.467 |
| 4 | 4 | Dragapult | Mega Lucario ex | 10 | 20 | 0 | 0 | 0 | 0.333 |
| 4 | 4 | Dragapult | Mega Abomasnow ex | 6 | 24 | 0 | 0 | 0 | 0.200 |
| 4 | 4 | Dragapult | Iono's Bellibolt ex | 22 | 8 | 0 | 0 | 0 | 0.733 |
| 5 | 9 | Dragapult Dudunsparce | Crustle | 13 | 17 | 0 | 0 | 0 | 0.433 |
| 5 | 9 | Dragapult Dudunsparce | Mega Lucario ex | 7 | 23 | 0 | 0 | 0 | 0.233 |
| 5 | 9 | Dragapult Dudunsparce | Mega Abomasnow ex | 5 | 25 | 0 | 0 | 0 | 0.167 |
| 5 | 9 | Dragapult Dudunsparce | Iono's Bellibolt ex | 18 | 12 | 0 | 0 | 0 | 0.600 |
| 6 | 10 | Hydrapple | Crustle | 14 | 16 | 0 | 0 | 0 | 0.467 |
| 6 | 10 | Hydrapple | Mega Lucario ex | 8 | 22 | 0 | 0 | 0 | 0.267 |
| 6 | 10 | Hydrapple | Mega Abomasnow ex | 17 | 13 | 0 | 0 | 0 | 0.567 |
| 6 | 10 | Hydrapple | Iono's Bellibolt ex | 17 | 13 | 0 | 0 | 0 | 0.567 |
| 7 | 11 | Raging Bolt Ogerpon | Crustle | 14 | 16 | 0 | 0 | 0 | 0.467 |
| 7 | 11 | Raging Bolt Ogerpon | Mega Lucario ex | 10 | 20 | 0 | 0 | 0 | 0.333 |
| 7 | 11 | Raging Bolt Ogerpon | Mega Abomasnow ex | 7 | 23 | 0 | 0 | 0 | 0.233 |
| 7 | 11 | Raging Bolt Ogerpon | Iono's Bellibolt ex | 18 | 12 | 0 | 0 | 0 | 0.600 |
| 8 | 18 | Dragapult Blaziken | Crustle | 11 | 19 | 0 | 0 | 0 | 0.367 |
| 8 | 18 | Dragapult Blaziken | Mega Lucario ex | 11 | 19 | 0 | 0 | 0 | 0.367 |
| 8 | 18 | Dragapult Blaziken | Mega Abomasnow ex | 10 | 20 | 0 | 0 | 0 | 0.333 |
| 8 | 18 | Dragapult Blaziken | Iono's Bellibolt ex | 24 | 6 | 0 | 0 | 0 | 0.800 |
| 9 | 22 | Ogerpon Box | Crustle | 15 | 15 | 0 | 0 | 0 | 0.500 |
| 9 | 22 | Ogerpon Box | Mega Lucario ex | 9 | 21 | 0 | 0 | 0 | 0.300 |
| 9 | 22 | Ogerpon Box | Mega Abomasnow ex | 13 | 17 | 0 | 0 | 0 | 0.433 |
| 9 | 22 | Ogerpon Box | Iono's Bellibolt ex | 16 | 14 | 0 | 0 | 0 | 0.533 |
