# Phase 4 Required Benchmark

Agent: `phase5-league:phase5-full`
Model: `models/rl/phase5_generalist_policy_13deck_10k.pt`
Games per matchup: 30
Max selections per game: 600

## Overall

- Games: 5070
- Wins: 2662
- Losses: 2390
- Draws: 18
- Timeouts: 50
- Errors: 0
- Win rate: 0.525

## Search Telemetry

- Searched decisions: 196748
- Search-started decisions: 196748
- Search-changed decisions: 37167
- Search change rate: 0.189
- Search errors: 0
- Search error rate: 0.000
- Candidate probes: 714636
- Candidate errors: 0
- Truncated candidates: 3476
- Average search seconds: 0.0629
- Max search seconds: 3.3680

## Search Telemetry By Matchup

| Deck | Opponent | Searched | Changed | Search errors | Candidate probes | Candidate errors | Truncated | Avg seconds |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Alakazam Dudunsparce | 1191 | 151 | 0 | 4160 | 0 | 28 | 0.0277 |
| 1 | Crustle | 1557 | 176 | 0 | 5664 | 0 | 105 | 0.0418 |
| 1 | Dragapult Dusknoir | 1432 | 194 | 0 | 5222 | 0 | 195 | 0.0509 |
| 1 | Dragapult | 1291 | 162 | 0 | 4552 | 0 | 195 | 0.0297 |
| 1 | Dragapult Dudunsparce | 1428 | 221 | 0 | 5216 | 0 | 265 | 0.0475 |
| 1 | Hydrapple | 1470 | 186 | 0 | 5366 | 0 | 176 | 0.0511 |
| 1 | Raging Bolt Ogerpon | 1317 | 201 | 0 | 4850 | 0 | 240 | 0.0742 |
| 1 | Dragapult Blaziken | 1363 | 163 | 0 | 4951 | 0 | 286 | 0.0448 |
| 1 | Ogerpon Box | 1467 | 182 | 0 | 5382 | 0 | 124 | 0.0536 |
| 1 | Crustle | 1462 | 189 | 0 | 5415 | 0 | 86 | 0.0622 |
| 1 | Mega Lucario ex | 1276 | 186 | 0 | 4697 | 0 | 133 | 0.0573 |
| 1 | Mega Abomasnow ex | 1233 | 139 | 0 | 4505 | 0 | 77 | 0.0326 |
| 1 | Iono's Bellibolt ex | 1347 | 146 | 0 | 4907 | 0 | 94 | 0.0401 |
| 2 | Alakazam Dudunsparce | 867 | 160 | 0 | 3056 | 0 | 0 | 0.0162 |
| 2 | Crustle | 1141 | 249 | 0 | 4082 | 0 | 0 | 0.0144 |
| 2 | Dragapult Dusknoir | 921 | 177 | 0 | 3275 | 0 | 0 | 0.0141 |
| 2 | Dragapult | 1105 | 239 | 0 | 3908 | 0 | 0 | 0.0133 |
| 2 | Dragapult Dudunsparce | 1177 | 307 | 0 | 4199 | 0 | 0 | 0.0151 |
| 2 | Hydrapple | 1118 | 271 | 0 | 3951 | 0 | 0 | 0.0141 |
| 2 | Raging Bolt Ogerpon | 937 | 205 | 0 | 3386 | 0 | 0 | 0.0157 |
| 2 | Dragapult Blaziken | 1158 | 284 | 0 | 4098 | 0 | 0 | 0.0148 |
| 2 | Ogerpon Box | 998 | 207 | 0 | 3528 | 0 | 0 | 0.0145 |
| 2 | Crustle | 1182 | 274 | 0 | 4275 | 0 | 0 | 0.0152 |
| 2 | Mega Lucario ex | 951 | 222 | 0 | 3362 | 0 | 0 | 0.0136 |
| 2 | Mega Abomasnow ex | 1071 | 223 | 0 | 3813 | 0 | 0 | 0.0125 |
| 2 | Iono's Bellibolt ex | 1086 | 240 | 0 | 3881 | 0 | 0 | 0.0145 |
| 3 | Alakazam Dudunsparce | 1192 | 184 | 0 | 4285 | 0 | 3 | 0.0811 |
| 3 | Crustle | 1300 | 236 | 0 | 4786 | 0 | 2 | 0.0733 |
| 3 | Dragapult Dusknoir | 1155 | 249 | 0 | 4246 | 0 | 24 | 0.0953 |
| 3 | Dragapult | 1149 | 233 | 0 | 4259 | 0 | 0 | 0.1102 |
| 3 | Dragapult Dudunsparce | 1149 | 252 | 0 | 4253 | 0 | 3 | 0.1205 |
| 3 | Hydrapple | 1314 | 271 | 0 | 4927 | 0 | 27 | 0.1386 |
| 3 | Raging Bolt Ogerpon | 1171 | 281 | 0 | 4374 | 0 | 25 | 0.1423 |
| 3 | Dragapult Blaziken | 1329 | 278 | 0 | 4958 | 0 | 6 | 0.1450 |
| 3 | Ogerpon Box | 1148 | 262 | 0 | 4333 | 0 | 5 | 0.1813 |
| 3 | Crustle | 1199 | 272 | 0 | 4436 | 0 | 7 | 0.0742 |
| 3 | Mega Lucario ex | 1059 | 275 | 0 | 3940 | 0 | 31 | 0.1232 |
| 3 | Mega Abomasnow ex | 1237 | 274 | 0 | 4625 | 0 | 11 | 0.0859 |
| 3 | Iono's Bellibolt ex | 1246 | 253 | 0 | 4640 | 0 | 17 | 0.1163 |
| 4 | Alakazam Dudunsparce | 1093 | 182 | 0 | 3911 | 0 | 0 | 0.0680 |
| 4 | Crustle | 1320 | 230 | 0 | 4980 | 0 | 0 | 0.0657 |
| 4 | Dragapult Dusknoir | 1274 | 186 | 0 | 4748 | 0 | 2 | 0.0845 |
| 4 | Dragapult | 1331 | 244 | 0 | 4951 | 0 | 0 | 0.0775 |
| 4 | Dragapult Dudunsparce | 1211 | 215 | 0 | 4573 | 0 | 0 | 0.1021 |
| 4 | Hydrapple | 1335 | 207 | 0 | 5041 | 0 | 0 | 0.0989 |
| 4 | Raging Bolt Ogerpon | 1168 | 213 | 0 | 4366 | 0 | 0 | 0.1088 |
| 4 | Dragapult Blaziken | 1505 | 270 | 0 | 5699 | 0 | 3 | 0.1082 |
| 4 | Ogerpon Box | 1388 | 280 | 0 | 5240 | 0 | 1 | 0.1280 |
| 4 | Crustle | 1414 | 246 | 0 | 5282 | 0 | 3 | 0.0722 |
| 4 | Mega Lucario ex | 1315 | 215 | 0 | 4945 | 0 | 1 | 0.0723 |
| 4 | Mega Abomasnow ex | 1410 | 243 | 0 | 5334 | 0 | 4 | 0.0626 |
| 4 | Iono's Bellibolt ex | 1372 | 217 | 0 | 5165 | 0 | 0 | 0.0803 |
| 5 | Alakazam Dudunsparce | 1262 | 216 | 0 | 4591 | 0 | 6 | 0.1072 |
| 5 | Crustle | 1342 | 265 | 0 | 4984 | 0 | 2 | 0.0855 |
| 5 | Dragapult Dusknoir | 1173 | 211 | 0 | 4320 | 0 | 1 | 0.0949 |
| 5 | Dragapult | 1212 | 233 | 0 | 4478 | 0 | 1 | 0.1081 |
| 5 | Dragapult Dudunsparce | 1388 | 311 | 0 | 5151 | 0 | 2 | 0.1290 |
| 5 | Hydrapple | 1288 | 239 | 0 | 4753 | 0 | 0 | 0.1214 |
| 5 | Raging Bolt Ogerpon | 1364 | 280 | 0 | 5103 | 0 | 5 | 0.1844 |
| 5 | Dragapult Blaziken | 1379 | 260 | 0 | 5137 | 0 | 0 | 0.1336 |
| 5 | Ogerpon Box | 1361 | 309 | 0 | 5095 | 0 | 5 | 0.1573 |
| 5 | Crustle | 1558 | 297 | 0 | 5846 | 0 | 3 | 0.0867 |
| 5 | Mega Lucario ex | 1421 | 279 | 0 | 5323 | 0 | 6 | 0.1016 |
| 5 | Mega Abomasnow ex | 1536 | 285 | 0 | 5710 | 0 | 7 | 0.0907 |
| 5 | Iono's Bellibolt ex | 1342 | 250 | 0 | 5013 | 0 | 1 | 0.1040 |
| 6 | Alakazam Dudunsparce | 991 | 176 | 0 | 3485 | 0 | 0 | 0.0314 |
| 6 | Crustle | 1016 | 194 | 0 | 3608 | 0 | 0 | 0.0827 |
| 6 | Dragapult Dusknoir | 1058 | 188 | 0 | 3726 | 0 | 0 | 0.0359 |
| 6 | Dragapult | 1121 | 222 | 0 | 4024 | 0 | 0 | 0.0428 |
| 6 | Dragapult Dudunsparce | 1145 | 226 | 0 | 4151 | 0 | 0 | 0.0539 |
| 6 | Hydrapple | 965 | 177 | 0 | 3472 | 0 | 0 | 0.0412 |
| 6 | Raging Bolt Ogerpon | 958 | 201 | 0 | 3508 | 0 | 0 | 0.0645 |
| 6 | Dragapult Blaziken | 1188 | 248 | 0 | 4296 | 0 | 0 | 0.0529 |
| 6 | Ogerpon Box | 1053 | 217 | 0 | 3877 | 0 | 0 | 0.0644 |
| 6 | Crustle | 1036 | 186 | 0 | 3721 | 0 | 0 | 0.0772 |
| 6 | Mega Lucario ex | 976 | 184 | 0 | 3523 | 0 | 0 | 0.0326 |
| 6 | Mega Abomasnow ex | 1141 | 230 | 0 | 4174 | 0 | 0 | 0.0359 |
| 6 | Iono's Bellibolt ex | 1172 | 233 | 0 | 4252 | 0 | 0 | 0.0447 |
| 7 | Alakazam Dudunsparce | 904 | 144 | 0 | 3181 | 0 | 0 | 0.0261 |
| 7 | Crustle | 1071 | 209 | 0 | 3873 | 0 | 0 | 0.0288 |
| 7 | Dragapult Dusknoir | 1070 | 213 | 0 | 3790 | 0 | 0 | 0.0303 |
| 7 | Dragapult | 1260 | 287 | 0 | 4443 | 0 | 0 | 0.0310 |
| 7 | Dragapult Dudunsparce | 1165 | 276 | 0 | 4196 | 0 | 12 | 0.0370 |
| 7 | Hydrapple | 1076 | 240 | 0 | 3843 | 0 | 3 | 0.0295 |
| 7 | Raging Bolt Ogerpon | 1105 | 244 | 0 | 3971 | 0 | 1 | 0.0313 |
| 7 | Dragapult Blaziken | 1188 | 249 | 0 | 4248 | 0 | 0 | 0.0368 |
| 7 | Ogerpon Box | 1114 | 288 | 0 | 3985 | 0 | 0 | 0.0341 |
| 7 | Crustle | 1188 | 242 | 0 | 4214 | 0 | 1 | 0.0251 |
| 7 | Mega Lucario ex | 1032 | 247 | 0 | 3622 | 0 | 3 | 0.0352 |
| 7 | Mega Abomasnow ex | 1092 | 204 | 0 | 3774 | 0 | 0 | 0.0271 |
| 7 | Iono's Bellibolt ex | 1193 | 261 | 0 | 4319 | 0 | 0 | 0.0331 |
| 8 | Alakazam Dudunsparce | 1429 | 213 | 0 | 5042 | 0 | 1 | 0.1049 |
| 8 | Crustle | 1475 | 236 | 0 | 5369 | 0 | 0 | 0.0908 |
| 8 | Dragapult Dusknoir | 1427 | 228 | 0 | 5151 | 0 | 2 | 0.1080 |
| 8 | Dragapult | 1364 | 230 | 0 | 4942 | 0 | 7 | 0.1295 |
| 8 | Dragapult Dudunsparce | 1226 | 222 | 0 | 4563 | 0 | 8 | 0.1765 |
| 8 | Hydrapple | 1346 | 207 | 0 | 4915 | 0 | 0 | 0.1853 |
| 8 | Raging Bolt Ogerpon | 1574 | 244 | 0 | 5792 | 0 | 9 | 0.2136 |
| 8 | Dragapult Blaziken | 1440 | 228 | 0 | 5297 | 0 | 17 | 0.1866 |
| 8 | Ogerpon Box | 1628 | 256 | 0 | 5978 | 0 | 7 | 0.2136 |
| 8 | Crustle | 1595 | 290 | 0 | 5819 | 0 | 3 | 0.0765 |
| 8 | Mega Lucario ex | 1281 | 237 | 0 | 4646 | 0 | 3 | 0.0746 |
| 8 | Mega Abomasnow ex | 1282 | 215 | 0 | 4713 | 0 | 2 | 0.0716 |
| 8 | Iono's Bellibolt ex | 1433 | 253 | 0 | 5305 | 0 | 2 | 0.1070 |
| 9 | Alakazam Dudunsparce | 890 | 152 | 0 | 3220 | 0 | 0 | 0.0342 |
| 9 | Crustle | 1157 | 201 | 0 | 4074 | 0 | 0 | 0.0257 |
| 9 | Dragapult Dusknoir | 1016 | 210 | 0 | 3556 | 0 | 1 | 0.0265 |
| 9 | Dragapult | 1179 | 256 | 0 | 4237 | 0 | 0 | 0.0345 |
| 9 | Dragapult Dudunsparce | 1190 | 264 | 0 | 4240 | 0 | 0 | 0.0329 |
| 9 | Hydrapple | 1171 | 232 | 0 | 4146 | 0 | 0 | 0.0328 |
| 9 | Raging Bolt Ogerpon | 1028 | 190 | 0 | 3649 | 0 | 0 | 0.0294 |
| 9 | Dragapult Blaziken | 1143 | 234 | 0 | 4054 | 0 | 0 | 0.0377 |
| 9 | Ogerpon Box | 1173 | 231 | 0 | 4202 | 0 | 0 | 0.0364 |
| 9 | Crustle | 1153 | 240 | 0 | 4050 | 0 | 0 | 0.0276 |
| 9 | Mega Lucario ex | 944 | 197 | 0 | 3341 | 0 | 0 | 0.0333 |
| 9 | Mega Abomasnow ex | 1108 | 204 | 0 | 3882 | 0 | 0 | 0.0291 |
| 9 | Iono's Bellibolt ex | 1128 | 255 | 0 | 4047 | 0 | 0 | 0.0288 |
| 10 | Alakazam Dudunsparce | 1010 | 180 | 0 | 3562 | 0 | 0 | 0.0157 |
| 10 | Crustle | 1110 | 236 | 0 | 3913 | 0 | 0 | 0.0150 |
| 10 | Dragapult Dusknoir | 1045 | 219 | 0 | 3715 | 0 | 0 | 0.0143 |
| 10 | Dragapult | 1180 | 287 | 0 | 4213 | 0 | 0 | 0.0153 |
| 10 | Dragapult Dudunsparce | 1270 | 327 | 0 | 4545 | 0 | 0 | 0.0164 |
| 10 | Hydrapple | 1197 | 262 | 0 | 4217 | 0 | 0 | 0.0153 |
| 10 | Raging Bolt Ogerpon | 1001 | 197 | 0 | 3555 | 0 | 0 | 0.0157 |
| 10 | Dragapult Blaziken | 1180 | 306 | 0 | 4265 | 0 | 0 | 0.0155 |
| 10 | Ogerpon Box | 1017 | 203 | 0 | 3684 | 0 | 0 | 0.0165 |
| 10 | Crustle | 1144 | 281 | 0 | 4135 | 0 | 0 | 0.0151 |
| 10 | Mega Lucario ex | 878 | 131 | 0 | 3057 | 0 | 0 | 0.0142 |
| 10 | Mega Abomasnow ex | 1199 | 238 | 0 | 4285 | 0 | 0 | 0.0130 |
| 10 | Iono's Bellibolt ex | 1142 | 277 | 0 | 4123 | 0 | 0 | 0.0152 |
| 11 | Alakazam Dudunsparce | 831 | 146 | 0 | 3083 | 0 | 0 | 0.0588 |
| 11 | Crustle | 1000 | 187 | 0 | 3722 | 0 | 0 | 0.0635 |
| 11 | Dragapult Dusknoir | 926 | 168 | 0 | 3461 | 0 | 0 | 0.0467 |
| 11 | Dragapult | 1005 | 206 | 0 | 3724 | 0 | 0 | 0.0637 |
| 11 | Dragapult Dudunsparce | 1070 | 265 | 0 | 4022 | 0 | 0 | 0.0666 |
| 11 | Hydrapple | 1008 | 183 | 0 | 3770 | 0 | 0 | 0.0685 |
| 11 | Raging Bolt Ogerpon | 831 | 159 | 0 | 3080 | 0 | 0 | 0.0782 |
| 11 | Dragapult Blaziken | 1218 | 281 | 0 | 4555 | 0 | 0 | 0.0673 |
| 11 | Ogerpon Box | 1028 | 203 | 0 | 3856 | 0 | 0 | 0.0802 |
| 11 | Crustle | 952 | 182 | 0 | 3547 | 0 | 0 | 0.0721 |
| 11 | Mega Lucario ex | 1086 | 244 | 0 | 4041 | 0 | 0 | 0.0550 |
| 11 | Mega Abomasnow ex | 1206 | 275 | 0 | 4582 | 0 | 0 | 0.0364 |
| 11 | Iono's Bellibolt ex | 985 | 195 | 0 | 3738 | 0 | 0 | 0.0513 |
| 12 | Alakazam Dudunsparce | 633 | 122 | 0 | 2253 | 0 | 0 | 0.0121 |
| 12 | Crustle | 671 | 170 | 0 | 2360 | 0 | 0 | 0.0105 |
| 12 | Dragapult Dusknoir | 639 | 140 | 0 | 2258 | 0 | 0 | 0.0110 |
| 12 | Dragapult | 616 | 130 | 0 | 2188 | 0 | 0 | 0.0105 |
| 12 | Dragapult Dudunsparce | 643 | 140 | 0 | 2307 | 0 | 0 | 0.0111 |
| 12 | Hydrapple | 650 | 140 | 0 | 2308 | 0 | 0 | 0.0115 |
| 12 | Raging Bolt Ogerpon | 518 | 126 | 0 | 1852 | 0 | 0 | 0.0115 |
| 12 | Dragapult Blaziken | 609 | 144 | 0 | 2164 | 0 | 0 | 0.0117 |
| 12 | Ogerpon Box | 473 | 132 | 0 | 1675 | 0 | 0 | 0.0119 |
| 12 | Crustle | 660 | 143 | 0 | 2305 | 0 | 0 | 0.0095 |
| 12 | Mega Lucario ex | 599 | 137 | 0 | 2108 | 0 | 0 | 0.0107 |
| 12 | Mega Abomasnow ex | 550 | 141 | 0 | 1958 | 0 | 0 | 0.0111 |
| 12 | Iono's Bellibolt ex | 615 | 142 | 0 | 2181 | 0 | 0 | 0.0104 |
| 13 | Alakazam Dudunsparce | 1259 | 165 | 0 | 4481 | 0 | 56 | 0.0414 |
| 13 | Crustle | 1541 | 299 | 0 | 5520 | 0 | 116 | 0.0421 |
| 13 | Dragapult Dusknoir | 1295 | 166 | 0 | 4613 | 0 | 49 | 0.0421 |
| 13 | Dragapult | 1408 | 191 | 0 | 4990 | 0 | 86 | 0.0450 |
| 13 | Dragapult Dudunsparce | 1617 | 240 | 0 | 5803 | 0 | 145 | 0.0484 |
| 13 | Hydrapple | 1441 | 204 | 0 | 5109 | 0 | 108 | 0.0480 |
| 13 | Raging Bolt Ogerpon | 1262 | 211 | 0 | 4502 | 0 | 57 | 0.0508 |
| 13 | Dragapult Blaziken | 1507 | 221 | 0 | 5337 | 0 | 79 | 0.0426 |
| 13 | Ogerpon Box | 1441 | 195 | 0 | 5058 | 0 | 131 | 0.0511 |
| 13 | Crustle | 1386 | 268 | 0 | 4993 | 0 | 82 | 0.0429 |
| 13 | Mega Lucario ex | 1105 | 166 | 0 | 3882 | 0 | 50 | 0.0347 |
| 13 | Mega Abomasnow ex | 1426 | 218 | 0 | 5072 | 0 | 66 | 0.0360 |
| 13 | Iono's Bellibolt ex | 1651 | 210 | 0 | 5901 | 0 | 151 | 0.0396 |

## Matchups

| Deck | Rank | Archetype | Opponent | Wins | Losses | Draws | Timeouts | Errors | Win rate |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | Alakazam Dudunsparce | Alakazam Dudunsparce | 17 | 13 | 0 | 2 | 0 | 0.567 |
| 1 | 1 | Alakazam Dudunsparce | Crustle | 0 | 30 | 0 | 0 | 0 | 0.000 |
| 1 | 1 | Alakazam Dudunsparce | Dragapult Dusknoir | 11 | 19 | 0 | 0 | 0 | 0.367 |
| 1 | 1 | Alakazam Dudunsparce | Dragapult | 5 | 25 | 0 | 0 | 0 | 0.167 |
| 1 | 1 | Alakazam Dudunsparce | Dragapult Dudunsparce | 7 | 23 | 0 | 1 | 0 | 0.233 |
| 1 | 1 | Alakazam Dudunsparce | Hydrapple | 0 | 30 | 0 | 0 | 0 | 0.000 |
| 1 | 1 | Alakazam Dudunsparce | Raging Bolt Ogerpon | 2 | 27 | 1 | 1 | 0 | 0.067 |
| 1 | 1 | Alakazam Dudunsparce | Dragapult Blaziken | 4 | 26 | 0 | 0 | 0 | 0.133 |
| 1 | 1 | Alakazam Dudunsparce | Ogerpon Box | 1 | 29 | 0 | 0 | 0 | 0.033 |
| 1 | 1 | Alakazam Dudunsparce | Crustle | 1 | 29 | 0 | 2 | 0 | 0.033 |
| 1 | 1 | Alakazam Dudunsparce | Mega Lucario ex | 0 | 30 | 0 | 0 | 0 | 0.000 |
| 1 | 1 | Alakazam Dudunsparce | Mega Abomasnow ex | 0 | 30 | 0 | 0 | 0 | 0.000 |
| 1 | 1 | Alakazam Dudunsparce | Iono's Bellibolt ex | 1 | 29 | 0 | 0 | 0 | 0.033 |
| 2 | 2 | Crustle | Alakazam Dudunsparce | 30 | 0 | 0 | 0 | 0 | 1.000 |
| 2 | 2 | Crustle | Crustle | 16 | 13 | 1 | 0 | 0 | 0.533 |
| 2 | 2 | Crustle | Dragapult Dusknoir | 26 | 4 | 0 | 0 | 0 | 0.867 |
| 2 | 2 | Crustle | Dragapult | 21 | 9 | 0 | 0 | 0 | 0.700 |
| 2 | 2 | Crustle | Dragapult Dudunsparce | 20 | 10 | 0 | 0 | 0 | 0.667 |
| 2 | 2 | Crustle | Hydrapple | 19 | 11 | 0 | 0 | 0 | 0.633 |
| 2 | 2 | Crustle | Raging Bolt Ogerpon | 21 | 9 | 0 | 0 | 0 | 0.700 |
| 2 | 2 | Crustle | Dragapult Blaziken | 18 | 12 | 0 | 0 | 0 | 0.600 |
| 2 | 2 | Crustle | Ogerpon Box | 19 | 11 | 0 | 0 | 0 | 0.633 |
| 2 | 2 | Crustle | Crustle | 16 | 14 | 0 | 0 | 0 | 0.533 |
| 2 | 2 | Crustle | Mega Lucario ex | 10 | 20 | 0 | 0 | 0 | 0.333 |
| 2 | 2 | Crustle | Mega Abomasnow ex | 21 | 9 | 0 | 0 | 0 | 0.700 |
| 2 | 2 | Crustle | Iono's Bellibolt ex | 21 | 9 | 0 | 0 | 0 | 0.700 |
| 3 | 3 | Dragapult Dusknoir | Alakazam Dudunsparce | 26 | 3 | 1 | 2 | 0 | 0.867 |
| 3 | 3 | Dragapult Dusknoir | Crustle | 8 | 22 | 0 | 0 | 0 | 0.267 |
| 3 | 3 | Dragapult Dusknoir | Dragapult Dusknoir | 15 | 15 | 0 | 0 | 0 | 0.500 |
| 3 | 3 | Dragapult Dusknoir | Dragapult | 8 | 22 | 0 | 0 | 0 | 0.267 |
| 3 | 3 | Dragapult Dusknoir | Dragapult Dudunsparce | 13 | 17 | 0 | 0 | 0 | 0.433 |
| 3 | 3 | Dragapult Dusknoir | Hydrapple | 11 | 19 | 0 | 0 | 0 | 0.367 |
| 3 | 3 | Dragapult Dusknoir | Raging Bolt Ogerpon | 16 | 14 | 0 | 0 | 0 | 0.533 |
| 3 | 3 | Dragapult Dusknoir | Dragapult Blaziken | 13 | 17 | 0 | 0 | 0 | 0.433 |
| 3 | 3 | Dragapult Dusknoir | Ogerpon Box | 15 | 15 | 0 | 0 | 0 | 0.500 |
| 3 | 3 | Dragapult Dusknoir | Crustle | 11 | 19 | 0 | 0 | 0 | 0.367 |
| 3 | 3 | Dragapult Dusknoir | Mega Lucario ex | 3 | 27 | 0 | 0 | 0 | 0.100 |
| 3 | 3 | Dragapult Dusknoir | Mega Abomasnow ex | 6 | 24 | 0 | 0 | 0 | 0.200 |
| 3 | 3 | Dragapult Dusknoir | Iono's Bellibolt ex | 15 | 15 | 0 | 0 | 0 | 0.500 |
| 4 | 4 | Dragapult | Alakazam Dudunsparce | 23 | 3 | 4 | 9 | 0 | 0.767 |
| 4 | 4 | Dragapult | Crustle | 12 | 18 | 0 | 0 | 0 | 0.400 |
| 4 | 4 | Dragapult | Dragapult Dusknoir | 15 | 15 | 0 | 0 | 0 | 0.500 |
| 4 | 4 | Dragapult | Dragapult | 12 | 18 | 0 | 0 | 0 | 0.400 |
| 4 | 4 | Dragapult | Dragapult Dudunsparce | 19 | 11 | 0 | 0 | 0 | 0.633 |
| 4 | 4 | Dragapult | Hydrapple | 12 | 18 | 0 | 0 | 0 | 0.400 |
| 4 | 4 | Dragapult | Raging Bolt Ogerpon | 13 | 17 | 0 | 0 | 0 | 0.433 |
| 4 | 4 | Dragapult | Dragapult Blaziken | 10 | 20 | 0 | 0 | 0 | 0.333 |
| 4 | 4 | Dragapult | Ogerpon Box | 10 | 20 | 0 | 0 | 0 | 0.333 |
| 4 | 4 | Dragapult | Crustle | 16 | 14 | 0 | 0 | 0 | 0.533 |
| 4 | 4 | Dragapult | Mega Lucario ex | 12 | 18 | 0 | 0 | 0 | 0.400 |
| 4 | 4 | Dragapult | Mega Abomasnow ex | 9 | 21 | 0 | 0 | 0 | 0.300 |
| 4 | 4 | Dragapult | Iono's Bellibolt ex | 14 | 16 | 0 | 0 | 0 | 0.467 |
| 5 | 9 | Dragapult Dudunsparce | Alakazam Dudunsparce | 25 | 3 | 2 | 6 | 0 | 0.833 |
| 5 | 9 | Dragapult Dudunsparce | Crustle | 14 | 15 | 1 | 0 | 0 | 0.467 |
| 5 | 9 | Dragapult Dudunsparce | Dragapult Dusknoir | 15 | 15 | 0 | 0 | 0 | 0.500 |
| 5 | 9 | Dragapult Dudunsparce | Dragapult | 11 | 19 | 0 | 0 | 0 | 0.367 |
| 5 | 9 | Dragapult Dudunsparce | Dragapult Dudunsparce | 13 | 17 | 0 | 0 | 0 | 0.433 |
| 5 | 9 | Dragapult Dudunsparce | Hydrapple | 12 | 18 | 0 | 0 | 0 | 0.400 |
| 5 | 9 | Dragapult Dudunsparce | Raging Bolt Ogerpon | 18 | 12 | 0 | 0 | 0 | 0.600 |
| 5 | 9 | Dragapult Dudunsparce | Dragapult Blaziken | 13 | 17 | 0 | 0 | 0 | 0.433 |
| 5 | 9 | Dragapult Dudunsparce | Ogerpon Box | 13 | 17 | 0 | 0 | 0 | 0.433 |
| 5 | 9 | Dragapult Dudunsparce | Crustle | 13 | 17 | 0 | 0 | 0 | 0.433 |
| 5 | 9 | Dragapult Dudunsparce | Mega Lucario ex | 12 | 18 | 0 | 0 | 0 | 0.400 |
| 5 | 9 | Dragapult Dudunsparce | Mega Abomasnow ex | 9 | 21 | 0 | 0 | 0 | 0.300 |
| 5 | 9 | Dragapult Dudunsparce | Iono's Bellibolt ex | 12 | 18 | 0 | 0 | 0 | 0.400 |
| 6 | 10 | Hydrapple | Alakazam Dudunsparce | 28 | 1 | 1 | 5 | 0 | 0.933 |
| 6 | 10 | Hydrapple | Crustle | 12 | 18 | 0 | 0 | 0 | 0.400 |
| 6 | 10 | Hydrapple | Dragapult Dusknoir | 18 | 12 | 0 | 0 | 0 | 0.600 |
| 6 | 10 | Hydrapple | Dragapult | 14 | 16 | 0 | 0 | 0 | 0.467 |
| 6 | 10 | Hydrapple | Dragapult Dudunsparce | 18 | 12 | 0 | 0 | 0 | 0.600 |
| 6 | 10 | Hydrapple | Hydrapple | 16 | 14 | 0 | 0 | 0 | 0.533 |
| 6 | 10 | Hydrapple | Raging Bolt Ogerpon | 20 | 10 | 0 | 0 | 0 | 0.667 |
| 6 | 10 | Hydrapple | Dragapult Blaziken | 15 | 15 | 0 | 0 | 0 | 0.500 |
| 6 | 10 | Hydrapple | Ogerpon Box | 19 | 11 | 0 | 0 | 0 | 0.633 |
| 6 | 10 | Hydrapple | Crustle | 16 | 14 | 0 | 0 | 0 | 0.533 |
| 6 | 10 | Hydrapple | Mega Lucario ex | 10 | 20 | 0 | 0 | 0 | 0.333 |
| 6 | 10 | Hydrapple | Mega Abomasnow ex | 17 | 13 | 0 | 0 | 0 | 0.567 |
| 6 | 10 | Hydrapple | Iono's Bellibolt ex | 15 | 15 | 0 | 0 | 0 | 0.500 |
| 7 | 11 | Raging Bolt Ogerpon | Alakazam Dudunsparce | 26 | 3 | 1 | 5 | 0 | 0.867 |
| 7 | 11 | Raging Bolt Ogerpon | Crustle | 14 | 16 | 0 | 0 | 0 | 0.467 |
| 7 | 11 | Raging Bolt Ogerpon | Dragapult Dusknoir | 20 | 10 | 0 | 0 | 0 | 0.667 |
| 7 | 11 | Raging Bolt Ogerpon | Dragapult | 17 | 13 | 0 | 0 | 0 | 0.567 |
| 7 | 11 | Raging Bolt Ogerpon | Dragapult Dudunsparce | 17 | 13 | 0 | 0 | 0 | 0.567 |
| 7 | 11 | Raging Bolt Ogerpon | Hydrapple | 15 | 15 | 0 | 0 | 0 | 0.500 |
| 7 | 11 | Raging Bolt Ogerpon | Raging Bolt Ogerpon | 21 | 9 | 0 | 0 | 0 | 0.700 |
| 7 | 11 | Raging Bolt Ogerpon | Dragapult Blaziken | 18 | 12 | 0 | 0 | 0 | 0.600 |
| 7 | 11 | Raging Bolt Ogerpon | Ogerpon Box | 21 | 9 | 0 | 0 | 0 | 0.700 |
| 7 | 11 | Raging Bolt Ogerpon | Crustle | 11 | 19 | 0 | 0 | 0 | 0.367 |
| 7 | 11 | Raging Bolt Ogerpon | Mega Lucario ex | 4 | 26 | 0 | 0 | 0 | 0.133 |
| 7 | 11 | Raging Bolt Ogerpon | Mega Abomasnow ex | 7 | 23 | 0 | 0 | 0 | 0.233 |
| 7 | 11 | Raging Bolt Ogerpon | Iono's Bellibolt ex | 18 | 12 | 0 | 0 | 0 | 0.600 |
| 8 | 18 | Dragapult Blaziken | Alakazam Dudunsparce | 25 | 5 | 0 | 2 | 0 | 0.833 |
| 8 | 18 | Dragapult Blaziken | Crustle | 13 | 17 | 0 | 0 | 0 | 0.433 |
| 8 | 18 | Dragapult Blaziken | Dragapult Dusknoir | 14 | 16 | 0 | 0 | 0 | 0.467 |
| 8 | 18 | Dragapult Blaziken | Dragapult | 17 | 13 | 0 | 0 | 0 | 0.567 |
| 8 | 18 | Dragapult Blaziken | Dragapult Dudunsparce | 19 | 11 | 0 | 0 | 0 | 0.633 |
| 8 | 18 | Dragapult Blaziken | Hydrapple | 16 | 14 | 0 | 0 | 0 | 0.533 |
| 8 | 18 | Dragapult Blaziken | Raging Bolt Ogerpon | 16 | 14 | 0 | 0 | 0 | 0.533 |
| 8 | 18 | Dragapult Blaziken | Dragapult Blaziken | 16 | 14 | 0 | 0 | 0 | 0.533 |
| 8 | 18 | Dragapult Blaziken | Ogerpon Box | 13 | 17 | 0 | 0 | 0 | 0.433 |
| 8 | 18 | Dragapult Blaziken | Crustle | 7 | 23 | 0 | 0 | 0 | 0.233 |
| 8 | 18 | Dragapult Blaziken | Mega Lucario ex | 11 | 19 | 0 | 0 | 0 | 0.367 |
| 8 | 18 | Dragapult Blaziken | Mega Abomasnow ex | 10 | 20 | 0 | 0 | 0 | 0.333 |
| 8 | 18 | Dragapult Blaziken | Iono's Bellibolt ex | 18 | 12 | 0 | 0 | 0 | 0.600 |
| 9 | 22 | Ogerpon Box | Alakazam Dudunsparce | 27 | 3 | 0 | 3 | 0 | 0.900 |
| 9 | 22 | Ogerpon Box | Crustle | 13 | 17 | 0 | 0 | 0 | 0.433 |
| 9 | 22 | Ogerpon Box | Dragapult Dusknoir | 23 | 7 | 0 | 0 | 0 | 0.767 |
| 9 | 22 | Ogerpon Box | Dragapult | 16 | 14 | 0 | 0 | 0 | 0.533 |
| 9 | 22 | Ogerpon Box | Dragapult Dudunsparce | 16 | 14 | 0 | 0 | 0 | 0.533 |
| 9 | 22 | Ogerpon Box | Hydrapple | 13 | 17 | 0 | 0 | 0 | 0.433 |
| 9 | 22 | Ogerpon Box | Raging Bolt Ogerpon | 21 | 9 | 0 | 0 | 0 | 0.700 |
| 9 | 22 | Ogerpon Box | Dragapult Blaziken | 14 | 16 | 0 | 0 | 0 | 0.467 |
| 9 | 22 | Ogerpon Box | Ogerpon Box | 15 | 15 | 0 | 0 | 0 | 0.500 |
| 9 | 22 | Ogerpon Box | Crustle | 9 | 21 | 0 | 0 | 0 | 0.300 |
| 9 | 22 | Ogerpon Box | Mega Lucario ex | 7 | 23 | 0 | 0 | 0 | 0.233 |
| 9 | 22 | Ogerpon Box | Mega Abomasnow ex | 9 | 21 | 0 | 0 | 0 | 0.300 |
| 9 | 22 | Ogerpon Box | Iono's Bellibolt ex | 17 | 13 | 0 | 0 | 0 | 0.567 |
| 10 | 1000 | Crustle | Alakazam Dudunsparce | 23 | 5 | 2 | 2 | 0 | 0.767 |
| 10 | 1000 | Crustle | Crustle | 9 | 21 | 0 | 0 | 0 | 0.300 |
| 10 | 1000 | Crustle | Dragapult Dusknoir | 20 | 10 | 0 | 0 | 0 | 0.667 |
| 10 | 1000 | Crustle | Dragapult | 21 | 9 | 0 | 0 | 0 | 0.700 |
| 10 | 1000 | Crustle | Dragapult Dudunsparce | 17 | 13 | 0 | 0 | 0 | 0.567 |
| 10 | 1000 | Crustle | Hydrapple | 8 | 21 | 1 | 0 | 0 | 0.267 |
| 10 | 1000 | Crustle | Raging Bolt Ogerpon | 19 | 11 | 0 | 0 | 0 | 0.633 |
| 10 | 1000 | Crustle | Dragapult Blaziken | 25 | 5 | 0 | 0 | 0 | 0.833 |
| 10 | 1000 | Crustle | Ogerpon Box | 24 | 6 | 0 | 0 | 0 | 0.800 |
| 10 | 1000 | Crustle | Crustle | 12 | 18 | 0 | 0 | 0 | 0.400 |
| 10 | 1000 | Crustle | Mega Lucario ex | 10 | 20 | 0 | 0 | 0 | 0.333 |
| 10 | 1000 | Crustle | Mega Abomasnow ex | 17 | 13 | 0 | 0 | 0 | 0.567 |
| 10 | 1000 | Crustle | Iono's Bellibolt ex | 17 | 13 | 0 | 0 | 0 | 0.567 |
| 11 | 1001 | Mega Lucario ex | Alakazam Dudunsparce | 28 | 0 | 2 | 8 | 0 | 0.933 |
| 11 | 1001 | Mega Lucario ex | Crustle | 27 | 3 | 0 | 0 | 0 | 0.900 |
| 11 | 1001 | Mega Lucario ex | Dragapult Dusknoir | 25 | 5 | 0 | 0 | 0 | 0.833 |
| 11 | 1001 | Mega Lucario ex | Dragapult | 19 | 11 | 0 | 0 | 0 | 0.633 |
| 11 | 1001 | Mega Lucario ex | Dragapult Dudunsparce | 21 | 9 | 0 | 0 | 0 | 0.700 |
| 11 | 1001 | Mega Lucario ex | Hydrapple | 24 | 6 | 0 | 0 | 0 | 0.800 |
| 11 | 1001 | Mega Lucario ex | Raging Bolt Ogerpon | 23 | 7 | 0 | 0 | 0 | 0.767 |
| 11 | 1001 | Mega Lucario ex | Dragapult Blaziken | 21 | 9 | 0 | 0 | 0 | 0.700 |
| 11 | 1001 | Mega Lucario ex | Ogerpon Box | 26 | 4 | 0 | 0 | 0 | 0.867 |
| 11 | 1001 | Mega Lucario ex | Crustle | 25 | 5 | 0 | 0 | 0 | 0.833 |
| 11 | 1001 | Mega Lucario ex | Mega Lucario ex | 17 | 12 | 1 | 0 | 0 | 0.567 |
| 11 | 1001 | Mega Lucario ex | Mega Abomasnow ex | 11 | 19 | 0 | 0 | 0 | 0.367 |
| 11 | 1001 | Mega Lucario ex | Iono's Bellibolt ex | 28 | 2 | 0 | 0 | 0 | 0.933 |
| 12 | 1002 | Mega Abomasnow ex | Alakazam Dudunsparce | 30 | 0 | 0 | 1 | 0 | 1.000 |
| 12 | 1002 | Mega Abomasnow ex | Crustle | 21 | 9 | 0 | 0 | 0 | 0.700 |
| 12 | 1002 | Mega Abomasnow ex | Dragapult Dusknoir | 26 | 4 | 0 | 0 | 0 | 0.867 |
| 12 | 1002 | Mega Abomasnow ex | Dragapult | 26 | 4 | 0 | 0 | 0 | 0.867 |
| 12 | 1002 | Mega Abomasnow ex | Dragapult Dudunsparce | 22 | 8 | 0 | 0 | 0 | 0.733 |
| 12 | 1002 | Mega Abomasnow ex | Hydrapple | 22 | 8 | 0 | 0 | 0 | 0.733 |
| 12 | 1002 | Mega Abomasnow ex | Raging Bolt Ogerpon | 20 | 10 | 0 | 0 | 0 | 0.667 |
| 12 | 1002 | Mega Abomasnow ex | Dragapult Blaziken | 23 | 7 | 0 | 0 | 0 | 0.767 |
| 12 | 1002 | Mega Abomasnow ex | Ogerpon Box | 22 | 8 | 0 | 0 | 0 | 0.733 |
| 12 | 1002 | Mega Abomasnow ex | Crustle | 20 | 10 | 0 | 0 | 0 | 0.667 |
| 12 | 1002 | Mega Abomasnow ex | Mega Lucario ex | 21 | 9 | 0 | 0 | 0 | 0.700 |
| 12 | 1002 | Mega Abomasnow ex | Mega Abomasnow ex | 16 | 14 | 0 | 0 | 0 | 0.533 |
| 12 | 1002 | Mega Abomasnow ex | Iono's Bellibolt ex | 25 | 5 | 0 | 0 | 0 | 0.833 |
| 13 | 1003 | Iono's Bellibolt ex | Alakazam Dudunsparce | 30 | 0 | 0 | 1 | 0 | 1.000 |
| 13 | 1003 | Iono's Bellibolt ex | Crustle | 12 | 18 | 0 | 0 | 0 | 0.400 |
| 13 | 1003 | Iono's Bellibolt ex | Dragapult Dusknoir | 21 | 9 | 0 | 0 | 0 | 0.700 |
| 13 | 1003 | Iono's Bellibolt ex | Dragapult | 13 | 17 | 0 | 0 | 0 | 0.433 |
| 13 | 1003 | Iono's Bellibolt ex | Dragapult Dudunsparce | 16 | 14 | 0 | 0 | 0 | 0.533 |
| 13 | 1003 | Iono's Bellibolt ex | Hydrapple | 17 | 13 | 0 | 0 | 0 | 0.567 |
| 13 | 1003 | Iono's Bellibolt ex | Raging Bolt Ogerpon | 19 | 11 | 0 | 0 | 0 | 0.633 |
| 13 | 1003 | Iono's Bellibolt ex | Dragapult Blaziken | 17 | 13 | 0 | 0 | 0 | 0.567 |
| 13 | 1003 | Iono's Bellibolt ex | Ogerpon Box | 13 | 17 | 0 | 0 | 0 | 0.433 |
| 13 | 1003 | Iono's Bellibolt ex | Crustle | 16 | 14 | 0 | 0 | 0 | 0.533 |
| 13 | 1003 | Iono's Bellibolt ex | Mega Lucario ex | 2 | 28 | 0 | 0 | 0 | 0.067 |
| 13 | 1003 | Iono's Bellibolt ex | Mega Abomasnow ex | 8 | 22 | 0 | 0 | 0 | 0.267 |
| 13 | 1003 | Iono's Bellibolt ex | Iono's Bellibolt ex | 21 | 9 | 0 | 0 | 0 | 0.700 |
