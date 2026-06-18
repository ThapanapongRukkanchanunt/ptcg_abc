# Sample Dragapult Benchmark

Opponent: `Kiyotah sample Dragapult ex deck`
Games per deck: 10
Max selections per game: 600

| Deck | Archetype | Wins | Losses | Draws | Timeouts | Errors | Win rate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Dragapult ex | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 2 | Dragapult ex | 2 | 8 | 0 | 0 | 0 | 0.200 |
| 3 | Dragapult ex | 1 | 9 | 0 | 0 | 0 | 0.100 |
| 4 | Dragapult ex | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 5 | Dragapult ex | 2 | 8 | 0 | 0 | 0 | 0.200 |
| 6 | Dragapult ex | 2 | 8 | 0 | 0 | 0 | 0.200 |
| 7 | Dragapult ex | 1 | 9 | 0 | 0 | 0 | 0.100 |
| 8 | Dragapult ex | 2 | 8 | 0 | 0 | 0 | 0.200 |
| 9 | Dragapult ex | 2 | 8 | 0 | 0 | 0 | 0.200 |
| 10 | Raging Bolt ex | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 11 | Raging Bolt ex | 2 | 8 | 0 | 0 | 0 | 0.200 |
| 12 | Alakazam Powerful Hand | 2 | 8 | 0 | 1 | 0 | 0.200 |
| 13 | Alakazam Powerful Hand | 1 | 9 | 0 | 2 | 0 | 0.100 |
| 14 | Festival Lead | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 15 | Festival Lead | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 16 | Mega Lopunny ex | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 17 | Mega Lopunny ex | 1 | 9 | 0 | 0 | 0 | 0.100 |
| 18 | Rocket's Mewtwo ex | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 19 | Rocket's Mewtwo ex | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 20 | Hydrapple ex | 2 | 8 | 0 | 0 | 0 | 0.200 |
| 21 | Hydrapple ex | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 22 | Ogerpon Box | 1 | 9 | 0 | 0 | 0 | 0.100 |
| 23 | Ogerpon Box | 1 | 9 | 0 | 0 | 0 | 0.100 |
| 24 | N's Zoroark ex | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 25 | N's Zoroark ex | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 26 | Cynthia's Garchomp ex | 1 | 9 | 0 | 0 | 0 | 0.100 |
| 27 | Cynthia's Garchomp ex | 0 | 10 | 0 | 0 | 0 | 0.000 |

## Deck Labels

| Deck | Label |
| ---: | --- |
| 1 | Dragapult ex / None / Hiromu Sasaki 1st |
| 2 | Dragapult ex / None / Andrew Hedrick 1st |
| 3 | Dragapult ex / Dragapult Dusknoir / Drake Z. 1st |
| 4 | Dragapult ex / Dragapult Dusknoir / Itsuki S. 1st |
| 5 | Dragapult ex / Dragapult Froslass / Paul Miller 235th |
| 6 | Dragapult ex / Dragapult Blaziken / Kaisei E. 2nd |
| 7 | Dragapult ex / Dragapult Blaziken / Lucas T. 2nd |
| 8 | Dragapult ex / Dragapult Dudunsparce / Diego Varea Barrera 1st |
| 9 | Dragapult ex / Dragapult Dudunsparce / Louisiana P. 1st |
| 10 | Raging Bolt ex / Raging Bolt Ogerpon / Charles R. 2nd |
| 11 | Raging Bolt ex / Raging Bolt Ogerpon / Donguk Jung 2nd |
| 12 | Alakazam Powerful Hand / Alakazam Dudunsparce / Cerys Jones 1st |
| 13 | Alakazam Powerful Hand / Alakazam Dudunsparce / Kamil B. 6th |
| 14 | Festival Lead / None / Miro J. 3rd |
| 15 | Festival Lead / None / Julian Nicolas Conde Borjas 4th |
| 16 | Mega Lopunny ex / Lopunny Dudunsparce / Miloslav Posledni 1st |
| 17 | Mega Lopunny ex / Lopunny Dudunsparce / Luigi A. 2nd |
| 18 | Rocket's Mewtwo ex / All / Bruno Barros 3rd |
| 19 | Rocket's Mewtwo ex / All / Valentin M. 3rd |
| 20 | Hydrapple ex / All / Matias Matricardi 1st |
| 21 | Hydrapple ex / All / Cass M. 2nd |
| 22 | Ogerpon Box / All / Clayton T. 1st |
| 23 | Ogerpon Box / All / Landon E. 5th |
| 24 | N's Zoroark ex / None / Attila R. 7th |
| 25 | N's Zoroark ex / None / Geon Oh 10th |
| 26 | Cynthia's Garchomp ex / All / Lucas A. 1st |
| 27 | Cynthia's Garchomp ex / All / Neddy Kosek 3rd |

## Debug Samples

These are compact traces from early losses, timeouts, or errors. They show the rule agent's selected option and the first prize-map route it was trying to serve.

### Deck 1, game 1: loss

- Label: `Dragapult ex / None / Hiromu Sasaki 1st`
- Our player index: 0
- Steps: 182
- Prize counts: `(3, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | PLAY#3 Ultra Ball | Dreepy -> atk 151 -> ACTIVE Dreepy (0 prizes, setup 900.0) | Fezandipiti ex |
| 2 | DISCARD | CARD#2 Crispin; CARD#3 Crispin | Dreepy -> atk 151 -> ACTIVE Dreepy (0 prizes, setup 900.0) | Fezandipiti ex |
| 2 | TO_HAND | CARD#0 Drakloak | none | Fezandipiti ex |
| 2 | MAIN | PLAY#1 Lillie's Determination | none | Fezandipiti ex |
| 2 | MAIN | PLAY#1 Moltres | none | Fezandipiti ex |
| 2 | MAIN | PLAY#0 Poké Pad | none | Fezandipiti ex |
| 2 | TO_HAND | CARD#0 Drakloak | none | Fezandipiti ex |
| 2 | MAIN | PLAY#0 Buddy-Buddy Poffin | none | Fezandipiti ex |
| 2 | MAIN | END#0  | none | Fezandipiti ex |
| 4 | MAIN | EVOLVE#1 Drakloak | none | Fezandipiti ex |

### Deck 2, game 1: loss

- Label: `Dragapult ex / None / Andrew Hedrick 1st`
- Our player index: 0
- Steps: 162
- Prize counts: `(6, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | PLAY#3 Buddy-Buddy Poffin | none | Dragapult ex |
| 2 | MAIN | PLAY#12 Poké Pad | none | Dragapult ex |
| 2 | TO_HAND | CARD#0 Drakloak | none | Dragapult ex |
| 2 | MAIN | ATTACH#0 Basic {D} Energy | none | Dragapult ex |
| 2 | MAIN | RETREAT#0  | none | Dragapult ex |
| 2 | SWITCH | CARD#0 Dreepy | none | Dragapult ex |
| 2 | MAIN | END#0  | none | Dragapult ex |
| 4 | MAIN | EVOLVE#9 Drakloak | Dreepy -> atk 150 -> ACTIVE Budew (0 prizes, setup 450.0) | Dragapult ex, Fezandipiti ex |
| 4 | MAIN | ABILITY#9  | none | Dragapult ex, Fezandipiti ex |
| 4 | TO_HAND | CARD#1 Fezandipiti ex | none | Dragapult ex, Fezandipiti ex |

### Deck 3, game 1: loss

- Label: `Dragapult ex / Dragapult Dusknoir / Drake Z. 1st`
- Our player index: 0
- Steps: 104
- Prize counts: `(6, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | ATTACH#4 Basic {R} Energy | Budew -> atk 323 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Fezandipiti ex |
| 2 | MAIN | PLAY#0 Dawn | Budew -> atk 323 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Fezandipiti ex |
| 2 | TO_HAND | CARD#6 Fezandipiti ex | Budew -> atk 323 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Fezandipiti ex |
| 2 | TO_HAND | CARD#0 Dusclops | Budew -> atk 323 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Fezandipiti ex |
| 2 | TO_HAND | CARD#0 Dragapult ex | Budew -> atk 323 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Fezandipiti ex |
| 2 | MAIN | PLAY#0 Fezandipiti ex | Budew -> atk 323 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Fezandipiti ex |
| 2 | MAIN | ATTACK#0  atk=323 | Budew -> atk 323 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Fezandipiti ex |
| 4 | MAIN | EVOLVE#0 Dusclops | Budew -> atk 323 -> ACTIVE Budew (0 prizes, setup 0.0) | Fezandipiti ex |
| 4 | MAIN | ABILITY#0  | Budew -> atk 323 -> ACTIVE Budew (0 prizes, setup 0.0) | Fezandipiti ex |
| 4 | DAMAGE_COUNTER | CARD#1 Latias ex | Budew -> atk 323 -> ACTIVE Budew (0 prizes, setup 0.0) | Fezandipiti ex |

### Deck 4, game 1: loss

- Label: `Dragapult ex / Dragapult Dusknoir / Itsuki S. 1st`
- Our player index: 0
- Steps: 98
- Prize counts: `(6, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | ATTACH#1 Basic {D} Energy | none | Fezandipiti ex |
| 2 | MAIN | PLAY#0 Boss’s Orders | none | Fezandipiti ex |
| 2 | SWITCH | CARD#0 Dreepy | none | Fezandipiti ex |
| 2 | MAIN | END#0  | none | Fezandipiti ex |
| 4 | MAIN | EVOLVE#0 Drakloak | none | Fezandipiti ex |
| 4 | MAIN | PLAY#0 Ultra Ball | none | Fezandipiti ex |
| 4 | TO_HAND | CARD#3 Fezandipiti ex | none | Fezandipiti ex |
| 4 | MAIN | PLAY#0 Fezandipiti ex | none | Fezandipiti ex |
| 4 | MAIN | ABILITY#0  | none | Fezandipiti ex |
| 4 | TO_HAND | CARD#0 Duskull | none | Fezandipiti ex |

### Deck 5, game 1: loss

- Label: `Dragapult ex / Dragapult Froslass / Paul Miller 235th`
- Our player index: 0
- Steps: 202
- Prize counts: `(5, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | PLAY#4 Buddy-Buddy Poffin | none | none |
| 2 | MAIN | ATTACH#1 Basic {R} Energy | none | none |
| 2 | MAIN | PLAY#2 Lillie's Determination | none | none |
| 2 | MAIN | PLAY#1 Buddy-Buddy Poffin | none | none |
| 2 | MAIN | PLAY#1 Poké Pad | none | none |
| 2 | TO_HAND | CARD#2 Drakloak | none | none |
| 2 | MAIN | PLAY#0 Ultra Ball | none | none |
| 2 | TO_HAND | CARD#3 Drakloak | none | none |
| 2 | MAIN | RETREAT#0  | none | none |
| 2 | SWITCH | CARD#2 Shaymin | none | none |

### Deck 6, game 1: loss

- Label: `Dragapult ex / Dragapult Blaziken / Kaisei E. 2nd`
- Our player index: 0
- Steps: 158
- Prize counts: `(5, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | ATTACH#1 Basic {P} Energy | Dreepy -> atk 150 -> ACTIVE Budew (0 prizes, setup 450.0) | Fezandipiti ex |
| 2 | MAIN | PLAY#0 Poké Pad | Dreepy -> atk 150 -> ACTIVE Budew (0 prizes, setup 0.0) | Fezandipiti ex |
| 2 | TO_HAND | CARD#2 Drakloak | Dreepy -> atk 150 -> ACTIVE Budew (0 prizes, setup 0.0) | Fezandipiti ex |
| 2 | MAIN | ATTACK#1  atk=150 | Dreepy -> atk 150 -> ACTIVE Budew (0 prizes, setup 0.0) | Fezandipiti ex |
| 4 | MAIN | EVOLVE#0 Drakloak | Dreepy -> atk 150 -> ACTIVE Budew (0 prizes, setup 0.0) | Fezandipiti ex |
| 4 | MAIN | EVOLVE#4 Drakloak | Dreepy -> atk 150 -> ACTIVE Budew (0 prizes, setup 950.0) | Fezandipiti ex |
| 4 | MAIN | ABILITY#4  | none | Fezandipiti ex |
| 4 | TO_HAND | CARD#1 Crispin | none | Fezandipiti ex |
| 4 | MAIN | ABILITY#5  | Drakloak -> atk 152 -> ACTIVE Budew (1 prizes, setup 450.0) | Fezandipiti ex |
| 4 | TO_HAND | CARD#0 Meowth ex | Drakloak -> atk 152 -> ACTIVE Budew (1 prizes, setup 450.0) | Fezandipiti ex |

### Deck 7, game 1: loss

- Label: `Dragapult ex / Dragapult Blaziken / Lucas T. 2nd`
- Our player index: 0
- Steps: 124
- Prize counts: `(6, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | PLAY#0 Poké Pad | none | Fezandipiti ex |
| 2 | TO_HAND | CARD#10 Combusken | none | Fezandipiti ex |
| 2 | MAIN | PLAY#0 Buddy-Buddy Poffin | none | Fezandipiti ex |
| 2 | MAIN | ATTACH#1 Basic {R} Energy | none | Fezandipiti ex |
| 2 | MAIN | PLAY#0 Judge | none | Fezandipiti ex |
| 2 | MAIN | PLAY#2 Chi-Yu | none | Fezandipiti ex |
| 2 | MAIN | PLAY#1 Poké Pad | none | Fezandipiti ex |
| 2 | TO_HAND | CARD#0 Drakloak | none | Fezandipiti ex |
| 2 | MAIN | PLAY#0 Ultra Ball | none | Fezandipiti ex |
| 2 | TO_HAND | CARD#1 Fezandipiti ex | none | Fezandipiti ex |

### Deck 8, game 1: loss

- Label: `Dragapult ex / Dragapult Dudunsparce / Diego Varea Barrera 1st`
- Our player index: 0
- Steps: 182
- Prize counts: `(3, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | ATTACH#3 Basic {P} Energy | Dreepy -> atk 151 -> ACTIVE Dreepy (0 prizes, setup 900.0) | Fezandipiti ex |
| 2 | MAIN | PLAY#0 Lillie's Determination | Dreepy -> atk 151 -> ACTIVE Dreepy (0 prizes, setup 450.0) | Fezandipiti ex |
| 2 | MAIN | PLAY#2 Munkidori | Dreepy -> atk 150 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Fezandipiti ex |
| 2 | MAIN | PLAY#1 Dreepy | Dreepy -> atk 150 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Fezandipiti ex |
| 2 | MAIN | PLAY#0 Poké Pad | Dreepy -> atk 150 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Fezandipiti ex |
| 2 | TO_HAND | CARD#2 Drakloak | Dreepy -> atk 150 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Fezandipiti ex |
| 2 | MAIN | ATTACK#1  atk=150 | Dreepy -> atk 150 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Fezandipiti ex |
| 4 | MAIN | EVOLVE#7 Drakloak | Dreepy -> atk 150 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Fezandipiti ex |
| 4 | MAIN | ATTACH#3 Basic {P} Energy | Dreepy -> atk 150 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Fezandipiti ex |
| 4 | MAIN | ABILITY#4  | Dreepy -> atk 150 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Fezandipiti ex |

### Deck 9, game 1: loss

- Label: `Dragapult ex / Dragapult Dudunsparce / Louisiana P. 1st`
- Our player index: 0
- Steps: 186
- Prize counts: `(3, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | PLAY#1 Poké Pad | Munkidori -> atk 141 -> ACTIVE Dreepy (0 prizes, setup 900.0) | Fezandipiti ex |
| 2 | TO_HAND | CARD#2 Drakloak | Munkidori -> atk 141 -> ACTIVE Dreepy (0 prizes, setup 900.0) | Fezandipiti ex |
| 2 | MAIN | PLAY#1 Poké Pad | Munkidori -> atk 141 -> ACTIVE Dreepy (0 prizes, setup 900.0) | Fezandipiti ex |
| 2 | TO_HAND | CARD#0 Drakloak | Munkidori -> atk 141 -> ACTIVE Dreepy (0 prizes, setup 900.0) | Fezandipiti ex |
| 2 | MAIN | PLAY#2 Buddy-Buddy Poffin | Munkidori -> atk 141 -> ACTIVE Dreepy (0 prizes, setup 900.0) | Fezandipiti ex |
| 2 | TO_BENCH | CARD#1 Dunsparce; CARD#4 Dunsparce | Munkidori -> atk 141 -> ACTIVE Dreepy (0 prizes, setup 900.0) | Fezandipiti ex |
| 2 | MAIN | PLAY#1 Crispin | Munkidori -> atk 141 -> ACTIVE Dreepy (0 prizes, setup 900.0) | Fezandipiti ex |
| 2 | TO_HAND | CARD#0 Basic {R} Energy | none | Fezandipiti ex |
| 2 | MAIN | ATTACH#4 Basic {R} Energy | Dunsparce -> atk 74 -> ACTIVE Dreepy (0 prizes, setup 950.0) | Fezandipiti ex |
| 2 | MAIN | PLAY#0 Risky Ruins | Dunsparce -> atk 74 -> ACTIVE Dreepy (0 prizes, setup 500.0) | Fezandipiti ex |

### Deck 10, game 1: loss

- Label: `Raging Bolt ex / Raging Bolt Ogerpon / Charles R. 2nd`
- Our player index: 0
- Steps: 118
- Prize counts: `(6, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | PLAY#8 Fezandipiti ex | none | Fezandipiti ex, Latias ex, Mega Kangaskhan ex |
| 2 | MAIN | ATTACH#2 Basic {P} Energy | none | Fezandipiti ex, Latias ex, Mega Kangaskhan ex |
| 2 | MAIN | PLAY#1 Ciphermaniac’s Codebreaking | none | Fezandipiti ex, Latias ex, Mega Kangaskhan ex |
| 2 | MAIN | PLAY#0 Area Zero Underdepths | none | Fezandipiti ex, Latias ex, Mega Kangaskhan ex |
| 2 | MAIN | RETREAT#0  | none | Fezandipiti ex, Latias ex, Mega Kangaskhan ex |
| 2 | SWITCH | CARD#1 Mega Kangaskhan ex | none | Fezandipiti ex, Latias ex, Mega Kangaskhan ex |
| 2 | MAIN | ABILITY#0  | none | Fezandipiti ex, Latias ex, Mega Kangaskhan ex |
| 2 | MAIN | PLAY#0 Mega Kangaskhan ex | none | Fezandipiti ex, Latias ex, Mega Kangaskhan ex |
| 2 | MAIN | PLAY#0 Mega Kangaskhan ex | none | Fezandipiti ex, Latias ex, Mega Kangaskhan ex |
| 2 | MAIN | END#0  | none | Fezandipiti ex, Latias ex, Mega Kangaskhan ex |

### Deck 11, game 1: loss

- Label: `Raging Bolt ex / Raging Bolt Ogerpon / Donguk Jung 2nd`
- Our player index: 0
- Steps: 91
- Prize counts: `(6, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | PLAY#5 Teal Mask Ogerpon ex | none | Raging Bolt, Latias ex, Mega Kangaskhan ex |
| 2 | MAIN | ATTACH#0 Basic {F} Energy | none | Raging Bolt, Latias ex, Mega Kangaskhan ex |
| 2 | MAIN | PLAY#0 Area Zero Underdepths | none | Raging Bolt, Latias ex, Mega Kangaskhan ex |
| 2 | MAIN | END#0  | none | Raging Bolt, Latias ex, Mega Kangaskhan ex |
| 4 | MAIN | END#0  | none | Raging Bolt, Latias ex, Mega Kangaskhan ex |
| 6 | MAIN | PLAY#0 Glass Trumpet | none | Iron Leaves ex, Latias ex, Mega Kangaskhan ex |
| 6 | MAIN | PLAY#0 Ultra Ball | none | Iron Leaves ex, Latias ex, Mega Kangaskhan ex |
| 6 | TO_HAND | CARD#2 Mega Kangaskhan ex | none | Iron Leaves ex, Latias ex, Mega Kangaskhan ex |
| 6 | MAIN | PLAY#0 Mega Kangaskhan ex | none | Iron Leaves ex, Latias ex, Mega Kangaskhan ex |
| 6 | MAIN | END#0  | none | Iron Leaves ex, Latias ex, Mega Kangaskhan ex |

### Deck 12, game 1: loss

- Label: `Alakazam Powerful Hand / Alakazam Dudunsparce / Cerys Jones 1st`
- Our player index: 0
- Steps: 140
- Prize counts: `(6, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | ATTACH#3 Telepath Psychic Energy | Abra -> atk 137 -> ACTIVE Dreepy (0 prizes, setup 450.0) | Fezandipiti ex |
| 2 | TO_BENCH | CARD#2 Dedenne; CARD#3 Elgyem | Abra -> atk 137 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Fezandipiti ex |
| 2 | MAIN | ABILITY#2  | Abra -> atk 137 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Fezandipiti ex |
| 2 | TO_ACTIVE | CARD#1 Elgyem | none | Fezandipiti ex |
| 2 | MAIN | PLAY#0 Dawn | none | Fezandipiti ex |
| 2 | TO_HAND | CARD#3 Fezandipiti ex | none | Fezandipiti ex |
| 2 | TO_HAND | CARD#0 Dudunsparce | none | Fezandipiti ex |
| 2 | TO_HAND | CARD#0 Alakazam | none | Fezandipiti ex |
| 2 | MAIN | PLAY#1 Fezandipiti ex | none | Fezandipiti ex |
| 2 | MAIN | PLAY#0 Nighttime Mine | none | Fezandipiti ex |

### Deck 13, game 1: loss

- Label: `Alakazam Powerful Hand / Alakazam Dudunsparce / Kamil B. 6th`
- Our player index: 0
- Steps: 174
- Prize counts: `(6, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | PLAY#2 Abra | none | Fezandipiti ex |
| 2 | MAIN | ABILITY#2  | none | Fezandipiti ex |
| 2 | TO_ACTIVE | CARD#0 Dedenne | none | Fezandipiti ex |
| 2 | MAIN | PLAY#0 Hilda | none | Fezandipiti ex |
| 2 | TO_HAND | CARD#1 Kadabra | none | Fezandipiti ex |
| 2 | TO_HAND | CARD#0 Telepath Psychic Energy | none | Fezandipiti ex |
| 2 | MAIN | ATTACH#1 Telepath Psychic Energy | Dedenne -> atk 301 -> ACTIVE Dreepy (0 prizes, setup 450.0) | Fezandipiti ex |
| 2 | TO_BENCH | CARD#0 Abra; CARD#1 Abra | Dedenne -> atk 301 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Fezandipiti ex |
| 2 | MAIN | ATTACK#2  atk=301 | Dedenne -> atk 301 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Fezandipiti ex |
| 3 | TO_ACTIVE | CARD#1 Abra | none | none |

### Deck 14, game 1: loss

- Label: `Festival Lead / None / Miro J. 3rd`
- Our player index: 0
- Steps: 131
- Prize counts: `(5, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | ATTACH#1 Basic {G} Energy | none | none |
| 2 | MAIN | PLAY#2 Buddy-Buddy Poffin | none | none |
| 2 | MAIN | PLAY#2 Poké Pad | none | none |
| 2 | TO_HAND | CARD#9 Thwackey | none | none |
| 2 | MAIN | PLAY#3 Buddy-Buddy Poffin | none | none |
| 2 | MAIN | PLAY#2 Lillie's Determination | none | none |
| 2 | MAIN | PLAY#0 Poké Pad | none | none |
| 2 | TO_HAND | CARD#6 Lilligant | none | none |
| 2 | MAIN | PLAY#0 Switch | none | none |
| 2 | SWITCH | CARD#0 Shaymin | none | none |

### Deck 15, game 1: loss

- Label: `Festival Lead / None / Julian Nicolas Conde Borjas 4th`
- Our player index: 0
- Steps: 177
- Prize counts: `(5, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | ATTACH#1 Basic {G} Energy | none | none |
| 2 | MAIN | PLAY#1 Poké Pad | none | none |
| 2 | TO_HAND | CARD#3 Thwackey | none | none |
| 2 | MAIN | PLAY#2 Poké Pad | none | none |
| 2 | TO_HAND | CARD#10 Thwackey | none | none |
| 2 | MAIN | PLAY#1 Lillie's Determination | none | none |
| 2 | MAIN | PLAY#3 Budew | none | none |
| 2 | MAIN | PLAY#5 Grookey | Budew -> atk 323 -> ACTIVE Dreepy (0 prizes, setup 500.0) | none |
| 2 | MAIN | PLAY#5 Applin | Budew -> atk 323 -> ACTIVE Dreepy (0 prizes, setup 500.0) | none |
| 2 | MAIN | ATTACH#1 Air Balloon | Budew -> atk 323 -> ACTIVE Dreepy (0 prizes, setup 500.0) | none |

### Deck 16, game 1: loss

- Label: `Mega Lopunny ex / Lopunny Dudunsparce / Miloslav Posledni 1st`
- Our player index: 0
- Steps: 145
- Prize counts: `(5, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | ATTACH#3 Basic {R} Energy | Fan Rotom -> atk 230 -> ACTIVE Dreepy (1 prizes, setup 450.0) | none |
| 2 | MAIN | PLAY#3 Dunsparce | Fan Rotom -> atk 230 -> ACTIVE Dreepy (1 prizes, setup 0.0) | none |
| 2 | MAIN | PLAY#2 Buddy-Buddy Poffin | Fan Rotom -> atk 230 -> ACTIVE Dreepy (1 prizes, setup 0.0) | none |
| 2 | TO_BENCH | CARD#0 Buneary; CARD#1 Buneary | Fan Rotom -> atk 230 -> ACTIVE Dreepy (1 prizes, setup 0.0) | none |
| 2 | MAIN | ATTACK#2  atk=230 | Fan Rotom -> atk 230 -> ACTIVE Dreepy (1 prizes, setup 0.0) | none |
| 2 | TO_HAND | CARD#0  | none | none |
| 4 | MAIN | PLAY#2 Hilda | Fan Rotom -> atk 230 -> ACTIVE Drakloak (0 prizes, setup 0.0) | none |
| 4 | TO_HAND | CARD#0 Dudunsparce | Fan Rotom -> atk 230 -> ACTIVE Drakloak (0 prizes, setup 0.0) | none |
| 4 | TO_HAND | CARD#0 Mist Energy | Fan Rotom -> atk 230 -> ACTIVE Drakloak (0 prizes, setup 0.0) | none |
| 4 | MAIN | EVOLVE#3 Dudunsparce | Fan Rotom -> atk 230 -> ACTIVE Drakloak (0 prizes, setup 0.0) | none |

### Deck 17, game 1: loss

- Label: `Mega Lopunny ex / Lopunny Dudunsparce / Luigi A. 2nd`
- Our player index: 0
- Steps: 120
- Prize counts: `(6, 1)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | ATTACH#0 Air Balloon | none | none |
| 2 | MAIN | PLAY#1 Hilda | none | none |
| 2 | TO_HAND | CARD#0 Dudunsparce | none | none |
| 2 | TO_HAND | CARD#0 Mist Energy | none | none |
| 2 | MAIN | ATTACH#1 Mist Energy | Dunsparce -> atk 74 -> ACTIVE Budew (0 prizes, setup 450.0) | none |
| 2 | MAIN | ATTACK#1  atk=74 | Dunsparce -> atk 74 -> ACTIVE Budew (0 prizes, setup 0.0) | none |
| 4 | MAIN | EVOLVE#1 Dudunsparce ex | Dunsparce -> atk 74 -> ACTIVE Budew (0 prizes, setup 0.0) | none |
| 4 | MAIN | PLAY#2 Abra | none | none |
| 4 | MAIN | PLAY#1 Hilda | none | none |
| 4 | TO_HAND | CARD#0 Dudunsparce | none | none |

### Deck 18, game 1: loss

- Label: `Rocket's Mewtwo ex / All / Bruno Barros 3rd`
- Our player index: 0
- Steps: 196
- Prize counts: `(6, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | PLAY#5 Ultra Ball | none | Team Rocket's Mewtwo ex |
| 2 | TO_HAND | CARD#3 Team Rocket's Mewtwo ex | none | Team Rocket's Mewtwo ex |
| 2 | MAIN | PLAY#4 Team Rocket's Mewtwo ex | none | Team Rocket's Mewtwo ex |
| 2 | MAIN | ATTACH#1 Basic {P} Energy | none | Team Rocket's Mewtwo ex |
| 2 | MAIN | PLAY#0 Team Rocket's Ariana | none | Team Rocket's Mewtwo ex |
| 2 | MAIN | PLAY#1 Ultra Ball | none | Team Rocket's Mewtwo ex |
| 2 | TO_HAND | CARD#4 Team Rocket's Mewtwo ex | none | Team Rocket's Mewtwo ex |
| 2 | MAIN | PLAY#5 Team Rocket's Mewtwo ex | none | Team Rocket's Mewtwo ex |
| 2 | MAIN | ATTACH#2 Brave Bangle | none | Team Rocket's Mewtwo ex |
| 2 | MAIN | ATTACH#2 Counter Gain | none | Team Rocket's Mewtwo ex |

### Deck 19, game 1: loss

- Label: `Rocket's Mewtwo ex / All / Valentin M. 3rd`
- Our player index: 0
- Steps: 213
- Prize counts: `(3, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | PLAY#0 Ultra Ball | none | Team Rocket's Kangaskhan ex, Team Rocket's Mewtwo ex |
| 2 | TO_HAND | CARD#1 Team Rocket's Mewtwo ex | none | Team Rocket's Kangaskhan ex, Team Rocket's Mewtwo ex |
| 2 | MAIN | PLAY#3 Team Rocket's Mewtwo ex | none | Team Rocket's Kangaskhan ex, Team Rocket's Mewtwo ex |
| 2 | MAIN | PLAY#0 Team Rocket's Proton | none | Team Rocket's Kangaskhan ex, Team Rocket's Mewtwo ex |
| 2 | TO_HAND | CARD#3 Team Rocket's Articuno; CARD#4 Team Rocket's Articuno; CARD#5 Team Rocket's Mimikyu | none | Team Rocket's Kangaskhan ex, Team Rocket's Mewtwo ex |
| 2 | MAIN | PLAY#1 Team Rocket's Articuno | none | Team Rocket's Kangaskhan ex, Team Rocket's Mewtwo ex |
| 2 | MAIN | PLAY#1 Team Rocket's Articuno | none | Team Rocket's Kangaskhan ex, Team Rocket's Mewtwo ex |
| 2 | MAIN | PLAY#0 Team Rocket's Factory | none | Team Rocket's Kangaskhan ex, Team Rocket's Mewtwo ex |
| 2 | MAIN | ABILITY#0  | none | Team Rocket's Kangaskhan ex, Team Rocket's Mewtwo ex |
| 2 | MAIN | ATTACH#0 Basic {P} Energy | none | Team Rocket's Kangaskhan ex, Team Rocket's Mewtwo ex |

### Deck 20, game 1: loss

- Label: `Hydrapple ex / All / Matias Matricardi 1st`
- Our player index: 0
- Steps: 154
- Prize counts: `(6, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | ATTACH#0 Basic {G} Energy | none | Fezandipiti ex, Tapu Bulu |
| 2 | MAIN | PLAY#0 Ciphermaniac’s Codebreaking | none | Fezandipiti ex, Tapu Bulu |
| 2 | MAIN | PLAY#0 Forest of Vitality | none | Fezandipiti ex, Tapu Bulu |
| 2 | MAIN | PLAY#0 Ultra Ball | none | Fezandipiti ex, Tapu Bulu |
| 2 | TO_HAND | CARD#14 Fezandipiti ex | none | Fezandipiti ex, Tapu Bulu |
| 2 | MAIN | PLAY#0 Fezandipiti ex | none | Fezandipiti ex, Tapu Bulu |
| 2 | MAIN | END#0  | none | Fezandipiti ex, Tapu Bulu |
| 4 | MAIN | END#0  | none | Fezandipiti ex, Tapu Bulu |
| 5 | TO_ACTIVE | CARD#0 Fezandipiti ex | none | Fezandipiti ex, Tapu Bulu |
| 6 | MAIN | ABILITY#1  | none | Fezandipiti ex, Tapu Bulu |

### Deck 21, game 1: loss

- Label: `Hydrapple ex / All / Cass M. 2nd`
- Our player index: 0
- Steps: 109
- Prize counts: `(6, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | PLAY#2 Meowth ex | none | Fezandipiti ex, Tapu Bulu |
| 2 | TO_HAND | CARD#0 Boss’s Orders | none | Fezandipiti ex, Tapu Bulu |
| 2 | MAIN | PLAY#0 Poké Pad | none | Fezandipiti ex, Tapu Bulu |
| 2 | TO_HAND | CARD#0 Tapu Bulu | none | Fezandipiti ex, Tapu Bulu |
| 2 | MAIN | PLAY#2 Tapu Bulu | none | Fezandipiti ex, Tapu Bulu |
| 2 | MAIN | PLAY#0 Lillie's Determination | none | Fezandipiti ex, Tapu Bulu |
| 2 | MAIN | PLAY#3 Chikorita | none | Fezandipiti ex, Tapu Bulu |
| 2 | MAIN | PLAY#2 Poké Pad | none | Fezandipiti ex, Tapu Bulu |
| 2 | TO_HAND | CARD#4 Bayleef | none | Fezandipiti ex, Tapu Bulu |
| 2 | MAIN | PLAY#0 Ultra Ball | none | Fezandipiti ex, Tapu Bulu |

### Deck 22, game 1: loss

- Label: `Ogerpon Box / All / Clayton T. 1st`
- Our player index: 0
- Steps: 168
- Prize counts: `(5, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | ATTACH#3 Basic {G} Energy | none | Latias ex, Mega Kangaskhan ex, Koraidon ex |
| 2 | MAIN | ATTACH#0 Hero’s Cape | none | Latias ex, Mega Kangaskhan ex, Koraidon ex |
| 2 | MAIN | PLAY#1 Lillie's Determination | none | Latias ex, Mega Kangaskhan ex, Koraidon ex |
| 2 | MAIN | PLAY#2 Latias ex | none | Latias ex, Mega Kangaskhan ex, Koraidon ex |
| 2 | MAIN | PLAY#0 Fezandipiti ex | none | Latias ex, Mega Kangaskhan ex, Koraidon ex |
| 2 | MAIN | PLAY#0 Lillie’s Clefairy ex | none | Latias ex, Mega Kangaskhan ex, Koraidon ex |
| 2 | MAIN | RETREAT#0  | none | Latias ex, Mega Kangaskhan ex, Koraidon ex |
| 2 | SWITCH | CARD#1 Latias ex | none | Latias ex, Mega Kangaskhan ex, Koraidon ex |
| 2 | MAIN | END#0  | none | Latias ex, Mega Kangaskhan ex, Koraidon ex |
| 4 | MAIN | ATTACH#4 Basic {G} Energy | Lillie’s Clefairy ex -> atk 371 -> BENCH Budew (0 prizes, setup 2250.0) | Latias ex, Mega Kangaskhan ex, Koraidon ex |

### Deck 23, game 1: loss

- Label: `Ogerpon Box / All / Landon E. 5th`
- Our player index: 0
- Steps: 104
- Prize counts: `(6, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | ATTACH#1 Basic {P} Energy | none | Fezandipiti ex, Latias ex, Mega Kangaskhan ex |
| 2 | MAIN | PLAY#2 Ciphermaniac’s Codebreaking | none | Fezandipiti ex, Latias ex, Mega Kangaskhan ex |
| 2 | MAIN | PLAY#0 Ultra Ball | none | Fezandipiti ex, Latias ex, Mega Kangaskhan ex |
| 2 | TO_HAND | CARD#3 Mega Kangaskhan ex | none | Fezandipiti ex, Latias ex, Mega Kangaskhan ex |
| 2 | MAIN | PLAY#0 Mega Kangaskhan ex | none | Fezandipiti ex, Latias ex, Mega Kangaskhan ex |
| 2 | MAIN | RETREAT#0  | none | Fezandipiti ex, Latias ex, Mega Kangaskhan ex |
| 2 | SWITCH | CARD#2 Mega Kangaskhan ex | none | Fezandipiti ex, Latias ex, Mega Kangaskhan ex |
| 2 | MAIN | END#0  | none | Fezandipiti ex, Latias ex, Mega Kangaskhan ex |
| 4 | MAIN | END#0  | none | Fezandipiti ex, Latias ex, Mega Kangaskhan ex |
| 6 | MAIN | ATTACH#0 Basic {P} Energy | none | Fezandipiti ex, Latias ex, Mega Kangaskhan ex |

### Deck 24, game 1: loss

- Label: `N's Zoroark ex / None / Attila R. 7th`
- Our player index: 0
- Steps: 211
- Prize counts: `(3, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | PLAY#4 Buddy-Buddy Poffin | none | Fezandipiti ex, N’s Reshiram, N's Zekrom |
| 2 | MAIN | ATTACH#0 Basic {D} Energy | none | Fezandipiti ex, N’s Reshiram, N's Zekrom |
| 2 | MAIN | END#0  | none | Fezandipiti ex, N’s Reshiram, N's Zekrom |
| 4 | MAIN | EVOLVE#1 N’s Zoroark ex | none | Fezandipiti ex, N’s Reshiram, N's Zekrom |
| 4 | MAIN | EVOLVE#5 N’s Zoroark ex | none | Fezandipiti ex, N’s Reshiram, N's Zekrom |
| 4 | MAIN | PLAY#5 N’s Zorua | none | Fezandipiti ex, N’s Reshiram, N's Zekrom |
| 4 | MAIN | ABILITY#6  | none | Fezandipiti ex, N’s Reshiram, N's Zekrom |
| 4 | MAIN | PLAY#6 Fezandipiti ex | none | Fezandipiti ex, N’s Reshiram, N's Zekrom |
| 4 | MAIN | ABILITY#7  | none | Fezandipiti ex, N’s Reshiram, N's Zekrom |
| 4 | MAIN | ATTACH#6 Basic {D} Energy | none | Fezandipiti ex, N’s Reshiram, N's Zekrom |

### Deck 25, game 1: loss

- Label: `N's Zoroark ex / None / Geon Oh 10th`
- Our player index: 0
- Steps: 156
- Prize counts: `(6, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | PLAY#1 Poké Pad | none | Fezandipiti ex, N’s Reshiram, N's Zekrom |
| 2 | TO_HAND | CARD#3 N’s Reshiram | none | Fezandipiti ex, N’s Reshiram, N's Zekrom |
| 2 | MAIN | PLAY#3 N’s Reshiram | none | Fezandipiti ex, N’s Reshiram, N's Zekrom |
| 2 | MAIN | PLAY#1 Buddy-Buddy Poffin | none | Fezandipiti ex, N’s Reshiram, N's Zekrom |
| 2 | MAIN | PLAY#0 Ciphermaniac’s Codebreaking | none | Fezandipiti ex, N’s Reshiram, N's Zekrom |
| 2 | MAIN | END#0  | none | Fezandipiti ex, N’s Reshiram, N's Zekrom |
| 4 | MAIN | EVOLVE#0 N’s Zoroark ex | none | Fezandipiti ex, N’s Reshiram, N's Zekrom |
| 4 | MAIN | EVOLVE#1 N’s Zoroark ex | none | Fezandipiti ex, N’s Reshiram, N's Zekrom |
| 4 | MAIN | ABILITY#1  | none | Fezandipiti ex, N’s Reshiram, N's Zekrom |
| 4 | MAIN | PLAY#1 N's Zekrom | none | Fezandipiti ex, N’s Reshiram, N's Zekrom |

### Deck 26, game 1: loss

- Label: `Cynthia's Garchomp ex / All / Lucas A. 1st`
- Our player index: 0
- Steps: 148
- Prize counts: `(5, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | ATTACH#0 Basic {F} Energy | Cynthia's Roselia -> atk 475 -> ACTIVE Dreepy (0 prizes, setup 450.0) | none |
| 2 | MAIN | ATTACH#0 Cynthia's Power Weight | Cynthia's Roselia -> atk 475 -> ACTIVE Dreepy (0 prizes, setup 0.0) | none |
| 2 | MAIN | ATTACH#1 Cynthia's Power Weight | Cynthia's Roselia -> atk 475 -> ACTIVE Dreepy (0 prizes, setup 0.0) | none |
| 2 | MAIN | PLAY#0 Team Rocket's Petrel | Cynthia's Roselia -> atk 475 -> ACTIVE Dreepy (0 prizes, setup 0.0) | none |
| 2 | TO_HAND | CARD#9 Buddy-Buddy Poffin | Cynthia's Roselia -> atk 475 -> ACTIVE Dreepy (0 prizes, setup 0.0) | none |
| 2 | MAIN | PLAY#0 Buddy-Buddy Poffin | Cynthia's Roselia -> atk 475 -> ACTIVE Dreepy (0 prizes, setup 0.0) | none |
| 2 | TO_BENCH | CARD#1 Cynthia's Gible; CARD#2 Cynthia's Gible | Cynthia's Roselia -> atk 475 -> ACTIVE Dreepy (0 prizes, setup 0.0) | none |
| 2 | MAIN | ATTACK#0  atk=475 | Cynthia's Roselia -> atk 475 -> ACTIVE Dreepy (0 prizes, setup 0.0) | none |
| 4 | MAIN | EVOLVE#4 Cynthia's Roserade | Cynthia's Roselia -> atk 475 -> ACTIVE Drakloak (0 prizes, setup 0.0) | none |
| 4 | MAIN | ATTACH#2 Basic {F} Energy | Cynthia's Gible -> atk 529 -> ACTIVE Drakloak (0 prizes, setup 950.0) | none |

### Deck 27, game 1: loss

- Label: `Cynthia's Garchomp ex / All / Neddy Kosek 3rd`
- Our player index: 0
- Steps: 151
- Prize counts: `(5, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | PLAY#3 Poké Pad | none | Cynthia's Garchomp ex |
| 2 | TO_HAND | CARD#3 Cynthia's Gible | none | Cynthia's Garchomp ex |
| 2 | MAIN | PLAY#4 Cynthia's Gible | none | Cynthia's Garchomp ex |
| 2 | MAIN | PLAY#4 Buddy-Buddy Poffin | none | Cynthia's Garchomp ex |
| 2 | MAIN | ATTACH#1 Cynthia's Power Weight | none | Cynthia's Garchomp ex |
| 2 | MAIN | PLAY#1 Larry’s Skill | none | Cynthia's Garchomp ex |
| 2 | TO_HAND | CARD#3 Cynthia's Garchomp ex | none | Cynthia's Garchomp ex |
| 2 | TO_HAND | CARD#0 Boss’s Orders | none | Cynthia's Garchomp ex |
| 2 | TO_HAND | CARD#0 Basic {F} Energy | none | Cynthia's Garchomp ex |
| 2 | MAIN | ATTACH#0 Basic {F} Energy | Cynthia's Roselia -> atk 475 -> ACTIVE Dreepy (0 prizes, setup 450.0) | Cynthia's Garchomp ex |

