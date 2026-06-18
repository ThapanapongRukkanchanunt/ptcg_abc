# Sample Dragapult Benchmark

Opponent: `Kiyotah sample Dragapult ex deck`
Games per deck: 10
Max selections per game: 600

| Deck | Archetype | Wins | Losses | Draws | Timeouts | Errors | Win rate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Dragapult ex | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 2 | Dragapult ex | 1 | 9 | 0 | 0 | 0 | 0.100 |
| 3 | Dragapult ex | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 4 | Dragapult ex | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 5 | Dragapult ex | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 6 | Dragapult ex | 1 | 9 | 0 | 0 | 0 | 0.100 |
| 7 | Dragapult ex | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 8 | Dragapult ex | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 9 | Dragapult ex | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 10 | Raging Bolt ex | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 11 | Raging Bolt ex | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 12 | Alakazam Powerful Hand | 1 | 9 | 0 | 0 | 0 | 0.100 |
| 13 | Alakazam Powerful Hand | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 14 | Festival Lead | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 15 | Festival Lead | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 16 | Mega Lopunny ex | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 17 | Mega Lopunny ex | 1 | 9 | 0 | 0 | 0 | 0.100 |
| 18 | Rocket's Mewtwo ex | 1 | 9 | 0 | 0 | 0 | 0.100 |
| 19 | Rocket's Mewtwo ex | 1 | 9 | 0 | 0 | 0 | 0.100 |
| 20 | Hydrapple ex | 1 | 9 | 0 | 0 | 0 | 0.100 |
| 21 | Hydrapple ex | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 22 | Ogerpon Box | 1 | 9 | 0 | 0 | 0 | 0.100 |
| 23 | Ogerpon Box | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 24 | N's Zoroark ex | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 25 | N's Zoroark ex | 0 | 10 | 0 | 0 | 0 | 0.000 |
| 26 | Cynthia's Garchomp ex | 1 | 9 | 0 | 0 | 0 | 0.100 |
| 27 | Cynthia's Garchomp ex | 1 | 9 | 0 | 0 | 0 | 0.100 |

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
- Steps: 152
- Prize counts: `(6, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | PLAY#3 Buddy-Buddy Poffin | none | Munkidori, Dreepy, Moltres |
| 2 | MAIN | PLAY#10 Poké Pad | none | Munkidori, Dreepy, Moltres |
| 2 | TO_HAND | CARD#3 Moltres | none | Munkidori, Dreepy, Moltres |
| 2 | MAIN | PLAY#10 Moltres | none | Munkidori, Dreepy, Moltres |
| 2 | MAIN | ATTACH#1 Basic {P} Energy | none | Munkidori, Dreepy, Moltres |
| 2 | MAIN | PLAY#0 Lillie's Determination | none | Munkidori, Dreepy, Moltres |
| 2 | MAIN | PLAY#1 Munkidori | none | Munkidori, Dreepy, Moltres |
| 2 | MAIN | PLAY#2 Poké Pad | none | Munkidori, Dreepy, Moltres |
| 2 | TO_HAND | CARD#3 Munkidori | none | Munkidori, Dreepy, Moltres |
| 2 | MAIN | PLAY#2 Munkidori | none | Munkidori, Dreepy, Moltres |

### Deck 2, game 1: loss

- Label: `Dragapult ex / None / Andrew Hedrick 1st`
- Our player index: 0
- Steps: 194
- Prize counts: `(4, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | PLAY#1 Ultra Ball | none | Dunsparce, Munkidori, Dreepy |
| 2 | TO_HAND | CARD#3 Dreepy | none | Dunsparce, Munkidori, Dreepy |
| 2 | MAIN | PLAY#2 Dreepy | none | Dunsparce, Munkidori, Dreepy |
| 2 | MAIN | ATTACH#1 Basic {P} Energy | none | Dunsparce, Munkidori, Dreepy |
| 2 | MAIN | PLAY#0 Crushing Hammer | none | Dunsparce, Munkidori, Dreepy |
| 2 | MAIN | RETREAT#0  | none | Dunsparce, Munkidori, Dreepy |
| 2 | SWITCH | CARD#0 Dreepy | none | Dunsparce, Munkidori, Dreepy |
| 2 | MAIN | END#0  | none | Dunsparce, Munkidori, Dreepy |
| 4 | MAIN | EVOLVE#0 Drakloak | none | Dunsparce, Munkidori, Dreepy |
| 4 | MAIN | ABILITY#0  | none | Dunsparce, Munkidori, Dreepy |

### Deck 3, game 1: loss

- Label: `Dragapult ex / Dragapult Dusknoir / Drake Z. 1st`
- Our player index: 0
- Steps: 108
- Prize counts: `(6, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | PLAY#5 Moltres | none | Dreepy, Duskull, Moltres |
| 2 | MAIN | ATTACH#1 Basic {P} Energy | none | Dreepy, Duskull, Moltres |
| 2 | MAIN | PLAY#0 Buddy-Buddy Poffin | none | Dreepy, Duskull, Moltres |
| 2 | MAIN | PLAY#0 Poké Pad | none | Dreepy, Duskull, Moltres |
| 2 | TO_HAND | CARD#8 Duskull | none | Dreepy, Duskull, Moltres |
| 2 | MAIN | PLAY#1 Duskull | none | Dreepy, Duskull, Moltres |
| 2 | MAIN | PLAY#0 Buddy-Buddy Poffin | none | Dreepy, Duskull, Moltres |
| 2 | MAIN | ATTACK#0  atk=169 | none | Dreepy, Duskull, Moltres |
| 4 | MAIN | ATTACH#0 Basic {P} Energy | none | Dreepy, Duskull, Moltres |
| 4 | MAIN | ATTACK#0  atk=169 | none | Dreepy, Duskull, Moltres |

### Deck 4, game 1: loss

- Label: `Dragapult ex / Dragapult Dusknoir / Itsuki S. 1st`
- Our player index: 0
- Steps: 136
- Prize counts: `(6, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | PLAY#0 Buddy-Buddy Poffin | Budew -> atk 323 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Munkidori, Dreepy, Duskull, Budew |
| 2 | TO_BENCH | CARD#1 Duskull; CARD#3 Duskull | Budew -> atk 323 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Munkidori, Dreepy, Duskull, Budew |
| 2 | MAIN | ATTACH#1 Basic {P} Energy | Budew -> atk 323 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Munkidori, Dreepy, Duskull, Budew |
| 2 | MAIN | ATTACK#0  atk=323 | Budew -> atk 323 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Munkidori, Dreepy, Duskull, Budew |
| 4 | MAIN | EVOLVE#2 Drakloak | Budew -> atk 323 -> ACTIVE Drakloak (0 prizes, setup 0.0) | Munkidori, Dreepy, Duskull, Budew |
| 4 | MAIN | ABILITY#0  | Budew -> atk 323 -> ACTIVE Drakloak (0 prizes, setup 0.0) | Munkidori, Dreepy, Duskull, Budew |
| 4 | TO_HAND | CARD#0 Meowth ex | Budew -> atk 323 -> ACTIVE Drakloak (0 prizes, setup 0.0) | Munkidori, Dreepy, Duskull, Budew |
| 4 | MAIN | PLAY#0 Meowth ex | Budew -> atk 323 -> ACTIVE Drakloak (0 prizes, setup 0.0) | Munkidori, Dreepy, Duskull, Budew |
| 4 | TO_HAND | CARD#0 Boss’s Orders | Budew -> atk 323 -> ACTIVE Drakloak (0 prizes, setup 0.0) | Munkidori, Dreepy, Duskull, Budew |
| 4 | MAIN | PLAY#0 Boss’s Orders | Budew -> atk 323 -> BENCH Budew (0 prizes, setup 850.0) | Munkidori, Dreepy, Duskull, Budew |

### Deck 5, game 1: loss

- Label: `Dragapult ex / Dragapult Froslass / Paul Miller 235th`
- Our player index: 0
- Steps: 153
- Prize counts: `(2, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | PLAY#2 Lillie's Determination | Budew -> atk 323 -> ACTIVE Fezandipiti ex (0 prizes, setup 0.0) | Munkidori, Budew |
| 2 | MAIN | PLAY#6 Shaymin | Budew -> atk 323 -> ACTIVE Fezandipiti ex (0 prizes, setup 0.0) | Budew |
| 2 | MAIN | PLAY#4 Poké Pad | Budew -> atk 323 -> ACTIVE Fezandipiti ex (0 prizes, setup 0.0) | Budew |
| 2 | TO_HAND | CARD#8 Budew | Budew -> atk 323 -> ACTIVE Fezandipiti ex (0 prizes, setup 0.0) | Budew |
| 2 | MAIN | PLAY#6 Budew | Budew -> atk 323 -> ACTIVE Fezandipiti ex (0 prizes, setup 0.0) | Budew |
| 2 | MAIN | ATTACH#0 Air Balloon | Budew -> atk 323 -> ACTIVE Fezandipiti ex (0 prizes, setup 0.0) | Budew |
| 2 | MAIN | ATTACK#2  atk=323 | Budew -> atk 323 -> ACTIVE Fezandipiti ex (0 prizes, setup 0.0) | Budew |
| 4 | MAIN | PLAY#3 Dreepy | Budew -> atk 323 -> ACTIVE Fezandipiti ex (0 prizes, setup 0.0) | Munkidori, Drakloak, Dragapult ex, Budew |
| 4 | MAIN | PLAY#0 Crispin | Budew -> atk 323 -> ACTIVE Fezandipiti ex (0 prizes, setup 0.0) | Drakloak, Dragapult ex, Budew, Mega Froslass ex |
| 4 | TO_HAND | CARD#0 Basic {P} Energy | Budew -> atk 323 -> ACTIVE Fezandipiti ex (0 prizes, setup 0.0) | Drakloak, Dragapult ex, Budew, Mega Froslass ex |

### Deck 6, game 1: loss

- Label: `Dragapult ex / Dragapult Blaziken / Kaisei E. 2nd`
- Our player index: 0
- Steps: 208
- Prize counts: `(6, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | PLAY#2 Poké Pad | none | Drakloak, Dragapult ex |
| 2 | TO_HAND | CARD#2 Drakloak | none | Drakloak, Dragapult ex |
| 2 | MAIN | PLAY#2 Buddy-Buddy Poffin | none | Drakloak, Dragapult ex |
| 2 | MAIN | PLAY#0 Lillie's Determination | none | Drakloak, Dragapult ex |
| 2 | MAIN | PLAY#0 Ultra Ball | none | Drakloak, Dragapult ex |
| 2 | TO_HAND | CARD#1 Drakloak | none | Drakloak, Dragapult ex |
| 2 | MAIN | ATTACH#0 Basic {P} Energy | none | Drakloak, Dragapult ex |
| 2 | MAIN | PLAY#0 Dreepy | none | Drakloak, Dragapult ex |
| 2 | MAIN | RETREAT#0  | none | Drakloak, Dragapult ex |
| 2 | SWITCH | CARD#2 Dreepy | none | Drakloak, Dragapult ex |

### Deck 7, game 1: loss

- Label: `Dragapult ex / Dragapult Blaziken / Lucas T. 2nd`
- Our player index: 0
- Steps: 102
- Prize counts: `(5, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | PLAY#1 Buddy-Buddy Poffin | Budew -> atk 323 -> ACTIVE Budew (0 prizes, setup 0.0) | Munkidori, Dreepy, Budew, Torchic |
| 2 | TO_BENCH | CARD#3 Torchic; CARD#4 Torchic | Budew -> atk 323 -> ACTIVE Budew (0 prizes, setup 0.0) | Munkidori, Dreepy, Budew, Torchic |
| 2 | MAIN | ATTACK#2  atk=323 | Budew -> atk 323 -> ACTIVE Budew (0 prizes, setup 0.0) | Munkidori, Dreepy, Budew, Torchic |
| 4 | MAIN | EVOLVE#0 Drakloak | Budew -> atk 323 -> ACTIVE Budew (0 prizes, setup 0.0) | Dreepy, Budew, Lillie’s Clefairy ex, Torchic |
| 4 | MAIN | ABILITY#1  | Budew -> atk 323 -> ACTIVE Budew (0 prizes, setup 0.0) | Dreepy, Budew, Lillie’s Clefairy ex, Torchic |
| 4 | TO_HAND | CARD#0 Buddy-Buddy Poffin | Budew -> atk 323 -> ACTIVE Budew (0 prizes, setup 0.0) | Dreepy, Budew, Lillie’s Clefairy ex, Torchic |
| 4 | MAIN | ATTACK#1  atk=323 | Budew -> atk 323 -> ACTIVE Budew (0 prizes, setup 0.0) | Dreepy, Budew, Lillie’s Clefairy ex, Torchic |
| 6 | MAIN | EVOLVE#1 Dragapult ex | Budew -> atk 323 -> ACTIVE Budew (1 prizes, setup 0.0) | Dreepy, Budew, Torchic |
| 6 | MAIN | PLAY#1 Lillie’s Clefairy ex | Budew -> atk 323 -> ACTIVE Budew (1 prizes, setup 0.0) | Dreepy, Budew, Torchic |
| 6 | MAIN | ATTACK#1  atk=323 | Budew -> atk 323 -> ACTIVE Budew (1 prizes, setup 0.0) | Dreepy, Budew, Torchic |

### Deck 8, game 1: loss

- Label: `Dragapult ex / Dragapult Dudunsparce / Diego Varea Barrera 1st`
- Our player index: 0
- Steps: 120
- Prize counts: `(6, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | PLAY#0 Boss’s Orders | Budew -> atk 323 -> BENCH Dreepy (0 prizes, setup 850.0) | Dunsparce, Dudunsparce, Drakloak, Budew |
| 2 | SWITCH | CARD#0 Dreepy | Budew -> atk 323 -> ACTIVE Latias ex (0 prizes, setup 0.0) | Dudunsparce, Drakloak, Budew, Dudunsparce ex |
| 2 | MAIN | PLAY#3 Poké Pad | Budew -> atk 323 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Dudunsparce, Drakloak, Budew, Dudunsparce ex |
| 2 | TO_HAND | CARD#3 Dudunsparce | Budew -> atk 323 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Dudunsparce, Drakloak, Budew, Dudunsparce ex |
| 2 | MAIN | ATTACH#2 Basic {P} Energy | Budew -> atk 323 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Dudunsparce, Drakloak, Budew, Dudunsparce ex |
| 2 | MAIN | ATTACK#1  atk=323 | Budew -> atk 323 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Dudunsparce, Drakloak, Budew, Dudunsparce ex |
| 4 | MAIN | EVOLVE#0 Dudunsparce ex | Budew -> atk 323 -> ACTIVE Budew (0 prizes, setup 0.0) | Munkidori, Dreepy, Budew, Moltres |
| 4 | MAIN | PLAY#0 Crispin | Budew -> atk 323 -> ACTIVE Budew (0 prizes, setup 0.0) | Munkidori, Dreepy, Budew, Moltres |
| 4 | TO_HAND | CARD#0 Basic {P} Energy | Budew -> atk 323 -> ACTIVE Budew (0 prizes, setup 0.0) | Munkidori, Dreepy, Budew, Moltres |
| 4 | ATTACH_TO | CARD#0 Basic {R} Energy | Budew -> atk 323 -> ACTIVE Budew (0 prizes, setup 0.0) | Munkidori, Dreepy, Budew, Moltres |

### Deck 9, game 1: loss

- Label: `Dragapult ex / Dragapult Dudunsparce / Louisiana P. 1st`
- Our player index: 0
- Steps: 156
- Prize counts: `(6, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | ATTACH#1 Basic {R} Energy | Dunsparce -> atk 74 -> ACTIVE Latias ex (0 prizes, setup 450.0) | Dunsparce, Munkidori, Dreepy, Moltres |
| 2 | MAIN | PLAY#1 Lillie's Determination | Dunsparce -> atk 74 -> ACTIVE Latias ex (0 prizes, setup 0.0) | Dunsparce, Munkidori, Dreepy, Moltres |
| 2 | MAIN | PLAY#3 Moltres | Dunsparce -> atk 74 -> ACTIVE Latias ex (0 prizes, setup 0.0) | Dunsparce, Munkidori, Dreepy, Moltres |
| 2 | MAIN | PLAY#3 Munkidori | Dunsparce -> atk 74 -> ACTIVE Latias ex (0 prizes, setup 0.0) | Dunsparce, Munkidori, Dreepy, Moltres |
| 2 | MAIN | PLAY#0 Poké Pad | Dunsparce -> atk 74 -> ACTIVE Latias ex (0 prizes, setup 0.0) | Dunsparce, Munkidori, Dreepy, Moltres |
| 2 | TO_HAND | CARD#2 Munkidori | Dunsparce -> atk 74 -> ACTIVE Latias ex (0 prizes, setup 0.0) | Dunsparce, Munkidori, Dreepy, Moltres |
| 2 | MAIN | ATTACK#0  atk=74 | Dunsparce -> atk 74 -> ACTIVE Latias ex (0 prizes, setup 0.0) | Dunsparce, Munkidori, Dreepy, Moltres |
| 4 | MAIN | EVOLVE#0 Dudunsparce | Dunsparce -> atk 74 -> ACTIVE Budew (0 prizes, setup 0.0) | Dunsparce, Dreepy, Dragapult ex, Moltres |
| 4 | MAIN | ABILITY#2  | Dudunsparce -> atk 76 -> ACTIVE Budew (1 prizes, setup 900.0) | Dudunsparce, Dreepy, Dragapult ex, Moltres |
| 4 | TO_ACTIVE | CARD#0 Dreepy | none | Dreepy, Dragapult ex, Moltres |

### Deck 10, game 1: loss

- Label: `Raging Bolt ex / Raging Bolt Ogerpon / Charles R. 2nd`
- Our player index: 0
- Steps: 110
- Prize counts: `(6, 6)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | ATTACH#0 Basic {G} Energy | Lillie’s Clefairy ex -> atk 371 -> ACTIVE Dreepy (0 prizes, setup 900.0) | Latias ex, Chien-Pao, Lillie’s Clefairy ex, Mega Kangaskhan ex |
| 2 | MAIN | ATTACH#2 Hero’s Cape | Lillie’s Clefairy ex -> atk 371 -> ACTIVE Dreepy (0 prizes, setup 450.0) | Latias ex, Chien-Pao, Lillie’s Clefairy ex, Mega Kangaskhan ex |
| 2 | MAIN | PLAY#1 Judge | Lillie’s Clefairy ex -> atk 371 -> ACTIVE Dreepy (0 prizes, setup 450.0) | Latias ex, Chien-Pao, Lillie’s Clefairy ex, Mega Kangaskhan ex |
| 2 | MAIN | PLAY#0 Meowth ex | none | Latias ex, Chien-Pao, Mega Kangaskhan ex |
| 2 | TO_HAND | CARD#0 Crispin | none | Latias ex, Chien-Pao, Mega Kangaskhan ex |
| 2 | MAIN | RETREAT#0  | none | Latias ex, Chien-Pao, Mega Kangaskhan ex |
| 2 | SWITCH | CARD#1 Meowth ex | none | Latias ex, Chien-Pao, Mega Kangaskhan ex |
| 2 | MAIN | END#0  | none | Latias ex, Chien-Pao, Mega Kangaskhan ex |
| 4 | MAIN | PLAY#6 Latias ex | none | Latias ex, Chien-Pao, Mega Kangaskhan ex |
| 4 | MAIN | ATTACH#4 Basic {L} Energy | Lillie’s Clefairy ex -> atk 371 -> ACTIVE Budew (0 prizes, setup 1400.0) | Latias ex, Chien-Pao, Lillie’s Clefairy ex, Mega Kangaskhan ex |

### Deck 11, game 1: loss

- Label: `Raging Bolt ex / Raging Bolt Ogerpon / Donguk Jung 2nd`
- Our player index: 0
- Steps: 156
- Prize counts: `(6, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | PLAY#6 Ultra Ball | none | Latias ex, Chien-Pao, Mega Kangaskhan ex |
| 2 | TO_HAND | CARD#3 Chien-Pao | none | Latias ex, Chien-Pao, Mega Kangaskhan ex |
| 2 | MAIN | PLAY#6 Chien-Pao | none | Latias ex, Chien-Pao, Mega Kangaskhan ex |
| 2 | MAIN | ATTACH#0 Basic {F} Energy | none | Latias ex, Chien-Pao, Mega Kangaskhan ex |
| 2 | MAIN | PLAY#0 Energy Switch | none | Latias ex, Chien-Pao, Mega Kangaskhan ex |
| 2 | MAIN | END#0  | none | Latias ex, Chien-Pao, Mega Kangaskhan ex |
| 4 | MAIN | PLAY#6 Mega Kangaskhan ex | none | Latias ex, Chien-Pao, Mega Kangaskhan ex |
| 4 | MAIN | ATTACH#0 Basic {L} Energy | none | Latias ex, Chien-Pao, Mega Kangaskhan ex |
| 4 | MAIN | RETREAT#0  | none | Latias ex, Chien-Pao, Mega Kangaskhan ex |
| 4 | SWITCH | CARD#2 Mega Kangaskhan ex | none | Latias ex, Chien-Pao, Mega Kangaskhan ex |

### Deck 12, game 1: loss

- Label: `Alakazam Powerful Hand / Alakazam Dudunsparce / Cerys Jones 1st`
- Our player index: 0
- Steps: 176
- Prize counts: `(5, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | ATTACH#1 Telepath Psychic Energy | Dedenne -> atk 301 -> ACTIVE Budew (1 prizes, setup 450.0) | Dunsparce, Genesect, Dedenne |
| 2 | TO_BENCH | CARD#1 Elgyem; CARD#0 Abra | Dedenne -> atk 301 -> ACTIVE Budew (1 prizes, setup 0.0) | Dunsparce, Genesect, Dedenne |
| 2 | MAIN | PLAY#1 Buddy-Buddy Poffin | Dedenne -> atk 301 -> ACTIVE Budew (1 prizes, setup 0.0) | Dunsparce, Genesect, Dedenne |
| 2 | TO_BENCH | CARD#2 Dunsparce | Dedenne -> atk 301 -> ACTIVE Budew (1 prizes, setup 0.0) | Dunsparce, Genesect, Dedenne |
| 2 | MAIN | ATTACK#3  atk=301 | Dedenne -> atk 301 -> ACTIVE Budew (1 prizes, setup 0.0) | Dunsparce, Genesect, Dedenne |
| 2 | TO_HAND | CARD#0  | none | Dudunsparce, Genesect |
| 3 | TO_ACTIVE | CARD#3 Dunsparce | none | Dudunsparce, Genesect |
| 4 | MAIN | PLAY#2 Dunsparce | none | Dudunsparce, Genesect |
| 4 | MAIN | ABILITY#7  | none | Dudunsparce, Genesect |
| 4 | MAIN | PLAY#7 Abra | none | Dudunsparce, Genesect |

### Deck 13, game 1: loss

- Label: `Alakazam Powerful Hand / Alakazam Dudunsparce / Kamil B. 6th`
- Our player index: 0
- Steps: 114
- Prize counts: `(6, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | ATTACH#0 Telepath Psychic Energy | Dunsparce -> atk 74 -> ACTIVE Fezandipiti ex (0 prizes, setup 450.0) | Dunsparce, Dudunsparce, Genesect |
| 2 | MAIN | PLAY#0 Poké Pad | Dunsparce -> atk 74 -> ACTIVE Fezandipiti ex (0 prizes, setup 0.0) | Dunsparce, Dudunsparce, Genesect |
| 2 | TO_HAND | CARD#4 Dudunsparce | Dunsparce -> atk 74 -> ACTIVE Fezandipiti ex (0 prizes, setup 0.0) | Dunsparce, Dudunsparce, Genesect |
| 2 | MAIN | ATTACK#0  atk=74 | Dunsparce -> atk 74 -> ACTIVE Fezandipiti ex (0 prizes, setup 0.0) | Dunsparce, Dudunsparce, Genesect |
| 4 | MAIN | EVOLVE#2 Dudunsparce | Dunsparce -> atk 74 -> ACTIVE Fezandipiti ex (0 prizes, setup 0.0) | Dunsparce, Dudunsparce, Genesect |
| 4 | MAIN | PLAY#2 Poké Pad | none | Dudunsparce, Genesect |
| 4 | TO_HAND | CARD#5 Genesect | none | Dudunsparce, Genesect |
| 4 | MAIN | PLAY#2 Genesect | none | Dudunsparce, Genesect |
| 4 | MAIN | ABILITY#3  | none | Dudunsparce, Genesect |
| 4 | TO_ACTIVE | CARD#1 Genesect | none | Dudunsparce, Genesect |

### Deck 14, game 1: loss

- Label: `Festival Lead / None / Miro J. 3rd`
- Our player index: 0
- Steps: 160
- Prize counts: `(6, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | PLAY#3 Buddy-Buddy Poffin | none | none |
| 2 | MAIN | ATTACH#2 Basic {G} Energy | none | none |
| 2 | MAIN | ATTACH#2 Brave Bangle | none | none |
| 2 | MAIN | PLAY#0 Lillie's Determination | none | none |
| 2 | MAIN | PLAY#0 Buddy-Buddy Poffin | none | none |
| 2 | MAIN | PLAY#0 Bug Catching Set | none | none |
| 2 | TO_HAND | CARD#2 Lilligant; CARD#1 Thwackey | none | none |
| 2 | MAIN | RETREAT#0  | none | none |
| 2 | SWITCH | CARD#1 Shaymin | none | none |
| 2 | MAIN | END#0  | none | none |

### Deck 15, game 1: loss

- Label: `Festival Lead / None / Julian Nicolas Conde Borjas 4th`
- Our player index: 0
- Steps: 39
- Prize counts: `(6, 5)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | PLAY#2 Festival Grounds | none | none |
| 2 | MAIN | PLAY#0 Boss’s Orders | none | none |
| 2 | SWITCH | CARD#0 Dreepy | none | none |
| 2 | MAIN | END#0  | none | none |
| 4 | MAIN | EVOLVE#1 Dipplin | none | none |
| 4 | MAIN | ATTACH#1 Basic {G} Energy | none | none |
| 4 | MAIN | ATTACK#1  atk=115 | none | none |
| 4 | ATTACK | ATTACK#0  atk=115 | none | none |

### Deck 16, game 1: loss

- Label: `Mega Lopunny ex / Lopunny Dudunsparce / Miloslav Posledni 1st`
- Our player index: 0
- Steps: 140
- Prize counts: `(4, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | ATTACH#5 Basic {R} Energy | Buneary -> atk 1096 -> ACTIVE Budew (0 prizes, setup 450.0) | Dunsparce, Fan Rotom, Buneary, Moltres |
| 2 | MAIN | ATTACH#0 Air Balloon | Buneary -> atk 1096 -> ACTIVE Budew (0 prizes, setup 0.0) | Dunsparce, Fan Rotom, Buneary, Moltres |
| 2 | MAIN | PLAY#0 Pokégear 3.0 | Buneary -> atk 1096 -> ACTIVE Budew (0 prizes, setup 0.0) | Dunsparce, Fan Rotom, Buneary, Moltres |
| 2 | TO_HAND | CARD#0 Wally's Compassion | Buneary -> atk 1096 -> ACTIVE Budew (0 prizes, setup 0.0) | Dunsparce, Fan Rotom, Buneary, Moltres |
| 2 | MAIN | PLAY#1 Pokégear 3.0 | Buneary -> atk 1096 -> ACTIVE Budew (0 prizes, setup 0.0) | Dunsparce, Fan Rotom, Buneary, Moltres |
| 2 | TO_HAND | CARD#0 Lillie's Determination | Buneary -> atk 1096 -> ACTIVE Budew (0 prizes, setup 0.0) | Dunsparce, Fan Rotom, Buneary, Moltres |
| 2 | MAIN | PLAY#2 Lillie's Determination | Buneary -> atk 1096 -> ACTIVE Budew (0 prizes, setup 0.0) | Dunsparce, Fan Rotom, Buneary, Moltres |
| 2 | MAIN | PLAY#0 Buneary | Buneary -> atk 1096 -> ACTIVE Budew (0 prizes, setup 0.0) | Dunsparce, Fan Rotom, Buneary, Moltres |
| 2 | MAIN | PLAY#1 Abra | Buneary -> atk 1096 -> ACTIVE Budew (0 prizes, setup 0.0) | Dunsparce, Fan Rotom, Buneary, Moltres |
| 2 | MAIN | ATTACK#4  atk=1096 | Buneary -> atk 1096 -> ACTIVE Budew (0 prizes, setup 0.0) | Dunsparce, Fan Rotom, Buneary, Moltres |

### Deck 17, game 1: loss

- Label: `Mega Lopunny ex / Lopunny Dudunsparce / Luigi A. 2nd`
- Our player index: 0
- Steps: 114
- Prize counts: `(5, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | ATTACH#5 Mist Energy | Dunsparce -> atk 74 -> ACTIVE Dreepy (0 prizes, setup 450.0) | Dunsparce, Fan Rotom, Dudunsparce ex, Mega Lopunny ex |
| 2 | MAIN | PLAY#3 Buddy-Buddy Poffin | Dunsparce -> atk 74 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Dunsparce, Fan Rotom, Dudunsparce ex, Mega Lopunny ex |
| 2 | TO_BENCH | CARD#2 Fan Rotom; CARD#3 Dunsparce | Dunsparce -> atk 74 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Dunsparce, Fan Rotom, Dudunsparce ex, Mega Lopunny ex |
| 2 | MAIN | ABILITY#4  | Dunsparce -> atk 74 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Dunsparce, Fan Rotom, Dudunsparce ex, Mega Lopunny ex |
| 2 | TO_HAND | CARD#0 Buneary; CARD#1 Buneary; CARD#2 Buneary | Dunsparce -> atk 74 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Dunsparce, Fan Rotom, Dudunsparce ex, Mega Lopunny ex |
| 2 | MAIN | PLAY#4 Buneary | Dunsparce -> atk 74 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Dunsparce, Fan Rotom, Dudunsparce ex, Mega Lopunny ex |
| 2 | MAIN | PLAY#4 Buneary | Dunsparce -> atk 74 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Dunsparce, Fan Rotom, Dudunsparce ex, Mega Lopunny ex |
| 2 | MAIN | PLAY#3 Lillie's Determination | Dunsparce -> atk 74 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Dunsparce, Fan Rotom, Dudunsparce ex, Mega Lopunny ex |
| 2 | MAIN | ATTACK#0  atk=74 | Dunsparce -> atk 74 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Dunsparce, Fan Rotom, Dudunsparce ex, Mega Lopunny ex |
| 4 | MAIN | EVOLVE#16 Mega Lopunny ex | Dunsparce -> atk 75 -> BENCH Budew (1 prizes, setup 1300.0) | Dunsparce, Fan Rotom, Mega Lopunny ex |

### Deck 18, game 1: loss

- Label: `Rocket's Mewtwo ex / All / Bruno Barros 3rd`
- Our player index: 0
- Steps: 223
- Prize counts: `(6, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | ATTACH#0 Basic {G} Energy | none | Team Rocket's Tarountula, Team Rocket's Articuno, Team Rocket's Mewtwo ex |
| 2 | MAIN | ATTACH#1 Counter Gain | none | Team Rocket's Tarountula, Team Rocket's Articuno, Team Rocket's Mewtwo ex |
| 2 | MAIN | PLAY#1 Team Rocket's Transceiver | none | Team Rocket's Tarountula, Team Rocket's Articuno, Team Rocket's Mewtwo ex |
| 2 | TO_HAND | CARD#0 Team Rocket's Ariana | none | Team Rocket's Tarountula, Team Rocket's Articuno, Team Rocket's Mewtwo ex |
| 2 | MAIN | PLAY#1 Team Rocket's Ariana | none | Team Rocket's Tarountula, Team Rocket's Articuno, Team Rocket's Mewtwo ex |
| 2 | MAIN | PLAY#5 Team Rocket's Tarountula | none | Team Rocket's Tarountula, Team Rocket's Articuno, Team Rocket's Mewtwo ex |
| 2 | MAIN | ATTACH#2 Lucky Helmet | none | Team Rocket's Tarountula, Team Rocket's Articuno, Team Rocket's Mewtwo ex |
| 2 | MAIN | PLAY#1 Secret Box | none | Team Rocket's Tarountula, Team Rocket's Articuno, Team Rocket's Mewtwo ex |
| 2 | TO_HAND | CARD#1 Ultra Ball | none | Team Rocket's Tarountula, Team Rocket's Articuno, Team Rocket's Mewtwo ex |
| 2 | TO_HAND | CARD#0 Brave Bangle | none | Team Rocket's Tarountula, Team Rocket's Articuno, Team Rocket's Mewtwo ex |

### Deck 19, game 1: loss

- Label: `Rocket's Mewtwo ex / All / Valentin M. 3rd`
- Our player index: 0
- Steps: 200
- Prize counts: `(6, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | PLAY#4 Lillie’s Clefairy ex | none | Team Rocket's Kangaskhan ex, Team Rocket's Mewtwo ex |
| 2 | MAIN | ATTACH#0 Basic {G} Energy | none | Team Rocket's Kangaskhan ex, Team Rocket's Mewtwo ex |
| 2 | MAIN | PLAY#1 Lillie's Determination | none | Team Rocket's Kangaskhan ex, Team Rocket's Mewtwo ex |
| 2 | MAIN | PLAY#0 Team Rocket's Tarountula | none | Team Rocket's Kangaskhan ex, Team Rocket's Mewtwo ex |
| 2 | MAIN | PLAY#0 Team Rocket's Transceiver | none | Team Rocket's Kangaskhan ex, Team Rocket's Mewtwo ex |
| 2 | TO_HAND | CARD#0 Team Rocket's Petrel | none | Team Rocket's Kangaskhan ex, Team Rocket's Mewtwo ex |
| 2 | MAIN | RETREAT#0  | none | Team Rocket's Kangaskhan ex, Team Rocket's Mewtwo ex |
| 2 | SWITCH | CARD#1 Team Rocket's Tarountula | none | Team Rocket's Kangaskhan ex, Team Rocket's Mewtwo ex |
| 2 | MAIN | END#0  | none | Team Rocket's Kangaskhan ex, Team Rocket's Mewtwo ex |
| 4 | MAIN | EVOLVE#5 Team Rocket's Spidops | Team Rocket's Tarountula -> atk 559 -> ACTIVE Latias ex (0 prizes, setup 450.0) | Team Rocket's Kangaskhan ex, Team Rocket's Tarountula, Team Rocket's Mewtwo ex |

### Deck 20, game 1: loss

- Label: `Hydrapple ex / All / Matias Matricardi 1st`
- Our player index: 0
- Steps: 118
- Prize counts: `(6, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | PLAY#3 Ultra Ball | none | Tapu Bulu |
| 2 | TO_HAND | CARD#7 Tapu Bulu | none | Tapu Bulu |
| 2 | MAIN | PLAY#1 Tapu Bulu | none | Tapu Bulu |
| 2 | MAIN | PLAY#0 Forest of Vitality | none | Tapu Bulu |
| 2 | MAIN | END#0  | none | Tapu Bulu |
| 4 | MAIN | EVOLVE#0 Bayleef | none | Tapu Bulu |
| 4 | MAIN | PLAY#0 Boss’s Orders | none | Tapu Bulu |
| 4 | SWITCH | CARD#0 Drakloak | none | Tapu Bulu |
| 4 | MAIN | END#0  | none | Tapu Bulu |
| 5 | TO_ACTIVE | CARD#1 Tapu Bulu | none | Tapu Bulu |

### Deck 21, game 1: loss

- Label: `Hydrapple ex / All / Cass M. 2nd`
- Our player index: 0
- Steps: 211
- Prize counts: `(3, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | ABILITY#6  | none | Applin, Celebi, Meowth ex |
| 2 | MAIN | ATTACH#3 Basic {G} Energy | none | Applin, Celebi, Meowth ex |
| 2 | MAIN | PLAY#2 Lillie's Determination | none | Applin, Celebi, Meowth ex |
| 2 | MAIN | PLAY#1 Meowth ex | none | Applin, Celebi, Meowth ex |
| 2 | MAIN | PLAY#1 Ultra Ball | none | Applin, Celebi, Meowth ex |
| 2 | TO_HAND | CARD#2 Celebi | none | Applin, Celebi, Meowth ex |
| 2 | MAIN | PLAY#1 Celebi | none | Applin, Celebi, Meowth ex |
| 2 | MAIN | PLAY#0 Bug Catching Set | none | Applin, Celebi, Meowth ex |
| 2 | TO_HAND | CARD#3 Dipplin; CARD#0 Basic {G} Energy | none | Applin, Celebi, Meowth ex |
| 2 | MAIN | RETREAT#0  | none | Applin, Celebi, Meowth ex |

### Deck 22, game 1: loss

- Label: `Ogerpon Box / All / Clayton T. 1st`
- Our player index: 0
- Steps: 120
- Prize counts: `(6, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | PLAY#1 Ultra Ball | none | Chien-Pao, Mega Kangaskhan ex, Koraidon ex |
| 2 | TO_HAND | CARD#8 Chien-Pao | none | Chien-Pao, Mega Kangaskhan ex, Koraidon ex |
| 2 | MAIN | PLAY#5 Chien-Pao | none | Chien-Pao, Mega Kangaskhan ex, Koraidon ex |
| 2 | MAIN | ABILITY#6  | none | Chien-Pao, Mega Kangaskhan ex, Koraidon ex |
| 2 | MAIN | PLAY#11 Fezandipiti ex | none | Chien-Pao, Mega Kangaskhan ex, Koraidon ex |
| 2 | MAIN | ABILITY#13  | none | Chien-Pao, Mega Kangaskhan ex, Koraidon ex |
| 2 | MAIN | ATTACH#1 Basic {P} Energy | none | Chien-Pao, Mega Kangaskhan ex, Koraidon ex |
| 2 | MAIN | PLAY#0 Judge | none | Chien-Pao, Mega Kangaskhan ex, Koraidon ex |
| 2 | MAIN | PLAY#0 Energy Switch | none | Chien-Pao, Mega Kangaskhan ex, Koraidon ex |
| 2 | MAIN | END#0  | none | Chien-Pao, Mega Kangaskhan ex, Koraidon ex |

### Deck 23, game 1: loss

- Label: `Ogerpon Box / All / Landon E. 5th`
- Our player index: 0
- Steps: 145
- Prize counts: `(5, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | ATTACH#4 Basic {G} Energy | Wellspring Mask Ogerpon ex -> atk 135 -> ACTIVE Budew (0 prizes, setup 950.0) | Wellspring Mask Ogerpon ex, Munkidori, Mega Kangaskhan ex, Moltres |
| 2 | MAIN | PLAY#0 Cyrano | Wellspring Mask Ogerpon ex -> atk 135 -> ACTIVE Budew (0 prizes, setup 500.0) | Wellspring Mask Ogerpon ex, Munkidori, Mega Kangaskhan ex, Moltres |
| 2 | TO_HAND | CARD#1 Mega Kangaskhan ex; CARD#7 Iron Leaves ex; CARD#5 Teal Mask Ogerpon ex | none | Munkidori, Mega Kangaskhan ex, Moltres |
| 2 | MAIN | PLAY#1 Iron Leaves ex | Wellspring Mask Ogerpon ex -> atk 135 -> ACTIVE Budew (0 prizes, setup 500.0) | Wellspring Mask Ogerpon ex, Munkidori, Mega Kangaskhan ex, Moltres |
| 2 | MAIN | PLAY#1 Teal Mask Ogerpon ex | none | Munkidori, Mega Kangaskhan ex, Moltres |
| 2 | MAIN | RETREAT#0  | none | Munkidori, Mega Kangaskhan ex, Moltres |
| 2 | SWITCH | CARD#0 Mega Kangaskhan ex | none | Munkidori, Mega Kangaskhan ex, Moltres |
| 2 | MAIN | ABILITY#0  | none | Munkidori, Mega Kangaskhan ex, Moltres |
| 2 | MAIN | PLAY#0 Ultra Ball | none | Munkidori, Mega Kangaskhan ex, Moltres |
| 2 | TO_HAND | CARD#7 Moltres | none | Munkidori, Mega Kangaskhan ex, Moltres |

### Deck 24, game 1: loss

- Label: `N's Zoroark ex / None / Attila R. 7th`
- Our player index: 0
- Steps: 174
- Prize counts: `(6, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | PLAY#4 Poké Pad | none | Munkidori, N’s Darumaka, Meowth ex |
| 2 | TO_HAND | CARD#7 Munkidori | none | Munkidori, N’s Darumaka, Meowth ex |
| 2 | MAIN | PLAY#9 Munkidori | none | Munkidori, N’s Darumaka, Meowth ex |
| 2 | MAIN | ATTACH#0 Basic {D} Energy | none | Munkidori, N’s Darumaka, Meowth ex |
| 2 | MAIN | PLAY#0 Lillie's Determination | none | Munkidori, N’s Darumaka, Meowth ex |
| 2 | MAIN | PLAY#4 Fezandipiti ex | none | Munkidori, N’s Darumaka, Meowth ex |
| 2 | MAIN | PLAY#3 N’s Zorua | none | Munkidori, N’s Darumaka, Meowth ex |
| 2 | MAIN | PLAY#0 Buddy-Buddy Poffin | none | Munkidori, N’s Darumaka, Meowth ex |
| 2 | MAIN | PLAY#0 Poké Pad | none | Munkidori, N’s Darumaka, Meowth ex |
| 2 | TO_HAND | CARD#5 N’s Darumaka | none | Munkidori, N’s Darumaka, Meowth ex |

### Deck 25, game 1: loss

- Label: `N's Zoroark ex / None / Geon Oh 10th`
- Our player index: 0
- Steps: 135
- Prize counts: `(5, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | ATTACH#1 Basic {D} Energy | N’s Zorua -> atk 402 -> ACTIVE Dreepy (0 prizes, setup 450.0) | N’s Zorua, N’s Reshiram, Yveltal, N's Zekrom |
| 2 | MAIN | PLAY#2 Ultra Ball | N’s Zorua -> atk 402 -> ACTIVE Dreepy (0 prizes, setup 0.0) | N’s Zorua, N’s Reshiram, Yveltal, N's Zekrom |
| 2 | DISCARD | CARD#1 Lillie's Determination; CARD#2 Lillie's Determination | N’s Zorua -> atk 402 -> ACTIVE Dreepy (0 prizes, setup 0.0) | N’s Zorua, N’s Reshiram, Yveltal, N's Zekrom |
| 2 | TO_HAND | CARD#2 N's Zekrom | N’s Zorua -> atk 402 -> ACTIVE Dreepy (0 prizes, setup 0.0) | N’s Zorua, N’s Reshiram, Yveltal, N's Zekrom |
| 2 | MAIN | PLAY#1 N's Zekrom | N’s Zorua -> atk 402 -> ACTIVE Dreepy (0 prizes, setup 0.0) | N’s Zorua, N’s Reshiram, Yveltal, N's Zekrom |
| 2 | MAIN | ATTACK#0  atk=402 | N’s Zorua -> atk 402 -> ACTIVE Dreepy (0 prizes, setup 0.0) | N’s Zorua, N’s Reshiram, Yveltal, N's Zekrom |
| 4 | MAIN | EVOLVE#0 N’s Zoroark ex | N’s Zorua -> atk 402 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Munkidori, N’s Zorua, Yveltal, N's Zekrom |
| 4 | MAIN | PLAY#0 Buddy-Buddy Poffin | none | Munkidori, Yveltal, N's Zekrom |
| 4 | MAIN | ABILITY#0  | none | Munkidori, Yveltal, N's Zekrom |
| 4 | MAIN | PLAY#0 Yveltal | none | Munkidori, Yveltal, N's Zekrom |

### Deck 26, game 1: loss

- Label: `Cynthia's Garchomp ex / All / Lucas A. 1st`
- Our player index: 0
- Steps: 181
- Prize counts: `(5, 0)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | ATTACH#1 Basic {F} Energy | Cynthia's Gible -> atk 529 -> ACTIVE Dreepy (0 prizes, setup 450.0) | Cynthia's Roserade, Cynthia's Gible, Cynthia's Garchomp ex |
| 2 | MAIN | PLAY#1 Cynthia's Spiritomb | Cynthia's Gible -> atk 529 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Cynthia's Roserade, Cynthia's Gible, Cynthia's Garchomp ex |
| 2 | MAIN | PLAY#0 Lillie's Determination | Cynthia's Gible -> atk 529 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Cynthia's Roserade, Cynthia's Gible, Cynthia's Garchomp ex |
| 2 | MAIN | PLAY#1 Buddy-Buddy Poffin | Cynthia's Gible -> atk 529 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Cynthia's Roserade, Cynthia's Gible, Cynthia's Garchomp ex |
| 2 | TO_BENCH | CARD#0 Cynthia's Roselia; CARD#2 Cynthia's Roselia | Cynthia's Gible -> atk 529 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Cynthia's Roserade, Cynthia's Gible, Cynthia's Garchomp ex |
| 2 | MAIN | PLAY#0 Premium Power Pro | Cynthia's Gible -> atk 529 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Cynthia's Roserade, Cynthia's Gible, Cynthia's Garchomp ex |
| 2 | MAIN | ATTACK#1  atk=529 | Cynthia's Gible -> atk 529 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Cynthia's Roserade, Cynthia's Gible, Cynthia's Garchomp ex |
| 4 | MAIN | EVOLVE#12 Cynthia's Roserade | Cynthia's Gible -> atk 529 -> ACTIVE Budew (0 prizes, setup 450.0) | Cynthia's Roselia, Cynthia's Gible, Cynthia's Garchomp ex |
| 4 | MAIN | ATTACH#1 Basic {F} Energy | Cynthia's Gible -> atk 529 -> ACTIVE Budew (0 prizes, setup 450.0) | Cynthia's Roselia, Cynthia's Gible, Cynthia's Garchomp ex |
| 4 | MAIN | PLAY#0 Boss’s Orders | Cynthia's Gible -> atk 529 -> ACTIVE Budew (0 prizes, setup 0.0) | Cynthia's Roselia, Cynthia's Gible, Cynthia's Garchomp ex |

### Deck 27, game 1: loss

- Label: `Cynthia's Garchomp ex / All / Neddy Kosek 3rd`
- Our player index: 0
- Steps: 85
- Prize counts: `(6, 2)`

| Turn | Context | Selected | Prize map | Key attackers |
| ---: | --- | --- | --- | --- |
| 2 | MAIN | ATTACH#2 Basic {F} Energy | Cynthia's Gible -> atk 529 -> ACTIVE Budew (0 prizes, setup 450.0) | Cynthia's Roserade, Cynthia's Gible, Cynthia's Gabite, Cynthia's Garchomp ex |
| 2 | MAIN | PLAY#0 Boss’s Orders | Cynthia's Gible -> atk 529 -> ACTIVE Budew (0 prizes, setup 0.0) | Cynthia's Roserade, Cynthia's Gible, Cynthia's Gabite, Cynthia's Garchomp ex |
| 2 | SWITCH | CARD#0 Dreepy | Cynthia's Gible -> atk 529 -> ACTIVE Budew (0 prizes, setup 0.0) | Cynthia's Roserade, Cynthia's Gible, Cynthia's Gabite, Cynthia's Garchomp ex |
| 2 | MAIN | PLAY#0 Buddy-Buddy Poffin | Cynthia's Gible -> atk 529 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Cynthia's Roserade, Cynthia's Gible, Cynthia's Gabite, Cynthia's Garchomp ex |
| 2 | TO_BENCH | CARD#0 Cynthia's Gible; CARD#4 Cynthia's Gible | Cynthia's Gible -> atk 529 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Cynthia's Roserade, Cynthia's Gible, Cynthia's Gabite, Cynthia's Garchomp ex |
| 2 | MAIN | ATTACK#0  atk=529 | Cynthia's Gible -> atk 529 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Cynthia's Roserade, Cynthia's Gible, Cynthia's Gabite, Cynthia's Garchomp ex |
| 4 | MAIN | EVOLVE#0 Cynthia's Gabite | Cynthia's Gible -> atk 529 -> ACTIVE Budew (0 prizes, setup 0.0) | Cynthia's Roserade, Cynthia's Gible, Cynthia's Gabite, Cynthia's Garchomp ex |
| 4 | MAIN | PLAY#0 Boss’s Orders | Cynthia's Gabite -> atk 530 -> ACTIVE Budew (1 prizes, setup 0.0) | Cynthia's Roserade, Cynthia's Gabite, Cynthia's Garchomp ex |
| 4 | SWITCH | CARD#1 Dreepy | Cynthia's Gabite -> atk 530 -> ACTIVE Budew (1 prizes, setup 0.0) | Cynthia's Roserade, Cynthia's Gabite, Cynthia's Garchomp ex |
| 4 | MAIN | ABILITY#6  | Cynthia's Gabite -> atk 530 -> ACTIVE Dreepy (0 prizes, setup 0.0) | Cynthia's Roserade, Cynthia's Gabite, Cynthia's Garchomp ex |

