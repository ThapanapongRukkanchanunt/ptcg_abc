# Damage-Prevention Pokemon

Date: 2026-06-18

## Method

Scanned Kaggle's local simulator metadata through `cg.api.all_card_data()` and
`cg.api.all_attack()`.

Included:

- Pokemon with an ability text that prevents incoming damage.
- Pokemon with an attack text that prevents incoming damage on a later turn or under a
  condition.

Excluded:

- Text where the user's own outgoing attack does no damage.
- Text that checks whether a Pokemon has no damage counters but does not prevent damage.

## Ability Or Attack Damage Prevention

| Card ID | Pokemon | Source | Name | Damage-prevention condition |
| ---: | --- | --- | --- | --- |
| 28 | Poltchageist | Ability | Storehouse Hideaway | While on Bench, prevents damage and effects from opponent attacks to itself. |
| 65 | Dunsparce | Attack | Dig | Coin-head protection next opponent turn; prevents damage and effects to itself. |
| 74 | Rabsca | Ability | Spherical Shield | Protects your Benched Pokemon from opponent attack damage and effects. |
| 83 | Farigiraf ex | Ability | Armor Tail | Prevents damage to itself from opponent Basic Pokemon ex attacks. |
| 117 | Cornerstone Mask Ogerpon ex | Ability | Cornerstone Stance | Prevents attack damage from opposing Pokemon with an Ability. |
| 158 | Drednaw | Ability | Impervious Shell | Prevents attack damage to itself when the incoming damage is 200 or more. |
| 176 | Terapagos ex | Attack | Crown Opal | Next opponent turn, prevents damage from Basic non-Colorless Pokemon attacks. |
| 185 | Flittle | Attack | Splashing Dodge | Coin-head protection next opponent turn; prevents damage and effects to itself. |
| 194 | Altaria | Attack | Cotton Wings | Coin-head protection next opponent turn; prevents attack damage to itself. |
| 207 | Milotic ex | Ability | Sparkling Scales | Prevents damage and effects from opponent Tera Pokemon attacks. |
| 253 | Metapod | Attack | Harden | Next opponent turn, prevents attack damage of 60 or less. |
| 330 | Sylveon | Ability | Safeguard | Prevents damage to itself from opponent Pokemon ex attacks. |
| 343 | Shaymin | Ability | Flower Curtain | Prevents opponent attack damage to your Benched non-Rule-Box Pokemon. |
| 345 | Crustle | Ability | Mysterious Rock Inn | Prevents damage to itself from opponent Pokemon ex attacks. |
| 362 | Misty's Magikarp | Ability | So Submerged | While on Bench, prevents damage and effects from opponent attacks to itself. |
| 365 | Cynthia's Feebas | Attack | Undulate | Coin-head protection next opponent turn; prevents damage and effects to itself. |
| 422 | Barraskewda | Attack | Dive | Coin-head protection next opponent turn; prevents damage and effects to itself. |
| 484 | Petilil | Attack | Hide | Coin-head protection next opponent turn; prevents damage and effects to itself. |
| 504 | Carracosta | Ability | Mighty Shell | Prevents damage and effects from opposing Pokemon with Special Energy attached. |
| 552 | Tranquill | Attack | Fly | Coin-head protection next opponent turn; prevents damage and effects to itself. |
| 553 | Unfezant | Attack | Swift Flight | Coin-head protection next opponent turn; prevents damage and effects to itself. |
| 599 | Roggenrola | Attack | Harden | Next opponent turn, prevents attack damage of 40 or less. |
| 681 | Marshadow | Attack | Shadowy Side Kick | If this attack knocks out an opposing Pokemon, protects itself next opponent turn. |
| 729 | Snom | Attack | Hide | Coin-head protection next opponent turn; prevents damage and effects to itself. |
| 737 | Mega Manectric ex | Attack | Flash Ray | Next opponent turn, prevents attack damage from Basic Pokemon. |
| 836 | Bronzor | Attack | Iron Defense | Coin-head protection next opponent turn; prevents attack damage to itself. |
| 840 | Archaludon | Attack | Coated Attack | Next opponent turn, prevents attack damage from Basic Pokemon. |
| 878 | Hop's Phantump | Attack | Splashing Dodge | Coin-head protection next opponent turn; prevents damage and effects to itself. |
| 908 | Noivern | Attack | Agility | Coin-head protection next opponent turn; prevents damage and effects to itself. |
| 921 | Dipplin | Attack | Coated Attack | Next opponent turn, prevents attack damage from Basic Pokemon. |
| 961 | Marill | Attack | Hide | Coin-head protection next opponent turn; prevents damage and effects to itself. |
| 970 | Fezandipiti | Ability | Adrena-Pheromone | With Darkness Energy attached, coin-head prevents incoming attack damage. |
| 979 | Koraidon ex | Attack | Tera | While on Bench, prevents attack damage to itself from both players. |
| 1018 | Spewpa | Attack | Hide | Coin-head protection next opponent turn; prevents damage and effects to itself. |

## Tera Bench-Damage Watchlist

Kaggle metadata also marks these Pokemon with `tera=True`. Treat them as a separate
watchlist for bench-damage rules, even when the damage-prevention text is not represented
as an ability or attack entry.

| Card ID | Pokemon |
| ---: | --- |
| 30 | Magcargo ex |
| 40 | Greninja ex |
| 52 | Wugtrio ex |
| 83 | Farigiraf ex |
| 96 | Teal Mask Ogerpon ex |
| 99 | Hearthflame Mask Ogerpon ex |
| 108 | Wellspring Mask Ogerpon ex |
| 117 | Cornerstone Mask Ogerpon ex |
| 121 | Dragapult ex |
| 130 | Revavroom ex |
| 153 | Cinderace ex |
| 154 | Lapras ex |
| 161 | Galvantula ex |
| 176 | Terapagos ex |
| 189 | Flygon ex |
| 193 | Alolan Exeggutor ex |
| 210 | Pikachu ex |
| 223 | PalossandEX |
| 229 | Hydreigon ex |
| 231 | Tatsugiri ex |
| 236 | Leafeon ex |
| 239 | Flareon ex |
| 241 | Vaporeon ex |
| 243 | Glaceon ex |
| 244 | Jolteon ex |
| 246 | Espeon ex |
| 248 | Umbreon ex |
| 249 | Eevee ex |
| 316 | Sylveon ex |
| 320 | Ceruledge ex |
| 957 | Miraidon ex |
| 979 | Koraidon ex |

## Future Rule Hooks

- Avoid choosing prevented attack targets when the prevention condition is active.
- Prefer non-ex or non-Basic attackers when facing Safeguard-style effects.
- Prefer Boss or switch effects to route around protected Bench Pokemon.
- For coin-based prevention attacks, track the attack log before assuming prevention is
  active.
- For Tera Bench protection, avoid bench-damage lines unless the target is Active or the
  effect places damage counters instead of doing attack damage.
