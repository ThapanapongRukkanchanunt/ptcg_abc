import unittest
from dataclasses import dataclass

from ptcg_abc.agent import RuleBasedAgent, select_option_indices


@dataclass
class FakeOption:
    type: int
    number: int | None = None
    area: int | None = None
    playerIndex: int | None = None


@dataclass
class FakeSelect:
    type: int
    context: int
    minCount: int
    maxCount: int
    option: list[FakeOption]


@dataclass
class FakeCurrent:
    yourIndex: int


@dataclass
class FakeObservation:
    select: FakeSelect | None
    current: FakeCurrent | None = None


@dataclass
class FakeCardData:
    cardId: int
    name: str
    cardType: int
    retreatCost: int = 0
    hp: int = 0
    weakness: int | None = None
    resistance: int | None = None
    energyType: int = 0
    basic: bool = False
    stage1: bool = False
    stage2: bool = False
    ex: bool = False
    megaEx: bool = False
    evolvesFrom: str | None = None
    skills: list | None = None
    attacks: list[int] | None = None


@dataclass
class FakeAttack:
    attackId: int
    damage: int
    energies: list[int]
    name: str = ""
    text: str = ""


@dataclass
class FakePokemon:
    id: int
    hp: int
    maxHp: int
    energies: list[int]
    energyCards: list
    tools: list


@dataclass
class FakePlayerState:
    active: list
    bench: list
    hand: list
    discard: list
    prize: list
    deckCount: int = 40


@dataclass
class FakeFullCurrent:
    yourIndex: int
    players: list[FakePlayerState]
    energyAttached: bool = False
    supporterPlayed: bool = False
    stadium: list | None = None


@dataclass
class FakeHandCard:
    id: int


class RuleBasedAgentTests(unittest.TestCase):
    def test_agent_returns_deck_for_initial_selection(self):
        deck = list(range(60))
        agent = RuleBasedAgent(deck)

        self.assertEqual(agent.act(FakeObservation(select=None)), deck)

    def test_main_selection_sequences_before_attack_and_end(self):
        select = FakeSelect(
            type=0,
            context=0,
            minCount=1,
            maxCount=1,
            option=[FakeOption(14), FakeOption(13), FakeOption(8)],
        )

        self.assertEqual(select_option_indices(select), [2])

    def test_count_selection_prefers_largest_number(self):
        select = FakeSelect(
            type=8,
            context=38,
            minCount=1,
            maxCount=1,
            option=[FakeOption(0, number=1), FakeOption(0, number=3), FakeOption(0, number=2)],
        )

        self.assertEqual(select_option_indices(select), [1])

    def test_discard_context_uses_minimum_allowed_count(self):
        select = FakeSelect(
            type=1,
            context=8,
            minCount=0,
            maxCount=2,
            option=[FakeOption(3), FakeOption(3)],
        )

        self.assertEqual(select_option_indices(select), [])

    def test_damage_context_prefers_opponent_target(self):
        select = FakeSelect(
            type=1,
            context=15,
            minCount=1,
            maxCount=1,
            option=[FakeOption(3, playerIndex=0), FakeOption(3, playerIndex=1)],
        )

        self.assertEqual(select_option_indices(select, current=FakeCurrent(yourIndex=0)), [1])

    def test_yes_no_prefers_going_second(self):
        select = FakeSelect(
            type=9,
            context=41,
            minCount=1,
            maxCount=1,
            option=[FakeOption(1), FakeOption(2)],
        )

        self.assertEqual(select_option_indices(select), [1])

    def test_metadata_scoring_prefers_ko_attack(self):
        select = FakeSelect(
            type=0,
            context=0,
            minCount=1,
            maxCount=1,
            option=[FakeOption(14), FakeOption(13)],
        )
        select.option[1].attackId = 99
        current = FakeFullCurrent(
            yourIndex=0,
            players=[
                FakePlayerState(
                    active=[FakePokemon(1, 100, 100, [1], [], [])],
                    bench=[],
                    hand=[],
                    discard=[],
                    prize=[1, 2, 3],
                ),
                FakePlayerState(
                    active=[FakePokemon(2, 50, 50, [], [], [])],
                    bench=[],
                    hand=[],
                    discard=[],
                    prize=[1, 2, 3],
                ),
            ],
            stadium=[],
        )
        card_by_id = {
            1: FakeCardData(cardId=1, name="Attacker", cardType=0, hp=100, attacks=[99]),
            2: FakeCardData(cardId=2, name="Target ex", cardType=0, hp=50, ex=True),
        }
        attack_by_id = {99: FakeAttack(attackId=99, damage=60, energies=[1])}

        self.assertEqual(
            select_option_indices(
                select,
                current=current,
                card_by_id=card_by_id,
                attack_by_id=attack_by_id,
            ),
            [1],
        )

    def test_prevention_rules(self):
        deck = [1] * 20 + [2] * 20 + [3] * 20
        card_data = [
            FakeCardData(cardId=1, name="Pikachu ex", cardType=0, basic=True, ex=True),
            FakeCardData(cardId=2, name="Charizard ex", cardType=0, stage2=True, ex=True),
            FakeCardData(cardId=3, name="Pidgeot", cardType=0, stage2=True, skills=[FakeOption(10)]),
        ]
        agent = RuleBasedAgent(deck, card_data=card_data)
        
        select = FakeSelect(
            type=0, context=0, minCount=1, maxCount=1,
            option=[FakeOption(13)] # ATTACK
        )
        select.option[0].attackId = 99
        
        current = FakeFullCurrent(
            yourIndex=0,
            players=[
                FakePlayerState(active=[FakePokemon(1, 100, 100, [1], [], [])], bench=[], hand=[], discard=[], prize=[1]),
                FakePlayerState(active=[FakePokemon(83, 100, 100, [], [], [])], bench=[], hand=[], discard=[], prize=[1]),
            ]
        )
        card_by_id = {
            1: FakeCardData(cardId=1, name="Pikachu ex", cardType=0, basic=True, ex=True, attacks=[99]),
            83: FakeCardData(cardId=83, name="Farigiraf ex", cardType=0, basic=True, ex=True),
            2: FakeCardData(cardId=2, name="Charizard ex", cardType=0, stage2=True, ex=True),
            3: FakeCardData(cardId=3, name="Pidgeot", cardType=0, stage2=True, skills=[FakeOption(10)]),
            330: FakeCardData(cardId=330, name="Sylveon", cardType=0, basic=True),
            117: FakeCardData(cardId=117, name="Cornerstone Mask Ogerpon ex", cardType=0, basic=True, ex=True, skills=[FakeOption(10)]),
        }
        attack_by_id = {99: FakeAttack(attackId=99, damage=50, energies=[1])}
        
        from ptcg_abc.agent.rule_based import _make_features, _score_option
        features = _make_features(select, current, card_by_id, attack_by_id)
        
        score = _score_option(0, select.option[0], select, features)
        self.assertEqual(score, 1.0)
        
        evolve_opt = FakeOption(9, area=2) # EVOLVE
        evolve_opt.index = 0
        evolve_opt.inPlayArea = 5
        evolve_opt.inPlayIndex = 0
        select_evolve = FakeSelect(type=7, context=37, minCount=1, maxCount=1, option=[evolve_opt])
        current_evolve = FakeFullCurrent(
            yourIndex=0,
            players=[
                FakePlayerState(active=[], bench=[FakePokemon(1, 100, 100, [], [], [])], hand=[FakePokemon(2, 120, 120, [], [], [])], discard=[], prize=[1]),
                FakePlayerState(active=[FakePokemon(83, 100, 100, [], [], [])], bench=[], hand=[], discard=[], prize=[1]),
            ]
        )
        features_evolve = _make_features(select_evolve, current_evolve, card_by_id, attack_by_id)
        score_evolve = _score_option(0, evolve_opt, select_evolve, features_evolve)
        self.assertGreater(score_evolve, 1.0)
        
        current_sylveon = FakeFullCurrent(
            yourIndex=0,
            players=[
                FakePlayerState(active=[FakePokemon(2, 100, 100, [1], [], [])], bench=[], hand=[], discard=[], prize=[1]),
                FakePlayerState(active=[FakePokemon(330, 100, 100, [], [], [])], bench=[], hand=[], discard=[], prize=[1]),
            ]
        )
        features_sylveon = _make_features(select, current_sylveon, card_by_id, attack_by_id)
        score_sylveon = _score_option(0, select.option[0], select, features_sylveon)
        self.assertEqual(score_sylveon, 1.0)
        
        current_ogerpon = FakeFullCurrent(
            yourIndex=0,
            players=[
                FakePlayerState(active=[FakePokemon(3, 100, 100, [1], [], [])], bench=[], hand=[], discard=[], prize=[1]),
                FakePlayerState(active=[FakePokemon(117, 100, 100, [], [], [])], bench=[], hand=[], discard=[], prize=[1]),
            ]
        )
        select_ability = FakeSelect(type=0, context=0, minCount=1, maxCount=1, option=[FakeOption(13)])
        select_ability.option[0].attackId = 99
        features_ogerpon = _make_features(select_ability, current_ogerpon, card_by_id, attack_by_id)
        score_ogerpon = _score_option(0, select_ability.option[0], select_ability, features_ogerpon)
        self.assertEqual(score_ogerpon, 1.0)

    def test_get_missing_energies(self):
        from ptcg_abc.agent.rule_based import get_missing_energies
        self.assertEqual(get_missing_energies([2], [2, 5]), [5])
        self.assertEqual(get_missing_energies([2, 2], [2, 5]), [5])
        self.assertEqual(get_missing_energies([2, 5], [2, 5]), [])
        self.assertEqual(get_missing_energies([2, 10], [2, 5]), [])
        self.assertEqual(get_missing_energies([2], [0]), [])
        self.assertEqual(get_missing_energies([], [0]), [0])
        self.assertEqual(get_missing_energies([2, 5], [2, 5, 0]), [0])
        self.assertEqual(get_missing_energies([2, 5, 1], [2, 5, 0]), [])

    def test_color_aware_energy_attaching(self):
        from ptcg_abc.agent.rule_based import _make_features, _score_option
        select = FakeSelect(
            type=0, context=0, minCount=1, maxCount=1,
            option=[
                FakeOption(8, area=2), # ATTACH Fire Energy to active
                FakeOption(8, area=2), # ATTACH Psychic Energy to active
            ]
        )
        select.option[0].index = 0
        select.option[0].inPlayArea = 4
        select.option[0].inPlayIndex = 0
        select.option[1].index = 1
        select.option[1].inPlayArea = 4
        select.option[1].inPlayIndex = 0
        
        current = FakeFullCurrent(
            yourIndex=0,
            players=[
                FakePlayerState(
                    active=[FakePokemon(1, 100, 100, [2], [], [])],
                    bench=[],
                    hand=[
                        FakeHandCard(101),
                        FakeHandCard(102),
                    ],
                    discard=[],
                    prize=[1],
                ),
                FakePlayerState(active=[FakePokemon(2, 100, 100, [], [], [])], bench=[], hand=[], discard=[], prize=[1]),
            ]
        )
        card_by_id = {
            1: FakeCardData(cardId=1, name="Dragapult ex", cardType=0, hp=100, attacks=[99]),
            2: FakeCardData(cardId=2, name="Target", cardType=0, hp=100),
            101: FakeCardData(cardId=101, name="Basic {R} Energy", cardType=5, energyType=2),
            102: FakeCardData(cardId=102, name="Basic {P} Energy", cardType=5, energyType=5),
        }
        attack_by_id = {99: FakeAttack(attackId=99, damage=100, energies=[2, 5])}
        
        features = _make_features(select, current, card_by_id, attack_by_id)
        
        score_fire = _score_option(0, select.option[0], select, features)
        self.assertEqual(score_fire, 100.0)
        
        score_psychic = _score_option(1, select.option[1], select, features)
        self.assertGreater(score_psychic, 100.0)

    def test_ability_priority_over_evolve(self):
        select = FakeSelect(
            type=0,  # MAIN
            context=0,
            minCount=1,
            maxCount=1,
            option=[
                FakeOption(9),  # EVOLVE
                FakeOption(10), # ABILITY
            ]
        )
        # Select options: since ABILITY is now priority 10 (base 9990) and EVOLVE is 20 (base 9980)
        # select_option_indices should rank index 1 (ABILITY) higher than index 0 (EVOLVE)
        indices = select_option_indices(select)
        self.assertEqual(indices, [1])

    def test_energy_acceleration_planning(self):
        from ptcg_abc.agent.rule_based import _make_features
        select = FakeSelect(
            type=0, context=0, minCount=1, maxCount=1,
            option=[FakeOption(14)] # END
        )
        
        # Scenario A: Dragapult ex has 0 energy, and Crispin is in hand.
        # We need Fire (2) and Psychic (5).
        # Dynamic attachments allows: 1 manual + 1 Crispin = 2.
        # This matches Fire (2) + Psychic (5) exactly because Crispin can search virtual types.
        current_with_crispin = FakeFullCurrent(
            yourIndex=0,
            players=[
                FakePlayerState(
                    active=[FakePokemon(1, 100, 100, [], [], [])], # 0 energy
                    bench=[],
                    hand=[
                        FakeHandCard(1198), # Crispin (ID 1198)
                    ],
                    discard=[],
                    prize=[1],
                ),
                FakePlayerState(active=[FakePokemon(2, 100, 100, [], [], [])], bench=[], hand=[], discard=[], prize=[1]),
            ],
            energyAttached=False,
            supporterPlayed=False
        )
        
        card_by_id = {
            1: FakeCardData(cardId=1, name="Dragapult ex", cardType=0, hp=100, attacks=[99]),
            2: FakeCardData(cardId=2, name="Target", cardType=0, hp=100),
            1198: FakeCardData(cardId=1198, name="Crispin", cardType=3),
        }
        attack_by_id = {99: FakeAttack(attackId=99, damage=100, energies=[2, 5])}
        
        features_with_crispin = _make_features(select, current_with_crispin, card_by_id, attack_by_id)
        # The plan should successfully select Phantom Dive (attackId 99)
        self.assertEqual(features_with_crispin.plan.attack_id, 99)
        self.assertTrue(features_with_crispin.plan.needs_energy)
        
        # Scenario B: Dragapult ex has 0 energy, but Crispin is NOT in hand (only random trainer is in hand)
        # Dynamic attachments only allows 1 manual. Phantom Dive needs 2, so it's not feasible.
        current_without_crispin = FakeFullCurrent(
            yourIndex=0,
            players=[
                FakePlayerState(
                    active=[FakePokemon(1, 100, 100, [], [], [])],
                    bench=[],
                    hand=[
                        FakeHandCard(999), # Some random trainer card
                    ],
                    discard=[],
                    prize=[1],
                ),
                FakePlayerState(active=[FakePokemon(2, 100, 100, [], [], [])], bench=[], hand=[], discard=[], prize=[1]),
            ],
            energyAttached=False,
            supporterPlayed=False
        )
        card_by_id_no_crispin = {
            1: FakeCardData(cardId=1, name="Dragapult ex", cardType=0, hp=100, attacks=[99]),
            2: FakeCardData(cardId=2, name="Target", cardType=0, hp=100),
            999: FakeCardData(cardId=999, name="Random Card", cardType=1),
        }
        
        features_no_crispin = _make_features(select, current_without_crispin, card_by_id_no_crispin, attack_by_id)
        # The plan should NOT choose Phantom Dive (attackId 99) because it's not powerable this turn
        self.assertEqual(features_no_crispin.plan.attack_id, -1)

    def test_active_retreat_cost_planning(self):
        from ptcg_abc.agent.rule_based import _make_features
        select = FakeSelect(
            type=0, context=0, minCount=1, maxCount=1,
            option=[FakeOption(14)] # END
        )
        # Active Dreepy (ID 119) with 0 energy attached, retreat cost 1.
        # Benched Dragapult ex (ID 1) with 0 energy, needs 1 energy (Psychic 5) to attack.
        # Hand has 1 Basic Psychic Energy card.
        # Total needed is 2, but we only have 1 allowed manual attachment this turn.
        # Dragapult ex should NOT be able to attack.
        current_1_energy = FakeFullCurrent(
            yourIndex=0,
            players=[
                FakePlayerState(
                    active=[FakePokemon(119, 60, 60, [], [], [])],
                    bench=[FakePokemon(1, 320, 320, [], [], [])],
                    hand=[
                        FakeHandCard(5), # Basic Psychic Energy
                    ],
                    discard=[],
                    prize=[1],
                ),
                FakePlayerState(active=[FakePokemon(2, 100, 100, [], [], [])], bench=[], hand=[], discard=[], prize=[1]),
            ],
            energyAttached=False,
            supporterPlayed=False
        )
        card_by_id = {
            1: FakeCardData(cardId=1, name="Dragapult ex", cardType=0, hp=320, attacks=[99]),
            119: FakeCardData(cardId=119, name="Dreepy", cardType=0, hp=60, retreatCost=1),
            2: FakeCardData(cardId=2, name="Target", cardType=0, hp=100),
            5: FakeCardData(cardId=5, name="Basic Psychic Energy", cardType=5, energyType=5),
        }
        attack_by_id = {99: FakeAttack(attackId=99, damage=100, energies=[5])}
        
        features_1_energy = _make_features(select, current_1_energy, card_by_id, attack_by_id)
        self.assertEqual(features_1_energy.plan.attack_id, -1)
        
        # Benched Dragapult ex already has 1 energy attached (needs 0 to attack).
        # Dreepy has 0 energy (needs 1 to retreat). Hand has 1 Basic Psychic Energy card.
        # Total needed is 1 attachment (for retreat cost), which is allowed (1 allowed).
        # Dragapult ex should be able to attack!
        current_ready = FakeFullCurrent(
            yourIndex=0,
            players=[
                FakePlayerState(
                    active=[FakePokemon(119, 60, 60, [], [], [])],
                    bench=[FakePokemon(1, 320, 320, [5], [], [])], # Dragapult has 1 attached Psychic
                    hand=[
                        FakeHandCard(5), # Basic Psychic Energy
                    ],
                    discard=[],
                    prize=[1],
                ),
                FakePlayerState(active=[FakePokemon(2, 100, 100, [], [], [])], bench=[], hand=[], discard=[], prize=[1]),
            ],
            energyAttached=False,
            supporterPlayed=False
        )
        features_ready = _make_features(select, current_ready, card_by_id, attack_by_id)
        self.assertEqual(features_ready.plan.attack_id, 99)
        self.assertTrue(features_ready.plan.needs_energy)

    def test_ability_target_restrictions(self):
        from ptcg_abc.agent.rule_based import _get_max_attachments_for_pokemon
        # Fake features
        option = FakeOption(type=10, number=1, area=4) # ABILITY, active ogerpon
        option.index = 0
        select = FakeSelect(
            type=0, context=0, minCount=1, maxCount=1,
            option=[option]
        )
        current = FakeFullCurrent(
            yourIndex=0,
            players=[
                FakePlayerState(
                    active=[FakePokemon(96, 210, 210, [], [], [])],
                    bench=[FakePokemon(1, 320, 320, [], [], [])],
                    hand=[],
                    discard=[],
                    prize=[1],
                ),
                FakePlayerState(active=[], bench=[], hand=[], discard=[], prize=[]),
            ],
            energyAttached=False,
            supporterPlayed=False
        )
        # Ogerpon ex has Teal Dance ability: "Once during your turn, you may attach a Basic Grass Energy card from your hand to this Pokémon..."
        card_by_id = {
            96: FakeCardData(
                cardId=96, name="Teal Mask Ogerpon ex", cardType=0, hp=210,
                skills=[{"name": "Teal Dance", "text": "Attach a Basic Grass Energy card from your hand to this Pokémon."}]
            ),
            1: FakeCardData(cardId=1, name="Dragapult ex", cardType=0, hp=320),
        }
        
        # Create features manually
        from ptcg_abc.agent.rule_based import _make_features
        features = _make_features(select, current, card_by_id, {})
        
        # Active is Ogerpon ex (ID 96). Benched is Dragapult ex (ID 1).
        active_pokemon = current.players[0].active[0]
        bench_pokemon = current.players[0].bench[0]
        
        # Evaluating Ogerpon ex (ability-user itself): Teal Dance ability should be allowed!
        allowed_self, _ = _get_max_attachments_for_pokemon(active_pokemon, features)
        self.assertEqual(allowed_self, 2) # 1 manual + 1 ability = 2
        
        # Evaluating Dragapult ex (non ability-user): Teal Dance should NOT be allowed!
        allowed_other, _ = _get_max_attachments_for_pokemon(bench_pokemon, features)
        self.assertEqual(allowed_other, 1) # 1 manual only

    def test_prize_map_prefers_boss_two_prize_route(self):
        from ptcg_abc.agent.rule_based import _make_features

        boss_option = FakeOption(7, area=2)
        boss_option.index = 0
        select = FakeSelect(type=0, context=0, minCount=1, maxCount=1, option=[boss_option])
        current = FakeFullCurrent(
            yourIndex=0,
            players=[
                FakePlayerState(
                    active=[FakePokemon(1, 100, 100, [1], [], [])],
                    bench=[],
                    hand=[FakeHandCard(1000)],
                    discard=[],
                    prize=[1, 2],
                ),
                FakePlayerState(
                    active=[FakePokemon(2, 100, 100, [], [], [])],
                    bench=[FakePokemon(3, 100, 100, [], [], [])],
                    hand=[],
                    discard=[],
                    prize=[1, 2],
                ),
            ],
            stadium=[],
        )
        card_by_id = {
            1: FakeCardData(cardId=1, name="Flexible Attacker ex", cardType=0, hp=100, basic=True, ex=True, attacks=[99]),
            2: FakeCardData(cardId=2, name="Single Prize Target", cardType=0, hp=100),
            3: FakeCardData(cardId=3, name="Bench ex", cardType=0, hp=100, ex=True),
            1000: FakeCardData(cardId=1000, name="Boss's Orders", cardType=3),
        }
        attack_by_id = {99: FakeAttack(attackId=99, damage=120, energies=[1])}

        features = _make_features(select, current, card_by_id, attack_by_id, deck_ids=[1] * 60)

        self.assertEqual(features.prize_map.attack_count, 1)
        self.assertEqual(features.prize_map.total_prizes, 2)
        self.assertEqual(features.prize_map.steps[0].target_area, "BENCH")
        self.assertEqual(features.plan.target_area, "BENCH")
        self.assertIn(1, features.key_attackers)

    def test_boss_play_is_boosted_when_prize_map_needs_bench_target(self):
        from ptcg_abc.agent.rule_based import _make_features, _score_option

        boss_option = FakeOption(7, area=2)
        boss_option.index = 0
        draw_option = FakeOption(7, area=2)
        draw_option.index = 1
        select = FakeSelect(type=0, context=0, minCount=1, maxCount=1, option=[boss_option, draw_option])
        current = FakeFullCurrent(
            yourIndex=0,
            players=[
                FakePlayerState(
                    active=[FakePokemon(1, 100, 100, [1], [], [])],
                    bench=[],
                    hand=[FakeHandCard(1000), FakeHandCard(1001)],
                    discard=[],
                    prize=[1, 2],
                ),
                FakePlayerState(
                    active=[FakePokemon(2, 100, 100, [], [], [])],
                    bench=[FakePokemon(3, 100, 100, [], [], [])],
                    hand=[],
                    discard=[],
                    prize=[1, 2],
                ),
            ],
            stadium=[],
        )
        card_by_id = {
            1: FakeCardData(cardId=1, name="Flexible Attacker ex", cardType=0, hp=100, basic=True, ex=True, attacks=[99]),
            2: FakeCardData(cardId=2, name="Single Prize Target", cardType=0, hp=100),
            3: FakeCardData(cardId=3, name="Bench ex", cardType=0, hp=100, ex=True),
            1000: FakeCardData(cardId=1000, name="Boss's Orders", cardType=3),
            1001: FakeCardData(cardId=1001, name="Draw Supporter", cardType=3),
        }
        attack_by_id = {99: FakeAttack(attackId=99, damage=120, energies=[1])}

        features = _make_features(select, current, card_by_id, attack_by_id, deck_ids=[1] * 60)

        self.assertGreater(
            _score_option(0, boss_option, select, features),
            _score_option(1, draw_option, select, features),
        )

    def test_key_attackers_are_identified_from_deck_metadata_for_search(self):
        select = FakeSelect(
            type=1,
            context=7,
            minCount=1,
            maxCount=1,
            option=[
                FakeOption(3, area=1),
                FakeOption(3, area=1),
            ],
        )
        select.option[0].index = 0
        select.option[1].index = 1
        select.deck = [FakeHandCard(10), FakeHandCard(20)]
        current = FakeFullCurrent(
            yourIndex=0,
            players=[
                FakePlayerState(active=[], bench=[], hand=[], discard=[], prize=[1, 2, 3]),
                FakePlayerState(
                    active=[FakePokemon(30, 180, 180, [], [], [])],
                    bench=[],
                    hand=[],
                    discard=[],
                    prize=[1, 2, 3],
                ),
            ],
            stadium=[],
        )
        card_by_id = {
            10: FakeCardData(cardId=10, name="Low Damage Basic", cardType=0, hp=70, basic=True, attacks=[10]),
            20: FakeCardData(cardId=20, name="Heavy Hitter ex", cardType=0, hp=220, basic=True, ex=True, attacks=[20]),
            30: FakeCardData(cardId=30, name="Opponent ex", cardType=0, hp=180, ex=True),
        }
        attack_by_id = {
            10: FakeAttack(attackId=10, damage=40, energies=[1]),
            20: FakeAttack(attackId=20, damage=200, energies=[1, 1]),
        }

        self.assertEqual(
            select_option_indices(
                select,
                current=current,
                card_by_id=card_by_id,
                attack_by_id=attack_by_id,
                deck_ids=[10] * 30 + [20] * 30,
            ),
            [1],
        )

    def test_low_damage_basic_is_not_promoted_to_key_attacker(self):
        from ptcg_abc.agent.rule_based import _make_features

        select = FakeSelect(type=0, context=0, minCount=1, maxCount=1, option=[FakeOption(14)])
        current = FakeFullCurrent(
            yourIndex=0,
            players=[
                FakePlayerState(active=[FakePokemon(10, 70, 70, [1], [], [])], bench=[], hand=[], discard=[], prize=[1, 2, 3]),
                FakePlayerState(active=[FakePokemon(30, 180, 180, [], [], [])], bench=[], hand=[], discard=[], prize=[1, 2, 3]),
            ],
            stadium=[],
        )
        card_by_id = {
            10: FakeCardData(cardId=10, name="Setup Basic", cardType=0, hp=70, basic=True, attacks=[10]),
            20: FakeCardData(cardId=20, name="Real Attacker ex", cardType=0, hp=220, basic=True, ex=True, attacks=[20]),
            30: FakeCardData(cardId=30, name="Opponent ex", cardType=0, hp=180, ex=True),
        }
        attack_by_id = {
            10: FakeAttack(attackId=10, damage=30, energies=[1]),
            20: FakeAttack(attackId=20, damage=200, energies=[1, 1]),
        }

        features = _make_features(select, current, card_by_id, attack_by_id, deck_ids=[10] * 30 + [20] * 30)

        self.assertNotIn(10, features.key_attackers)
        self.assertIn(20, features.key_attackers)

    def test_low_damage_basic_prize_route_does_not_override_plan(self):
        from ptcg_abc.agent.rule_based import _make_features, _route_is_actionable

        attack_option = FakeOption(13)
        attack_option.attackId = 10
        select = FakeSelect(type=0, context=0, minCount=1, maxCount=1, option=[attack_option])
        current = FakeFullCurrent(
            yourIndex=0,
            players=[
                FakePlayerState(active=[FakePokemon(10, 70, 70, [1], [], [])], bench=[], hand=[], discard=[], prize=[1]),
                FakePlayerState(active=[FakePokemon(30, 30, 30, [], [], [])], bench=[], hand=[], discard=[], prize=[1]),
            ],
            stadium=[],
        )
        card_by_id = {
            10: FakeCardData(cardId=10, name="Setup Basic", cardType=0, hp=70, basic=True, attacks=[10]),
            30: FakeCardData(cardId=30, name="Small Target", cardType=0, hp=30),
        }
        attack_by_id = {10: FakeAttack(attackId=10, damage=30, energies=[1])}

        features = _make_features(select, current, card_by_id, attack_by_id, deck_ids=[10] * 60)

        self.assertEqual(features.prize_map.total_prizes, 1)
        self.assertFalse(_route_is_actionable(features))
        self.assertNotIn(10, features.key_attackers)

    def test_phantom_dive_style_counters_count_multi_target_prizes(self):
        from ptcg_abc.agent.rule_based import _make_features

        attack_option = FakeOption(13)
        attack_option.attackId = 154
        select = FakeSelect(type=0, context=0, minCount=1, maxCount=1, option=[attack_option])
        current = FakeFullCurrent(
            yourIndex=0,
            players=[
                FakePlayerState(
                    active=[FakePokemon(121, 320, 320, [2, 5], [], [])],
                    bench=[],
                    hand=[],
                    discard=[],
                    prize=[1, 2, 3, 4],
                ),
                FakePlayerState(
                    active=[FakePokemon(40, 200, 200, [], [], [])],
                    bench=[
                        FakePokemon(41, 30, 30, [], [], []),
                        FakePokemon(42, 30, 30, [], [], []),
                    ],
                    hand=[],
                    discard=[],
                    prize=[1, 2, 3, 4],
                ),
            ],
            stadium=[],
        )
        card_by_id = {
            121: FakeCardData(cardId=121, name="Dragapult ex", cardType=0, hp=320, stage2=True, ex=True, attacks=[154]),
            40: FakeCardData(cardId=40, name="Active ex", cardType=0, hp=200, ex=True),
            41: FakeCardData(cardId=41, name="Bench A", cardType=0, hp=30),
            42: FakeCardData(cardId=42, name="Bench B", cardType=0, hp=30),
        }
        attack_by_id = {
            154: FakeAttack(
                attackId=154,
                damage=200,
                energies=[2, 5],
                name="Phantom Dive",
                text="Put 6 damage counters on your opponent's Benched Pokemon in any way you like.",
            )
        }

        features = _make_features(select, current, card_by_id, attack_by_id, deck_ids=[121] * 60)

        self.assertEqual(features.prize_map.attack_count, 1)
        self.assertEqual(features.prize_map.total_prizes, 4)
        self.assertEqual(features.prize_map.steps[0].prizes_taken, 4)
        self.assertEqual(features.prize_map.steps[0].target_damages[("ACTIVE", 0)], 200)
        self.assertEqual(features.prize_map.steps[0].target_damages[("BENCH", 0)], 30)
        self.assertEqual(features.prize_map.steps[0].target_damages[("BENCH", 1)], 30)

    def test_search_prefers_key_evolution_chain_basic(self):
        select = FakeSelect(
            type=1,
            context=7,
            minCount=1,
            maxCount=1,
            option=[
                FakeOption(3, area=1),
                FakeOption(3, area=1),
            ],
        )
        select.option[0].index = 0
        select.option[1].index = 1
        select.deck = [FakeHandCard(10), FakeHandCard(40)]
        current = FakeFullCurrent(
            yourIndex=0,
            players=[
                FakePlayerState(active=[], bench=[], hand=[], discard=[], prize=[1, 2, 3, 4]),
                FakePlayerState(
                    active=[FakePokemon(50, 220, 220, [], [], [])],
                    bench=[],
                    hand=[],
                    discard=[],
                    prize=[1, 2, 3, 4],
                ),
            ],
            stadium=[],
        )
        card_by_id = {
            10: FakeCardData(cardId=10, name="Setup Basic", cardType=0, hp=70, basic=True, attacks=[10]),
            20: FakeCardData(cardId=20, name="Middle Link", cardType=0, hp=90, stage1=True, evolvesFrom="Setup Basic", attacks=[20]),
            30: FakeCardData(cardId=30, name="Stage 2 Attacker ex", cardType=0, hp=320, stage2=True, ex=True, evolvesFrom="Middle Link", attacks=[30]),
            40: FakeCardData(cardId=40, name="Support ex", cardType=0, hp=210, basic=True, ex=True, skills=[FakeOption(10)], attacks=[40]),
            50: FakeCardData(cardId=50, name="Opponent ex", cardType=0, hp=220, ex=True),
        }
        attack_by_id = {
            10: FakeAttack(attackId=10, damage=30, energies=[1]),
            20: FakeAttack(attackId=20, damage=70, energies=[1]),
            30: FakeAttack(attackId=30, damage=220, energies=[1, 1]),
            40: FakeAttack(attackId=40, damage=60, energies=[1, 1, 1]),
        }

        self.assertEqual(
            select_option_indices(
                select,
                current=current,
                card_by_id=card_by_id,
                attack_by_id=attack_by_id,
                deck_ids=[10] * 4 + [20] * 3 + [30] * 3 + [40],
            ),
            [0],
        )

    def test_opening_active_prefers_key_basic_over_support_ex(self):
        select = FakeSelect(
            type=1,
            context=1,
            minCount=1,
            maxCount=1,
            option=[
                FakeOption(3, area=2, playerIndex=0),
                FakeOption(3, area=2, playerIndex=0),
            ],
        )
        select.option[0].index = 0
        select.option[1].index = 1
        current = FakeFullCurrent(
            yourIndex=0,
            players=[
                FakePlayerState(
                    active=[],
                    bench=[],
                    hand=[FakeHandCard(40), FakeHandCard(10)],
                    discard=[],
                    prize=[1, 2, 3, 4],
                ),
                FakePlayerState(
                    active=[FakePokemon(50, 220, 220, [], [], [])],
                    bench=[],
                    hand=[],
                    discard=[],
                    prize=[1, 2, 3, 4],
                ),
            ],
            stadium=[],
        )
        card_by_id = {
            10: FakeCardData(cardId=10, name="Setup Basic", cardType=0, hp=70, basic=True, attacks=[10]),
            20: FakeCardData(cardId=20, name="Middle Link", cardType=0, hp=90, stage1=True, evolvesFrom="Setup Basic", attacks=[20]),
            30: FakeCardData(cardId=30, name="Stage 2 Attacker ex", cardType=0, hp=320, stage2=True, ex=True, evolvesFrom="Middle Link", attacks=[30]),
            40: FakeCardData(cardId=40, name="Support ex", cardType=0, hp=210, basic=True, ex=True, skills=[FakeOption(10)], attacks=[40]),
            50: FakeCardData(cardId=50, name="Opponent ex", cardType=0, hp=220, ex=True),
        }
        attack_by_id = {
            10: FakeAttack(attackId=10, damage=30, energies=[1]),
            20: FakeAttack(attackId=20, damage=70, energies=[1]),
            30: FakeAttack(attackId=30, damage=220, energies=[1, 1]),
            40: FakeAttack(attackId=40, damage=60, energies=[1, 1, 1]),
        }

        self.assertEqual(
            select_option_indices(
                select,
                current=current,
                card_by_id=card_by_id,
                attack_by_id=attack_by_id,
                deck_ids=[10] * 4 + [20] * 3 + [30] * 3 + [40],
            ),
            [1],
        )

    def test_repeat_attach_ability_can_power_iono_style_attack(self):
        from ptcg_abc.agent.rule_based import _make_features

        ability_option = FakeOption(10, area=4)
        ability_option.index = 0
        select = FakeSelect(type=0, context=0, minCount=1, maxCount=1, option=[ability_option])
        current = FakeFullCurrent(
            yourIndex=0,
            players=[
                FakePlayerState(
                    active=[FakePokemon(269, 280, 280, [], [], [])],
                    bench=[],
                    hand=[FakeHandCard(4), FakeHandCard(4), FakeHandCard(4), FakeHandCard(4)],
                    discard=[],
                    prize=[1, 2, 3],
                ),
                FakePlayerState(
                    active=[FakePokemon(50, 230, 230, [], [], [])],
                    bench=[],
                    hand=[],
                    discard=[],
                    prize=[1, 2, 3],
                ),
            ],
            energyAttached=False,
            supporterPlayed=False,
            stadium=[],
        )
        card_by_id = {
            4: FakeCardData(cardId=4, name="Basic Lightning Energy", cardType=5, energyType=4),
            269: FakeCardData(
                cardId=269,
                name="Iono's Bellibolt ex",
                cardType=0,
                hp=280,
                stage1=True,
                ex=True,
                evolvesFrom="Iono's Tadbulb",
                skills=[
                    {
                        "name": "Electric Streamer",
                        "text": "As often as you like during your turn, you may attach a Basic {L} Energy card from your hand to 1 of your Iono's Pokemon.",
                    }
                ],
                attacks=[368],
            ),
            50: FakeCardData(cardId=50, name="Opponent ex", cardType=0, hp=230, ex=True),
        }
        attack_by_id = {
            368: FakeAttack(attackId=368, damage=230, energies=[4, 4, 4, 0])
        }

        features = _make_features(select, current, card_by_id, attack_by_id, deck_ids=[269] * 4 + [4] * 20)

        self.assertEqual(features.plan.attack_id, 368)
        self.assertTrue(features.plan.needs_energy)


if __name__ == "__main__":
    unittest.main()
