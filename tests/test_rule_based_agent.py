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


if __name__ == "__main__":
    unittest.main()
