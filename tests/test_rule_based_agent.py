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


if __name__ == "__main__":
    unittest.main()
