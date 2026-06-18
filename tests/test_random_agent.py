import unittest
from dataclasses import dataclass

from ptcg_abc.agent import RandomAgent


@dataclass
class FakeOption:
    type: int


@dataclass
class FakeSelect:
    minCount: int
    maxCount: int
    option: list[FakeOption]


@dataclass
class FakeObservation:
    select: FakeSelect | None


class RandomAgentTests(unittest.TestCase):
    def test_returns_deck_for_initial_selection(self):
        deck = list(range(60))
        agent = RandomAgent(deck, seed=1)

        self.assertEqual(agent.act(FakeObservation(select=None)), deck)

    def test_random_selection_respects_bounds_and_uniqueness(self):
        agent = RandomAgent(list(range(60)), seed=1)
        obs = FakeObservation(
            select=FakeSelect(
                minCount=1,
                maxCount=3,
                option=[FakeOption(3), FakeOption(3), FakeOption(3), FakeOption(3)],
            )
        )

        choice = agent.act(obs)

        self.assertGreaterEqual(len(choice), 1)
        self.assertLessEqual(len(choice), 3)
        self.assertEqual(len(choice), len(set(choice)))
        self.assertTrue(all(0 <= index < 4 for index in choice))


if __name__ == "__main__":
    unittest.main()
