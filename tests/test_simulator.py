import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ptcg_abc.simulator import run_battle


class SimulatorSeedTests(unittest.TestCase):
    def test_run_battle_seeds_engine_randomness_before_start(self):
        observed: list[float] = []
        cg_module = types.ModuleType("cg")
        cg_module.__path__ = []
        api_module = types.ModuleType("cg.api")
        api_module.to_observation_class = lambda value: value
        game_module = types.ModuleType("cg.game")

        def battle_start(deck0, deck1):
            observed.append(random.random())
            return None, SimpleNamespace(errorPlayer=0, errorType=0)

        game_module.battle_start = battle_start
        game_module.battle_select = lambda choice: None
        game_module.battle_finish = lambda: None
        agent = SimpleNamespace(act=lambda observation: [])

        with tempfile.TemporaryDirectory() as tmp:
            sample_dir = Path(tmp)
            (sample_dir / "cg").mkdir()
            (sample_dir / "cg" / "game.py").write_text("", encoding="utf-8")
            with patch.dict(
                sys.modules,
                {
                    "cg": cg_module,
                    "cg.api": api_module,
                    "cg.game": game_module,
                },
            ):
                result = run_battle(
                    [1] * 60,
                    [2] * 60,
                    sample_dir=sample_dir,
                    agent0=agent,
                    agent1=agent,
                    card_data=[],
                    attack_data=[],
                    seed=1234,
                )

        self.assertFalse(result.started)
        self.assertEqual(observed, [random.Random(1234).random()])
