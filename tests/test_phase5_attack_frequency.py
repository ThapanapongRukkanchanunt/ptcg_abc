import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.analysis.phase5_attack_frequency import summarize


class Phase5AttackFrequencyTests(unittest.TestCase):
    def test_counts_selected_attacks_by_name(self):
        rows = [
            _row(1, "rule_deck_01", chosen=[1], attacks=[(1, 10), (2, 11)]),
            _row(1, "rule_deck_01", chosen=[2], attacks=[(1, 10), (2, 11)]),
            _row(2, "rule_deck_02", chosen=[1], attacks=[(1, 10)]),
        ]
        attacks = [
            SimpleNamespace(attackId=10, name="Phantom Dive"),
            SimpleNamespace(attackId=11, name="Jet Headbutt"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with patch(
                "scripts.analysis.phase5_attack_frequency.load_engine_metadata",
                return_value=([], attacks),
            ):
                result = summarize(path, Path(tmp))

        self.assertEqual(result["games"], 2)
        self.assertEqual(result["total_attack_commands"], 3)
        self.assertEqual(result["phantom_dive_count"], 2)
        self.assertEqual(result["attacks"][0]["attack_name"], "Phantom Dive")
        self.assertEqual(result["attacks"][0]["opponent_counts"]["rule_deck_01"], 1)
        self.assertEqual(result["attacks"][0]["opponent_counts"]["rule_deck_02"], 1)


def _row(game: int, opponent: str, *, chosen: list[int], attacks: list[tuple[int, int]]):
    return {
        "chosen_indices": chosen,
        "decision": {
            "reward_metadata": {
                "game_index": game,
                "opponent_agent_key": opponent,
            },
            "legal_options": [
                {
                    "index": index,
                    "option_type": "ATTACK",
                    "attack_id": attack_id,
                }
                for index, attack_id in attacks
            ],
        },
    }


if __name__ == "__main__":
    unittest.main()
