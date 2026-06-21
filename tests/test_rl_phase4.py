import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from ptcg_abc.agent import HybridRlAgent
from ptcg_abc.cli import build_parser
from ptcg_abc.rl.featurizer import BOARD_IMAGE_HEIGHT, BOARD_IMAGE_WIDTH, make_decision_frame
from ptcg_abc.rl.model import LinearOptionModel, train_behavior_cloning_model
from ptcg_abc.rl.records import DecisionFrame
from ptcg_abc.rl.rewards import RewardConfig, reward_from_result_metadata
from ptcg_abc.rl.torch_backend import (
    TORCH_AVAILABLE,
    TorchBackendUnavailable,
    collect_feature_names,
    linear_model_from_actor_params,
    train_torch_bc_model,
)
from ptcg_abc.rl.workflow import build_selfplay_deck_plan, selfplay_pair_for_game


@dataclass
class FakeOption:
    type: int
    number: int | None = None


@dataclass
class FakeSelect:
    type: int
    context: int
    minCount: int
    maxCount: int
    option: list[FakeOption]


@dataclass
class FakeCurrent:
    yourIndex: int = 0
    turn: int = 1
    players: list | None = None
    energyAttached: bool = False
    supporterPlayed: bool = False


@dataclass
class FakePlayer:
    active: list | None = None
    bench: list | None = None
    hand: list | None = None
    handCount: int = 0
    discard: list | None = None
    prize: list | None = None
    deckCount: int = 0


@dataclass
class FakeObservation:
    select: FakeSelect | None
    current: FakeCurrent


@dataclass
class FakePreparedDeck:
    index: int
    label: str


class Phase4RlTests(unittest.TestCase):
    def test_decision_frame_contains_options_scores_and_board_image(self):
        obs = FakeObservation(
            select=FakeSelect(
                type=0,
                context=0,
                minCount=1,
                maxCount=1,
                option=[FakeOption(14), FakeOption(13), FakeOption(8)],
            ),
            current=FakeCurrent(players=[]),
        )

        frame = make_decision_frame(obs)

        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(frame.select_type, "MAIN")
        self.assertEqual(frame.target_count, 1)
        self.assertEqual(len(frame.legal_options), 3)
        self.assertEqual(frame.rule_selected_indices, [2])
        self.assertEqual(len(frame.board_image), BOARD_IMAGE_HEIGHT)
        self.assertEqual(len(frame.board_image[0]), BOARD_IMAGE_WIDTH)

    def test_board_image_uses_opponent_hidden_hand_count(self):
        obs = FakeObservation(
            select=FakeSelect(
                type=0,
                context=0,
                minCount=1,
                maxCount=1,
                option=[FakeOption(14)],
            ),
            current=FakeCurrent(
                players=[
                    FakePlayer(hand=[], handCount=0),
                    FakePlayer(hand=None, handCount=7),
                ],
            ),
        )

        frame = make_decision_frame(obs)

        assert frame is not None
        self.assertEqual(frame.board["opponent_hand_count"], 7)
        self.assertTrue(any(value > 0 for row in frame.board_image[:6] for value in row))

    def test_decision_frame_round_trips(self):
        obs = FakeObservation(
            select=FakeSelect(
                type=8,
                context=38,
                minCount=1,
                maxCount=1,
                option=[FakeOption(0, number=1), FakeOption(0, number=3)],
            ),
            current=FakeCurrent(players=[]),
        )
        frame = make_decision_frame(obs)
        assert frame is not None

        restored = DecisionFrame.from_dict(frame.to_dict())

        self.assertEqual(restored.to_dict(), frame.to_dict())

    def test_behavior_cloning_model_learns_teacher_choice(self):
        obs = FakeObservation(
            select=FakeSelect(
                type=0,
                context=0,
                minCount=1,
                maxCount=1,
                option=[FakeOption(14), FakeOption(13), FakeOption(8)],
            ),
            current=FakeCurrent(players=[]),
        )
        frame = make_decision_frame(obs)
        assert frame is not None

        model, summary = train_behavior_cloning_model([frame] * 5, epochs=3)
        scores = model.score_frame(frame)

        self.assertGreater(summary.actions, 0)
        self.assertEqual(max(range(len(scores)), key=lambda index: scores[index]), 2)

    def test_hybrid_agent_can_use_exported_model(self):
        obs = FakeObservation(
            select=FakeSelect(
                type=0,
                context=0,
                minCount=1,
                maxCount=1,
                option=[FakeOption(14), FakeOption(13)],
            ),
            current=FakeCurrent(players=[]),
        )
        model = LinearOptionModel(weights={"is_end": 10.0})
        agent = HybridRlAgent(
            list(range(60)),
            model=model,
            rule_weight=0.0,
            guidance_rules=(),
        )

        self.assertEqual(agent.act(obs), [0])

    def test_model_save_load_round_trip(self):
        model = LinearOptionModel(weights={"rule_score": 1.5}, bias=-0.2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            model.save(path)

            restored = LinearOptionModel.load(path)

        self.assertEqual(restored.to_dict(), model.to_dict())

    def test_reward_metadata_uses_terminal_and_prize_shape(self):
        reward = reward_from_result_metadata(
            {
                "player_index": 0,
                "winner": 0,
                "previous_prizes": [6, 6],
                "prize_counts": [5, 6],
            },
            RewardConfig(),
        )

        self.assertGreater(reward, 1.0)

    def test_cli_exposes_phase4_commands(self):
        parser = build_parser()

        args = parser.parse_args(["rl-train-bc", "--dataset", "demo.jsonl", "--backend", "torch"])

        self.assertEqual(args.command, "rl-train-bc")
        self.assertEqual(args.backend, "torch")

        snapshot_args = parser.parse_args(
            ["rl-board-snapshots", "--turns-per-player", "2", "--record-player", "both"]
        )

        self.assertEqual(snapshot_args.command, "rl-board-snapshots")
        self.assertEqual(snapshot_args.turns_per_player, 2)
        self.assertEqual(snapshot_args.record_player, "both")

        progression_args = parser.parse_args(
            [
                "rl-image-progression",
                "--image-size",
                "256",
                "--iterations",
                "1",
                "--selfplay-games",
                "2",
                "--eval-games-per-matchup",
                "1",
            ]
        )

        self.assertEqual(progression_args.command, "rl-image-progression")
        self.assertEqual(progression_args.image_size, [256])
        self.assertEqual(progression_args.iterations, 1)
        self.assertEqual(progression_args.selfplay_games, 2)
        self.assertIsNone(progression_args.deck_a_index)
        self.assertIsNone(progression_args.deck_b_index)
        self.assertEqual(progression_args.selfplay_deck_index, [])

        subset_progression_args = parser.parse_args(
            [
                "rl-image-progression",
                "--selfplay-deck-index",
                "1",
                "--selfplay-deck-index",
                "9",
            ]
        )

        self.assertEqual(subset_progression_args.selfplay_deck_index, [1, 9])

    def test_selfplay_default_plan_rotates_all_ordered_deck_pairs(self):
        decks = [FakePreparedDeck(index, f"deck-{index}") for index in range(1, 10)]

        plan = build_selfplay_deck_plan(decks)

        self.assertEqual(plan.mode, "rotate")
        self.assertEqual(plan.deck_indices, list(range(1, 10)))
        self.assertEqual(len(plan.pairs), 81)
        self.assertEqual(
            [(deck_a.index, deck_b.index) for deck_a, deck_b in plan.pairs[:3]],
            [(1, 1), (1, 2), (1, 3)],
        )
        self.assertEqual((plan.pairs[-1][0].index, plan.pairs[-1][1].index), (9, 9))
        self.assertEqual(
            (
                selfplay_pair_for_game(plan, 81)[0].index,
                selfplay_pair_for_game(plan, 81)[1].index,
            ),
            (1, 1),
        )

    def test_selfplay_fixed_plan_requires_explicit_two_deck_pair(self):
        decks = [FakePreparedDeck(index, f"deck-{index}") for index in range(1, 10)]

        plan = build_selfplay_deck_plan(decks, deck_a_index=9, deck_b_index=9)

        self.assertEqual(plan.mode, "fixed")
        self.assertEqual(plan.deck_indices, [9, 9])
        self.assertEqual(len(plan.pairs), 1)
        self.assertEqual((plan.pairs[0][0].index, plan.pairs[0][1].index), (9, 9))

    def test_torch_actor_export_equation_matches_linear_model(self):
        obs = FakeObservation(
            select=FakeSelect(
                type=0,
                context=0,
                minCount=1,
                maxCount=1,
                option=[FakeOption(14), FakeOption(13), FakeOption(8)],
            ),
            current=FakeCurrent(players=[]),
        )
        frame = make_decision_frame(obs)
        assert frame is not None
        feature_names = collect_feature_names([frame])
        weights = [0.05 * (index + 1) for index, _ in enumerate(feature_names)]
        bias = -0.25
        model = linear_model_from_actor_params(feature_names, weights, bias)

        for action in frame.legal_options:
            expected = bias + sum(
                action.features.get(name, 0.0) * weight
                for name, weight in zip(feature_names, weights, strict=True)
            )
            self.assertAlmostEqual(model.score_action(action), expected)

    def test_torch_backend_reports_missing_dependency_cleanly(self):
        if TORCH_AVAILABLE:
            self.skipTest("PyTorch is installed in this environment.")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TorchBackendUnavailable):
                train_torch_bc_model(
                    [],
                    checkpoint_path=Path(tmp) / "checkpoint.pt",
                    export_model_path=Path(tmp) / "model.json",
                )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed.")
    def test_torch_training_exports_linear_inference_model(self):
        obs = FakeObservation(
            select=FakeSelect(
                type=0,
                context=0,
                minCount=1,
                maxCount=1,
                option=[FakeOption(14), FakeOption(13), FakeOption(8)],
            ),
            current=FakeCurrent(players=[]),
        )
        frame = make_decision_frame(obs)
        assert frame is not None
        with tempfile.TemporaryDirectory() as tmp:
            summary = train_torch_bc_model(
                [frame] * 2,
                checkpoint_path=Path(tmp) / "checkpoint.pt",
                export_model_path=Path(tmp) / "model.json",
                epochs=1,
            )

            self.assertGreater(summary.actions, 0)
            self.assertTrue((Path(tmp) / "checkpoint.pt").exists())
            self.assertTrue((Path(tmp) / "model.json").exists())


if __name__ == "__main__":
    unittest.main()
