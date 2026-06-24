import tempfile
import unittest
from pathlib import Path

from ptcg_abc.cli import build_parser
from ptcg_abc.rl.dataset import write_decision_jsonl
from ptcg_abc.rl.phase5_encoder import Phase5SymbolicEncoder
from ptcg_abc.rl.phase5_policy import TORCH_AVAILABLE
from ptcg_abc.rl.phase5_symbolic_training import (
    build_phase5_symbolic_dataset,
    decision_frame_to_legal_actions,
    decision_frame_to_state,
    phase5_symbolic_record_from_decision,
    read_phase5_symbolic_jsonl,
    train_phase5_symbolic_policy_from_decisions,
)
from ptcg_abc.rl.records import ActionFrame, DecisionFrame


def _phase5_frame(
    *,
    step_index: int,
    selected: list[int],
    search: list[int] | None = None,
    baseline: list[int] | None = None,
    changed: bool = False,
    turn: int = 2,
) -> DecisionFrame:
    search = list(search if search is not None else selected)
    baseline = list(baseline if baseline is not None else selected)
    return DecisionFrame(
        select_type="MAIN",
        context="MAIN",
        min_count=1,
        max_count=1,
        target_count=1,
        legal_options=[
            ActionFrame(
                index=0,
                option_type="PLAY",
                features={"old": 1.0},
                rule_score=40000.0,
                rule_rank=1,
                card_id=200,
                card_name="Trainer",
                area="HAND",
                area_index=0,
            ),
            ActionFrame(
                index=1,
                option_type="ATTACK",
                features={"search": 1.0},
                rule_score=10000.0,
                rule_rank=2,
                attack_id=42,
            ),
            ActionFrame(
                index=2,
                option_type="END",
                features={"end": 1.0},
                rule_score=-1000.0,
                rule_rank=3,
            ),
        ],
        rule_selected_indices=list(selected),
        board={
            "turn": turn,
            "your_index": 0,
            "energy_attached": False,
            "supporter_played": False,
            "stadium_count": 0,
            "looking_count": 0,
            "my_active_id": 100,
            "my_active_hp": 120,
            "my_active_max_hp": 150,
            "my_active_energy_count": 2,
            "my_active_is_ex": True,
            "my_bench_count": 1,
            "my_bench_0_id": 101,
            "my_bench_0_hp": 80,
            "my_hand_count": 4,
            "my_deck_count": 42,
            "my_prizes": 6,
            "opponent_active_id": 110,
            "opponent_active_hp": 90,
            "opponent_active_max_hp": 100,
            "opponent_hand_count": 5,
            "opponent_deck_count": 43,
            "opponent_prizes": 6,
        },
        board_image=[],
        reward_metadata={
            "collector": "phase5_search",
            "game_index": 7,
            "step_index": step_index,
            "deck_index": 3,
            "player_index": 0,
            "phase5_baseline_indices": baseline,
            "phase5_search_indices": search,
            "phase5_search_changed": changed,
        },
    )


class Phase5SymbolicTrainingTests(unittest.TestCase):
    def test_decision_frame_bridge_emits_state_actions_and_record(self):
        frame = _phase5_frame(
            step_index=1,
            selected=[1],
            search=[1],
            baseline=[0],
            changed=True,
        )
        state = decision_frame_to_state(frame)
        actions = decision_frame_to_legal_actions(frame)
        record = phase5_symbolic_record_from_decision(
            frame,
            encoder=Phase5SymbolicEncoder(max_entities=8, max_actions=4),
            previous_action_features=[],
            max_previous_actions=3,
            changed_weight=5.0,
            unchanged_weight=0.25,
        )

        self.assertEqual(state.us.hand_count, 4)
        self.assertEqual(state.us.active[0].card_id, 100)
        self.assertEqual(state.opponent.active[0].card_id, 110)
        self.assertEqual([action.local_index for action in actions], [0, 1, 2])
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.target_indices, [1])
        self.assertEqual(record.target_positions, [1])
        self.assertEqual(record.weight, 5.0)
        self.assertEqual(sum(record.previous_action_mask), 0.0)

    def test_symbolic_dataset_builder_tracks_previous_turn_actions(self):
        frames = [
            _phase5_frame(step_index=1, selected=[1], search=[1], baseline=[0], changed=True),
            _phase5_frame(step_index=2, selected=[0], search=[0], baseline=[0], changed=False),
            _phase5_frame(step_index=3, selected=[2], search=[2], baseline=[2], turn=3),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "decisions.jsonl"
            output_path = Path(tmp) / "symbolic.jsonl"
            write_decision_jsonl(frames, dataset_path)

            summary = build_phase5_symbolic_dataset(
                dataset_path=dataset_path,
                output_path=output_path,
                max_entities=8,
                max_actions=4,
                max_previous_actions=3,
                changed_weight=3.0,
                unchanged_weight=0.5,
            )
            records = read_phase5_symbolic_jsonl(output_path)

        self.assertEqual(summary.records_written, 3)
        self.assertEqual(summary.changed_records, 1)
        self.assertEqual(len(records), 3)
        self.assertEqual(sum(records[0].previous_action_mask), 0.0)
        self.assertEqual(sum(records[1].previous_action_mask), 1.0)
        self.assertEqual(sum(records[2].previous_action_mask), 0.0)
        self.assertEqual(records[1].weight, 0.5)

    def test_cli_exposes_symbolic_commands(self):
        parser = build_parser()
        build_args = parser.parse_args(
            [
                "rl-build-phase5-symbolic-dataset",
                "--dataset",
                "in.jsonl",
                "--output",
                "out.jsonl",
            ]
        )
        train_args = parser.parse_args(
            [
                "rl-train-phase5-symbolic",
                "--dataset",
                "in.jsonl",
                "--checkpoint",
                "model.pt",
            ]
        )

        self.assertEqual(build_args.func.__name__, "command_rl_build_phase5_symbolic")
        self.assertEqual(train_args.func.__name__, "command_rl_train_phase5_symbolic")

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed.")
    def test_symbolic_trainer_writes_checkpoint(self):
        frames = [
            _phase5_frame(step_index=1, selected=[1], search=[1], baseline=[0], changed=True),
            _phase5_frame(step_index=2, selected=[0], search=[0], baseline=[0]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "decisions.jsonl"
            checkpoint_path = Path(tmp) / "policy.pt"
            report_path = Path(tmp) / "report.json"
            write_decision_jsonl(frames, dataset_path)

            summary = train_phase5_symbolic_policy_from_decisions(
                dataset_path=dataset_path,
                checkpoint_path=checkpoint_path,
                report_path=report_path,
                epochs=1,
                batch_size=2,
                d_model=32,
                max_entities=8,
                max_actions=4,
                max_previous_actions=3,
            )

            self.assertEqual(summary.examples, 2)
            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(report_path.exists())


if __name__ == "__main__":
    unittest.main()
