import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ptcg_abc.cli import build_parser
from ptcg_abc.rl.phase5_alpha_league import (
    PHASE5_ALPHA_DECK_INDICES,
    alpha_iteration_dir,
    cleanup_phase5_alpha_raw_train,
    generate_phase5_alpha_league_iteration,
    train_phase5_deck_specialists_ppo,
)


class Phase5AlphaLeagueTests(unittest.TestCase):
    def test_cli_exposes_alpha_league_commands(self):
        parser = build_parser()

        bootstrap_args = parser.parse_args(
            [
                "rl-generate-phase5-alpha-bootstrap",
                "--iteration",
                "3",
                "--games-per-pair",
                "2",
                "--deck-index",
                "1",
                "--deck-index",
                "13",
            ]
        )
        specialist_args = parser.parse_args(
            [
                "rl-train-phase5-deck-specialists",
                "--no-decision-dataset",
                "--selfplay-dataset",
                "bootstrap.jsonl",
                "--deck-index",
                "2",
                "--decision-limit",
                "10",
                "--iteration",
                "3",
            ]
        )
        iteration_args = parser.parse_args(
            [
                "rl-generate-phase5-alpha-league-iteration",
                "--iteration",
                "4",
                "--specialist-model-dir",
                "models/specialists",
                "--agent",
                "phase5-rl",
                "--games-per-deck",
                "5",
                "--deck-index",
                "1",
            ]
        )
        ppo_specialist_args = parser.parse_args(
            [
                "rl-train-phase5-alpha-ppo-specialists",
                "--trajectory-dataset",
                "selfplay.jsonl",
                "--source-checkpoint-dir",
                "models/source",
                "--output-checkpoint-dir",
                "models/output",
                "--report-dir",
                "reports/decks",
                "--report-json",
                "reports/update.json",
                "--iteration",
                "4",
                "--deck-index",
                "1",
            ]
        )
        cleanup_args = parser.parse_args(
            [
                "rl-clean-phase5-alpha-iteration",
                "--iteration-dir",
                "iter-0003",
                "--update-report",
                "update.json",
            ]
        )
        full_eval_args = parser.parse_args(
            [
                "rl-evaluate-phase5-league",
                "--agent",
                "phase5-full",
                "--model",
                "model.pt",
                "--games-per-matchup",
                "30",
            ]
        )
        filtered_generalist_args = parser.parse_args(
            [
                "rl-train-phase5-generalist",
                "--decision-dataset",
                "decisions.jsonl",
                "--deck-index-filter",
                "7",
            ]
        )

        self.assertEqual(
            bootstrap_args.func.__name__,
            "command_rl_generate_phase5_alpha_bootstrap",
        )
        self.assertEqual(bootstrap_args.deck_index, [1, 13])
        self.assertEqual(
            specialist_args.func.__name__,
            "command_rl_train_phase5_deck_specialists",
        )
        self.assertTrue(specialist_args.no_decision_dataset)
        self.assertEqual(specialist_args.iteration, 3)
        self.assertEqual(
            iteration_args.func.__name__,
            "command_rl_generate_phase5_alpha_league_iteration",
        )
        self.assertEqual(iteration_args.iteration, 4)
        self.assertEqual(iteration_args.specialist_model_dir, Path("models/specialists"))
        self.assertEqual(iteration_args.agent, "phase5-rl")
        self.assertEqual(iteration_args.games_per_deck, 5)
        self.assertEqual(
            ppo_specialist_args.func.__name__,
            "command_rl_train_phase5_alpha_ppo_specialists",
        )
        self.assertEqual(ppo_specialist_args.iteration, 4)
        self.assertFalse(ppo_specialist_args.allow_off_policy_trajectories)
        self.assertEqual(cleanup_args.func.__name__, "command_rl_clean_phase5_alpha_iteration")
        self.assertEqual(full_eval_args.agent, "phase5-full")
        self.assertEqual(full_eval_args.games_per_matchup, 30)
        self.assertEqual(filtered_generalist_args.deck_index_filter, 7)

    def test_alpha_iteration_dir_and_default_decks(self):
        self.assertEqual(
            alpha_iteration_dir(Path("/tmp/alpha"), 12).as_posix(),
            "/tmp/alpha/iterations/iter-0012",
        )
        self.assertEqual(PHASE5_ALPHA_DECK_INDICES[0], 1)
        self.assertEqual(PHASE5_ALPHA_DECK_INDICES[-1], 13)
        self.assertEqual(len(PHASE5_ALPHA_DECK_INDICES), 13)

    def test_cleanup_requires_update_report_and_removes_raw_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            iteration_dir = root / "iterations" / "iter-0001"
            raw_train = iteration_dir / "raw_train"
            raw_train.mkdir(parents=True)
            raw_train.joinpath("data.jsonl").write_text("x\n", encoding="utf-8")
            cleanup_report = root / "cleanup.json"

            with self.assertRaises(ValueError):
                cleanup_phase5_alpha_raw_train(
                    iteration_dir=iteration_dir,
                    cleanup_report_path=cleanup_report,
                    update_report_path=None,
                    require_update_report=True,
                )

            update_report = root / "update.json"
            update_report.write_text("{}", encoding="utf-8")
            summary = cleanup_phase5_alpha_raw_train(
                iteration_dir=iteration_dir,
                cleanup_report_path=cleanup_report,
                update_report_path=update_report,
                require_update_report=True,
            )

            self.assertTrue(summary.removed)
            self.assertEqual(summary.files_removed, 1)
            self.assertGreater(summary.bytes_removed, 0)
            self.assertFalse(raw_train.exists())
            payload = json.loads(cleanup_report.read_text(encoding="utf-8"))
            self.assertEqual(payload["raw_train_dir"], raw_train.as_posix())

    def test_league_iteration_uses_specialist_dir_and_writes_report(self):
        class FakeSelfPlaySummary:
            def to_dict(self):
                return {
                    "agent": "phase5-rl",
                    "games_requested": 10,
                    "games_started": 10,
                    "errors": 0,
                    "specialist_model_dir": "models/specialists",
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specialist_dir = root / "models" / "specialists"
            specialist_dir.mkdir(parents=True)
            for index in (1, 2):
                specialist_dir.joinpath(f"deck-{index:02d}.pt").write_bytes(b"pt")
            iteration_dir = root / "iterations" / "iter-0004"
            report_path = root / "reports" / "iter-0004_league_iteration.json"

            with patch(
                "ptcg_abc.rl.phase5_alpha_league.rollout_selfplay_games",
                return_value=FakeSelfPlaySummary(),
            ) as rollout:
                summary = generate_phase5_alpha_league_iteration(
                    sample_dir=root / "sample",
                    iteration_dir=iteration_dir,
                    report_path=report_path,
                    specialist_model_dir=specialist_dir,
                    games_per_deck=5,
                    deck_indices=[1, 2],
                    game_offset=40,
                    max_steps=300,
                    agent_kind="phase5-rl",
                    search_trace_game_limit=2,
                )

            rollout.assert_called_once()
            kwargs = rollout.call_args.kwargs
            self.assertEqual(kwargs["agent_kind"], "phase5-rl")
            self.assertEqual(kwargs["games"], 10)
            self.assertEqual(kwargs["game_offset"], 40)
            self.assertEqual(kwargs["specialist_model_dir"], specialist_dir)
            self.assertEqual(kwargs["selfplay_deck_indices"], [1, 2])
            self.assertEqual(
                kwargs["output_path"],
                iteration_dir / "raw_train" / "phase5_alpha_league_selfplay.jsonl",
            )
            self.assertEqual(summary.iteration, 4)
            self.assertEqual(summary.games_per_deck, 5)
            self.assertEqual(summary.games_requested, 10)
            self.assertEqual(summary.specialist_model_paths["1"], (specialist_dir / "deck-01.pt").as_posix())
            self.assertTrue(report_path.exists())
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["selfplay"]["games_started"], 10)
            self.assertTrue(payload["cleanup_required"])

    def test_ppo_specialist_update_filters_each_deck_and_writes_report(self):
        class FakePPOSummary:
            def __init__(self, deck_index: int):
                self.examples = 7
                self.deck_index = deck_index

            def to_dict(self):
                return {
                    "examples": self.examples,
                    "deck_index_filter": self.deck_index,
                    "require_on_policy": True,
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trajectory_path = root / "selfplay.jsonl"
            trajectory_path.write_text("{}\n", encoding="utf-8")
            source_dir = root / "source"
            source_dir.mkdir()
            for index in (1, 2):
                source_dir.joinpath(f"deck-{index:02d}.pt").write_bytes(b"pt")
            output_dir = root / "output"
            report_dir = root / "reports"
            aggregate_path = root / "aggregate.json"

            def fake_train(**kwargs):
                return FakePPOSummary(kwargs["deck_index_filter"])

            with patch(
                "ptcg_abc.rl.phase5_alpha_league.train_phase5_ppo_policy_from_trajectories",
                side_effect=fake_train,
            ) as trainer:
                summary = train_phase5_deck_specialists_ppo(
                    trajectory_dataset_paths=[trajectory_path],
                    source_checkpoint_dir=source_dir,
                    output_checkpoint_dir=output_dir,
                    report_dir=report_dir,
                    aggregate_report_path=aggregate_path,
                    iteration=4,
                    deck_indices=[1, 2],
                    epochs=1,
                )

            self.assertEqual(trainer.call_count, 2)
            first_kwargs = trainer.call_args_list[0].kwargs
            self.assertEqual(first_kwargs["deck_index_filter"], 1)
            self.assertTrue(first_kwargs["require_on_policy"])
            self.assertEqual(first_kwargs["output_checkpoint_path"], output_dir / "deck-01.pt")
            self.assertEqual(summary.iteration, 4)
            self.assertEqual(summary.summaries["1"]["examples"], 7)
            payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["summaries"]["2"]["deck_index_filter"], 2)


if __name__ == "__main__":
    unittest.main()
