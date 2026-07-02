import json
import tempfile
import unittest
from pathlib import Path

from ptcg_abc.cli import build_parser
from ptcg_abc.rl.phase5_alpha_league import (
    PHASE5_ALPHA_DECK_INDICES,
    alpha_iteration_dir,
    cleanup_phase5_alpha_raw_train,
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


if __name__ == "__main__":
    unittest.main()
