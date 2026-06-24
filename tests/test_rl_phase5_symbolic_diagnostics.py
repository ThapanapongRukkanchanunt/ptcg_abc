import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from ptcg_abc.cli import build_parser, command_rl_diagnose_phase5_symbolic
from ptcg_abc.rl.phase5_symbolic_diagnostics import (
    SymbolicAgreementStats,
    _symbolic_recommendations,
)


class Phase5SymbolicDiagnosticsTests(unittest.TestCase):
    def test_symbolic_stats_tracks_changed_third_action_drift(self):
        stats = SymbolicAgreementStats()
        stats.update(
            action_count=3,
            predicted={2},
            search={1},
            baseline={0},
            changed=True,
            model_margin=-0.5,
            rule_margin=-100.0,
            model_score_range=2.0,
        )
        payload = stats.to_dict()

        self.assertEqual(payload["frames"], 1)
        self.assertEqual(payload["search_hit_rate"], 0.0)
        self.assertEqual(payload["baseline_hit_rate"], 0.0)
        self.assertEqual(payload["predicted_third_when_changed_rate"], 1.0)
        self.assertLess(payload["mean_model_search_minus_baseline_score"], 0)

    def test_symbolic_recommendations_flag_bad_changed_decisions(self):
        recommendations = _symbolic_recommendations(
            {"search_hit_rate": 0.8},
            {
                "search_hit_rate": 0.2,
                "predicted_third_when_changed_rate": 0.5,
                "mean_model_search_minus_baseline_score": -0.25,
                "model_score_flat_rate": 0.0,
            },
        )

        self.assertGreaterEqual(len(recommendations), 3)

    def test_cli_exposes_symbolic_diagnostic_command(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "rl-diagnose-phase5-symbolic",
                "--dataset",
                "decisions.jsonl",
                "--checkpoint",
                "policy.pt",
            ]
        )

        self.assertEqual(args.func.__name__, "command_rl_diagnose_phase5_symbolic")

    def test_symbolic_diagnostic_missing_checkpoint_fails_cleanly(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "decisions.jsonl"
            dataset.write_text("", encoding="utf-8")
            args = parser.parse_args(
                [
                    "rl-diagnose-phase5-symbolic",
                    "--dataset",
                    str(dataset),
                    "--checkpoint",
                    str(Path(tmp) / "missing.pt"),
                ]
            )
            with contextlib.redirect_stderr(io.StringIO()):
                status = command_rl_diagnose_phase5_symbolic(args)

        self.assertEqual(status, 2)


if __name__ == "__main__":
    unittest.main()
