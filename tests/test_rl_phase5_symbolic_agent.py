import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from ptcg_abc.agent.phase5_symbolic import Phase5SymbolicPolicyAgent
from ptcg_abc.cli import build_parser, command_rl_evaluate, command_rl_rollout


class Phase5SymbolicAgentTests(unittest.TestCase):
    def test_agent_requires_checkpoint_path(self):
        with self.assertRaises(ValueError):
            Phase5SymbolicPolicyAgent([1] * 60)

    def test_cli_accepts_phase5_symbolic_agent(self):
        parser = build_parser()
        evaluate_args = parser.parse_args(
            [
                "rl-evaluate",
                "--agent",
                "phase5-symbolic",
                "--model",
                "models/rl/phase5_symbolic_policy_10shards.pt",
            ]
        )
        rollout_args = parser.parse_args(
            [
                "rl-rollout",
                "--agent",
                "phase5-symbolic",
                "--model",
                "models/rl/phase5_symbolic_policy_10shards.pt",
            ]
        )

        self.assertEqual(evaluate_args.agent, "phase5-symbolic")
        self.assertEqual(rollout_args.agent, "phase5-symbolic")

    def test_evaluate_fails_cleanly_when_symbolic_checkpoint_missing(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            args = parser.parse_args(
                [
                    "rl-evaluate",
                    "--sample-dir",
                    tmp,
                    "--agent",
                    "phase5-symbolic",
                    "--model",
                    str(Path(tmp) / "missing.pt"),
                ]
            )
            with contextlib.redirect_stderr(io.StringIO()):
                status = command_rl_evaluate(args)

        self.assertEqual(status, 2)

    def test_rollout_fails_cleanly_when_symbolic_checkpoint_missing(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            args = parser.parse_args(
                [
                    "rl-rollout",
                    "--sample-dir",
                    tmp,
                    "--agent",
                    "phase5-symbolic",
                    "--model",
                    str(Path(tmp) / "missing.pt"),
                ]
            )
            with contextlib.redirect_stderr(io.StringIO()):
                status = command_rl_rollout(args)

        self.assertEqual(status, 2)


if __name__ == "__main__":
    unittest.main()
