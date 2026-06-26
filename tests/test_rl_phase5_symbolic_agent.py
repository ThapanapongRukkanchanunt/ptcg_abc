import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ptcg_abc.agent.phase5_search import Phase5SearchPolicyAgent
from ptcg_abc.agent.phase5_symbolic import Phase5SymbolicPolicyAgent
from ptcg_abc.cli import build_parser, command_rl_evaluate, command_rl_rollout
from ptcg_abc.rl.phase5_search import RootSearchConfig


class Phase5SymbolicAgentTests(unittest.TestCase):
    def test_search_config_default_uses_promoted_rollout_cap(self):
        self.assertEqual(RootSearchConfig().max_rollout_steps, 30)

    def test_agent_requires_checkpoint_path(self):
        with self.assertRaises(ValueError):
            Phase5SymbolicPolicyAgent([1] * 60)

    def test_search_agent_requires_opponent_deck(self):
        with self.assertRaises(ValueError):
            Phase5SearchPolicyAgent(
                [1] * 60,
                opponent_deck_ids=[2] * 59,
                sample_dir=".",
                checkpoint_path="missing.pt",
            )

    def test_search_agent_keeps_policy_choice_when_root_search_fails(self):
        agent = object.__new__(Phase5SearchPolicyAgent)
        agent.deck_ids = [1] * 60
        agent.opponent_deck_ids = [2] * 60
        agent.card_by_id = {}
        agent.config = RootSearchConfig(top_k=2)
        agent.use_rule_tiebreak = False
        agent.traces = []
        agent._search_begin = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
        agent._search_end = lambda: None

        frame = SimpleNamespace(
            legal_options=[
                SimpleNamespace(option_type="PLAY", card_name="A", attack_id=None),
                SimpleNamespace(option_type="END", card_name="", attack_id=None),
            ],
            board={"your_index": 0, "turn": 1},
            select_type="MAIN",
            context="MAIN",
        )
        encoded = SimpleNamespace(
            legal_action_indices=[0, 1],
            legal_action_mask=[1.0, 1.0],
        )
        legal_actions = [
            SimpleNamespace(rule_score=0.0, local_index=0),
            SimpleNamespace(rule_score=0.0, local_index=1),
        ]

        selected, trace = agent._search_decision(
            SimpleNamespace(),
            frame,
            baseline=[1],
            encoded=encoded,
            legal_actions=legal_actions,
            scores=[10.0, 0.0],
        )

        self.assertEqual(selected, [1])
        self.assertIn("RuntimeError", trace.search_error or "")
        self.assertFalse(trace.changed)

    def test_search_agent_reports_telemetry_rates(self):
        agent = object.__new__(Phase5SearchPolicyAgent)
        agent.search_decisions = 4
        agent.search_started_decisions = 4
        agent.changed_decisions = 1
        agent.search_errors = 1
        agent.candidate_probes = 12
        agent.candidate_errors = 2
        agent.truncated_candidates = 3
        agent.search_elapsed_seconds = 2.0
        agent.max_search_elapsed_seconds = 0.8

        telemetry = agent.search_telemetry()

        self.assertEqual(telemetry["searched_decisions"], 4)
        self.assertEqual(telemetry["changed_decisions"], 1)
        self.assertEqual(telemetry["candidate_probes"], 12)
        self.assertAlmostEqual(telemetry["change_rate"], 0.25)
        self.assertAlmostEqual(telemetry["search_error_rate"], 0.25)
        self.assertAlmostEqual(telemetry["candidate_error_rate"], 2 / 12)
        self.assertAlmostEqual(telemetry["truncated_candidate_rate"], 3 / 12)
        self.assertAlmostEqual(telemetry["avg_search_seconds"], 0.5)
        self.assertAlmostEqual(telemetry["max_search_seconds"], 0.8)

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
        search_eval_args = parser.parse_args(
            [
                "rl-evaluate",
                "--agent",
                "phase5-search",
                "--model",
                "models/rl/phase5_symbolic_policy_10shards.pt",
                "--search-trace-output",
                "experiments/rl/phase5_search_eval_traces.jsonl",
                "--search-top-k",
                "6",
                "--search-rollout-steps",
                "30",
            ]
        )
        search_selfplay_args = parser.parse_args(
            [
                "rl-generate-phase5-search-selfplay",
                "--model",
                "models/rl/phase5_symbolic_policy_10shards.pt",
                "--games",
                "4",
                "--game-offset",
                "12",
                "--deck-index",
                "1",
                "--deck-index",
                "3",
                "--search-top-k",
                "5",
                "--search-rollout-steps",
                "30",
                "--search-trace-output",
                "experiments/rl/phase5_search_selfplay/traces.jsonl",
                "--search-trace-games",
                "2",
            ]
        )

        self.assertEqual(evaluate_args.agent, "phase5-symbolic")
        self.assertEqual(rollout_args.agent, "phase5-symbolic")
        self.assertEqual(search_eval_args.agent, "phase5-search")
        self.assertEqual(
            search_eval_args.search_trace_output,
            Path("experiments/rl/phase5_search_eval_traces.jsonl"),
        )
        self.assertEqual(search_eval_args.search_top_k, 6)
        self.assertEqual(search_eval_args.search_rollout_steps, 30)
        self.assertEqual(search_selfplay_args.command, "rl-generate-phase5-search-selfplay")
        self.assertEqual(search_selfplay_args.games, 4)
        self.assertEqual(search_selfplay_args.game_offset, 12)
        self.assertEqual(search_selfplay_args.deck_index, [1, 3])
        self.assertEqual(search_selfplay_args.search_top_k, 5)
        self.assertEqual(search_selfplay_args.search_rollout_steps, 30)
        self.assertEqual(search_selfplay_args.search_trace_games, 2)
        self.assertEqual(
            search_selfplay_args.search_trace_output,
            Path("experiments/rl/phase5_search_selfplay/traces.jsonl"),
        )

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

    def test_evaluate_fails_cleanly_when_search_checkpoint_missing(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            args = parser.parse_args(
                [
                    "rl-evaluate",
                    "--sample-dir",
                    tmp,
                    "--agent",
                    "phase5-search",
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
