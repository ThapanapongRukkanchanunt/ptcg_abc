import tempfile
import unittest
from pathlib import Path

from ptcg_abc.cli import build_parser
from ptcg_abc.public_agents import (
    LoadedPublicAgent,
    PublicAgentStatus,
    PublicAgentSource,
    load_public_agent,
    public_agent_sources,
)
from ptcg_abc.evaluation import Phase3RequiredBenchmarkRow
from ptcg_abc.rl.records import ActionFrame, DecisionFrame
from ptcg_abc.rl.public_opponents import (
    PublicAgentTacticalRewardConfig,
    _filter_public_opponents,
    _tactical_reward_for_frame,
    summarize_public_agent_gate,
)


class PublicAgentRosterTests(unittest.TestCase):
    def test_builtin_roster_has_public_20_plus_sample_4(self):
        all_sources = public_agent_sources()
        public_sources = public_agent_sources(include_samples=False)
        sample_sources = public_agent_sources(include_public=False)

        self.assertEqual(len(all_sources), 24)
        self.assertEqual(len(public_sources), 20)
        self.assertEqual(len(sample_sources), 4)
        self.assertIn("sample_abomasnow", {source.key for source in sample_sources})
        self.assertIn("sample_iono", {source.key for source in sample_sources})

    def test_loads_local_python_agent_by_key(self):
        source = PublicAgentSource(
            key="fake_agent",
            label="Fake Agent",
            source_ref="tester/fake-agent",
            url="https://example.test/fake-agent",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_dir = root / "fake_agent"
            agent_dir.mkdir()
            (agent_dir / "submission.py").write_text(
                "\n".join(
                    [
                        "deck_ids = [1] * 60",
                        "def agent(obs):",
                        "    return [0]",
                    ]
                ),
                encoding="utf-8",
            )

            loaded = load_public_agent(
                source,
                roots=[root],
                sample_dir=Path("."),
                include_builtin_samples=False,
            )

            self.assertEqual(loaded.deck_ids, [1] * 60)
            self.assertEqual(loaded.make_agent().act(object()), [0])

    def test_cli_exposes_public_agent_commands(self):
        parser = build_parser()

        roster_args = parser.parse_args(
            ["phase5-public-agent-roster", "--public-agent-key", "sample_dragapult"]
        )
        self.assertEqual(roster_args.func.__name__, "command_phase5_public_agent_roster")
        self.assertEqual(roster_args.public_agent_key, ["sample_dragapult"])

        eval_args = parser.parse_args(
            [
                "rl-evaluate-phase5-public-agents",
                "--deck-index",
                "12",
                "--public-agent-key",
                "sample_dragapult",
                "--search-trace-output",
                "experiments/public-score-traces.jsonl",
                "--search-trace-games",
                "5",
            ]
        )
        self.assertEqual(
            eval_args.func.__name__,
            "command_rl_evaluate_phase5_public_agents",
        )
        self.assertEqual(eval_args.deck_index, [12])
        self.assertEqual(eval_args.public_agent_key, ["sample_dragapult"])
        self.assertEqual(
            eval_args.search_trace_output,
            Path("experiments/public-score-traces.jsonl"),
        )
        self.assertEqual(eval_args.search_trace_games, 5)

        traj_args = parser.parse_args(
            [
                "rl-generate-phase5-public-agent-trajectories",
                "--controlled-public-agent-key",
                "sample_dragapult",
                "--controlled-deck-index",
                "101",
                "--public-agent-key",
                "sample_lucario",
                "--agent",
                "phase5-epsilon",
                "--policy-epsilon",
                "0.75",
                "--outcome-reward-scale",
                "0.25",
                "--tactical-reward-mode",
                "basic",
                "--teacher-agent",
                "rule",
            ]
        )
        self.assertEqual(
            traj_args.func.__name__,
            "command_rl_generate_phase5_public_agent_trajectories",
        )
        self.assertEqual(traj_args.controlled_public_agent_key, "sample_dragapult")
        self.assertEqual(traj_args.controlled_deck_index, 101)
        self.assertEqual(traj_args.public_agent_key, ["sample_lucario"])
        self.assertEqual(traj_args.agent, "phase5-epsilon")
        self.assertEqual(traj_args.policy_epsilon, 0.75)
        self.assertEqual(traj_args.outcome_reward_scale, 0.25)
        self.assertEqual(traj_args.tactical_reward_mode, "basic")
        self.assertEqual(traj_args.teacher_agent, "rule")

        eval_public_args = parser.parse_args(
            [
                "rl-evaluate-phase5-public-agents",
                "--controlled-public-agent-key",
                "sample_dragapult",
                "--controlled-deck-index",
                "101",
                "--public-agent-key",
                "sample_lucario",
                "--agent",
                "phase5-symbolic",
                "--replay-output-dir",
                "experiments/replays",
                "--saved-win-replays",
                "1",
                "--saved-loss-replays",
                "1",
            ]
        )
        self.assertEqual(eval_public_args.controlled_public_agent_key, "sample_dragapult")
        self.assertEqual(eval_public_args.controlled_deck_index, 101)
        self.assertEqual(eval_public_args.agent, "phase5-symbolic")
        self.assertEqual(eval_public_args.saved_win_replays, 1)
        self.assertEqual(eval_public_args.saved_loss_replays, 1)

        init_args = parser.parse_args(
            [
                "rl-init-phase5-policy-checkpoint",
                "--checkpoint",
                "models/rl/scratch/deck-101.pt",
                "--deck-index",
                "101",
                "--controlled-public-agent-key",
                "sample_dragapult",
            ]
        )
        self.assertEqual(
            init_args.func.__name__,
            "command_rl_init_phase5_policy_checkpoint",
        )
        self.assertEqual(init_args.deck_index, 101)

    def test_public_agent_key_filter_preserves_selected_statuses(self):
        source_a = PublicAgentSource(
            key="sample_dragapult",
            label="Sample Dragapult",
            source_ref="tester/dragapult",
            url="https://example.test/dragapult",
        )
        source_b = PublicAgentSource(
            key="sample_abomasnow",
            label="Sample Abomasnow",
            source_ref="tester/abomasnow",
            url="https://example.test/abomasnow",
        )
        loaded = LoadedPublicAgent(
            source=source_a,
            path=None,
            deck_ids=[1] * 60,
            make_agent=lambda: None,
            built_in=True,
        )
        missing = PublicAgentStatus(
            source=source_b,
            status="missing",
            error="not exported",
        )

        opponents, statuses = _filter_public_opponents(
            [loaded],
            [loaded.to_status(), missing],
            ["sample_dragapult"],
        )

        self.assertEqual([opponent.key for opponent in opponents], ["sample_dragapult"])
        self.assertEqual([status.source.key for status in statuses], ["sample_dragapult"])
        with self.assertRaisesRegex(ValueError, "Unknown public-agent key"):
            _filter_public_opponents([loaded], [loaded.to_status()], ["missing_key"])

    def test_basic_tactical_reward_marks_attack_and_attach_choices(self):
        frame = _public_tactical_frame()
        config = PublicAgentTacticalRewardConfig(mode="basic")

        attack_reward, attack_meta = _tactical_reward_for_frame(frame, [0], config)
        self.assertAlmostEqual(attack_reward, 0.04)
        self.assertTrue(attack_meta["tactical_attack_taken"])
        self.assertTrue(attack_meta["tactical_missed_attach"])

        attach_reward, attach_meta = _tactical_reward_for_frame(frame, [1], config)
        self.assertAlmostEqual(attach_reward, 0.06)
        self.assertTrue(attach_meta["tactical_attach_taken"])
        self.assertFalse(attach_meta["tactical_missed_attack"])

        end_reward, end_meta = _tactical_reward_for_frame(frame, [2], config)
        self.assertAlmostEqual(end_reward, -0.16)
        self.assertTrue(end_meta["tactical_missed_attack"])
        self.assertTrue(end_meta["tactical_missed_attach"])

    def test_tactical_reward_default_is_noop_with_metadata(self):
        reward, metadata = _tactical_reward_for_frame(
            _public_tactical_frame(),
            [2],
            PublicAgentTacticalRewardConfig(),
        )

        self.assertEqual(reward, 0.0)
        self.assertEqual(metadata["tactical_reward_mode"], "none")
        self.assertTrue(metadata["tactical_attack_available"])

    def test_public_agent_gate_summarizes_opponents_and_decks(self):
        rows = [
            Phase3RequiredBenchmarkRow(
                deck_index=1,
                deck_label="Deck One",
                archetype="Deck One",
                tournament_rank=1,
                opponent="agent-a",
                opponent_deck_label="Agent A",
                games=10,
                wins=6,
                losses=4,
                win_rate=0.6,
            ),
            Phase3RequiredBenchmarkRow(
                deck_index=1,
                deck_label="Deck One",
                archetype="Deck One",
                tournament_rank=1,
                opponent="agent-b",
                opponent_deck_label="Agent B",
                games=10,
                wins=4,
                losses=6,
                win_rate=0.4,
            ),
        ]

        summary = summarize_public_agent_gate(rows, min_win_rate=0.5)

        self.assertFalse(summary["passed"])
        self.assertEqual(summary["worst_opponent"]["key"], "agent-b")
        self.assertEqual(summary["controlled_decks"][0]["win_rate"], 0.5)
        self.assertEqual(len(summary["failing_opponents"]), 1)


def _public_tactical_frame() -> DecisionFrame:
    return DecisionFrame(
        select_type="MAIN",
        context="MAIN",
        min_count=1,
        max_count=1,
        target_count=1,
        legal_options=[
            ActionFrame(index=0, option_type="ATTACK", features={}),
            ActionFrame(index=1, option_type="ATTACH", features={}),
            ActionFrame(index=2, option_type="END", features={}),
        ],
        rule_selected_indices=[0],
        board={},
        board_image=[],
    )


if __name__ == "__main__":
    unittest.main()
