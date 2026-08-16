import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ptcg_abc.cli import build_parser, command_rl_evaluate_phase5_public_agents
from ptcg_abc.public_agents import (
    LoadedPublicAgent,
    PublicAgentStatus,
    PublicAgentSource,
    load_public_agent,
    public_agent_sources,
)
from ptcg_abc.evaluation import Phase3RequiredBenchmarkRow
from ptcg_abc.rl.records import ActionFrame, DecisionFrame
from ptcg_abc.rl.featurizer import summarize_board
from ptcg_abc.rl.public_opponents import (
    PublicAgentTacticalRewardConfig,
    _balanced_matchup_game_counts,
    _deck_shaped_turn_targets,
    _deck_reward_potential_components,
    _filter_public_opponents,
    _summarize_turn_prize_games,
    _tactical_reward_for_frame,
    _turn_prize_game_summary,
    _turn_prize_targets,
    summarize_public_agent_gate,
)
from ptcg_abc.rl.workflow import RecordedPolicyFrame, _policy_metadata


class PublicAgentRosterTests(unittest.TestCase):
    def test_board_summary_exposes_only_observable_reward_signals(self):
        card_by_id = {
            2: SimpleNamespace(cardType=2, energyType=1),
            109: SimpleNamespace(cardType=1, hp=70),
            500: SimpleNamespace(cardType=3, rarity="ACE SPEC"),
            999: SimpleNamespace(cardType=1, hp=210, weakness=1, ex=True),
        }
        abra = SimpleNamespace(
            id=109,
            hp=70,
            maxHp=70,
            energies=[1],
            energyCards=[SimpleNamespace(id=2)],
            tools=[],
        )
        target = SimpleNamespace(
            id=999,
            hp=210,
            maxHp=210,
            energies=[],
            energyCards=[],
            tools=[],
        )
        mine = SimpleNamespace(
            active=[abra],
            bench=[],
            hand=[SimpleNamespace(id=742)],
            discard=[],
            prize=[1] * 6,
            deckCount=40,
        )
        opponent = SimpleNamespace(
            active=[target],
            bench=[],
            hand=[SimpleNamespace(id=123)],
            handCount=1,
            discard=[SimpleNamespace(id=500)],
            prize=[1] * 6,
            deckCount=40,
        )
        current = SimpleNamespace(
            yourIndex=0,
            players=[mine, opponent],
            turn=1,
            stadium=[],
        )

        board = summarize_board(current, card_by_id=card_by_id)

        self.assertEqual(board["my_hand_card_ids"], [742])
        self.assertEqual(board["opponent_hand_card_ids"], [])
        self.assertEqual(board["my_active_card"]["energy_card_ids"], [2])
        self.assertTrue(board["opponent_active_card"]["weak_to_fire"])
        self.assertTrue(board["opponent_ace_spec_seen"])

    def test_alakazam_deck_potential_matches_finalized_reward(self):
        board = {
            "my_active_card": _card_state(109, energies=[5]),
            "my_active_id": 109,
            "my_bench_cards": [
                _card_state(742),
                _card_state(245),
                _card_state(142, tools=1),
                _card_state(858),
                _card_state(66),
            ],
            "my_hand_card_ids": [742, 245, 1079],
            "opponent_active_card": _card_state(132),
            "opponent_bench_cards": [],
            "opponent_ace_spec_seen": False,
        }

        components = _deck_reward_potential_components(board, deck_index=1)

        self.assertEqual(components["abra_in_play"], 1.0)
        self.assertEqual(components["kadabra_in_play"], 4.0)
        self.assertEqual(components["alakazam_in_play_binary"], 10.0)
        self.assertEqual(components["psychic_energy_on_alakazam_line"], 1.0)
        self.assertEqual(components["kadabra_hand_abra_play_pairs"], 2.0)
        self.assertEqual(components["alakazam_candy_abra_sets"], 3.0)
        self.assertEqual(components["alakazam_hand_kadabra_play_pairs"], 4.0)
        self.assertEqual(components["genesect_tool_before_opponent_ace_spec"], 1.0)
        self.assertEqual(components["psyduck_into_dusknoir_line"], 1.0)
        self.assertEqual(components["dudunsparce_or_fezandipiti"], 0.5)
        self.assertEqual(sum(components.values()), 27.5)

        board["opponent_ace_spec_seen"] = True
        components = _deck_reward_potential_components(board, deck_index=1)
        self.assertEqual(components["genesect_tool_before_opponent_ace_spec"], 0.0)

    def test_alakazam_binary_and_capped_rewards_cannot_be_farmed_by_copies(self):
        board = {
            "my_active_card": _card_state(245),
            "my_bench_cards": [
                _card_state(245),
                _card_state(66),
                _card_state(66),
                _card_state(140),
                _card_state(140),
                _card_state(140),
            ],
        }

        components = _deck_reward_potential_components(board, deck_index=1)

        self.assertEqual(components["alakazam_in_play_binary"], 10.0)
        self.assertEqual(components["dudunsparce_or_fezandipiti"], 2.0)

    def test_dragapult_deck_potential_matches_finalized_reward(self):
        board = {
            "my_active_id": 119,
            "my_active_card": _card_state(119, energies=[2]),
            "my_bench_cards": [
                _card_state(120, energies=[5]),
                _card_state(121, energies=[2, 5]),
                _card_state(131),
                _card_state(132),
                _card_state(133),
                _card_state(112, energies=[7]),
                _card_state(140),
                _card_state(791, energies=[2]),
                _card_state(1071),
            ],
            "my_hand_card_ids": [120, 121],
            "opponent_active_card": _card_state(
                999,
                hp=210,
                is_ex=True,
                weak_to_fire=True,
            ),
        }

        components = _deck_reward_potential_components(board, deck_index=3)

        self.assertEqual(components["dreepy_in_play"], 1.0)
        self.assertEqual(components["drakloak_in_play"], 4.0)
        self.assertEqual(components["dragapult_in_play_binary"], 10.0)
        self.assertEqual(components["drakloak_hand_dreepy_play_pairs"], 2.0)
        self.assertEqual(components["dragapult_hand_drakloak_play_pairs"], 4.0)
        self.assertEqual(components["fire_psychic_on_dragapult_line"], 5.0)
        self.assertEqual(components["dusknoir_line_in_play"], 1.5)
        self.assertEqual(components["munkidori_with_darkness"], 0.5)
        self.assertEqual(components["fezandipiti_in_play"], 0.5)
        self.assertEqual(components["budew_active_before_powered_dragapult"], 0.0)
        self.assertEqual(components["moltres_fire_ko_window"], 5.0)
        self.assertNotIn("meowth", components)

    def test_dragapult_budew_reward_stops_after_dragapult_is_powered(self):
        board = {
            "my_active_id": 235,
            "my_active_card": _card_state(235),
            "my_bench_cards": [_card_state(121, energies=[2])],
        }
        self.assertEqual(
            _deck_reward_potential_components(board, deck_index=3)[
                "budew_active_before_powered_dragapult"
            ],
            2.0,
        )

        board["my_bench_cards"] = [_card_state(121, energies=[2, 5])]
        self.assertEqual(
            _deck_reward_potential_components(board, deck_index=3)[
                "budew_active_before_powered_dragapult"
            ],
            0.0,
        )

    def test_deck_shaping_uses_potential_difference_prizes_and_timeout_once(self):
        first = _recorded_turn(turn=1, prizes=6)
        first.frame.board.update(
            {"my_active_id": 109, "my_active_card": _card_state(109)}
        )
        second = _recorded_turn(turn=2, prizes=5)
        second.frame.board.update(
            {"my_active_id": 742, "my_active_card": _card_state(742)}
        )

        targets = _deck_shaped_turn_targets(
            [first, second],
            final_prize_count=5,
            gamma=0.9,
            deck_index=1,
            timed_out=True,
        )

        self.assertAlmostEqual(targets[0]["immediate_reward"], 12.6)
        self.assertAlmostEqual(targets[1]["immediate_reward"], -14.0)
        self.assertAlmostEqual(targets[0]["return"], 0.0)
        self.assertEqual(targets[0]["timeout_penalty"], 0.0)
        self.assertEqual(targets[1]["timeout_penalty"], -10.0)
        self.assertAlmostEqual(targets[0]["discounted_prize_return"], 1.0)

    def test_alakazam_rewards_ready_kadabra_promotion_after_opponent_prize(self):
        first = _recorded_turn(turn=1, prizes=6)
        first.frame.board.update({"opponent_prizes": 6})
        second = _recorded_promotion(
            turn=2,
            prizes=6,
            opponent_prizes=5,
            chosen_card_id=742,
            chosen_state=_card_state(742),
            hand_ids=[245, 5],
        )

        targets = _deck_shaped_turn_targets(
            [first, second],
            final_prize_count=6,
            gamma=0.9,
            deck_index=1,
            timed_out=False,
        )

        self.assertEqual(targets[0]["event_reward"], 2.0)
        self.assertAlmostEqual(targets[0]["immediate_reward"], 9.2)
        self.assertEqual(targets[1]["event_reward"], 0.0)

        second.frame.board["my_hand_card_ids"] = [245]
        second.frame.board["my_bench_cards"] = [_card_state(742, energies=[19])]
        targets = _deck_shaped_turn_targets(
            [first, second],
            final_prize_count=6,
            gamma=0.9,
            deck_index=1,
            timed_out=False,
        )
        self.assertEqual(targets[0]["event_reward"], 2.0)

    def test_alakazam_kadabra_promotion_bonus_requires_full_window(self):
        first = _recorded_turn(turn=1, prizes=6)
        first.frame.board.update({"opponent_prizes": 6})
        cases = [
            {"opponent_prizes": 6, "chosen_card_id": 742, "hand_ids": [245, 5]},
            {"opponent_prizes": 5, "chosen_card_id": 109, "hand_ids": [245, 5]},
            {"opponent_prizes": 5, "chosen_card_id": 742, "hand_ids": [5]},
            {"opponent_prizes": 5, "chosen_card_id": 742, "hand_ids": [245]},
        ]
        for case in cases:
            with self.subTest(case=case):
                second = _recorded_promotion(
                    turn=2,
                    prizes=6,
                    opponent_prizes=case["opponent_prizes"],
                    chosen_card_id=case["chosen_card_id"],
                    chosen_state=_card_state(case["chosen_card_id"]),
                    hand_ids=case["hand_ids"],
                )
                targets = _deck_shaped_turn_targets(
                    [first, second],
                    final_prize_count=6,
                    gamma=0.9,
                    deck_index=1,
                    timed_out=False,
                )
                self.assertEqual(targets[0]["event_reward"], 0.0)

    def test_discounted_turn_prizes_use_exact_prize_deltas(self):
        records = [
            _recorded_turn(turn=1, prizes=6),
            _recorded_turn(turn=1, prizes=6),
            _recorded_turn(turn=2, prizes=5),
        ]

        targets = _turn_prize_targets(records, final_prize_count=2, gamma=0.9)

        self.assertEqual([target["record_index"] for target in targets], [0, 2])
        self.assertEqual([target["prize_reward"] for target in targets], [1, 3])
        self.assertAlmostEqual(targets[0]["return"], 3.7)
        self.assertAlmostEqual(targets[1]["return"], 3.0)

    def test_discounted_turn_prizes_skip_setup_only_decisions(self):
        setup = _recorded_turn(turn=0, prizes=6)

        targets = _turn_prize_targets(
            [setup],
            final_prize_count=6,
            gamma=0.97,
        )

        self.assertEqual(targets, [])

    def test_turn_prize_summary_reports_primary_progress_metrics(self):
        fast = _turn_prize_game_summary(
            _turn_prize_targets(
                [_recorded_turn(turn=1, prizes=6), _recorded_turn(turn=2, prizes=3)],
                final_prize_count=0,
                gamma=0.9,
            ),
            final_prize_count=0,
            finished=True,
        )
        partial = _turn_prize_game_summary(
            _turn_prize_targets(
                [_recorded_turn(turn=1, prizes=6)],
                final_prize_count=4,
                gamma=0.9,
            ),
            final_prize_count=4,
            finished=False,
        )

        summary = _summarize_turn_prize_games([fast, partial])

        self.assertEqual(summary["total_prizes_taken"], 8)
        self.assertEqual(summary["average_prizes_taken"], 4.0)
        self.assertEqual(summary["games_reaching_six_prizes"], 1)
        self.assertEqual(summary["average_turns_to_six"], 2.0)

    def test_prize_pipeline_cli_defaults_and_overrides(self):
        args = build_parser().parse_args(
            [
                "rl-generate-phase5-public-agent-trajectories",
                "--reward-objective",
                "discounted-turn-prizes",
                "--turn-prize-discount-gamma",
                "0.95",
            ]
        )
        eval_args = build_parser().parse_args(
            ["rl-evaluate-phase5-public-agents"]
        )

        self.assertEqual(args.reward_objective, "discounted-turn-prizes")
        self.assertAlmostEqual(args.turn_prize_discount_gamma, 0.95)
        self.assertAlmostEqual(eval_args.prize_discount_gamma, 0.97)

        shaped_args = build_parser().parse_args(
            [
                "rl-generate-phase5-public-agent-trajectories",
                "--reward-objective",
                "deck-shaped-prizes",
            ]
        )
        self.assertEqual(shaped_args.reward_objective, "deck-shaped-prizes")

    def test_rule_roster_cli_and_exact_balanced_game_budget(self):
        args = build_parser().parse_args(
            [
                "rl-generate-phase5-public-agent-trajectories",
                "--opponent-pool",
                "league-rule",
                "--games-total",
                "1000",
            ]
        )
        eval_args = build_parser().parse_args(
            [
                "rl-evaluate-phase5-public-agents",
                "--opponent-pool",
                "league-rule",
            ]
        )

        first = _balanced_matchup_game_counts(
            games_total=1000,
            matchup_count=13,
            game_offset=0,
        )
        second = _balanced_matchup_game_counts(
            games_total=1000,
            matchup_count=13,
            game_offset=1000,
        )

        self.assertEqual(args.opponent_pool, "league-rule")
        self.assertEqual(args.games_total, 1000)
        self.assertEqual(eval_args.opponent_pool, "league-rule")
        self.assertEqual(sum(first), 1000)
        self.assertEqual(set(first), {76, 77})
        self.assertEqual(sum(second), 1000)
        self.assertNotEqual(first, second)

    def test_policy_metadata_preserves_search_correction_markers(self):
        agent = SimpleNamespace(
            last_policy_metadata={
                "logprob": 0.0,
                "value": 0.0,
                "on_policy": False,
                "mode": "deterministic",
                "phase5_search_applied": True,
                "phase5_baseline_indices": [0],
                "phase5_search_indices": [1],
                "phase5_search_changed": True,
                "phase5_search_error": None,
                "phase5_search_baseline_score": 0.2,
                "phase5_search_selected_score": 0.5,
                "phase5_search_score_margin": 0.3,
            }
        )

        metadata = _policy_metadata(agent)

        self.assertTrue(metadata["phase5_search_applied"])
        self.assertEqual(metadata["phase5_baseline_indices"], [0])
        self.assertEqual(metadata["phase5_search_indices"], [1])
        self.assertTrue(metadata["phase5_search_changed"])
        self.assertIsNone(metadata["phase5_search_error"])
        self.assertAlmostEqual(metadata["phase5_search_score_margin"], 0.3)

    def test_rule_public_eval_ignores_specialist_checkpoint_directory(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            args = parser.parse_args(
                [
                    "rl-evaluate-phase5-public-agents",
                    "--sample-dir",
                    tmp,
                    "--agent",
                    "rule",
                    "--specialist-model-dir",
                    str(Path(tmp) / "missing-specialists"),
                ]
            )
            with patch(
                "ptcg_abc.cli.run_phase5_public_agent_benchmark",
                side_effect=RuntimeError("reached evaluator"),
            ):
                with self.assertRaisesRegex(RuntimeError, "reached evaluator"):
                    command_rl_evaluate_phase5_public_agents(args)

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
                "phase5-epsilon-mixture",
                "--policy-epsilon",
                "0.75",
                "--policy-seed",
                "1234",
                "--outcome-reward-scale",
                "0.25",
                "--outcome-reward-assignment",
                "terminal",
                "--tactical-reward-mode",
                "basic-fractional-prize",
                "--tactical-fractional-prize-weight",
                "0.25",
                "--tactical-fractional-opponent-weight",
                "0.5",
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
        self.assertEqual(traj_args.agent, "phase5-epsilon-mixture")
        self.assertEqual(traj_args.policy_epsilon, 0.75)
        self.assertEqual(traj_args.policy_seed, 1234)
        self.assertEqual(traj_args.outcome_reward_scale, 0.25)
        self.assertEqual(traj_args.outcome_reward_assignment, "terminal")
        self.assertEqual(traj_args.tactical_reward_mode, "basic-fractional-prize")
        self.assertEqual(traj_args.tactical_fractional_prize_weight, 0.25)
        self.assertEqual(traj_args.tactical_fractional_opponent_weight, 0.5)
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
                "--game-seed",
                "20260727",
            ]
        )
        self.assertEqual(eval_public_args.controlled_public_agent_key, "sample_dragapult")
        self.assertEqual(eval_public_args.controlled_deck_index, 101)
        self.assertEqual(eval_public_args.agent, "phase5-symbolic")
        self.assertEqual(eval_public_args.saved_win_replays, 1)
        self.assertEqual(eval_public_args.saved_loss_replays, 1)
        self.assertEqual(eval_public_args.game_seed, 20260727)

        rule_eval_args = parser.parse_args(
            [
                "rl-evaluate-phase5-public-agents",
                "--controlled-public-agent-key",
                "sample_dragapult",
                "--public-agent-key",
                "sample_lucario",
                "--agent",
                "rule",
            ]
        )
        self.assertEqual(rule_eval_args.agent, "rule")

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

    def test_fractional_prize_reward_values_partial_prize_progress(self):
        before = _public_tactical_frame(
            board={
                "my_prizes": 6,
                "opponent_prizes": 6,
                "my_active_card": {"hp": 200, "max_hp": 200},
                "my_bench_cards": [],
                "opponent_active_card": {
                    "hp": 340,
                    "max_hp": 340,
                    "is_ex": True,
                    "is_mega_ex": True,
                },
                "opponent_bench_cards": [
                    {"hp": 80, "max_hp": 80},
                    {"hp": 110, "max_hp": 110},
                ],
            }
        )
        riolu_after = _public_tactical_frame(
            board={
                "my_prizes": 3,
                "opponent_prizes": 6,
                "my_active_card": {"hp": 200, "max_hp": 200},
                "my_bench_cards": [],
                "opponent_active_card": {},
                "opponent_bench_cards": [
                    {"hp": 20, "max_hp": 80},
                    {"hp": 110, "max_hp": 110},
                ],
            }
        )
        solrock_after = _public_tactical_frame(
            board={
                "my_prizes": 3,
                "opponent_prizes": 6,
                "my_active_card": {"hp": 200, "max_hp": 200},
                "my_bench_cards": [],
                "opponent_active_card": {},
                "opponent_bench_cards": [
                    {"hp": 80, "max_hp": 80},
                    {"hp": 50, "max_hp": 110},
                ],
            }
        )
        config = PublicAgentTacticalRewardConfig(mode="fractional-prize")

        riolu_reward, riolu_meta = _tactical_reward_for_frame(
            before,
            [0],
            config,
            next_frame=riolu_after,
        )
        solrock_reward, solrock_meta = _tactical_reward_for_frame(
            before,
            [0],
            config,
            next_frame=solrock_after,
        )

        self.assertAlmostEqual(riolu_reward, 3.75)
        self.assertAlmostEqual(solrock_reward, 3.0 + (60.0 / 110.0))
        self.assertGreater(riolu_reward, solrock_reward)
        self.assertAlmostEqual(riolu_meta["tactical_fractional_prize_delta"], 3.75)
        self.assertEqual(
            riolu_meta["tactical_fractional_prize_after_source"],
            "next-frame",
        )
        self.assertAlmostEqual(
            solrock_meta["tactical_fractional_prize_delta"],
            3.0 + (60.0 / 110.0),
        )
        self.assertEqual(riolu_meta["tactical_basic_reward"], 0.0)

    def test_fractional_prize_reward_prefers_post_action_board(self):
        before = _public_tactical_frame(
            board={
                "my_prizes": 6,
                "opponent_prizes": 6,
                "opponent_active_card": {
                    "hp": 340,
                    "max_hp": 340,
                    "is_ex": True,
                    "is_mega_ex": True,
                },
                "opponent_bench_cards": [
                    {"hp": 80, "max_hp": 80},
                    {"hp": 110, "max_hp": 110},
                ],
            }
        )
        next_frame = _public_tactical_frame(
            board={
                "my_prizes": 3,
                "opponent_prizes": 6,
                "opponent_active_card": {},
                "opponent_bench_cards": [
                    {"hp": 80, "max_hp": 80},
                    {"hp": 50, "max_hp": 110},
                ],
            }
        )
        post_action_board = {
            "my_prizes": 3,
            "opponent_prizes": 6,
            "opponent_active_card": {},
            "opponent_bench_cards": [
                {"hp": 20, "max_hp": 80},
                {"hp": 110, "max_hp": 110},
            ],
        }

        reward, metadata = _tactical_reward_for_frame(
            before,
            [0],
            PublicAgentTacticalRewardConfig(mode="fractional-prize"),
            post_action_board=post_action_board,
            next_frame=next_frame,
        )

        self.assertAlmostEqual(reward, 3.75)
        self.assertEqual(
            metadata["tactical_fractional_prize_after_source"],
            "post-action",
        )

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


def _public_tactical_frame(board: dict | None = None) -> DecisionFrame:
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
        board=board or {},
        board_image=[],
    )


def _recorded_turn(*, turn: int, prizes: int) -> RecordedPolicyFrame:
    return RecordedPolicyFrame(
        frame=_public_tactical_frame(board={"turn": turn, "my_prizes": prizes}),
        chosen_indices=[0],
    )


def _recorded_promotion(
    *,
    turn: int,
    prizes: int,
    opponent_prizes: int,
    chosen_card_id: int,
    chosen_state: dict,
    hand_ids: list[int],
) -> RecordedPolicyFrame:
    return RecordedPolicyFrame(
        frame=DecisionFrame(
            select_type="CARD",
            context="TO_ACTIVE",
            min_count=1,
            max_count=1,
            target_count=1,
            legal_options=[
                ActionFrame(
                    index=0,
                    option_type="CARD",
                    features={},
                    card_id=chosen_card_id,
                    area="BENCH",
                    area_index=0,
                )
            ],
            rule_selected_indices=[0],
            board={
                "turn": turn,
                "my_prizes": prizes,
                "opponent_prizes": opponent_prizes,
                "my_bench_cards": [chosen_state],
                "my_hand_card_ids": list(hand_ids),
            },
            board_image=[],
        ),
        chosen_indices=[0],
    )


def _card_state(
    card_id: int,
    *,
    energies: list[int] | None = None,
    tools: int = 0,
    hp: int = 0,
    is_ex: bool = False,
    weak_to_fire: bool = False,
) -> dict:
    return {
        "id": card_id,
        "energy_card_ids": list(energies or []),
        "tool_count": tools,
        "hp": hp,
        "is_ex": is_ex,
        "weak_to_fire": weak_to_fire,
    }


if __name__ == "__main__":
    unittest.main()
