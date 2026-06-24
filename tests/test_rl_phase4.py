import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from ptcg_abc.agent import HybridRlAgent
from ptcg_abc.cli import build_parser
from ptcg_abc.rl.dataset import write_decision_jsonl
from ptcg_abc.rl.featurizer import BOARD_IMAGE_HEIGHT, BOARD_IMAGE_WIDTH, make_decision_frame
from ptcg_abc.rl.model import LinearOptionModel, train_behavior_cloning_model
from ptcg_abc.rl.phase5_diagnostics import diagnose_search_distillation, diagnose_search_traces
from ptcg_abc.rl.phase5_search import (
    RootSearchConfig,
    _best_candidate_indices,
    _candidate_evaluations,
    _replace_frame_selection,
    _score_candidates,
    merge_search_data,
    phase5_absolute_game_index,
    phase5_matchup_for_game_index,
    sample_hidden_state,
)
from ptcg_abc.rl.records import ActionFrame, DecisionFrame
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


@dataclass
class FakeCard:
    id: int
    playerIndex: int


@dataclass
class FakePokemon:
    id: int
    playerIndex: int
    energyCards: list | None = None
    tools: list | None = None
    preEvolution: list | None = None


def _diagnostic_frame(
    *,
    baseline: list[int],
    search: list[int],
    changed: bool,
    step_index: int,
) -> DecisionFrame:
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
                features={"prefer_baseline": 1.0, "prefer_search": 0.0},
                rule_score=1.0,
                card_name="Baseline Card",
            ),
            ActionFrame(
                index=1,
                option_type="PLAY",
                features={"prefer_baseline": 0.0, "prefer_search": 1.0},
                rule_score=0.1,
                card_name="Search Card",
            ),
        ],
        rule_selected_indices=list(search),
        board={},
        board_image=[],
        reward_metadata={
            "collector": "phase5_search",
            "game_index": 1,
            "step_index": step_index,
            "deck_index": 1,
            "deck_label": "Diagnostic Deck",
            "opponent": "Diagnostic Opponent",
            "phase5_search_applied": True,
            "phase5_baseline_indices": list(baseline),
            "phase5_search_indices": list(search),
            "phase5_search_changed": changed,
        },
    )


def _diagnostic_three_action_frame(
    *,
    baseline: list[int],
    search: list[int],
    changed: bool,
    step_index: int,
) -> DecisionFrame:
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
                features={"prefer_baseline": 1.0, "prefer_search": 0.0, "prefer_third": 0.0},
                rule_score=1.0,
                card_name="Baseline Card",
            ),
            ActionFrame(
                index=1,
                option_type="PLAY",
                features={"prefer_baseline": 0.0, "prefer_search": 1.0, "prefer_third": 0.0},
                rule_score=0.1,
                card_name="Search Card",
            ),
            ActionFrame(
                index=2,
                option_type="RETREAT",
                features={"prefer_baseline": 0.0, "prefer_search": 0.0, "prefer_third": 1.0},
                rule_score=-1.0,
                card_name="",
            ),
        ],
        rule_selected_indices=list(search),
        board={},
        board_image=[],
        reward_metadata={
            "collector": "phase5_search",
            "game_index": 1,
            "step_index": step_index,
            "deck_index": 1,
            "deck_label": "Diagnostic Deck",
            "opponent": "Diagnostic Opponent",
            "phase5_search_applied": True,
            "phase5_baseline_indices": list(baseline),
            "phase5_search_indices": list(search),
            "phase5_search_changed": changed,
        },
    )


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

    def test_behavior_cloning_can_overweight_search_changed_frames(self):
        changed = _diagnostic_frame(baseline=[0], search=[1], changed=True, step_index=1)
        unchanged = _diagnostic_frame(baseline=[0], search=[0], changed=False, step_index=2)

        model, summary = train_behavior_cloning_model(
            [changed] + [unchanged] * 10,
            epochs=5,
            changed_weight=20.0,
            unchanged_weight=1.0,
            excluded_features=["rule_score", "rule_rank_inv"],
        )
        scores = model.score_frame(changed)

        self.assertGreater(summary.actions, 0)
        self.assertGreater(scores[1], scores[0])
        self.assertNotIn("rule_score", model.weights)
        self.assertNotIn("rule_rank_inv", model.weights)

    def test_behavior_cloning_pairwise_changed_prefers_search_over_baseline(self):
        changed = _diagnostic_frame(baseline=[0], search=[1], changed=True, step_index=1)
        unchanged = _diagnostic_frame(baseline=[0], search=[0], changed=False, step_index=2)

        model, summary = train_behavior_cloning_model(
            [changed] + [unchanged] * 10,
            epochs=8,
            changed_weight=4.0,
            unchanged_weight=0.1,
            excluded_features=["rule_score", "rule_rank_inv"],
            pairwise_changed=True,
            pairwise_margin=1.0,
        )
        scores = model.score_frame(changed)

        self.assertGreater(summary.actions, 0)
        self.assertGreater(scores[1] - scores[0], 0.5)
        self.assertTrue(model.metadata["pairwise_changed"])
        self.assertGreater(model.metadata["pairwise_pairs"], 0)

    def test_behavior_cloning_pairwise_all_prefers_search_over_all_legal_actions(self):
        changed = _diagnostic_three_action_frame(
            baseline=[0],
            search=[1],
            changed=True,
            step_index=1,
        )

        model, summary = train_behavior_cloning_model(
            [changed] * 4,
            epochs=8,
            changed_weight=2.0,
            unchanged_weight=0.1,
            excluded_features=["rule_score", "rule_rank_inv"],
            pairwise_changed=True,
            pairwise_margin=1.0,
            pairwise_negatives="all",
        )
        scores = model.score_frame(changed)

        self.assertGreater(summary.actions, 0)
        self.assertGreater(scores[1] - scores[0], 0.5)
        self.assertGreater(scores[1] - scores[2], 0.5)
        self.assertEqual(model.metadata["pairwise_negatives"], "all")
        self.assertGreater(model.metadata["pairwise_pairs"], 0)

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

        search_args = parser.parse_args(
            [
                "rl-generate-search-data",
                "--games",
                "1",
                "--top-k",
                "3",
                "--shard-index",
                "2",
                "--shard-count",
                "8",
                "--require-changed",
            ]
        )

        self.assertEqual(search_args.command, "rl-generate-search-data")
        self.assertEqual(search_args.games, 1)
        self.assertEqual(search_args.top_k, 3)
        self.assertEqual(search_args.shard_index, 2)
        self.assertEqual(search_args.shard_count, 8)
        self.assertTrue(search_args.require_changed)

        merge_args = parser.parse_args(
            [
                "rl-merge-search-data",
                "--input",
                "data/shard-*.jsonl",
                "--trace-input",
                "experiments/trace-*.jsonl",
            ]
        )

        self.assertEqual(merge_args.command, "rl-merge-search-data")
        self.assertEqual(merge_args.input, ["data/shard-*.jsonl"])
        self.assertEqual(merge_args.trace_input, ["experiments/trace-*.jsonl"])

        diagnose_args = parser.parse_args(
            [
                "rl-diagnose-search-distill",
                "--dataset",
                "data/phase5.jsonl",
                "--model",
                "models/phase5.json",
                "--trace-input",
                "experiments/traces.jsonl",
            ]
        )

        self.assertEqual(diagnose_args.command, "rl-diagnose-search-distill")
        self.assertEqual(diagnose_args.dataset, Path("data/phase5.jsonl"))
        self.assertEqual(diagnose_args.model, Path("models/phase5.json"))
        self.assertEqual(diagnose_args.trace_input, Path("experiments/traces.jsonl"))

        weighted_train_args = parser.parse_args(
            [
                "rl-train-bc",
                "--changed-weight",
                "12",
                "--unchanged-weight",
                "1",
                "--exclude-feature",
                "rule_score",
                "--exclude-feature",
                "rule_rank_inv",
                "--pairwise-changed",
                "--pairwise-margin",
                "1.5",
                "--pairwise-negatives",
                "all",
            ]
        )

        self.assertEqual(weighted_train_args.changed_weight, 12.0)
        self.assertEqual(weighted_train_args.unchanged_weight, 1.0)
        self.assertEqual(weighted_train_args.exclude_feature, ["rule_score", "rule_rank_inv"])
        self.assertTrue(weighted_train_args.pairwise_changed)
        self.assertEqual(weighted_train_args.pairwise_margin, 1.5)
        self.assertEqual(weighted_train_args.pairwise_negatives, "all")

    def test_phase5_shard_game_indices_interleave_without_overlap(self):
        shard0 = [
            phase5_absolute_game_index(local, shard_index=0, shard_count=3)
            for local in range(4)
        ]
        shard1 = [
            phase5_absolute_game_index(local, shard_index=1, shard_count=3)
            for local in range(4)
        ]
        shard2 = [
            phase5_absolute_game_index(local, shard_index=2, shard_count=3)
            for local in range(4)
        ]

        self.assertEqual(shard0, [0, 3, 6, 9])
        self.assertEqual(shard1, [1, 4, 7, 10])
        self.assertEqual(shard2, [2, 5, 8, 11])
        self.assertEqual(sorted(shard0 + shard1 + shard2), list(range(12)))

    def test_phase5_matchup_rotation_uses_global_game_index(self):
        matchups = [("deck-a", "bench-1"), ("deck-b", "bench-2")]

        self.assertEqual(phase5_matchup_for_game_index(matchups, 0), ("deck-a", "bench-1", True))
        self.assertEqual(phase5_matchup_for_game_index(matchups, 1), ("deck-b", "bench-2", False))
        self.assertEqual(phase5_matchup_for_game_index(matchups, 2), ("deck-a", "bench-1", True))

    def test_phase5_merge_search_data_combines_shards_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_a = root / "decisions-a.jsonl"
            shard_b = root / "decisions-b.jsonl"
            trace_a = root / "traces-a.jsonl"
            trace_b = root / "traces-b.jsonl"
            shard_a.write_text('{"decision": 1}\n\n', encoding="utf-8")
            shard_b.write_text('{"decision": 2}\n', encoding="utf-8")
            trace_a.write_text('{"trace": 1}\n', encoding="utf-8")
            trace_b.write_text('{"trace": 2}\n{"trace": 3}\n', encoding="utf-8")

            summary = merge_search_data(
                decision_inputs=[str(root / "decisions-*.jsonl")],
                trace_inputs=[str(root / "traces-*.jsonl")],
                output_path=root / "merged-decisions.jsonl",
                trace_path=root / "merged-traces.jsonl",
                manifest_path=root / "manifest.json",
            )

            self.assertEqual(summary.decision_files, 2)
            self.assertEqual(summary.trace_files, 2)
            self.assertEqual(summary.decision_records, 2)
            self.assertEqual(summary.trace_records, 3)
            self.assertEqual(
                (root / "merged-decisions.jsonl").read_text(encoding="utf-8").splitlines(),
                ['{"decision": 1}', '{"decision": 2}'],
            )
            self.assertTrue((root / "manifest.json").exists())

    def test_phase5_candidate_scoring_can_relabel_baseline_choice(self):
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
        frame = make_decision_frame(obs, selected_indices=[0])
        assert frame is not None
        candidates = _candidate_evaluations(frame, [0], top_k=3)
        for candidate in candidates:
            candidate.tactical_score = 2.0 if candidate.option_index == 1 else 0.0

        _score_candidates(candidates, RootSearchConfig(rule_prior_weight=0.0))
        selected = _best_candidate_indices(candidates, [0])
        relabeled = _replace_frame_selection(
            frame,
            selected,
            {
                "phase5_baseline_indices": [0],
                "phase5_search_indices": selected,
                "phase5_search_changed": selected != [0],
            },
        )

        self.assertEqual(selected, [1])
        self.assertEqual(relabeled.rule_selected_indices, [1])
        self.assertEqual(relabeled.reward_metadata["phase5_baseline_indices"], [0])
        self.assertTrue(relabeled.reward_metadata["phase5_search_changed"])

    def test_phase5_hidden_state_sampler_matches_visible_counts(self):
        own_deck = list(range(1, 61))
        opponent_deck = list(range(101, 161))
        obs = FakeObservation(
            select=FakeSelect(type=0, context=0, minCount=1, maxCount=1, option=[FakeOption(14)]),
            current=FakeCurrent(
                yourIndex=0,
                players=[
                    FakePlayer(
                        active=[FakePokemon(1, 0, energyCards=[FakeCard(2, 0)])],
                        bench=[FakePokemon(3, 0)],
                        hand=[FakeCard(4, 0), FakeCard(5, 0)],
                        handCount=2,
                        discard=[FakeCard(6, 0)],
                        prize=[None, None, None, None, None, None],
                        deckCount=50,
                    ),
                    FakePlayer(
                        active=[FakePokemon(101, 1)],
                        bench=[FakePokemon(102, 1)],
                        hand=None,
                        handCount=5,
                        discard=[FakeCard(103, 1)],
                        prize=[None, None, None, None, None, None],
                        deckCount=47,
                    ),
                ],
            ),
        )

        hidden = sample_hidden_state(
            obs,
            own_deck_ids=own_deck,
            opponent_deck_ids=opponent_deck,
            card_by_id={},
        )

        self.assertEqual(len(hidden.your_deck), 50)
        self.assertEqual(len(hidden.your_prize), 6)
        self.assertEqual(len(hidden.opponent_deck), 47)
        self.assertEqual(len(hidden.opponent_prize), 6)
        self.assertEqual(len(hidden.opponent_hand), 5)
        self.assertEqual(hidden.opponent_active, [])

    def test_phase5_search_distill_diagnostics_focuses_changed_decisions(self):
        changed = _diagnostic_frame(
            baseline=[0],
            search=[1],
            changed=True,
            step_index=1,
        )
        unchanged = _diagnostic_frame(
            baseline=[0],
            search=[0],
            changed=False,
            step_index=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "decisions.jsonl"
            model_path = root / "model.json"
            trace_path = root / "traces.jsonl"
            report_json = root / "report.json"
            report_md = root / "report.md"
            write_decision_jsonl([changed, unchanged], dataset)
            LinearOptionModel(weights={"prefer_baseline": 1.0, "prefer_search": -1.0}).save(
                model_path
            )
            trace_path.write_text(
                "\n".join(
                    [
                        '{"baseline_indices":[0],"search_indices":[1],"changed":true,'
                        '"search_error":null,"candidates":['
                        '{"indices":[0],"combined_score":0.4,"tactical_score":0.2},'
                        '{"indices":[1],"combined_score":0.9,"tactical_score":0.7}]}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = diagnose_search_distillation(
                dataset_path=dataset,
                model_path=model_path,
                trace_path=trace_path,
                report_json_path=report_json,
                report_md_path=report_md,
                example_limit=5,
            )

            payload = report.to_dict()
            self.assertEqual(payload["overall"]["frames"], 2)
            self.assertEqual(payload["search_changed"]["frames"], 1)
            self.assertEqual(payload["search_changed"]["search_hit"], 0)
            self.assertEqual(payload["search_changed"]["baseline_hit"], 1)
            self.assertLess(
                payload["search_changed"]["mean_model_search_minus_baseline_score"],
                0,
            )
            self.assertGreater(
                payload["trace"]["mean_search_minus_baseline_combined_score"],
                0,
            )
            self.assertTrue(payload["examples"])
            self.assertTrue(report_json.exists())
            self.assertTrue(report_md.exists())

    def test_phase5_trace_diagnostics_counts_candidate_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "traces.jsonl"
            trace.write_text(
                "\n".join(
                    [
                        '{"baseline_indices":[0],"search_indices":[1],"changed":true,'
                        '"search_error":"boom","candidates":['
                        '{"indices":[0],"combined_score":1.0,"tactical_score":0.5},'
                        '{"indices":[1],"combined_score":0.5,"tactical_score":0.25,'
                        '"error":"bad","truncated":true}]}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            diagnostics = diagnose_search_traces(trace)

        self.assertEqual(diagnostics.records, 1)
        self.assertEqual(diagnostics.changed_records, 1)
        self.assertEqual(diagnostics.search_errors, 1)
        self.assertEqual(diagnostics.candidate_errors, 1)
        self.assertEqual(diagnostics.truncated_candidates, 1)
        self.assertLess(diagnostics.mean_search_minus_baseline_combined_score, 0)

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
