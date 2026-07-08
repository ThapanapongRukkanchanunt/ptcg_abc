import json
import os
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path

from ptcg_abc.cli import build_parser
from ptcg_abc.rl.phase5_belief import (
    OpponentDeckPrior,
    card_ids_from_value,
    infer_opponent_deck,
    visible_opponent_card_ids,
)
from ptcg_abc.rl.phase5_reports import compare_benchmark_reports
from ptcg_abc.rl.phase5_search import CandidateEvaluation, RootSearchConfig, _score_candidates
from ptcg_abc.rl.workflow import _policy_pool_model_path
from ptcg_abc.submission import PHASE5_SEARCH_MAIN_PY, build_phase5_search_submission_bundle


class Phase5FullAgentScaffoldTests(unittest.TestCase):
    def test_opponent_prior_infers_visible_league_deck(self):
        priors = [
            OpponentDeckPrior(index=1, label="Alpha", card_ids=(1, 1, 2, 3)),
            OpponentDeckPrior(index=2, label="Beta", card_ids=(4, 5, 5, 6)),
        ]
        observation = {
            "current": {
                "yourIndex": 0,
                "players": [
                    {},
                    {
                        "active": [{"id": 4}],
                        "bench": [{"cardID": 5}],
                        "discard": [{"cardId": 5}],
                    },
                ],
            }
        }

        inference = infer_opponent_deck(observation, priors)

        self.assertEqual(visible_opponent_card_ids(observation), [4, 5, 5])
        self.assertEqual(inference.index, 2)
        self.assertEqual(inference.overlap, 3)
        self.assertFalse(inference.used_default)

    def test_opponent_prior_defaults_when_nothing_visible(self):
        priors = [
            OpponentDeckPrior(index=1, label="Alpha", card_ids=(1,)),
            OpponentDeckPrior(
                index=2,
                label="Crustle / Required benchmark sample",
                card_ids=(2,),
            ),
        ]

        inference = infer_opponent_deck({"current": {"players": [{}, {}]}}, priors)

        self.assertEqual(inference.index, 2)
        self.assertTrue(inference.used_default)
        self.assertEqual(card_ids_from_value({"cards": [{"cardIds": [7, 8]}]}), [7, 8])

    def test_search_candidate_scoring_accepts_neural_priors(self):
        candidates = [
            CandidateEvaluation(
                indices=[0],
                option_index=0,
                option_type="PLAY",
                card_name="A",
                attack_id=None,
                rule_score=0.0,
                rule_rank=1,
                tactical_score=0.0,
                policy_score=0.0,
                neural_action_value=0.0,
            ),
            CandidateEvaluation(
                indices=[1],
                option_index=1,
                option_type="PLAY",
                card_name="B",
                attack_id=None,
                rule_score=0.0,
                rule_rank=2,
                tactical_score=0.0,
                policy_score=1.0,
                neural_action_value=3.0,
            ),
        ]

        _score_candidates(
            candidates,
            RootSearchConfig(
                rule_prior_weight=0.0,
                policy_prior_weight=0.5,
                neural_action_value_weight=0.5,
            ),
        )

        self.assertEqual(candidates[0].combined_score, 0.0)
        self.assertEqual(candidates[1].policy_prior, 1.0)
        self.assertEqual(candidates[1].neural_action_value_prior, 1.0)
        self.assertEqual(candidates[1].combined_score, 1.0)

    def test_search_candidate_scoring_accepts_normalized_leaf_value(self):
        candidates = [
            CandidateEvaluation(
                indices=[0],
                option_index=0,
                option_type="END",
                card_name="",
                attack_id=None,
                rule_score=0.0,
                rule_rank=1,
                tactical_score=20.0,
                leaf_state_value=-0.4,
            ),
            CandidateEvaluation(
                indices=[1],
                option_index=1,
                option_type="PLAY",
                card_name="B",
                attack_id=None,
                rule_score=0.0,
                rule_rank=2,
                tactical_score=10.0,
                leaf_state_value=0.6,
            ),
        ]

        _score_candidates(
            candidates,
            RootSearchConfig(
                rule_prior_weight=0.0,
                tactical_score_weight=0.25,
                normalize_tactical_score=True,
                leaf_state_value_weight=0.75,
            ),
        )

        self.assertEqual(candidates[0].tactical_score_prior, 1.0)
        self.assertEqual(candidates[1].leaf_state_value_prior, 1.0)
        self.assertAlmostEqual(candidates[0].combined_score, 0.25)
        self.assertAlmostEqual(candidates[1].combined_score, 0.75)

    def test_policy_pool_rotation_is_deterministic(self):
        paths = [Path("a.pt"), Path("b.pt"), Path("c.pt")]

        self.assertEqual(
            _policy_pool_model_path(
                paths,
                absolute_game_index=0,
                player_slot=0,
                default_model_path=Path("default.pt"),
            ),
            Path("a.pt"),
        )
        self.assertEqual(
            _policy_pool_model_path(
                paths,
                absolute_game_index=1,
                player_slot=1,
                default_model_path=Path("default.pt"),
            ),
            Path("a.pt"),
        )
        self.assertEqual(
            _policy_pool_model_path(
                [],
                absolute_game_index=1,
                player_slot=1,
                default_model_path=Path("default.pt"),
            ),
            Path("default.pt"),
        )

    def test_compare_benchmark_reports_writes_delta(self):
        baseline = {
            "rows": [
                {"deck_index": 1, "archetype": "A", "opponent": "X", "games": 2, "wins": 1},
                {"deck_index": 2, "archetype": "B", "opponent": "X", "games": 2, "wins": 0},
            ]
        }
        candidate = {
            "rows": [
                {"deck_index": 1, "archetype": "A", "opponent": "X", "games": 2, "wins": 2},
                {"deck_index": 2, "archetype": "B", "opponent": "X", "games": 2, "wins": 1},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_path = root / "baseline.json"
            candidate_path = root / "candidate.json"
            output_md = root / "compare.md"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

            payload = compare_benchmark_reports(
                baseline_path=baseline_path,
                candidate_path=candidate_path,
                output_md=output_md,
            )
            markdown = output_md.read_text(encoding="utf-8")

        self.assertEqual(payload["overall_delta"]["wins"], 2)
        self.assertAlmostEqual(payload["overall_delta"]["win_rate"], 0.5)
        self.assertIn("Phase 5 Benchmark Comparison", markdown)

    def test_phase5_package_builds_direct_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "sample"
            sample_dir.joinpath("cg").mkdir(parents=True)
            sample_dir.joinpath("cg", "__init__.py").write_text("", encoding="utf-8")
            model_path = root / "model.pt"
            model_path.write_bytes(b"checkpoint")
            src_root = root / "src"
            src_root.joinpath("ptcg_abc").mkdir(parents=True)
            src_root.joinpath("ptcg_abc", "__init__.py").write_text("", encoding="utf-8")
            zip_path = root / "submission.zip"

            result = build_phase5_search_submission_bundle(
                deck_ids=[1] * 60,
                sample_dir=sample_dir,
                output_dir=root / "out",
                model_path=model_path,
                zip_path=zip_path,
                src_root=src_root,
            )

            with zipfile.ZipFile(result.zip_path) as archive:
                names = set(archive.namelist())

        self.assertIn("main.py", names)
        self.assertIn("deck.csv", names)
        self.assertIn("model.pt", names)
        self.assertIn("cg/__init__.py", names)
        self.assertIn("ptcg_abc/__init__.py", names)

    def test_phase5_template_execs_without_file_global(self):
        fake_cg = types.ModuleType("cg")
        fake_api = types.ModuleType("cg.api")
        fake_api.all_attack = lambda: []
        fake_api.all_card_data = lambda: []
        fake_api.to_observation_class = lambda obs: obs
        fake_agent_module = types.ModuleType("ptcg_abc.agent")
        fake_agent_module.Phase5SearchPolicyAgent = object
        previous = {
            name: sys.modules.get(name)
            for name in ("cg", "cg.api", "ptcg_abc.agent")
        }
        sys.modules["cg"] = fake_cg
        sys.modules["cg.api"] = fake_api
        sys.modules["ptcg_abc.agent"] = fake_agent_module
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.joinpath("cg").mkdir()
            root.joinpath("deck.csv").write_text("\n".join(["1"] * 60), encoding="utf-8")
            old_cwd = os.getcwd()
            try:
                os.chdir(root)
                namespace = {"__name__": "phase5_submission_main"}
                exec(compile(PHASE5_SEARCH_MAIN_PY, "main.py", "exec"), namespace)
                self.assertEqual(namespace["_agent_root"](), str(root))
                self.assertEqual(len(namespace["read_deck_csv"]()), 60)
            finally:
                os.chdir(old_cwd)
                for name, module in previous.items():
                    if module is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = module

    def test_cli_exposes_full_agent_commands(self):
        parser = build_parser()
        league_args = parser.parse_args(
            [
                "rl-evaluate-phase5-league",
                "--agent",
                "phase5-search",
                "--policy-prior-weight",
                "0.1",
                "--neural-action-value-weight",
                "0.2",
                "--normalize-tactical-score",
                "--tactical-score-weight",
                "0.5",
                "--leaf-state-value-weight",
                "0.75",
                "--specialist-model-dir",
                "models/rl/phase5_league_alpha/iter-0000/specialists",
            ]
        )
        ppo_args = parser.parse_args(
            [
                "rl-train-phase5-ppo",
                "--trajectory-dataset",
                "selfplay.jsonl",
                "--checkpoint",
                "model.pt",
            ]
        )
        compare_args = parser.parse_args(
            [
                "phase5-compare-benchmarks",
                "--baseline",
                "old.json",
                "--candidate",
                "new.json",
            ]
        )
        selfplay_args = parser.parse_args(
            [
                "rl-generate-phase5-search-selfplay",
                "--model",
                "model.pt",
                "--policy-pool-model",
                "old.pt",
                "--policy-pool-model",
                "new.pt",
            ]
        )
        package_args = parser.parse_args(
            [
                "phase5-package",
                "--deck-pool",
                "league-13",
                "--deck-index",
                "11",
                "--model-dir",
                "models/rl/phase5_league_alpha/iter-0000/specialists",
            ]
        )

        self.assertEqual(league_args.policy_prior_weight, 0.1)
        self.assertEqual(league_args.neural_action_value_weight, 0.2)
        self.assertTrue(league_args.normalize_tactical_score)
        self.assertEqual(league_args.tactical_score_weight, 0.5)
        self.assertEqual(league_args.leaf_state_value_weight, 0.75)
        self.assertEqual(
            league_args.specialist_model_dir,
            Path("models/rl/phase5_league_alpha/iter-0000/specialists"),
        )
        self.assertEqual(ppo_args.func.__name__, "command_rl_train_phase5_ppo")
        self.assertEqual(compare_args.func.__name__, "command_phase5_compare_benchmarks")
        self.assertEqual(selfplay_args.policy_pool_model, [Path("old.pt"), Path("new.pt")])
        self.assertEqual(package_args.deck_pool, "league-13")
        self.assertEqual(package_args.deck_index, [11])
        self.assertEqual(
            package_args.model_dir,
            Path("models/rl/phase5_league_alpha/iter-0000/specialists"),
        )


if __name__ == "__main__":
    unittest.main()
