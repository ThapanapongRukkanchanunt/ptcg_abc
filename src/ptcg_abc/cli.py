from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from ptcg_abc.config import (
    COMPETITION_SLUG,
    KAGGLE_INPUT_DIR,
    KAGGLE_RAW_DIR,
    LEGAL_CARDS_PATH,
    LIMITLESS_FORMAT,
    PROCESSED_DIR,
    REPORTS_DIR,
)
from ptcg_abc.card_db import load_card_id_lookup
from ptcg_abc.corpus import (
    deck_card_names,
    deck_label,
    deck_to_card_ids,
    get_deck_by_index,
    load_deck_corpus,
    write_deck_csv,
)
from ptcg_abc.evaluation import (
    Phase3CloseoutResult,
    choose_deck_from_best_archetype,
    phase5_league_prepared_decks,
    phase3_tournament_559_prepared_decks,
    prepare_decks,
    required_phase3_prepared_decks,
    run_archetype_sweep,
    run_phase3_required_benchmark,
    run_random_evaluation,
    run_sample_dragapult_benchmark,
    sample_dragapult_benchmark_from_dict,
    write_phase3_required_benchmark_report,
    write_sample_dragapult_comparison_report,
    write_sample_dragapult_benchmark_report,
    write_closeout_reports,
)
from ptcg_abc.kaggle_api import KaggleCredentialsError, setup_kaggle_data
from ptcg_abc.legal_cards import (
    choose_legal_card_candidate,
    discover_legal_card_candidates,
    load_legal_cards,
    write_candidate_report,
    write_legal_cards,
)
from ptcg_abc.simulator import run_battle_smoke
from ptcg_abc.submission import (
    build_hybrid_rl_submission_bundle,
    build_phase5_search_submission_bundle,
    build_submission_bundle,
)
from ptcg_abc.rl.workflow import (
    PHASE5_SELFPLAY_DECK_POOLS,
    collect_bc_demonstrations,
    rollout_games,
    rollout_selfplay_games,
    run_image_progression_experiment,
    run_phase5_league_benchmark,
    run_phase4_required_benchmark,
    train_bc_from_jsonl,
    train_ppo_from_rollouts,
    train_torch_bc_from_jsonl,
    write_phase4_benchmark_report,
)
from ptcg_abc.rl.phase5_search import (
    RootSearchConfig,
    generate_search_improved_data,
    merge_search_data,
)
from ptcg_abc.rl.phase5_diagnostics import (
    diagnose_search_distillation,
    diagnose_search_score_components,
    diagnose_search_traces,
    write_search_score_component_markdown,
    write_trace_diagnostic_markdown,
)
from ptcg_abc.rl.phase5_policy import Phase5PolicyUnavailable
from ptcg_abc.rl.phase5_alpha_league import (
    PHASE5_ALPHA_DECK_INDICES,
    cleanup_phase5_alpha_raw_train,
    generate_phase5_alpha_league_iteration,
    generate_phase5_alpha_rule_bootstrap,
    train_phase5_deck_specialists,
    train_phase5_deck_specialists_ppo,
)
from ptcg_abc.rl.public_opponents import (
    PublicAgentTacticalRewardConfig,
    discover_phase5_public_opponents,
    format_public_agent_gate_markdown,
    generate_phase5_public_agent_trajectories,
    run_phase5_public_agent_benchmark,
    summarize_public_agent_gate,
    write_public_agent_status_report,
    write_public_agent_trajectory_report,
)
from ptcg_abc.rl.phase5_reports import compare_benchmark_reports
from ptcg_abc.rl.snapshots import run_rule_vs_benchmark_snapshots
from ptcg_abc.rl.phase5_symbolic_diagnostics import diagnose_phase5_symbolic_policy
from ptcg_abc.rl.phase5_symbolic_training import (
    build_phase5_symbolic_dataset,
    initialize_phase5_policy_checkpoint,
    train_phase5_bc_policy_from_trajectories,
    train_phase5_bc_ppo_policy_from_trajectories,
    train_phase5_ppo_policy_from_trajectories,
    train_phase5_generalist_policy,
    train_phase5_symbolic_policy_from_decisions,
)
from ptcg_abc.rl.torch_backend import TorchBackendUnavailable


def _path(value: str) -> Path:
    return Path(value)


def _slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "-" for char in value]
    return "-".join(part for part in "".join(chars).split("-") if part)


def _root_search_config_from_args(args: argparse.Namespace) -> RootSearchConfig | None:
    fields = {
        "top_k": getattr(args, "search_top_k", None),
        "max_rollout_steps": getattr(args, "search_rollout_steps", None),
        "tactical_score_weight": getattr(args, "tactical_score_weight", None),
        "normalize_tactical_score": (
            True if getattr(args, "normalize_tactical_score", False) else None
        ),
        "policy_prior_weight": getattr(args, "policy_prior_weight", None),
        "neural_action_value_weight": getattr(args, "neural_action_value_weight", None),
        "neural_tactical_weight": getattr(args, "neural_tactical_weight", None),
        "leaf_state_value_weight": getattr(args, "leaf_state_value_weight", None),
    }
    if all(value is None for value in fields.values()):
        return None
    base_config = RootSearchConfig()
    return RootSearchConfig(
        top_k=fields["top_k"] if fields["top_k"] is not None else base_config.top_k,
        max_rollout_steps=(
            fields["max_rollout_steps"]
            if fields["max_rollout_steps"] is not None
            else base_config.max_rollout_steps
        ),
        tactical_score_weight=(
            fields["tactical_score_weight"]
            if fields["tactical_score_weight"] is not None
            else base_config.tactical_score_weight
        ),
        normalize_tactical_score=(
            fields["normalize_tactical_score"]
            if fields["normalize_tactical_score"] is not None
            else base_config.normalize_tactical_score
        ),
        policy_prior_weight=(
            fields["policy_prior_weight"]
            if fields["policy_prior_weight"] is not None
            else base_config.policy_prior_weight
        ),
        neural_action_value_weight=(
            fields["neural_action_value_weight"]
            if fields["neural_action_value_weight"] is not None
            else base_config.neural_action_value_weight
        ),
        neural_tactical_weight=(
            fields["neural_tactical_weight"]
            if fields["neural_tactical_weight"] is not None
            else base_config.neural_tactical_weight
        ),
        leaf_state_value_weight=(
            fields["leaf_state_value_weight"]
            if fields["leaf_state_value_weight"] is not None
            else base_config.leaf_state_value_weight
        ),
    )


def _add_common_limitless_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--snapshot-date", default=date.today().isoformat())
    parser.add_argument("--top-archetypes", type=int, default=10)
    parser.add_argument("--lists-per-variant", type=int, default=2)
    parser.add_argument("--candidate-limit", type=int, default=250)
    parser.add_argument("--delay-seconds", type=float, default=0.2)
    parser.add_argument("--limitless-format", default=LIMITLESS_FORMAT)
    parser.add_argument("--refresh", action="store_true")


def _add_phase5_search_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--search-top-k",
        type=int,
        default=None,
        help="Override phase5-search root candidate count.",
    )
    parser.add_argument(
        "--search-rollout-steps",
        type=int,
        default=None,
        help="Override phase5-search one-turn rollout step cap.",
    )
    parser.add_argument(
        "--policy-prior-weight",
        type=float,
        default=None,
        help="Optional normalized policy-logit prior weight inside root-search scoring.",
    )
    parser.add_argument(
        "--tactical-score-weight",
        type=float,
        default=None,
        help="Weight for root-search rollout tactical score.",
    )
    parser.add_argument(
        "--normalize-tactical-score",
        action="store_true",
        help="Normalize rollout tactical score across candidates before scoring.",
    )
    parser.add_argument(
        "--neural-action-value-weight",
        type=float,
        default=None,
        help="Optional normalized action-Q prior weight inside root-search scoring.",
    )
    parser.add_argument(
        "--neural-tactical-weight",
        type=float,
        default=None,
        help="Optional normalized neural tactical-head prior weight inside root-search scoring.",
    )
    parser.add_argument(
        "--leaf-state-value-weight",
        type=float,
        default=None,
        help="Optional normalized leaf state-value weight inside root-search scoring.",
    )


def command_kaggle_setup(args: argparse.Namespace) -> int:
    try:
        setup = setup_kaggle_data(
            args.raw_dir,
            args.input_dir,
            competition=args.competition,
            archive_path=args.archive,
            refresh=args.refresh,
        )
    except KaggleCredentialsError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    candidates = discover_legal_card_candidates(args.legal_source or args.input_dir)
    if args.candidate_report:
        write_candidate_report(candidates, args.candidate_report)

    candidate = choose_legal_card_candidate(args.input_dir, source=args.legal_source)
    write_legal_cards(candidate.names, args.legal_cards)
    print(json.dumps(setup, indent=2))
    print(
        f"Wrote {candidate.count} legal card names to {args.legal_cards} "
        f"from {candidate.path}."
    )
    return 0


def command_discover_legal_cards(args: argparse.Namespace) -> int:
    candidates = discover_legal_card_candidates(args.input_dir)
    if args.candidate_report:
        write_candidate_report(candidates, args.candidate_report)
    candidate = choose_legal_card_candidate(args.input_dir, source=args.legal_source)
    write_legal_cards(candidate.names, args.legal_cards)
    print(
        f"Wrote {candidate.count} legal card names to {args.legal_cards} "
        f"from {candidate.path}."
    )
    return 0


def command_missing_limitless(args: argparse.Namespace) -> int:
    from ptcg_abc.limitless import (
        collect_limitless_decks,
        deck_collection_summary,
        write_deck_collection,
        write_missing_report,
    )

    if not args.legal_cards.exists():
        print(
            f"Legal card list not found at {args.legal_cards}. "
            "Run `python -m ptcg_abc kaggle-setup` first.",
            file=sys.stderr,
        )
        return 2

    legal_cards = load_legal_cards(args.legal_cards)
    collection = collect_limitless_decks(
        snapshot_date=args.snapshot_date,
        top_archetypes=args.top_archetypes,
        lists_per_variant=args.lists_per_variant,
        refresh=args.refresh,
        candidate_limit=args.candidate_limit,
        delay_seconds=args.delay_seconds,
        limitless_format=args.limitless_format,
    )
    write_deck_collection(
        collection,
        snapshot_date=args.snapshot_date,
        limitless_format=args.limitless_format,
        top_archetypes=args.top_archetypes,
        lists_per_variant=args.lists_per_variant,
        candidate_limit=args.candidate_limit,
    )
    report_path = write_missing_report(
        collection, legal_cards, args.output, limitless_format=args.limitless_format
    )
    print(deck_collection_summary(collection))
    print(f"Wrote missing-card report to {report_path}.")
    return 0


def command_collect_corpus(args: argparse.Namespace) -> int:
    from ptcg_abc.limitless import (
        collect_limitless_decks,
        deck_collection_summary,
        write_deck_collection,
    )

    collection = collect_limitless_decks(
        snapshot_date=args.snapshot_date,
        top_archetypes=args.top_archetypes,
        lists_per_variant=args.lists_per_variant,
        refresh=args.refresh,
        candidate_limit=args.candidate_limit,
        delay_seconds=args.delay_seconds,
        limitless_format=args.limitless_format,
    )
    outputs = write_deck_collection(
        collection,
        snapshot_date=args.snapshot_date,
        limitless_format=args.limitless_format,
        top_archetypes=args.top_archetypes,
        lists_per_variant=args.lists_per_variant,
        candidate_limit=args.candidate_limit,
    )
    print(deck_collection_summary(collection))
    print(f"Wrote corpus outputs to {outputs['output_dir']}.")
    return 0


def command_agent_smoke(args: argparse.Namespace) -> int:
    if not args.corpus.exists():
        print(
            f"Deck corpus not found at {args.corpus}. "
            "Run `python -m ptcg_abc collect-corpus` first.",
            file=sys.stderr,
        )
        return 2
    if not args.card_data.exists():
        print(
            f"Kaggle card data not found at {args.card_data}. "
            "Run `python -m ptcg_abc kaggle-setup` first.",
            file=sys.stderr,
        )
        return 2
    if not args.sample_dir.exists():
        print(
            f"Kaggle sample submission not found at {args.sample_dir}. "
            "Run `python -m ptcg_abc kaggle-setup` first.",
            file=sys.stderr,
        )
        return 2

    decks = load_deck_corpus(args.corpus)
    if len(decks) < 2:
        print("The corpus needs at least two decks for a battle smoke check.", file=sys.stderr)
        return 2

    try:
        deck0 = get_deck_by_index(decks, args.deck_index)
        deck1 = get_deck_by_index(decks, args.opponent_index)
    except IndexError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    lookup = load_card_id_lookup(args.card_data)
    try:
        deck0_ids = deck_to_card_ids(deck0, lookup)
        deck1_ids = deck_to_card_ids(deck1, lookup)
    except (KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.write_deck_csv:
        write_deck_csv(deck0_ids, args.write_deck_csv)

    selected_names = list(deck_card_names(deck0)) + list(deck_card_names(deck1))
    ambiguous = lookup.ambiguous_names(selected_names)
    result = run_battle_smoke(
        deck0_ids,
        deck1_ids,
        sample_dir=args.sample_dir,
        max_steps=args.max_steps,
    )

    print(f"Deck 0: {deck_label(deck0)}")
    print(f"Deck 1: {deck_label(deck1)}")
    print(f"Ambiguous names resolved by lowest Kaggle Card ID: {len(ambiguous)}")
    for name, card_ids in list(ambiguous.items())[:10]:
        print(f"  {name}: {', '.join(str(card_id) for card_id in card_ids)}")
    if len(ambiguous) > 10:
        print(f"  ... {len(ambiguous) - 10} more")
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.started and result.error is None else 1


def command_phase3_closeout(args: argparse.Namespace) -> int:
    if not args.corpus.exists():
        print(
            f"Deck corpus not found at {args.corpus}. "
            "Run `python -m ptcg_abc collect-corpus` first.",
            file=sys.stderr,
        )
        return 2
    if not args.card_data.exists():
        print(
            f"Kaggle card data not found at {args.card_data}. "
            "Run `python -m ptcg_abc kaggle-setup` first.",
            file=sys.stderr,
        )
        return 2
    if not args.sample_dir.exists():
        print(
            f"Kaggle sample submission not found at {args.sample_dir}. "
            "Run `python -m ptcg_abc kaggle-setup` first.",
            file=sys.stderr,
        )
        return 2

    decks = load_deck_corpus(args.corpus)
    lookup = load_card_id_lookup(args.card_data)
    try:
        prepared = prepare_decks(decks, lookup)
    except (KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if len({deck.archetype for deck in prepared}) < 2:
        print("The corpus needs at least two archetypes for the Phase 3 sweep.", file=sys.stderr)
        return 2

    print(
        f"Running archetype sweep: archetypes={len({deck.archetype for deck in prepared})} "
        f"games_per_matchup={args.games_per_matchup} max_steps={args.max_steps}"
    )
    scores, matchups = run_archetype_sweep(
        prepared,
        sample_dir=args.sample_dir,
        games_per_matchup=args.games_per_matchup,
        max_steps=args.max_steps,
    )
    selected = choose_deck_from_best_archetype(prepared, scores, seed=args.seed)

    print(
        f"Testing selected archetype against random agent: "
        f"{selected.archetype}, games={args.random_games}"
    )
    random_eval = run_random_evaluation(
        selected,
        sample_dir=args.sample_dir,
        games=args.random_games,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    submission = build_submission_bundle(
        deck_ids=selected.card_ids,
        sample_dir=args.sample_dir,
        output_dir=args.output_dir,
        tar_path=args.submission_tar,
    )

    result = Phase3CloseoutResult(
        selected_archetype=selected.archetype,
        selected_deck_index=selected.index,
        selected_deck_label=selected.label,
        submission_tar=str(submission.tar_path.as_posix()),
        random_eval=random_eval,
        archetype_scores=scores,
        matchups=matchups,
        games_per_matchup=args.games_per_matchup,
        random_games=args.random_games,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    write_closeout_reports(result, json_path=args.report_json, markdown_path=args.report_md)

    print(json.dumps(result.to_dict(), indent=2))
    print(f"Wrote submission bundle to {submission.tar_path}.")
    print(f"Wrote closeout report to {args.report_md}.")
    return 0


def command_benchmark_sample_dragapult(args: argparse.Namespace) -> int:
    if not args.corpus.exists():
        print(
            f"Deck corpus not found at {args.corpus}. "
            "Run `python -m ptcg_abc collect-corpus` first.",
            file=sys.stderr,
        )
        return 2
    if not args.card_data.exists():
        print(
            f"Kaggle card data not found at {args.card_data}. "
            "Run `python -m ptcg_abc kaggle-setup` first.",
            file=sys.stderr,
        )
        return 2
    if not args.sample_dir.exists():
        print(
            f"Kaggle sample submission not found at {args.sample_dir}. "
            "Run `python -m ptcg_abc kaggle-setup` first.",
            file=sys.stderr,
        )
        return 2

    decks = load_deck_corpus(args.corpus)
    lookup = load_card_id_lookup(args.card_data)
    try:
        prepared = prepare_decks(decks, lookup)
    except (KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not args.skip_required_decks:
        prepared.extend(required_phase3_prepared_decks(start_index=len(prepared) + 1))

    print(
        f"Running sample Dragapult benchmark: decks={len(prepared)} "
        f"games_per_deck={args.games_per_deck} max_steps={args.max_steps}"
    )
    result = run_sample_dragapult_benchmark(
        prepared,
        sample_dir=args.sample_dir,
        games_per_deck=args.games_per_deck,
        max_steps=args.max_steps,
        debug_limit_per_deck=args.debug_limit_per_deck,
        trace_limit=args.trace_limit,
    )
    write_sample_dragapult_benchmark_report(
        result,
        json_path=args.report_json,
        markdown_path=args.report_md,
    )
    if args.baseline_json and args.baseline_json.exists():
        baseline = sample_dragapult_benchmark_from_dict(
            json.loads(args.baseline_json.read_text(encoding="utf-8"))
        )
        write_sample_dragapult_comparison_report(
            baseline,
            result,
            markdown_path=args.comparison_md,
        )
        print(f"Wrote benchmark comparison to {args.comparison_md}.")
    games = sum(row.games for row in result.rows)
    wins = sum(row.wins for row in result.rows)
    losses = sum(row.losses for row in result.rows)
    draws = sum(row.draws for row in result.rows)
    timeouts = sum(row.timeouts for row in result.rows)
    errors = sum(row.errors for row in result.rows)
    print(
        json.dumps(
            {
                "games": games,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "timeouts": timeouts,
                "errors": errors,
                "win_rate": wins / games if games else 0.0,
                "debug_games": len(result.debug_games),
            },
            indent=2,
        )
    )
    print(f"Wrote benchmark report to {args.report_md}.")
    return 0


def command_benchmark_phase3_required(args: argparse.Namespace) -> int:
    if not args.sample_dir.exists():
        print(
            f"Kaggle sample submission not found at {args.sample_dir}. "
            "Run `python -m ptcg_abc kaggle-setup` first.",
            file=sys.stderr,
        )
        return 2

    our_decks = phase3_tournament_559_prepared_decks()
    benchmark_decks = required_phase3_prepared_decks(start_index=1)
    print(
        f"Running Phase 3 required benchmark: our_decks={len(our_decks)} "
        f"benchmark_decks={len(benchmark_decks)} "
        f"games_per_matchup={args.games_per_matchup} max_steps={args.max_steps}"
    )
    result = run_phase3_required_benchmark(
        our_decks,
        benchmark_decks,
        sample_dir=args.sample_dir,
        games_per_matchup=args.games_per_matchup,
        max_steps=args.max_steps,
        debug_limit_per_matchup=args.debug_limit_per_matchup,
        trace_limit=args.trace_limit,
    )
    write_phase3_required_benchmark_report(
        result,
        json_path=args.report_json,
        markdown_path=args.report_md,
    )
    games = sum(row.games for row in result.rows)
    wins = sum(row.wins for row in result.rows)
    losses = sum(row.losses for row in result.rows)
    draws = sum(row.draws for row in result.rows)
    timeouts = sum(row.timeouts for row in result.rows)
    errors = sum(row.errors for row in result.rows)
    print(
        json.dumps(
            {
                "games": games,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "timeouts": timeouts,
                "errors": errors,
                "win_rate": wins / games if games else 0.0,
                "debug_games": len(result.debug_games),
            },
            indent=2,
        )
    )
    print(f"Wrote Phase 3 required benchmark report to {args.report_md}.")
    return 0


def command_rl_collect_bc(args: argparse.Namespace) -> int:
    if not args.sample_dir.exists():
        print(
            f"Kaggle sample submission not found at {args.sample_dir}. "
            "Run `python -m ptcg_abc kaggle-setup` first.",
            file=sys.stderr,
        )
        return 2
    summary = collect_bc_demonstrations(
        sample_dir=args.sample_dir,
        output_path=args.output,
        games=args.games,
        max_steps=args.max_steps,
    )
    print(json.dumps(summary.to_dict(), indent=2))
    return 0 if summary.errors == 0 else 1


def command_rl_train_bc(args: argparse.Namespace) -> int:
    if not args.dataset.exists():
        print(f"BC dataset not found at {args.dataset}. Run `rl-collect-bc` first.", file=sys.stderr)
        return 2
    try:
        if args.backend == "torch":
            summary = train_torch_bc_from_jsonl(
                dataset_path=args.dataset,
                checkpoint_path=args.checkpoint,
                export_model_path=args.model,
                report_path=args.report_json,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                changed_weight=args.changed_weight,
                unchanged_weight=args.unchanged_weight,
                excluded_features=args.exclude_feature,
                pairwise_changed=args.pairwise_changed,
                pairwise_margin=args.pairwise_margin,
                pairwise_negatives=args.pairwise_negatives,
            )
        else:
            summary = train_bc_from_jsonl(
                dataset_path=args.dataset,
                model_path=args.model,
                report_path=args.report_json,
                epochs=args.epochs,
                changed_weight=args.changed_weight,
                unchanged_weight=args.unchanged_weight,
                excluded_features=args.exclude_feature,
                pairwise_changed=args.pairwise_changed,
                pairwise_margin=args.pairwise_margin,
                pairwise_negatives=args.pairwise_negatives,
            )
    except TorchBackendUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


def command_rl_build_phase5_symbolic(args: argparse.Namespace) -> int:
    if not args.dataset.exists():
        print(f"Phase 5 decision dataset not found at {args.dataset}.", file=sys.stderr)
        return 2
    limit = None if args.limit == 0 else args.limit
    summary = build_phase5_symbolic_dataset(
        dataset_path=args.dataset,
        output_path=args.output,
        limit=limit,
        max_entities=args.max_entities,
        max_actions=args.max_actions,
        max_previous_actions=args.max_previous_actions,
        target_source=args.target_source,
        changed_weight=args.changed_weight,
        unchanged_weight=args.unchanged_weight,
    )
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


def command_rl_train_phase5_symbolic(args: argparse.Namespace) -> int:
    if not args.dataset.exists():
        print(f"Phase 5 decision dataset not found at {args.dataset}.", file=sys.stderr)
        return 2
    limit = None if args.limit == 0 else args.limit
    try:
        summary = train_phase5_symbolic_policy_from_decisions(
            dataset_path=args.dataset,
            checkpoint_path=args.checkpoint,
            report_path=args.report_json,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_entities=args.max_entities,
            max_actions=args.max_actions,
            max_previous_actions=args.max_previous_actions,
            d_model=args.d_model,
            target_source=args.target_source,
            changed_weight=args.changed_weight,
            unchanged_weight=args.unchanged_weight,
            pairwise_changed=args.pairwise_changed,
            pairwise_weight=args.pairwise_weight,
            pairwise_margin=args.pairwise_margin,
            pairwise_negatives=args.pairwise_negatives,
            value_loss_weight=args.value_loss_weight,
            limit=limit,
            changed_only=args.changed_only,
        )
    except Phase5PolicyUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


def command_rl_train_phase5_generalist(args: argparse.Namespace) -> int:
    decision_dataset = args.decision_dataset
    if args.no_decision_dataset:
        decision_dataset = None
    if decision_dataset is not None and not decision_dataset.exists():
        print(f"Phase 5 decision dataset not found at {decision_dataset}.", file=sys.stderr)
        return 2
    selfplay_datasets = list(args.selfplay_dataset or [])
    for path in selfplay_datasets:
        if not path.exists():
            print(f"Phase 5 self-play dataset not found at {path}.", file=sys.stderr)
            return 2
    decision_limit = None if args.decision_limit == 0 else args.decision_limit
    selfplay_limit = None if args.selfplay_limit == 0 else args.selfplay_limit
    try:
        summary = train_phase5_generalist_policy(
            decision_dataset_path=decision_dataset,
            selfplay_dataset_paths=selfplay_datasets,
            checkpoint_path=args.checkpoint,
            initial_checkpoint_path=args.initial_checkpoint,
            report_path=args.report_json,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_entities=args.max_entities,
            max_actions=args.max_actions,
            max_previous_actions=args.max_previous_actions,
            d_model=args.d_model,
            target_source=args.target_source,
            rule_demo_target_source=args.rule_demo_target_source,
            changed_weight=args.changed_weight,
            unchanged_weight=args.unchanged_weight,
            search_decision_weight=args.search_decision_weight,
            rule_demo_weight=args.rule_demo_weight,
            selfplay_weight=args.selfplay_weight,
            pairwise_changed=args.pairwise_changed,
            pairwise_weight=args.pairwise_weight,
            pairwise_margin=args.pairwise_margin,
            pairwise_negatives=args.pairwise_negatives,
            value_loss_weight=args.value_loss_weight,
            action_value_loss_weight=args.action_value_loss_weight,
            tactical_loss_weight=args.tactical_loss_weight,
            decision_limit=decision_limit,
            selfplay_limit=selfplay_limit,
            deck_index_filter=args.deck_index_filter,
        )
    except Phase5PolicyUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


def command_rl_rollout(args: argparse.Namespace) -> int:
    if not args.sample_dir.exists():
        print(
            f"Kaggle sample submission not found at {args.sample_dir}. "
            "Run `python -m ptcg_abc kaggle-setup` first.",
            file=sys.stderr,
        )
        return 2
    model_path = args.model if args.model and args.model.exists() else None
    if args.agent in {"phase5-symbolic", "phase5-search", "phase5-full", "phase5-rl"} and model_path is None:
        print(
            f"Phase 5 checkpoint not found at {args.model}.",
            file=sys.stderr,
        )
        return 2
    summary = rollout_games(
        sample_dir=args.sample_dir,
        output_path=args.output,
        agent_kind=args.agent,
        model_path=model_path,
        games=args.games,
        max_steps=args.max_steps,
    )
    print(json.dumps(summary.to_dict(), indent=2))
    return 0 if summary.errors == 0 else 1


def command_rl_generate_phase5_search_selfplay(args: argparse.Namespace) -> int:
    if not args.sample_dir.exists():
        print(
            f"Kaggle sample submission not found at {args.sample_dir}. "
            "Run `python -m ptcg_abc kaggle-setup` first.",
            file=sys.stderr,
        )
        return 2
    model_path = args.model if args.model and args.model.exists() else None
    if model_path is None:
        print(f"Phase 5 checkpoint not found at {args.model}.", file=sys.stderr)
        return 2
    search_config = _root_search_config_from_args(args)
    try:
        summary = rollout_selfplay_games(
            sample_dir=args.sample_dir,
            output_path=args.output,
            agent_kind="phase5-search",
            model_path=model_path,
            games=args.games,
            game_offset=args.game_offset,
            max_steps=args.max_steps,
            deck_pool=args.deck_pool,
            deck_a_index=args.deck_a_index,
            deck_b_index=args.deck_b_index,
            selfplay_deck_indices=args.deck_index or None,
            search_config=search_config,
            search_trace_path=args.search_trace_output,
            search_trace_game_limit=args.search_trace_games,
            policy_pool_paths=args.policy_pool_model or (),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    payload = summary.to_dict()
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    print(f"Wrote Phase 5 search self-play report to {args.report_json}.")
    return 0 if summary.errors == 0 else 1


def command_rl_generate_search_data(args: argparse.Namespace) -> int:
    if not args.sample_dir.exists():
        print(
            f"Kaggle sample submission not found at {args.sample_dir}. "
            "Run `python -m ptcg_abc kaggle-setup` first.",
            file=sys.stderr,
        )
        return 2
    config = RootSearchConfig(
        top_k=args.top_k,
        max_rollout_steps=args.rollout_steps,
        min_candidates=args.min_candidates,
    )
    summary = generate_search_improved_data(
        sample_dir=args.sample_dir,
        output_path=args.output,
        trace_path=args.trace_output,
        games=args.games,
        game_offset=args.game_offset,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        max_steps=args.max_steps,
        append=args.append,
        config=config,
    )
    print(json.dumps(summary.to_dict(), indent=2))
    if summary.errors:
        return 1
    if args.require_changed and summary.changed_decisions <= 0:
        print(
            "No Phase 5 root-search decision changes were observed. "
            "Increase --games, --max-steps, --top-k, or --rollout-steps.",
            file=sys.stderr,
        )
        return 1
    return 0


def command_rl_merge_search_data(args: argparse.Namespace) -> int:
    try:
        summary = merge_search_data(
            decision_inputs=args.input,
            trace_inputs=args.trace_input,
            output_path=args.output,
            trace_path=args.trace_output,
            manifest_path=args.manifest,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


def command_rl_diagnose_search_distill(args: argparse.Namespace) -> int:
    if not args.dataset.exists():
        print(
            f"Phase 5 search dataset not found at {args.dataset}. "
            "Run `rl-merge-search-data` first.",
            file=sys.stderr,
        )
        return 2
    checkpoint_path = args.checkpoint if args.checkpoint and args.checkpoint.exists() else None
    if args.checkpoint and checkpoint_path is None:
        print(f"Phase 5 torch checkpoint not found at {args.checkpoint}.", file=sys.stderr)
        return 2
    model_path = args.model if args.model and args.model.exists() else None
    if checkpoint_path is None and model_path is None:
        print(
            f"Phase 5 exported model not found at {args.model}. "
            "Set --checkpoint to diagnose a torch checkpoint, or run "
            "`rl-train-bc --backend torch` first.",
            file=sys.stderr,
        )
        return 2
    trace_path = args.trace_input if args.trace_input and args.trace_input.exists() else None
    try:
        diagnostics = diagnose_search_distillation(
            dataset_path=args.dataset,
            model_path=model_path,
            checkpoint_path=checkpoint_path,
            trace_path=trace_path,
            report_json_path=args.report_json,
            report_md_path=args.report_md,
            example_limit=args.examples,
        )
    except (TorchBackendUnavailable, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(diagnostics.to_dict(), indent=2))
    print(f"Wrote Phase 5 search-distillation diagnostics to {args.report_md}.")
    return 0


def command_rl_diagnose_search_traces(args: argparse.Namespace) -> int:
    if not args.trace_input.exists():
        print(f"Phase 5 search trace file not found at {args.trace_input}.", file=sys.stderr)
        return 2
    diagnostics = diagnose_search_traces(args.trace_input)
    payload = diagnostics.to_dict()
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_trace_diagnostic_markdown(
        diagnostics,
        args.report_md,
        trace_path=args.trace_input,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote Phase 5 search trace diagnostics to {args.report_md}.")
    return 0


def command_rl_diagnose_search_score_components(args: argparse.Namespace) -> int:
    if not args.trace_input.exists():
        print(f"Phase 5 search trace file not found at {args.trace_input}.", file=sys.stderr)
        return 2
    diagnostics = diagnose_search_score_components(args.trace_input)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_search_score_component_markdown(
        diagnostics,
        args.report_md,
        trace_path=args.trace_input,
    )
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    print(f"Wrote Phase 5 search score-component diagnostics to {args.report_md}.")
    return 0


def command_rl_diagnose_phase5_symbolic(args: argparse.Namespace) -> int:
    if not args.dataset.exists():
        print(f"Phase 5 decision dataset not found at {args.dataset}.", file=sys.stderr)
        return 2
    if not args.checkpoint.exists():
        print(f"Phase 5 symbolic checkpoint not found at {args.checkpoint}.", file=sys.stderr)
        return 2
    limit = None if args.limit == 0 else args.limit
    try:
        diagnostics = diagnose_phase5_symbolic_policy(
            dataset_path=args.dataset,
            checkpoint_path=args.checkpoint,
            report_json_path=args.report_json,
            report_md_path=args.report_md,
            limit=limit,
            batch_size=args.batch_size,
            example_limit=args.examples,
        )
    except (Phase5PolicyUnavailable, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(diagnostics.to_dict(), indent=2))
    print(f"Wrote Phase 5 symbolic diagnostics to {args.report_md}.")
    return 0


def command_rl_train_ppo(args: argparse.Namespace) -> int:
    if not args.rollout.exists():
        print(f"Rollout dataset not found at {args.rollout}. Run `rl-rollout` first.", file=sys.stderr)
        return 2
    summary = train_ppo_from_rollouts(
        rollout_path=args.rollout,
        model_path=args.model,
        output_path=args.output_model,
        report_path=args.report_json,
        epochs=args.epochs,
    )
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


def command_rl_train_phase5_ppo(args: argparse.Namespace) -> int:
    if not args.checkpoint.exists():
        print(f"Phase 5 checkpoint not found at {args.checkpoint}.", file=sys.stderr)
        return 2
    trajectory_datasets = list(args.trajectory_dataset or [])
    for path in trajectory_datasets:
        if not path.exists():
            print(f"Phase 5 trajectory dataset not found at {path}.", file=sys.stderr)
            return 2
    selfplay_limit = None if args.selfplay_limit == 0 else args.selfplay_limit
    try:
        summary = train_phase5_ppo_policy_from_trajectories(
            trajectory_dataset_paths=trajectory_datasets,
            checkpoint_path=args.checkpoint,
            output_checkpoint_path=args.output_checkpoint,
            report_path=args.report_json,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            clip_epsilon=args.clip_epsilon,
            policy_loss_weight=args.policy_loss_weight,
            value_loss_weight=args.value_loss_weight,
            entropy_weight=args.entropy_weight,
            selfplay_limit=selfplay_limit,
            deck_index_filter=args.deck_index_filter,
            require_on_policy=args.require_on_policy,
        )
    except (Phase5PolicyUnavailable, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


def command_rl_train_phase5_bc_ppo(args: argparse.Namespace) -> int:
    if not args.checkpoint.exists():
        print(f"Phase 5 checkpoint not found at {args.checkpoint}.", file=sys.stderr)
        return 2
    rule_datasets = list(args.rule_trajectory_dataset or [])
    on_policy_datasets = list(args.on_policy_trajectory_dataset or [])
    for path in [*rule_datasets, *on_policy_datasets]:
        if not path.exists():
            print(f"Phase 5 trajectory dataset not found at {path}.", file=sys.stderr)
            return 2
    rule_limit = None if args.rule_step_limit == 0 else args.rule_step_limit
    on_policy_limit = (
        None if args.on_policy_step_limit == 0 else args.on_policy_step_limit
    )
    try:
        summary = train_phase5_bc_ppo_policy_from_trajectories(
            rule_trajectory_dataset_paths=rule_datasets,
            on_policy_trajectory_dataset_paths=on_policy_datasets,
            checkpoint_path=args.checkpoint,
            output_checkpoint_path=args.output_checkpoint,
            report_path=args.report_json,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            clip_epsilon=args.clip_epsilon,
            bc_loss_weight=args.bc_loss_weight,
            ppo_policy_loss_weight=args.ppo_policy_loss_weight,
            ppo_value_loss_weight=args.ppo_value_loss_weight,
            entropy_weight=args.entropy_weight,
            rule_step_limit=rule_limit,
            on_policy_step_limit=on_policy_limit,
            deck_index_filter=args.deck_index_filter,
            update_schedule=args.update_schedule,
            rule_anchor_fraction=args.rule_anchor_fraction,
            gradient_diagnostic_batches=args.gradient_diagnostic_batches,
        )
    except (Phase5PolicyUnavailable, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


def command_rl_train_phase5_trajectory_bc(args: argparse.Namespace) -> int:
    if not args.checkpoint.exists():
        print(f"Phase 5 checkpoint not found at {args.checkpoint}.", file=sys.stderr)
        return 2
    rule_datasets = list(args.rule_trajectory_dataset or [])
    for path in rule_datasets:
        if not path.exists():
            print(f"Phase 5 trajectory dataset not found at {path}.", file=sys.stderr)
            return 2
    rule_limit = None if args.rule_step_limit == 0 else args.rule_step_limit
    try:
        summary = train_phase5_bc_policy_from_trajectories(
            rule_trajectory_dataset_paths=rule_datasets,
            checkpoint_path=args.checkpoint,
            output_checkpoint_path=args.output_checkpoint,
            report_path=args.report_json,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            rule_step_limit=rule_limit,
            deck_index_filter=args.deck_index_filter,
        )
    except (Phase5PolicyUnavailable, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


def command_rl_train_phase5_alpha_ppo_specialists(args: argparse.Namespace) -> int:
    trajectory_datasets = list(args.trajectory_dataset or [])
    for path in trajectory_datasets:
        if not path.exists():
            print(f"Phase 5 trajectory dataset not found at {path}.", file=sys.stderr)
            return 2
    selfplay_limit = None if args.selfplay_limit == 0 else args.selfplay_limit
    try:
        summary = train_phase5_deck_specialists_ppo(
            trajectory_dataset_paths=trajectory_datasets,
            source_checkpoint_dir=args.source_checkpoint_dir,
            output_checkpoint_dir=args.output_checkpoint_dir,
            report_dir=args.report_dir,
            aggregate_report_path=args.report_json,
            iteration=args.iteration,
            deck_indices=args.deck_index or PHASE5_ALPHA_DECK_INDICES,
            allow_empty_decks=args.allow_empty_decks,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            clip_epsilon=args.clip_epsilon,
            policy_loss_weight=args.policy_loss_weight,
            value_loss_weight=args.value_loss_weight,
            entropy_weight=args.entropy_weight,
            selfplay_limit=selfplay_limit,
            require_on_policy=not args.allow_off_policy_trajectories,
        )
    except (Phase5PolicyUnavailable, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


def command_rl_init_phase5_policy_checkpoint(args: argparse.Namespace) -> int:
    try:
        summary = initialize_phase5_policy_checkpoint(
            checkpoint_path=args.checkpoint,
            report_path=args.report_json,
            max_entities=args.max_entities,
            max_actions=args.max_actions,
            max_previous_actions=args.max_previous_actions,
            d_model=args.d_model,
            seed=args.seed,
            metadata={
                "deck_index": args.deck_index,
                "controlled_public_agent_key": args.controlled_public_agent_key,
                "experiment": args.experiment,
            },
            overwrite=args.overwrite,
        )
    except (Phase5PolicyUnavailable, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


def command_rl_generate_phase5_alpha_bootstrap(args: argparse.Namespace) -> int:
    if not args.sample_dir.exists():
        print(
            f"Kaggle sample submission not found at {args.sample_dir}. "
            "Run `python -m ptcg_abc kaggle-setup` first.",
            file=sys.stderr,
        )
        return 2
    iteration_dir = args.iteration_dir
    if iteration_dir is None:
        iteration_dir = args.league_root / "iterations" / f"iter-{args.iteration:04d}"
    report_json = args.report_json
    if report_json is None:
        report_json = (
            Path("experiments")
            / "rl"
            / "phase5_league_alpha"
            / f"iter-{args.iteration:04d}_rule_bootstrap_report.json"
        )
    try:
        summary = generate_phase5_alpha_rule_bootstrap(
            sample_dir=args.sample_dir,
            iteration_dir=iteration_dir,
            report_path=report_json,
            games_per_pair=args.games_per_pair,
            max_steps=args.max_steps,
            deck_indices=args.deck_index or PHASE5_ALPHA_DECK_INDICES,
            game_offset=args.game_offset,
            allow_existing_raw=args.allow_existing_raw,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary.to_dict(), indent=2))
    print(f"Wrote Phase 5 Alpha bootstrap report to {report_json}.")
    return 0 if summary.selfplay.get("errors", 0) == 0 else 1


def command_rl_generate_phase5_alpha_league_iteration(args: argparse.Namespace) -> int:
    if not args.sample_dir.exists():
        print(
            f"Kaggle sample submission not found at {args.sample_dir}. "
            "Run `python -m ptcg_abc kaggle-setup` first.",
            file=sys.stderr,
        )
        return 2
    iteration_dir = args.iteration_dir
    if iteration_dir is None:
        iteration_dir = args.league_root / "iterations" / f"iter-{args.iteration:04d}"
    report_json = args.report_json
    if report_json is None:
        report_json = (
            Path("experiments")
            / "rl"
            / "phase5_league_alpha"
            / f"iter-{args.iteration:04d}_league_iteration_report.json"
        )
    try:
        summary = generate_phase5_alpha_league_iteration(
            sample_dir=args.sample_dir,
            iteration_dir=iteration_dir,
            report_path=report_json,
            specialist_model_dir=args.specialist_model_dir,
            games_per_deck=args.games_per_deck,
            max_steps=args.max_steps,
            deck_indices=args.deck_index or PHASE5_ALPHA_DECK_INDICES,
            game_offset=args.game_offset,
            allow_existing_raw=args.allow_existing_raw,
            agent_kind=args.agent,
            search_trace_path=args.search_trace_output,
            search_trace_game_limit=args.search_trace_games,
            search_config=_root_search_config_from_args(args),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary.to_dict(), indent=2))
    print(f"Wrote Phase 5 Alpha league-iteration report to {report_json}.")
    return 0 if summary.selfplay.get("errors", 0) == 0 else 1


def _public_agent_roots_from_args(args: argparse.Namespace) -> list[Path]:
    roots = list(getattr(args, "public_agent_root", []) or [])
    return roots or [Path("data") / "public_agents"]


def _public_controlled_deck_indices_from_args(args: argparse.Namespace) -> list[int]:
    controlled_key = getattr(args, "controlled_public_agent_key", None)
    if controlled_key:
        return [int(getattr(args, "controlled_deck_index", 101))]
    return list(getattr(args, "deck_index", []) or PHASE5_ALPHA_DECK_INDICES)


def command_phase5_public_agent_roster(args: argparse.Namespace) -> int:
    if not args.sample_dir.exists():
        print(
            f"Kaggle sample submission not found at {args.sample_dir}. "
            "Run `python -m ptcg_abc kaggle-setup` first.",
            file=sys.stderr,
        )
        return 2
    try:
        loaded, statuses = discover_phase5_public_opponents(
            sample_dir=args.sample_dir,
            public_agent_roots=_public_agent_roots_from_args(args),
            roster_notebook=args.roster_notebook,
            include_public=not args.samples_only,
            include_samples=not args.public_only,
            include_builtin_samples=not args.no_builtin_samples,
            public_agent_keys=args.public_agent_key or None,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    write_public_agent_status_report(args.report_json, statuses)
    payload = {
        "available": len(loaded),
        "unavailable": len([status for status in statuses if status.status != "available"]),
        "statuses": [status.to_dict() for status in statuses],
        "report_json": args.report_json.as_posix(),
    }
    print(json.dumps(payload, indent=2))
    print(f"Wrote Phase 5 public-agent roster report to {args.report_json}.")
    return 0 if loaded else 1


def command_rl_evaluate_phase5_public_agents(args: argparse.Namespace) -> int:
    if not args.sample_dir.exists():
        print(
            f"Kaggle sample submission not found at {args.sample_dir}. "
            "Run `python -m ptcg_abc kaggle-setup` first.",
            file=sys.stderr,
        )
        return 2
    model_path = args.model if args.model and args.model.exists() else None
    specialist_model_dir = None if args.agent == "rule" else args.specialist_model_dir
    if specialist_model_dir is not None:
        missing = [
            specialist_model_dir / f"deck-{deck_index:02d}.pt"
            for deck_index in _public_controlled_deck_indices_from_args(args)
            if not (specialist_model_dir / f"deck-{deck_index:02d}.pt").exists()
        ]
        if missing:
            preview = ", ".join(path.as_posix() for path in missing[:3])
            suffix = "..." if len(missing) > 3 else ""
            print(
                f"Missing Phase 5 specialist checkpoint(s): {preview}{suffix}",
                file=sys.stderr,
            )
            return 2
    if (
        args.agent
        in {
            "phase5-symbolic",
            "phase5-search",
            "phase5-full",
            "phase5-rl",
            "phase5-epsilon",
            "phase5-epsilon-mixture",
        }
        and model_path is None
        and specialist_model_dir is None
    ):
        print(f"Phase 5 checkpoint not found at {args.model}.", file=sys.stderr)
        return 2
    try:
        result, statuses = run_phase5_public_agent_benchmark(
            sample_dir=args.sample_dir,
            public_agent_roots=_public_agent_roots_from_args(args),
            roster_notebook=args.roster_notebook,
            include_public=not args.samples_only,
            include_samples=not args.public_only,
            include_builtin_samples=not args.no_builtin_samples,
            public_agent_keys=args.public_agent_key or None,
            require_min_opponents=args.require_min_opponents,
            controlled_public_agent_key=args.controlled_public_agent_key,
            controlled_deck_index=args.controlled_deck_index,
            agent_kind=args.agent,
            model_path=model_path,
            specialist_model_dir=specialist_model_dir,
            deck_indices=args.deck_index or None,
            games_per_matchup=args.games_per_matchup,
            max_steps=args.max_steps,
            search_config=_root_search_config_from_args(args),
            search_trace_path=args.search_trace_output,
            search_trace_game_limit=args.search_trace_games,
            policy_epsilon=args.policy_epsilon,
            replay_output_dir=args.replay_output_dir,
            saved_win_replays=args.saved_win_replays,
            saved_loss_replays=args.saved_loss_replays,
            replay_trace_limit=args.replay_trace_limit,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    gate_summary = summarize_public_agent_gate(
        result.rows,
        min_win_rate=args.min_opponent_win_rate,
    )
    write_phase4_benchmark_report(
        result,
        json_path=args.report_json,
        markdown_path=args.report_md,
        agent_kind=f"phase5-public-agents:{args.agent}",
        model_path=specialist_model_dir or model_path,
        extra_json={"public_agent_gate": gate_summary},
        extra_markdown_sections=[format_public_agent_gate_markdown(gate_summary)],
    )
    write_public_agent_status_report(args.status_json, statuses)
    totals = _benchmark_totals(result.rows)
    totals["public_agent_gate"] = gate_summary
    totals["available_public_agents"] = len(
        [status for status in statuses if status.status == "available"]
    )
    totals["unavailable_public_agents"] = len(
        [status for status in statuses if status.status != "available"]
    )
    print(json.dumps(totals, indent=2))
    print(f"Wrote Phase 5 public-agent benchmark report to {args.report_md}.")
    print(f"Wrote Phase 5 public-agent availability report to {args.status_json}.")
    gate_failed = args.fail_on_gate and not gate_summary["passed"]
    return 0 if totals["errors"] == 0 and not gate_failed else 1


def command_rl_generate_phase5_public_agent_trajectories(args: argparse.Namespace) -> int:
    if not args.sample_dir.exists():
        print(
            f"Kaggle sample submission not found at {args.sample_dir}. "
            "Run `python -m ptcg_abc kaggle-setup` first.",
            file=sys.stderr,
        )
        return 2
    model_path = args.model if args.model and args.model.exists() else None
    specialist_model_dir = None if args.agent == "rule" else args.specialist_model_dir
    if specialist_model_dir is not None:
        missing = [
            specialist_model_dir / f"deck-{deck_index:02d}.pt"
            for deck_index in _public_controlled_deck_indices_from_args(args)
            if not (specialist_model_dir / f"deck-{deck_index:02d}.pt").exists()
        ]
        if missing:
            preview = ", ".join(path.as_posix() for path in missing[:3])
            suffix = "..." if len(missing) > 3 else ""
            print(
                f"Missing Phase 5 specialist checkpoint(s): {preview}{suffix}",
                file=sys.stderr,
            )
            return 2
    if (
        args.agent
        in {
            "phase5-symbolic",
            "phase5-search",
            "phase5-full",
            "phase5-rl",
            "phase5-epsilon",
            "phase5-epsilon-mixture",
        }
        and model_path is None
        and args.specialist_model_dir is None
    ):
        print(f"Phase 5 checkpoint not found at {args.model}.", file=sys.stderr)
        return 2
    try:
        summary = generate_phase5_public_agent_trajectories(
            sample_dir=args.sample_dir,
            public_agent_roots=_public_agent_roots_from_args(args),
            output_path=args.output,
            roster_notebook=args.roster_notebook,
            include_public=not args.samples_only,
            include_samples=not args.public_only,
            include_builtin_samples=not args.no_builtin_samples,
            public_agent_keys=args.public_agent_key or None,
            require_min_opponents=args.require_min_opponents,
            controlled_public_agent_key=args.controlled_public_agent_key,
            controlled_deck_index=args.controlled_deck_index,
            agent_kind=args.agent,
            model_path=model_path,
            specialist_model_dir=specialist_model_dir,
            deck_indices=args.deck_index or None,
            games_per_matchup=args.games_per_matchup,
            max_steps=args.max_steps,
            game_offset=args.game_offset,
            search_config=_root_search_config_from_args(args),
            overwrite=args.overwrite,
            outcome_reward_scale=args.outcome_reward_scale,
            tactical_reward_config=PublicAgentTacticalRewardConfig(
                mode=args.tactical_reward_mode,
                attack_bonus=args.tactical_attack_bonus,
                attach_bonus=args.tactical_attach_bonus,
                missed_attack_penalty=args.tactical_missed_attack_penalty,
                missed_attach_penalty=args.tactical_missed_attach_penalty,
                fractional_prize_weight=args.tactical_fractional_prize_weight,
                fractional_opponent_weight=args.tactical_fractional_opponent_weight,
            ),
            policy_epsilon=args.policy_epsilon,
            policy_seed=args.policy_seed,
            teacher_agent_kind=args.teacher_agent,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    write_public_agent_trajectory_report(args.report_json, summary)
    print(json.dumps(summary.to_dict(), indent=2))
    print(f"Wrote Phase 5 public-agent trajectories to {args.output}.")
    print(f"Wrote Phase 5 public-agent trajectory report to {args.report_json}.")
    return 0 if summary.errors == 0 else 1


def command_rl_train_phase5_deck_specialists(args: argparse.Namespace) -> int:
    decision_dataset = None if args.no_decision_dataset else args.decision_dataset
    if decision_dataset is not None and not decision_dataset.exists():
        print(f"Phase 5 decision dataset not found at {decision_dataset}.", file=sys.stderr)
        return 2
    selfplay_datasets = list(args.selfplay_dataset or [])
    for path in selfplay_datasets:
        if not path.exists():
            print(f"Phase 5 self-play dataset not found at {path}.", file=sys.stderr)
            return 2
    decision_limit = None if args.decision_limit == 0 else args.decision_limit
    selfplay_limit = None if args.selfplay_limit == 0 else args.selfplay_limit
    try:
        summary = train_phase5_deck_specialists(
            decision_dataset_path=decision_dataset,
            selfplay_dataset_paths=selfplay_datasets,
            checkpoint_dir=args.checkpoint_dir,
            report_dir=args.report_dir,
            aggregate_report_path=args.report_json,
            iteration=args.iteration,
            deck_indices=args.deck_index or PHASE5_ALPHA_DECK_INDICES,
            allow_empty_decks=args.allow_empty_decks,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_entities=args.max_entities,
            max_actions=args.max_actions,
            max_previous_actions=args.max_previous_actions,
            d_model=args.d_model,
            target_source=args.target_source,
            rule_demo_target_source=args.rule_demo_target_source,
            changed_weight=args.changed_weight,
            unchanged_weight=args.unchanged_weight,
            search_decision_weight=args.search_decision_weight,
            rule_demo_weight=args.rule_demo_weight,
            selfplay_weight=args.selfplay_weight,
            pairwise_changed=args.pairwise_changed,
            pairwise_weight=args.pairwise_weight,
            pairwise_margin=args.pairwise_margin,
            pairwise_negatives=args.pairwise_negatives,
            value_loss_weight=args.value_loss_weight,
            action_value_loss_weight=args.action_value_loss_weight,
            tactical_loss_weight=args.tactical_loss_weight,
            decision_limit=decision_limit,
            selfplay_limit=selfplay_limit,
        )
    except (Phase5PolicyUnavailable, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


def command_rl_clean_phase5_alpha_iteration(args: argparse.Namespace) -> int:
    try:
        summary = cleanup_phase5_alpha_raw_train(
            iteration_dir=args.iteration_dir,
            cleanup_report_path=args.report_json,
            update_report_path=args.update_report,
            require_update_report=not args.allow_missing_update_report,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary.to_dict(), indent=2))
    print(f"Wrote Phase 5 Alpha cleanup report to {args.report_json}.")
    return 0


def command_rl_evaluate(args: argparse.Namespace) -> int:
    if not args.sample_dir.exists():
        print(
            f"Kaggle sample submission not found at {args.sample_dir}. "
            "Run `python -m ptcg_abc kaggle-setup` first.",
            file=sys.stderr,
        )
        return 2
    model_path = args.model if args.model and args.model.exists() else None
    if args.agent in {"phase5-symbolic", "phase5-search", "phase5-full", "phase5-rl"} and model_path is None:
        print(
            f"Phase 5 checkpoint not found at {args.model}.",
            file=sys.stderr,
        )
        return 2
    search_config = _root_search_config_from_args(args)
    result = run_phase4_required_benchmark(
        sample_dir=args.sample_dir,
        agent_kind=args.agent,
        model_path=model_path,
        games_per_matchup=args.games_per_matchup,
        max_steps=args.max_steps,
        search_trace_path=args.search_trace_output,
        search_config=search_config,
    )
    write_phase4_benchmark_report(
        result,
        json_path=args.report_json,
        markdown_path=args.report_md,
        agent_kind=args.agent,
        model_path=model_path,
    )
    totals = _benchmark_totals(result.rows)
    print(json.dumps(totals, indent=2))
    print(f"Wrote Phase 4 benchmark report to {args.report_md}.")
    if args.search_trace_output is not None:
        print(f"Wrote Phase 5 search traces to {args.search_trace_output}.")
    return 0 if totals["errors"] == 0 else 1


def command_rl_evaluate_phase5_league(args: argparse.Namespace) -> int:
    if not args.sample_dir.exists():
        print(
            f"Kaggle sample submission not found at {args.sample_dir}. "
            "Run `python -m ptcg_abc kaggle-setup` first.",
            file=sys.stderr,
        )
        return 2
    model_path = args.model if args.model and args.model.exists() else None
    specialist_model_dir = args.specialist_model_dir
    if specialist_model_dir is not None:
        missing = [
            specialist_model_dir / f"deck-{deck_index:02d}.pt"
            for deck_index in PHASE5_ALPHA_DECK_INDICES
            if not (specialist_model_dir / f"deck-{deck_index:02d}.pt").exists()
        ]
        if missing:
            preview = ", ".join(path.as_posix() for path in missing[:3])
            suffix = "..." if len(missing) > 3 else ""
            print(
                f"Missing Phase 5 specialist checkpoint(s): {preview}{suffix}",
                file=sys.stderr,
            )
            return 2
    if (
        args.agent in {"phase5-symbolic", "phase5-search", "phase5-full", "phase5-rl"}
        and model_path is None
        and specialist_model_dir is None
    ):
        print(f"Phase 5 checkpoint not found at {args.model}.", file=sys.stderr)
        return 2
    result = run_phase5_league_benchmark(
        sample_dir=args.sample_dir,
        agent_kind=args.agent,
        model_path=model_path,
        specialist_model_dir=specialist_model_dir,
        games_per_matchup=args.games_per_matchup,
        max_steps=args.max_steps,
        search_trace_path=args.search_trace_output,
        search_config=_root_search_config_from_args(args),
    )
    write_phase4_benchmark_report(
        result,
        json_path=args.report_json,
        markdown_path=args.report_md,
        agent_kind=f"phase5-league:{args.agent}",
        model_path=specialist_model_dir or model_path,
    )
    totals = _benchmark_totals(result.rows)
    print(json.dumps(totals, indent=2))
    print(f"Wrote Phase 5 league benchmark report to {args.report_md}.")
    return 0 if totals["errors"] == 0 else 1


def command_rl_image_progression(args: argparse.Namespace) -> int:
    if not args.sample_dir.exists():
        print(
            f"Kaggle sample submission not found at {args.sample_dir}. "
            "Run `python -m ptcg_abc kaggle-setup` first.",
            file=sys.stderr,
        )
        return 2
    payload = []
    image_sizes = args.image_size or [1024, 512, 256]
    for image_size in image_sizes:
        try:
            summary = run_image_progression_experiment(
                sample_dir=args.sample_dir,
                image_size=image_size,
                iterations=args.iterations,
                selfplay_games=args.selfplay_games,
                eval_games_per_matchup=args.eval_games_per_matchup,
                max_steps=args.max_steps,
                deck_a_index=args.deck_a_index,
                deck_b_index=args.deck_b_index,
                selfplay_deck_indices=args.selfplay_deck_index or None,
                saved_replays_per_matchup=args.saved_replays_per_matchup,
                replay_trace_limit=args.replay_trace_limit,
                update_epochs=args.update_epochs,
                base_model_path=args.base_model,
                dataset_root=args.dataset_root,
                model_root=args.model_root,
                report_root=args.report_root,
                output_root=args.output_root,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        payload.append(summary.to_dict())
    print(json.dumps({"experiments": payload}, indent=2))
    return 0


def command_rl_evaluate_guidance(args: argparse.Namespace) -> int:
    if not args.sample_dir.exists():
        print(
            f"Kaggle sample submission not found at {args.sample_dir}. "
            "Run `python -m ptcg_abc kaggle-setup` first.",
            file=sys.stderr,
        )
        return 2
    model_path = args.model if args.model and args.model.exists() else None
    result = run_phase4_required_benchmark(
        sample_dir=args.sample_dir,
        agent_kind="hybrid",
        model_path=model_path,
        games_per_matchup=args.games_per_matchup,
        max_steps=args.max_steps,
        guidance_rules=[args.guidance_rule],
    )
    write_phase4_benchmark_report(
        result,
        json_path=args.report_json,
        markdown_path=args.report_md,
        agent_kind=f"hybrid:{args.guidance_rule}",
        model_path=model_path,
    )
    totals = _benchmark_totals(result.rows)
    payload = {
        "guidance_rule": args.guidance_rule,
        **totals,
        "accepted_against_baseline": None,
    }
    if args.baseline_json and args.baseline_json.exists():
        baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        baseline_wins = sum(int(row.get("wins", 0)) for row in baseline.get("rows", []))
        payload["baseline_wins"] = baseline_wins
        payload["accepted_against_baseline"] = totals["wins"] >= baseline_wins + 2
    print(json.dumps(payload, indent=2))
    return 0 if totals["errors"] == 0 else 1


def command_rl_package(args: argparse.Namespace) -> int:
    if not args.sample_dir.exists():
        print(
            f"Kaggle sample submission not found at {args.sample_dir}. "
            "Run `python -m ptcg_abc kaggle-setup` first.",
            file=sys.stderr,
        )
        return 2
    decks = {deck.index: deck for deck in phase3_tournament_559_prepared_decks()}
    selected_indices = args.deck_index or [1]
    outputs = []
    for deck_index in selected_indices:
        if deck_index not in decks:
            print(f"Unknown Tournament 559 deck index: {deck_index}", file=sys.stderr)
            return 2
        deck = decks[deck_index]
        deck_dir = args.output_dir / f"deck-{deck.index}"
        tar_path = deck_dir / "submission.tar.gz"
        result = build_hybrid_rl_submission_bundle(
            deck_ids=deck.card_ids,
            sample_dir=args.sample_dir,
            output_dir=deck_dir,
            model_path=args.model if args.model and args.model.exists() else None,
            tar_path=tar_path,
        )
        outputs.append(
            {
                "deck_index": deck.index,
                "deck_label": deck.label,
                "tar_path": str(result.tar_path.as_posix()),
            }
        )
    print(json.dumps({"packages": outputs}, indent=2))
    return 0


def command_phase5_package(args: argparse.Namespace) -> int:
    if not args.sample_dir.exists():
        print(
            f"Kaggle sample submission not found at {args.sample_dir}. "
            "Run `python -m ptcg_abc kaggle-setup` first.",
            file=sys.stderr,
        )
        return 2
    if args.model_dir is not None:
        if not args.model_dir.exists():
            print(
                f"Phase 5 specialist checkpoint directory not found at {args.model_dir}.",
                file=sys.stderr,
            )
            return 2
    elif not args.model.exists():
        print(f"Phase 5 checkpoint not found at {args.model}.", file=sys.stderr)
        return 2
    if args.deck_pool == "league-13":
        prepared_decks = phase5_league_prepared_decks()
        pool_label = "Phase 5 league"
    else:
        prepared_decks = phase3_tournament_559_prepared_decks()
        pool_label = "Tournament 559"
    decks = {deck.index: deck for deck in prepared_decks}
    selected_indices = args.deck_index or [2, 8]
    outputs = []
    for deck_index in selected_indices:
        if deck_index not in decks:
            print(f"Unknown {pool_label} deck index: {deck_index}", file=sys.stderr)
            return 2
        deck = decks[deck_index]
        model_path = (
            args.model_dir / f"deck-{deck.index:02d}.pt"
            if args.model_dir is not None
            else args.model
        )
        if not model_path.exists():
            print(f"Phase 5 checkpoint not found at {model_path}.", file=sys.stderr)
            return 2
        deck_dir = args.output_dir / f"deck-{deck.index:02d}-{_slug(deck.archetype)}"
        tar_path = deck_dir / "submission.tar.gz"
        zip_path = args.output_dir / f"deck-{deck.index:02d}-{_slug(deck.archetype)}-phase5-search-submission.zip"
        result = build_phase5_search_submission_bundle(
            deck_ids=deck.card_ids,
            sample_dir=args.sample_dir,
            output_dir=deck_dir,
            model_path=model_path,
            tar_path=tar_path,
            zip_path=zip_path,
        )
        outputs.append(
            {
                "deck_index": deck.index,
                "deck_pool": args.deck_pool,
                "deck_label": deck.label,
                "agent": "phase5-search",
                "model": str(model_path.as_posix()),
                "tar_path": str(result.tar_path.as_posix()),
                "zip_path": str(result.zip_path.as_posix()) if result.zip_path else None,
            }
        )
    print(json.dumps({"packages": outputs}, indent=2))
    return 0


def command_phase5_compare_benchmarks(args: argparse.Namespace) -> int:
    if not args.baseline.exists():
        print(f"Baseline report not found at {args.baseline}.", file=sys.stderr)
        return 2
    if not args.candidate.exists():
        print(f"Candidate report not found at {args.candidate}.", file=sys.stderr)
        return 2
    payload = compare_benchmark_reports(
        baseline_path=args.baseline,
        candidate_path=args.candidate,
        output_json=args.report_json,
        output_md=args.report_md,
    )
    print(json.dumps(payload["overall_delta"], indent=2))
    print(f"Wrote Phase 5 benchmark comparison to {args.report_md}.")
    return 0


def command_rl_board_snapshots(args: argparse.Namespace) -> int:
    if not args.sample_dir.exists():
        print(
            f"Kaggle sample submission not found at {args.sample_dir}. "
            "Run `python -m ptcg_abc kaggle-setup` first.",
            file=sys.stderr,
        )
        return 2
    try:
        manifest = run_rule_vs_benchmark_snapshots(
            sample_dir=args.sample_dir,
            output_dir=args.output_dir,
            our_deck_index=args.our_deck_index,
            benchmark_index=args.benchmark_index,
            max_steps=args.max_steps,
            record_player=args.record_player,
            image_limit=args.image_limit,
            turns_per_player=args.turns_per_player,
            card_art_pdf=args.card_art_pdf,
            card_art_dir=args.card_art_dir,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(manifest.to_dict(), indent=2))
    print(f"Wrote {len(manifest.images)} board snapshots to {args.output_dir}.")
    return 0 if manifest.battle_result.get("error") is None else 1


def _benchmark_totals(rows: list) -> dict[str, float | int]:
    games = sum(row.games for row in rows)
    wins = sum(row.wins for row in rows)
    losses = sum(row.losses for row in rows)
    draws = sum(row.draws for row in rows)
    timeouts = sum(row.timeouts for row in rows)
    errors = sum(row.errors for row in rows)
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "timeouts": timeouts,
        "errors": errors,
        "win_rate": wins / games if games else 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ptcg-abc",
        description="Pokemon TCG AI Battle Challenge tooling.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    kaggle_setup = subparsers.add_parser(
        "kaggle-setup",
        help="Download Kaggle competition files and discover the legal card list.",
    )
    kaggle_setup.add_argument("--competition", default=COMPETITION_SLUG)
    kaggle_setup.add_argument("--raw-dir", type=_path, default=KAGGLE_RAW_DIR)
    kaggle_setup.add_argument("--input-dir", type=_path, default=KAGGLE_INPUT_DIR)
    kaggle_setup.add_argument(
        "--archive",
        type=_path,
        help="Use an already-downloaded Kaggle competition zip instead of the API.",
    )
    kaggle_setup.add_argument("--legal-cards", type=_path, default=LEGAL_CARDS_PATH)
    kaggle_setup.add_argument("--legal-source", type=_path)
    kaggle_setup.add_argument(
        "--candidate-report",
        type=_path,
        default=REPORTS_DIR / "legal_card_candidates.md",
    )
    kaggle_setup.add_argument("--refresh", action="store_true")
    kaggle_setup.set_defaults(func=command_kaggle_setup)

    discover = subparsers.add_parser(
        "discover-legal-cards",
        help="Discover legal card names from already-downloaded Kaggle files.",
    )
    discover.add_argument("--input-dir", type=_path, default=KAGGLE_INPUT_DIR)
    discover.add_argument("--legal-cards", type=_path, default=LEGAL_CARDS_PATH)
    discover.add_argument("--legal-source", type=_path)
    discover.add_argument(
        "--candidate-report",
        type=_path,
        default=REPORTS_DIR / "legal_card_candidates.md",
    )
    discover.set_defaults(func=command_discover_legal_cards)

    missing = subparsers.add_parser(
        "missing-limitless",
        help="Report Limitless card names missing from the Kaggle legal card list.",
    )
    missing.add_argument("--legal-cards", type=_path, default=LEGAL_CARDS_PATH)
    missing.add_argument("--output", type=_path, default=REPORTS_DIR / "missing_limitless_cards.md")
    _add_common_limitless_args(missing)
    missing.set_defaults(func=command_missing_limitless)

    collect = subparsers.add_parser(
        "collect-corpus",
        help="Collect Limitless decks and write JSONL, CSV, TXT, and manifest outputs.",
    )
    _add_common_limitless_args(collect)
    collect.set_defaults(func=command_collect_corpus)

    smoke = subparsers.add_parser(
        "agent-smoke",
        help="Run the first rule-based agent against two corpus decks in the Kaggle simulator.",
    )
    smoke.add_argument(
        "--corpus",
        type=_path,
        default=PROCESSED_DIR / date.today().isoformat() / "deck_corpus.jsonl",
    )
    smoke.add_argument(
        "--card-data",
        type=_path,
        default=KAGGLE_INPUT_DIR / "EN_Card_Data.csv",
    )
    smoke.add_argument(
        "--sample-dir",
        type=_path,
        default=KAGGLE_INPUT_DIR / "sample_submission",
    )
    smoke.add_argument("--deck-index", type=int, default=1)
    smoke.add_argument("--opponent-index", type=int, default=2)
    smoke.add_argument("--max-steps", type=int, default=50)
    smoke.add_argument("--write-deck-csv", type=_path)
    smoke.set_defaults(func=command_agent_smoke)

    closeout = subparsers.add_parser(
        "phase3-closeout",
        help="Run Phase 3 evaluation, select the final archetype/deck, and build submission.tar.gz.",
    )
    closeout.add_argument(
        "--corpus",
        type=_path,
        default=PROCESSED_DIR / date.today().isoformat() / "deck_corpus.jsonl",
    )
    closeout.add_argument(
        "--card-data",
        type=_path,
        default=KAGGLE_INPUT_DIR / "EN_Card_Data.csv",
    )
    closeout.add_argument(
        "--sample-dir",
        type=_path,
        default=KAGGLE_INPUT_DIR / "sample_submission",
    )
    closeout.add_argument("--games-per-matchup", type=int, default=10)
    closeout.add_argument("--random-games", type=int, default=20)
    closeout.add_argument("--max-steps", type=int, default=600)
    closeout.add_argument("--seed", type=int, default=20260617)
    closeout.add_argument("--output-dir", type=_path, default=Path("submissions") / "phase3")
    closeout.add_argument(
        "--submission-tar",
        type=_path,
        default=Path("submissions") / "phase3" / "submission.tar.gz",
    )
    closeout.add_argument(
        "--report-json",
        type=_path,
        default=REPORTS_DIR / "phase3_closeout.json",
    )
    closeout.add_argument(
        "--report-md",
        type=_path,
        default=REPORTS_DIR / "phase3_closeout.md",
    )
    closeout.set_defaults(func=command_phase3_closeout)

    sample_dragapult = subparsers.add_parser(
        "benchmark-sample-dragapult",
        help="Run our rule-based agent with every corpus deck against the copied sample Dragapult agent.",
    )
    sample_dragapult.add_argument(
        "--corpus",
        type=_path,
        default=PROCESSED_DIR / date.today().isoformat() / "deck_corpus.jsonl",
    )
    sample_dragapult.add_argument(
        "--card-data",
        type=_path,
        default=KAGGLE_INPUT_DIR / "EN_Card_Data.csv",
    )
    sample_dragapult.add_argument(
        "--sample-dir",
        type=_path,
        default=KAGGLE_INPUT_DIR / "sample_submission",
    )
    sample_dragapult.add_argument("--games-per-deck", type=int, default=10)
    sample_dragapult.add_argument("--max-steps", type=int, default=600)
    sample_dragapult.add_argument("--debug-limit-per-deck", type=int, default=2)
    sample_dragapult.add_argument("--trace-limit", type=int, default=80)
    sample_dragapult.add_argument(
        "--skip-required-decks",
        action="store_true",
        help="Benchmark only the collected corpus and omit the four required Phase 3 sample decks.",
    )
    sample_dragapult.add_argument(
        "--baseline-json",
        type=_path,
        default=None,
    )
    sample_dragapult.add_argument(
        "--comparison-md",
        type=_path,
        default=REPORTS_DIR / "sample_dragapult_benchmark_comparison.md",
    )
    sample_dragapult.add_argument(
        "--report-json",
        type=_path,
        default=REPORTS_DIR / "sample_dragapult_benchmark.json",
    )
    sample_dragapult.add_argument(
        "--report-md",
        type=_path,
        default=REPORTS_DIR / "sample_dragapult_benchmark.md",
    )
    sample_dragapult.set_defaults(func=command_benchmark_sample_dragapult)

    phase3_required = subparsers.add_parser(
        "benchmark-phase3-required",
        help="Run the nine Tournament 559 decks against the four fixed Phase 3 benchmark decks.",
    )
    phase3_required.add_argument(
        "--sample-dir",
        type=_path,
        default=KAGGLE_INPUT_DIR / "sample_submission",
    )
    phase3_required.add_argument("--games-per-matchup", type=int, default=10)
    phase3_required.add_argument("--max-steps", type=int, default=600)
    phase3_required.add_argument("--debug-limit-per-matchup", type=int, default=0)
    phase3_required.add_argument("--trace-limit", type=int, default=60)
    phase3_required.add_argument(
        "--report-json",
        type=_path,
        default=REPORTS_DIR / "phase3_required_benchmark.json",
    )
    phase3_required.add_argument(
        "--report-md",
        type=_path,
        default=REPORTS_DIR / "phase3_required_benchmark.md",
    )
    phase3_required.set_defaults(func=command_benchmark_phase3_required)

    rl_collect = subparsers.add_parser(
        "rl-collect-bc",
        help="Collect rule-agent behavior-cloning decisions for the Phase 4 benchmark grid.",
    )
    rl_collect.add_argument(
        "--sample-dir",
        type=_path,
        default=KAGGLE_INPUT_DIR / "sample_submission",
    )
    rl_collect.add_argument("--games", type=int, default=36)
    rl_collect.add_argument("--max-steps", type=int, default=120)
    rl_collect.add_argument(
        "--output",
        type=_path,
        default=Path("data") / "datasets" / "rl" / "bc_decisions.jsonl",
    )
    rl_collect.set_defaults(func=command_rl_collect_bc)

    rl_train_bc = subparsers.add_parser(
        "rl-train-bc",
        help="Train the exported Phase 4 option-ranker from behavior-cloning decisions.",
    )
    rl_train_bc.add_argument(
        "--dataset",
        type=_path,
        default=Path("data") / "datasets" / "rl" / "bc_decisions.jsonl",
    )
    rl_train_bc.add_argument(
        "--model",
        type=_path,
        default=Path("models") / "rl" / "bc_model.json",
    )
    rl_train_bc.add_argument(
        "--checkpoint",
        type=_path,
        default=Path("models") / "rl" / "bc_torch.pt",
        help="Torch checkpoint path when `--backend torch` is used.",
    )
    rl_train_bc.add_argument("--backend", choices=["linear", "torch"], default="linear")
    rl_train_bc.add_argument("--epochs", type=int, default=1)
    rl_train_bc.add_argument("--learning-rate", type=float, default=0.02)
    rl_train_bc.add_argument(
        "--changed-weight",
        type=float,
        default=1.0,
        help="Frame weight for Phase 5 decisions where root search changed the rule baseline.",
    )
    rl_train_bc.add_argument(
        "--unchanged-weight",
        type=float,
        default=1.0,
        help="Frame weight for unchanged decisions.",
    )
    rl_train_bc.add_argument(
        "--exclude-feature",
        action="append",
        default=[],
        help="Action feature to exclude from training/export. Repeat for multiple features.",
    )
    rl_train_bc.add_argument(
        "--pairwise-changed",
        action="store_true",
        help="Train changed Phase 5 search decisions with search-vs-baseline pairwise updates.",
    )
    rl_train_bc.add_argument(
        "--pairwise-margin",
        type=float,
        default=0.0,
        help="Desired score margin for `--pairwise-changed` updates.",
    )
    rl_train_bc.add_argument(
        "--pairwise-negatives",
        choices=["baseline", "all"],
        default="baseline",
        help="Negative action set for `--pairwise-changed` updates.",
    )
    rl_train_bc.add_argument(
        "--report-json",
        type=_path,
        default=Path("experiments") / "rl" / "bc_train_report.json",
    )
    rl_train_bc.set_defaults(func=command_rl_train_bc)

    rl_build_phase5_symbolic = subparsers.add_parser(
        "rl-build-phase5-symbolic-dataset",
        help="Convert Phase 5 DecisionFrame JSONL into bounded symbolic tensor JSONL.",
    )
    rl_build_phase5_symbolic.add_argument(
        "--dataset",
        type=_path,
        default=Path("data") / "datasets" / "rl" / "phase5_search_decisions_merged.jsonl",
    )
    rl_build_phase5_symbolic.add_argument(
        "--output",
        type=_path,
        default=Path("data") / "datasets" / "rl" / "phase5_symbolic_decisions_smoke.jsonl",
    )
    rl_build_phase5_symbolic.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum records to write. Use 0 to convert the full input.",
    )
    rl_build_phase5_symbolic.add_argument("--max-entities", type=int, default=96)
    rl_build_phase5_symbolic.add_argument("--max-actions", type=int, default=128)
    rl_build_phase5_symbolic.add_argument("--max-previous-actions", type=int, default=16)
    rl_build_phase5_symbolic.add_argument(
        "--target-source",
        choices=["search", "baseline", "rule"],
        default="search",
    )
    rl_build_phase5_symbolic.add_argument("--changed-weight", type=float, default=1.0)
    rl_build_phase5_symbolic.add_argument("--unchanged-weight", type=float, default=1.0)
    rl_build_phase5_symbolic.set_defaults(func=command_rl_build_phase5_symbolic)

    rl_train_phase5_symbolic = subparsers.add_parser(
        "rl-train-phase5-symbolic",
        help="Train the AlphaStar-style Phase 5 symbolic legal-action policy.",
    )
    rl_train_phase5_symbolic.add_argument(
        "--dataset",
        type=_path,
        default=Path("data") / "datasets" / "rl" / "phase5_search_decisions_merged.jsonl",
    )
    rl_train_phase5_symbolic.add_argument(
        "--checkpoint",
        type=_path,
        default=Path("models") / "rl" / "phase5_symbolic_policy.pt",
    )
    rl_train_phase5_symbolic.add_argument(
        "--report-json",
        type=_path,
        default=Path("experiments") / "rl" / "phase5_symbolic_train_report.json",
    )
    rl_train_phase5_symbolic.add_argument("--epochs", type=int, default=1)
    rl_train_phase5_symbolic.add_argument("--batch-size", type=int, default=64)
    rl_train_phase5_symbolic.add_argument("--learning-rate", type=float, default=1.0e-4)
    rl_train_phase5_symbolic.add_argument("--d-model", type=int, default=128)
    rl_train_phase5_symbolic.add_argument("--max-entities", type=int, default=96)
    rl_train_phase5_symbolic.add_argument("--max-actions", type=int, default=128)
    rl_train_phase5_symbolic.add_argument("--max-previous-actions", type=int, default=16)
    rl_train_phase5_symbolic.add_argument(
        "--target-source",
        choices=["search", "baseline", "rule"],
        default="search",
    )
    rl_train_phase5_symbolic.add_argument("--changed-weight", type=float, default=1.0)
    rl_train_phase5_symbolic.add_argument("--unchanged-weight", type=float, default=1.0)
    rl_train_phase5_symbolic.add_argument(
        "--pairwise-changed",
        action="store_true",
        help="Add pairwise ranking loss on root-search changed frames.",
    )
    rl_train_phase5_symbolic.add_argument("--pairwise-weight", type=float, default=1.0)
    rl_train_phase5_symbolic.add_argument("--pairwise-margin", type=float, default=1.0)
    rl_train_phase5_symbolic.add_argument(
        "--pairwise-negatives",
        choices=["all", "baseline"],
        default="all",
        help="Negative actions for changed-frame pairwise loss.",
    )
    rl_train_phase5_symbolic.add_argument("--value-loss-weight", type=float, default=0.0)
    rl_train_phase5_symbolic.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum source frames per epoch. Use 0 for the full input.",
    )
    rl_train_phase5_symbolic.add_argument(
        "--changed-only",
        action="store_true",
        help="Train only on root-search changed frames.",
    )
    rl_train_phase5_symbolic.set_defaults(func=command_rl_train_phase5_symbolic)

    rl_train_phase5_generalist = subparsers.add_parser(
        "rl-train-phase5-generalist",
        help=(
            "Train Phase 5 policy/value/Q/tactical heads from search decisions, "
            "rule demonstrations, and phase5-search self-play trajectories."
        ),
    )
    rl_train_phase5_generalist.add_argument(
        "--decision-dataset",
        type=_path,
        default=Path("data") / "datasets" / "rl" / "phase5_search_decisions_merged.jsonl",
    )
    rl_train_phase5_generalist.add_argument(
        "--no-decision-dataset",
        action="store_true",
        help="Train only from self-play trajectory datasets.",
    )
    rl_train_phase5_generalist.add_argument(
        "--selfplay-dataset",
        type=_path,
        action="append",
        default=[],
        help="Phase 5 search self-play trajectory JSONL. Repeat for multiple shards.",
    )
    rl_train_phase5_generalist.add_argument(
        "--checkpoint",
        type=_path,
        default=Path("models") / "rl" / "phase5_generalist_policy.pt",
    )
    rl_train_phase5_generalist.add_argument(
        "--initial-checkpoint",
        type=_path,
        default=None,
        help="Optional source checkpoint to continue training from.",
    )
    rl_train_phase5_generalist.add_argument(
        "--report-json",
        type=_path,
        default=Path("experiments") / "rl" / "phase5_generalist_train_report.json",
    )
    rl_train_phase5_generalist.add_argument("--epochs", type=int, default=1)
    rl_train_phase5_generalist.add_argument("--batch-size", type=int, default=64)
    rl_train_phase5_generalist.add_argument("--learning-rate", type=float, default=1.0e-4)
    rl_train_phase5_generalist.add_argument("--d-model", type=int, default=128)
    rl_train_phase5_generalist.add_argument("--max-entities", type=int, default=96)
    rl_train_phase5_generalist.add_argument("--max-actions", type=int, default=128)
    rl_train_phase5_generalist.add_argument("--max-previous-actions", type=int, default=16)
    rl_train_phase5_generalist.add_argument(
        "--target-source",
        choices=["search", "baseline", "rule"],
        default="search",
    )
    rl_train_phase5_generalist.add_argument(
        "--rule-demo-target-source",
        choices=["baseline", "rule"],
        default="baseline",
    )
    rl_train_phase5_generalist.add_argument("--changed-weight", type=float, default=2.0)
    rl_train_phase5_generalist.add_argument("--unchanged-weight", type=float, default=0.5)
    rl_train_phase5_generalist.add_argument("--search-decision-weight", type=float, default=1.0)
    rl_train_phase5_generalist.add_argument("--rule-demo-weight", type=float, default=0.25)
    rl_train_phase5_generalist.add_argument("--selfplay-weight", type=float, default=1.0)
    rl_train_phase5_generalist.add_argument(
        "--pairwise-changed",
        action="store_true",
        help="Add pairwise ranking loss on root-search changed decision frames.",
    )
    rl_train_phase5_generalist.add_argument("--pairwise-weight", type=float, default=0.25)
    rl_train_phase5_generalist.add_argument("--pairwise-margin", type=float, default=1.0)
    rl_train_phase5_generalist.add_argument(
        "--pairwise-negatives",
        choices=["all", "baseline"],
        default="baseline",
    )
    rl_train_phase5_generalist.add_argument("--value-loss-weight", type=float, default=0.25)
    rl_train_phase5_generalist.add_argument(
        "--action-value-loss-weight",
        type=float,
        default=0.25,
    )
    rl_train_phase5_generalist.add_argument("--tactical-loss-weight", type=float, default=0.05)
    rl_train_phase5_generalist.add_argument(
        "--decision-limit",
        type=int,
        default=0,
        help="Maximum decision frames per epoch. Use 0 for the full input.",
    )
    rl_train_phase5_generalist.add_argument(
        "--selfplay-limit",
        type=int,
        default=0,
        help="Maximum self-play trajectory steps per epoch. Use 0 for the full input.",
    )
    rl_train_phase5_generalist.add_argument(
        "--deck-index-filter",
        type=int,
        default=None,
        help="Optional Phase 5 league deck index to train from.",
    )
    rl_train_phase5_generalist.set_defaults(func=command_rl_train_phase5_generalist)

    rl_rollout = subparsers.add_parser(
        "rl-rollout",
        help="Generate Phase 4 trajectory records with rule, RL, or hybrid agents.",
    )
    rl_rollout.add_argument(
        "--sample-dir",
        type=_path,
        default=KAGGLE_INPUT_DIR / "sample_submission",
    )
    rl_rollout.add_argument(
        "--agent",
        choices=["rule", "rl", "hybrid", "phase5-symbolic", "phase5-search", "phase5-full", "phase5-rl"],
        default="hybrid",
    )
    rl_rollout.add_argument(
        "--model",
        type=_path,
        default=Path("models") / "rl" / "bc_model.json",
    )
    rl_rollout.add_argument("--games", type=int, default=36)
    rl_rollout.add_argument("--max-steps", type=int, default=120)
    rl_rollout.add_argument(
        "--output",
        type=_path,
        default=Path("data") / "datasets" / "rl" / "rollout_steps.jsonl",
    )
    rl_rollout.set_defaults(func=command_rl_rollout)

    rl_phase5_selfplay = subparsers.add_parser(
        "rl-generate-phase5-search-selfplay",
        help="Generate Phase 5 phase5-search self-play trajectory data with outcome targets.",
    )
    rl_phase5_selfplay.add_argument(
        "--sample-dir",
        type=_path,
        default=KAGGLE_INPUT_DIR / "sample_submission",
    )
    rl_phase5_selfplay.add_argument(
        "--model",
        type=_path,
        default=Path("models") / "rl" / "phase5_symbolic_policy_10shards.pt",
    )
    rl_phase5_selfplay.add_argument("--games", type=int, default=36)
    rl_phase5_selfplay.add_argument(
        "--game-offset",
        type=int,
        default=0,
        help="Absolute game offset for deterministic array shards.",
    )
    rl_phase5_selfplay.add_argument("--max-steps", type=int, default=600)
    rl_phase5_selfplay.add_argument(
        "--deck-pool",
        choices=PHASE5_SELFPLAY_DECK_POOLS,
        default="tournament-9",
        help=(
            "Prepared deck pool for self-play: tournament-9 preserves the current "
            "Phase 5 data path; league-13 adds the four required sample decks."
        ),
    )
    rl_phase5_selfplay.add_argument(
        "--deck-a-index",
        type=int,
        default=None,
        help="Optional fixed self-play deck A index.",
    )
    rl_phase5_selfplay.add_argument(
        "--deck-b-index",
        type=int,
        default=None,
        help="Optional fixed self-play deck B index.",
    )
    rl_phase5_selfplay.add_argument(
        "--deck-index",
        type=int,
        action="append",
        default=[],
        help="Prepared deck index included in rotating self-play. Repeat to choose a subset.",
    )
    _add_phase5_search_config_args(rl_phase5_selfplay)
    rl_phase5_selfplay.add_argument(
        "--policy-pool-model",
        type=_path,
        action="append",
        default=[],
        help=(
            "Checkpoint path included in the self-play policy pool. Repeat to "
            "rotate older/current policies by game and player slot."
        ),
    )
    rl_phase5_selfplay.add_argument(
        "--output",
        type=_path,
        default=Path("data") / "datasets" / "rl" / "phase5_search_selfplay.jsonl",
    )
    rl_phase5_selfplay.add_argument(
        "--report-json",
        type=_path,
        default=Path("experiments") / "rl" / "phase5_search_selfplay_report.json",
    )
    rl_phase5_selfplay.add_argument(
        "--search-trace-output",
        type=_path,
        default=None,
        help="Optional JSONL output for sampled phase5-search self-play traces.",
    )
    rl_phase5_selfplay.add_argument(
        "--search-trace-games",
        type=int,
        default=3,
        help=(
            "Number of local self-play games to trace when --search-trace-output "
            "is set; use 0 for all."
        ),
    )
    rl_phase5_selfplay.set_defaults(func=command_rl_generate_phase5_search_selfplay)

    rl_alpha_bootstrap = subparsers.add_parser(
        "rl-generate-phase5-alpha-bootstrap",
        help="Generate rule-agent 13-deck bootstrap trajectories for the Alpha league track.",
    )
    rl_alpha_bootstrap.add_argument(
        "--sample-dir",
        type=_path,
        default=KAGGLE_INPUT_DIR / "sample_submission",
    )
    rl_alpha_bootstrap.add_argument("--iteration", type=int, default=0)
    rl_alpha_bootstrap.add_argument(
        "--league-root",
        type=_path,
        default=Path("/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha"),
    )
    rl_alpha_bootstrap.add_argument(
        "--iteration-dir",
        type=_path,
        default=None,
        help="Explicit iteration directory. Defaults under --league-root.",
    )
    rl_alpha_bootstrap.add_argument(
        "--games-per-pair",
        type=int,
        default=2,
        help="Rule-bootstrap games for each ordered 13-deck pair.",
    )
    rl_alpha_bootstrap.add_argument("--game-offset", type=int, default=0)
    rl_alpha_bootstrap.add_argument("--max-steps", type=int, default=600)
    rl_alpha_bootstrap.add_argument(
        "--deck-index",
        type=int,
        action="append",
        default=[],
        help="Phase 5 league deck index to include. Repeat; defaults to all 13.",
    )
    rl_alpha_bootstrap.add_argument(
        "--allow-existing-raw",
        action="store_true",
        help="Allow writing into an existing nonempty raw_train directory.",
    )
    rl_alpha_bootstrap.add_argument(
        "--report-json",
        type=_path,
        default=None,
        help="Bootstrap summary path. Defaults under experiments/rl/phase5_league_alpha.",
    )
    rl_alpha_bootstrap.set_defaults(func=command_rl_generate_phase5_alpha_bootstrap)

    rl_alpha_iteration = subparsers.add_parser(
        "rl-generate-phase5-alpha-league-iteration",
        help="Generate learned-agent Alpha league trajectories using deck specialists.",
    )
    rl_alpha_iteration.add_argument(
        "--sample-dir",
        type=_path,
        default=KAGGLE_INPUT_DIR / "sample_submission",
    )
    rl_alpha_iteration.add_argument("--iteration", type=int, default=1)
    rl_alpha_iteration.add_argument(
        "--league-root",
        type=_path,
        default=Path("/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha"),
    )
    rl_alpha_iteration.add_argument(
        "--iteration-dir",
        type=_path,
        default=None,
        help="Explicit iteration directory. Defaults under --league-root.",
    )
    rl_alpha_iteration.add_argument(
        "--specialist-model-dir",
        type=_path,
        required=True,
        help="Directory containing deck-01.pt through deck-13.pt specialist checkpoints.",
    )
    rl_alpha_iteration.add_argument(
        "--agent",
        choices=["phase5-search", "phase5-full", "phase5-rl"],
        default="phase5-full",
    )
    rl_alpha_iteration.add_argument(
        "--games-per-deck",
        type=int,
        default=100,
        help="League training games scheduled per selected deck.",
    )
    rl_alpha_iteration.add_argument("--game-offset", type=int, default=0)
    rl_alpha_iteration.add_argument("--max-steps", type=int, default=600)
    rl_alpha_iteration.add_argument(
        "--deck-index",
        type=int,
        action="append",
        default=[],
        help="Phase 5 league deck index to include. Repeat; defaults to all 13.",
    )
    rl_alpha_iteration.add_argument(
        "--allow-existing-raw",
        action="store_true",
        help="Allow writing into an existing nonempty raw_train directory.",
    )
    _add_phase5_search_config_args(rl_alpha_iteration)
    rl_alpha_iteration.add_argument(
        "--report-json",
        type=_path,
        default=None,
        help="League iteration summary path. Defaults under experiments/rl/phase5_league_alpha.",
    )
    rl_alpha_iteration.add_argument(
        "--search-trace-output",
        type=_path,
        default=None,
        help="Optional JSONL output for sampled phase5-full league traces.",
    )
    rl_alpha_iteration.add_argument(
        "--search-trace-games",
        type=int,
        default=3,
        help="Number of local league games to trace when --search-trace-output is set; use 0 for all.",
    )
    rl_alpha_iteration.set_defaults(
        func=command_rl_generate_phase5_alpha_league_iteration
    )

    phase5_public_roster = subparsers.add_parser(
        "phase5-public-agent-roster",
        help="Discover locally available public/specialized Kaggle agents from the Phase 5 roster.",
    )
    phase5_public_roster.add_argument(
        "--sample-dir",
        type=_path,
        default=KAGGLE_INPUT_DIR / "sample_submission",
    )
    phase5_public_roster.add_argument(
        "--public-agent-root",
        type=_path,
        action="append",
        default=[],
        help="Directory containing downloaded/exported public agents. Repeatable.",
    )
    phase5_public_roster.add_argument(
        "--roster-notebook",
        type=_path,
        default=None,
        help="Optional notebook containing AGENT_SOURCES; defaults to built-in public-20-plus-sample-4 roster.",
    )
    phase5_public_roster.add_argument("--public-only", action="store_true")
    phase5_public_roster.add_argument("--samples-only", action="store_true")
    phase5_public_roster.add_argument(
        "--public-agent-key",
        action="append",
        default=[],
        help="Restrict discovery output to a specific public-agent key. Repeatable.",
    )
    phase5_public_roster.add_argument(
        "--no-builtin-samples",
        action="store_true",
        help="Disable repo-bundled sample-agent adapters during discovery.",
    )
    phase5_public_roster.add_argument(
        "--report-json",
        type=_path,
        default=REPORTS_DIR / "phase5_public_agent_roster.json",
    )
    phase5_public_roster.set_defaults(func=command_phase5_public_agent_roster)

    rl_public_trajectories = subparsers.add_parser(
        "rl-generate-phase5-public-agent-trajectories",
        help="Generate our-agent trajectories against available public/specialized rule agents.",
    )
    rl_public_trajectories.add_argument(
        "--sample-dir",
        type=_path,
        default=KAGGLE_INPUT_DIR / "sample_submission",
    )
    rl_public_trajectories.add_argument(
        "--public-agent-root",
        type=_path,
        action="append",
        default=[],
        help="Directory containing downloaded/exported public agents. Repeatable.",
    )
    rl_public_trajectories.add_argument("--roster-notebook", type=_path, default=None)
    rl_public_trajectories.add_argument("--public-only", action="store_true")
    rl_public_trajectories.add_argument("--samples-only", action="store_true")
    rl_public_trajectories.add_argument("--no-builtin-samples", action="store_true")
    rl_public_trajectories.add_argument(
        "--public-agent-key",
        action="append",
        default=[],
        help="Restrict trajectory generation to a specific public-agent key. Repeatable.",
    )
    rl_public_trajectories.add_argument(
        "--controlled-public-agent-key",
        default=None,
        help="Use this public/sample agent's deck as the model-controlled deck.",
    )
    rl_public_trajectories.add_argument(
        "--controlled-deck-index",
        type=int,
        default=101,
        help="Synthetic deck index for --controlled-public-agent-key trajectory records.",
    )
    rl_public_trajectories.add_argument("--require-min-opponents", type=int, default=1)
    rl_public_trajectories.add_argument(
        "--agent",
        choices=[
            "rule",
            "phase5-symbolic",
            "phase5-search",
            "phase5-full",
            "phase5-rl",
            "phase5-epsilon",
            "phase5-epsilon-mixture",
        ],
        default="phase5-rl",
    )
    rl_public_trajectories.add_argument(
        "--teacher-agent",
        choices=["rule"],
        default=None,
        help=(
            "Label model-visited states with this teacher while executing "
            "the selected --agent behavior."
        ),
    )
    rl_public_trajectories.add_argument(
        "--policy-epsilon",
        type=float,
        default=0.0,
        help=(
            "Exploration probability for phase5-epsilon or the differentiable "
            "phase5-epsilon-mixture policy."
        ),
    )
    rl_public_trajectories.add_argument(
        "--policy-seed",
        type=int,
        default=None,
        help="Base exploration seed; each game uses base seed plus game index.",
    )
    rl_public_trajectories.add_argument(
        "--model",
        type=_path,
        default=Path("models") / "rl" / "phase5_generalist_policy_13deck_10k.pt",
    )
    rl_public_trajectories.add_argument(
        "--specialist-model-dir",
        type=_path,
        default=None,
        help="Directory containing deck-01.pt through deck-13.pt specialist checkpoints.",
    )
    rl_public_trajectories.add_argument(
        "--deck-index",
        type=int,
        action="append",
        default=[],
        help="Phase 5 league deck index to include. Repeat; defaults to all 13.",
    )
    rl_public_trajectories.add_argument("--games-per-matchup", type=int, default=2)
    rl_public_trajectories.add_argument("--game-offset", type=int, default=0)
    rl_public_trajectories.add_argument("--max-steps", type=int, default=600)
    rl_public_trajectories.add_argument(
        "--outcome-reward-scale",
        type=float,
        default=1.0,
        help="Scale terminal outcome reward before adding tactical step rewards.",
    )
    rl_public_trajectories.add_argument(
        "--tactical-reward-mode",
        choices=["none", "basic", "fractional-prize", "basic-fractional-prize"],
        default="none",
        help=(
            "Optional dense reward for public-agent trajectory steps. "
            "Fractional-prize modes reward changes in prize and partial-KO progress."
        ),
    )
    rl_public_trajectories.add_argument("--tactical-attack-bonus", type=float, default=0.10)
    rl_public_trajectories.add_argument("--tactical-attach-bonus", type=float, default=0.06)
    rl_public_trajectories.add_argument(
        "--tactical-missed-attack-penalty",
        type=float,
        default=-0.10,
    )
    rl_public_trajectories.add_argument(
        "--tactical-missed-attach-penalty",
        type=float,
        default=-0.06,
    )
    rl_public_trajectories.add_argument(
        "--tactical-fractional-prize-weight",
        type=float,
        default=1.0,
        help="Scale fractional prize-progress reward before adding it to the step reward.",
    )
    rl_public_trajectories.add_argument(
        "--tactical-fractional-opponent-weight",
        type=float,
        default=1.0,
        help="Scale opponent fractional prize-progress deltas before subtracting them.",
    )
    _add_phase5_search_config_args(rl_public_trajectories)
    rl_public_trajectories.add_argument(
        "--output",
        type=_path,
        default=Path("data") / "datasets" / "rl" / "phase5_public_agent_trajectories.jsonl",
    )
    rl_public_trajectories.add_argument(
        "--report-json",
        type=_path,
        default=Path("experiments")
        / "rl"
        / "phase5_league_alpha"
        / "phase5_public_agent_trajectories_report.json",
    )
    rl_public_trajectories.add_argument("--overwrite", action="store_true")
    rl_public_trajectories.set_defaults(
        func=command_rl_generate_phase5_public_agent_trajectories
    )

    rl_train_specialists = subparsers.add_parser(
        "rl-train-phase5-deck-specialists",
        help="Train one Phase 5 specialist checkpoint per selected league deck.",
    )
    rl_train_specialists.add_argument(
        "--decision-dataset",
        type=_path,
        default=Path("data") / "datasets" / "rl" / "phase5_search_decisions_10shards.jsonl",
    )
    rl_train_specialists.add_argument(
        "--no-decision-dataset",
        action="store_true",
        help="Train specialists only from trajectory datasets.",
    )
    rl_train_specialists.add_argument(
        "--selfplay-dataset",
        type=_path,
        action="append",
        default=[],
        help="Phase 5 trajectory JSONL. Repeat for multiple shards.",
    )
    rl_train_specialists.add_argument(
        "--checkpoint-dir",
        type=_path,
        default=Path("models") / "rl" / "phase5_league_alpha" / "specialists",
    )
    rl_train_specialists.add_argument(
        "--report-dir",
        type=_path,
        default=Path("experiments") / "rl" / "phase5_league_alpha" / "specialists",
    )
    rl_train_specialists.add_argument(
        "--report-json",
        type=_path,
        default=Path("experiments")
        / "rl"
        / "phase5_league_alpha"
        / "deck_specialists_report.json",
    )
    rl_train_specialists.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Alpha league iteration number to record in the aggregate report.",
    )
    rl_train_specialists.add_argument(
        "--deck-index",
        type=int,
        action="append",
        default=[],
        help="Phase 5 league deck index to train. Repeat; defaults to all 13.",
    )
    rl_train_specialists.add_argument(
        "--allow-empty-decks",
        action="store_true",
        help="Write reports even if a selected deck has zero examples.",
    )
    rl_train_specialists.add_argument("--epochs", type=int, default=1)
    rl_train_specialists.add_argument("--batch-size", type=int, default=64)
    rl_train_specialists.add_argument("--learning-rate", type=float, default=1.0e-4)
    rl_train_specialists.add_argument("--d-model", type=int, default=128)
    rl_train_specialists.add_argument("--max-entities", type=int, default=96)
    rl_train_specialists.add_argument("--max-actions", type=int, default=128)
    rl_train_specialists.add_argument("--max-previous-actions", type=int, default=16)
    rl_train_specialists.add_argument(
        "--target-source",
        choices=["search", "baseline", "rule"],
        default="search",
    )
    rl_train_specialists.add_argument(
        "--rule-demo-target-source",
        choices=["baseline", "rule"],
        default="baseline",
    )
    rl_train_specialists.add_argument("--changed-weight", type=float, default=2.0)
    rl_train_specialists.add_argument("--unchanged-weight", type=float, default=0.5)
    rl_train_specialists.add_argument("--search-decision-weight", type=float, default=1.0)
    rl_train_specialists.add_argument("--rule-demo-weight", type=float, default=0.25)
    rl_train_specialists.add_argument("--selfplay-weight", type=float, default=1.0)
    rl_train_specialists.add_argument("--pairwise-changed", action="store_true")
    rl_train_specialists.add_argument("--pairwise-weight", type=float, default=0.25)
    rl_train_specialists.add_argument("--pairwise-margin", type=float, default=1.0)
    rl_train_specialists.add_argument(
        "--pairwise-negatives",
        choices=["all", "baseline"],
        default="baseline",
    )
    rl_train_specialists.add_argument("--value-loss-weight", type=float, default=0.25)
    rl_train_specialists.add_argument(
        "--action-value-loss-weight",
        type=float,
        default=0.25,
    )
    rl_train_specialists.add_argument("--tactical-loss-weight", type=float, default=0.05)
    rl_train_specialists.add_argument("--decision-limit", type=int, default=0)
    rl_train_specialists.add_argument("--selfplay-limit", type=int, default=0)
    rl_train_specialists.set_defaults(func=command_rl_train_phase5_deck_specialists)

    rl_train_alpha_ppo_specialists = subparsers.add_parser(
        "rl-train-phase5-alpha-ppo-specialists",
        help="Apply on-policy PPO updates to one Phase 5 specialist per league deck.",
    )
    rl_train_alpha_ppo_specialists.add_argument(
        "--trajectory-dataset",
        type=_path,
        action="append",
        default=[],
        help="On-policy Phase 5 trajectory JSONL. Repeat for multiple raw windows.",
    )
    rl_train_alpha_ppo_specialists.add_argument(
        "--source-checkpoint-dir",
        type=_path,
        required=True,
        help="Directory containing source deck-01.pt through deck-13.pt checkpoints.",
    )
    rl_train_alpha_ppo_specialists.add_argument(
        "--output-checkpoint-dir",
        type=_path,
        required=True,
        help="Directory for updated deck-XX.pt checkpoints.",
    )
    rl_train_alpha_ppo_specialists.add_argument(
        "--report-dir",
        type=_path,
        required=True,
        help="Directory for per-deck PPO reports.",
    )
    rl_train_alpha_ppo_specialists.add_argument(
        "--report-json",
        type=_path,
        required=True,
        help="Aggregate PPO specialist update report.",
    )
    rl_train_alpha_ppo_specialists.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Alpha league iteration number to record in the aggregate report.",
    )
    rl_train_alpha_ppo_specialists.add_argument(
        "--deck-index",
        type=int,
        action="append",
        default=[],
        help="Phase 5 league deck index to update. Repeat; defaults to all 13.",
    )
    rl_train_alpha_ppo_specialists.add_argument(
        "--allow-empty-decks",
        action="store_true",
        help="Write reports even if a selected deck has zero PPO examples.",
    )
    rl_train_alpha_ppo_specialists.add_argument(
        "--allow-off-policy-trajectories",
        action="store_true",
        help="Allow legacy trajectories without policy_on_policy=true.",
    )
    rl_train_alpha_ppo_specialists.add_argument("--epochs", type=int, default=1)
    rl_train_alpha_ppo_specialists.add_argument("--batch-size", type=int, default=64)
    rl_train_alpha_ppo_specialists.add_argument("--learning-rate", type=float, default=5.0e-5)
    rl_train_alpha_ppo_specialists.add_argument("--clip-epsilon", type=float, default=0.2)
    rl_train_alpha_ppo_specialists.add_argument("--policy-loss-weight", type=float, default=1.0)
    rl_train_alpha_ppo_specialists.add_argument("--value-loss-weight", type=float, default=0.5)
    rl_train_alpha_ppo_specialists.add_argument("--entropy-weight", type=float, default=0.01)
    rl_train_alpha_ppo_specialists.add_argument(
        "--selfplay-limit",
        type=int,
        default=0,
        help="Maximum trajectory steps per epoch per deck. Use 0 for all input.",
    )
    rl_train_alpha_ppo_specialists.set_defaults(
        func=command_rl_train_phase5_alpha_ppo_specialists
    )

    rl_init_phase5_policy = subparsers.add_parser(
        "rl-init-phase5-policy-checkpoint",
        help="Create a scratch Phase 5 symbolic policy checkpoint.",
    )
    rl_init_phase5_policy.add_argument("--checkpoint", type=_path, required=True)
    rl_init_phase5_policy.add_argument("--report-json", type=_path, default=None)
    rl_init_phase5_policy.add_argument("--deck-index", type=int, default=None)
    rl_init_phase5_policy.add_argument("--controlled-public-agent-key", default=None)
    rl_init_phase5_policy.add_argument("--experiment", default="")
    rl_init_phase5_policy.add_argument("--max-entities", type=int, default=96)
    rl_init_phase5_policy.add_argument("--max-actions", type=int, default=128)
    rl_init_phase5_policy.add_argument("--max-previous-actions", type=int, default=16)
    rl_init_phase5_policy.add_argument("--d-model", type=int, default=128)
    rl_init_phase5_policy.add_argument("--seed", type=int, default=None)
    rl_init_phase5_policy.add_argument("--overwrite", action="store_true")
    rl_init_phase5_policy.set_defaults(func=command_rl_init_phase5_policy_checkpoint)

    rl_alpha_clean = subparsers.add_parser(
        "rl-clean-phase5-alpha-iteration",
        help="Remove raw Alpha league training data after a successful update.",
    )
    rl_alpha_clean.add_argument("--iteration-dir", type=_path, required=True)
    rl_alpha_clean.add_argument(
        "--update-report",
        type=_path,
        default=None,
        help="Training/update report proving raw data was consumed.",
    )
    rl_alpha_clean.add_argument(
        "--allow-missing-update-report",
        action="store_true",
        help="Allow cleanup without an update report. Use only for failed smoke cleanup.",
    )
    rl_alpha_clean.add_argument(
        "--report-json",
        type=_path,
        default=Path("experiments") / "rl" / "phase5_league_alpha" / "cleanup_report.json",
    )
    rl_alpha_clean.set_defaults(func=command_rl_clean_phase5_alpha_iteration)

    rl_search = subparsers.add_parser(
        "rl-generate-search-data",
        help="Generate Phase 5 search-improved decision records with one-turn root-search traces.",
    )
    rl_search.add_argument(
        "--sample-dir",
        type=_path,
        default=KAGGLE_INPUT_DIR / "sample_submission",
    )
    rl_search.add_argument("--games", type=int, default=2)
    rl_search.add_argument(
        "--game-offset",
        type=int,
        default=0,
        help="First global game index before shard interleaving.",
    )
    rl_search.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index for deterministic SLURM array generation.",
    )
    rl_search.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Total shard count for deterministic SLURM array generation.",
    )
    rl_search.add_argument("--max-steps", type=int, default=80)
    rl_search.add_argument("--top-k", type=int, default=4)
    rl_search.add_argument(
        "--rollout-steps",
        type=int,
        default=RootSearchConfig().max_rollout_steps,
    )
    rl_search.add_argument("--min-candidates", type=int, default=2)
    rl_search.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output JSONL files instead of overwriting them.",
    )
    rl_search.add_argument(
        "--require-changed",
        action="store_true",
        help="Return a failing status if no root-search decision differs from the rule baseline.",
    )
    rl_search.add_argument(
        "--output",
        type=_path,
        default=Path("data") / "datasets" / "rl" / "phase5_search_decisions.jsonl",
    )
    rl_search.add_argument(
        "--trace-output",
        type=_path,
        default=Path("experiments") / "rl" / "phase5_search_traces.jsonl",
    )
    rl_search.set_defaults(func=command_rl_generate_search_data)

    rl_merge_search = subparsers.add_parser(
        "rl-merge-search-data",
        help="Merge Phase 5 search-data shards into one decision JSONL and one trace JSONL.",
    )
    rl_merge_search.add_argument(
        "--input",
        action="append",
        required=True,
        help="Decision shard path or glob. Repeat for multiple patterns.",
    )
    rl_merge_search.add_argument(
        "--trace-input",
        action="append",
        required=True,
        help="Trace shard path or glob. Repeat for multiple patterns.",
    )
    rl_merge_search.add_argument(
        "--output",
        type=_path,
        default=Path("data") / "datasets" / "rl" / "phase5_search_decisions_merged.jsonl",
    )
    rl_merge_search.add_argument(
        "--trace-output",
        type=_path,
        default=Path("experiments") / "rl" / "phase5_search_traces_merged.jsonl",
    )
    rl_merge_search.add_argument(
        "--manifest",
        type=_path,
        default=Path("experiments") / "rl" / "phase5_search_merge_manifest.json",
    )
    rl_merge_search.set_defaults(func=command_rl_merge_search_data)

    rl_diagnose_search = subparsers.add_parser(
        "rl-diagnose-search-distill",
        help="Diagnose a Phase 5 search-distilled model against search-improved decision data.",
    )
    rl_diagnose_search.add_argument(
        "--dataset",
        type=_path,
        default=Path("data") / "datasets" / "rl" / "phase5_search_decisions_merged.jsonl",
    )
    rl_diagnose_search.add_argument(
        "--model",
        type=_path,
        default=Path("models") / "rl" / "phase5_search_distill.json",
    )
    rl_diagnose_search.add_argument(
        "--checkpoint",
        type=_path,
        default=None,
        help="Optional torch checkpoint to diagnose instead of the exported JSON model.",
    )
    rl_diagnose_search.add_argument(
        "--trace-input",
        type=_path,
        default=None,
        help="Optional merged Phase 5 trace JSONL for rollout-score diagnostics.",
    )
    rl_diagnose_search.add_argument("--examples", type=int, default=20)
    rl_diagnose_search.add_argument(
        "--report-json",
        type=_path,
        default=Path("reports") / "phase5_search_distill_diagnostics.json",
    )
    rl_diagnose_search.add_argument(
        "--report-md",
        type=_path,
        default=Path("reports") / "phase5_search_distill_diagnostics.md",
    )
    rl_diagnose_search.set_defaults(func=command_rl_diagnose_search_distill)

    rl_diagnose_traces = subparsers.add_parser(
        "rl-diagnose-search-traces",
        help="Summarize Phase 5 root-search trace JSONL files.",
    )
    rl_diagnose_traces.add_argument(
        "--trace-input",
        type=_path,
        default=Path("experiments") / "rl" / "phase5_search_traces_merged.jsonl",
    )
    rl_diagnose_traces.add_argument(
        "--report-json",
        type=_path,
        default=Path("reports") / "phase5_search_trace_diagnostics.json",
    )
    rl_diagnose_traces.add_argument(
        "--report-md",
        type=_path,
        default=Path("reports") / "phase5_search_trace_diagnostics.md",
    )
    rl_diagnose_traces.set_defaults(func=command_rl_diagnose_search_traces)

    rl_diagnose_scores = subparsers.add_parser(
        "rl-diagnose-search-score-components",
        help="Summarize root-search candidate score components by game outcome.",
    )
    rl_diagnose_scores.add_argument(
        "--trace-input",
        type=_path,
        default=Path("experiments") / "rl" / "phase5_public_agent_score_traces.jsonl",
    )
    rl_diagnose_scores.add_argument(
        "--report-json",
        type=_path,
        default=Path("reports") / "phase5_search_score_components.json",
    )
    rl_diagnose_scores.add_argument(
        "--report-md",
        type=_path,
        default=Path("reports") / "phase5_search_score_components.md",
    )
    rl_diagnose_scores.set_defaults(func=command_rl_diagnose_search_score_components)

    rl_diagnose_symbolic = subparsers.add_parser(
        "rl-diagnose-phase5-symbolic",
        help="Diagnose a Phase 5 symbolic checkpoint against search-improved decisions.",
    )
    rl_diagnose_symbolic.add_argument(
        "--dataset",
        type=_path,
        default=Path("data") / "datasets" / "rl" / "phase5_search_decisions_merged.jsonl",
    )
    rl_diagnose_symbolic.add_argument(
        "--checkpoint",
        type=_path,
        default=Path("models") / "rl" / "phase5_symbolic_policy_10shards.pt",
    )
    rl_diagnose_symbolic.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum source frames to diagnose. Use 0 for the full input.",
    )
    rl_diagnose_symbolic.add_argument("--batch-size", type=int, default=128)
    rl_diagnose_symbolic.add_argument("--examples", type=int, default=20)
    rl_diagnose_symbolic.add_argument(
        "--report-json",
        type=_path,
        default=Path("reports") / "phase5_symbolic_diagnostics.json",
    )
    rl_diagnose_symbolic.add_argument(
        "--report-md",
        type=_path,
        default=Path("reports") / "phase5_symbolic_diagnostics.md",
    )
    rl_diagnose_symbolic.set_defaults(func=command_rl_diagnose_phase5_symbolic)

    rl_train_ppo = subparsers.add_parser(
        "rl-train-ppo",
        help="Apply a reward-weighted PPO-style update from rollout JSONL chunks.",
    )
    rl_train_ppo.add_argument(
        "--rollout",
        type=_path,
        default=Path("data") / "datasets" / "rl" / "rollout_steps.jsonl",
    )
    rl_train_ppo.add_argument(
        "--model",
        type=_path,
        default=Path("models") / "rl" / "bc_model.json",
    )
    rl_train_ppo.add_argument(
        "--output-model",
        type=_path,
        default=Path("models") / "rl" / "ppo_model.json",
    )
    rl_train_ppo.add_argument("--epochs", type=int, default=1)
    rl_train_ppo.add_argument(
        "--report-json",
        type=_path,
        default=Path("experiments") / "rl" / "ppo_train_report.json",
    )
    rl_train_ppo.set_defaults(func=command_rl_train_ppo)

    rl_train_phase5_ppo = subparsers.add_parser(
        "rl-train-phase5-ppo",
        help="Apply a Phase 5 legal-action PPO-style update from trajectory JSONL.",
    )
    rl_train_phase5_ppo.add_argument(
        "--trajectory-dataset",
        type=_path,
        action="append",
        default=[],
        help="Phase 5 trajectory JSONL. Repeat for multiple self-play shards.",
    )
    rl_train_phase5_ppo.add_argument(
        "--checkpoint",
        type=_path,
        default=Path("models") / "rl" / "phase5_generalist_policy_13deck_10k.pt",
    )
    rl_train_phase5_ppo.add_argument(
        "--output-checkpoint",
        type=_path,
        default=Path("models") / "rl" / "phase5_generalist_policy_13deck_ppo.pt",
    )
    rl_train_phase5_ppo.add_argument(
        "--report-json",
        type=_path,
        default=Path("experiments") / "rl" / "phase5_ppo_train_report.json",
    )
    rl_train_phase5_ppo.add_argument("--epochs", type=int, default=1)
    rl_train_phase5_ppo.add_argument("--batch-size", type=int, default=64)
    rl_train_phase5_ppo.add_argument("--learning-rate", type=float, default=5.0e-5)
    rl_train_phase5_ppo.add_argument("--clip-epsilon", type=float, default=0.2)
    rl_train_phase5_ppo.add_argument("--policy-loss-weight", type=float, default=1.0)
    rl_train_phase5_ppo.add_argument("--value-loss-weight", type=float, default=0.5)
    rl_train_phase5_ppo.add_argument("--entropy-weight", type=float, default=0.01)
    rl_train_phase5_ppo.add_argument(
        "--deck-index-filter",
        type=int,
        default=None,
        help="Only train from trajectories whose reward metadata deck_index matches.",
    )
    rl_train_phase5_ppo.add_argument(
        "--require-on-policy",
        action="store_true",
        help="Skip trajectories without policy_on_policy=true metadata.",
    )
    rl_train_phase5_ppo.add_argument(
        "--selfplay-limit",
        type=int,
        default=0,
        help="Maximum trajectory steps per epoch. Use 0 for all input.",
    )
    rl_train_phase5_ppo.set_defaults(func=command_rl_train_phase5_ppo)

    rl_train_phase5_trajectory_bc = subparsers.add_parser(
        "rl-train-phase5-trajectory-bc",
        help="Behavior-clone a Phase 5 policy once from rule trajectories.",
    )
    rl_train_phase5_trajectory_bc.add_argument(
        "--rule-trajectory-dataset",
        type=_path,
        action="append",
        default=[],
        help="Rule-demonstration trajectory JSONL. Repeat for multiple datasets.",
    )
    rl_train_phase5_trajectory_bc.add_argument(
        "--checkpoint",
        type=_path,
        default=Path("models") / "rl" / "phase5_policy_scratch.pt",
    )
    rl_train_phase5_trajectory_bc.add_argument(
        "--output-checkpoint",
        type=_path,
        default=Path("models") / "rl" / "phase5_policy_rule_bc.pt",
    )
    rl_train_phase5_trajectory_bc.add_argument(
        "--report-json",
        type=_path,
        default=Path("experiments") / "rl" / "phase5_trajectory_bc_report.json",
    )
    rl_train_phase5_trajectory_bc.add_argument("--epochs", type=int, default=1)
    rl_train_phase5_trajectory_bc.add_argument("--batch-size", type=int, default=64)
    rl_train_phase5_trajectory_bc.add_argument(
        "--learning-rate",
        type=float,
        default=5.0e-5,
    )
    rl_train_phase5_trajectory_bc.add_argument(
        "--deck-index-filter",
        type=int,
        default=None,
        help="Only train from trajectories whose reward metadata deck_index matches.",
    )
    rl_train_phase5_trajectory_bc.add_argument(
        "--rule-step-limit",
        type=int,
        default=0,
        help="Maximum raw rule trajectory steps to scan. Use 0 for all input.",
    )
    rl_train_phase5_trajectory_bc.set_defaults(
        func=command_rl_train_phase5_trajectory_bc
    )

    rl_train_phase5_bc_ppo = subparsers.add_parser(
        "rl-train-phase5-bc-ppo",
        help=(
            "Apply rule behavior cloning and valid on-policy Phase 5 PPO "
            "with a balanced or PPO-led update schedule."
        ),
    )
    rl_train_phase5_bc_ppo.add_argument(
        "--rule-trajectory-dataset",
        type=_path,
        action="append",
        default=[],
        help="Rule-demonstration trajectory JSONL. Repeat for multiple datasets.",
    )
    rl_train_phase5_bc_ppo.add_argument(
        "--on-policy-trajectory-dataset",
        type=_path,
        action="append",
        default=[],
        help=(
            "Trajectory JSONL from policy_mode=sample or epsilon_mixture with "
            "policy_on_policy=true. Repeat for multiple datasets."
        ),
    )
    rl_train_phase5_bc_ppo.add_argument(
        "--checkpoint",
        type=_path,
        default=Path("models") / "rl" / "phase5_generalist_policy_13deck_10k.pt",
    )
    rl_train_phase5_bc_ppo.add_argument(
        "--output-checkpoint",
        type=_path,
        default=Path("models") / "rl" / "phase5_generalist_policy_bc_ppo.pt",
    )
    rl_train_phase5_bc_ppo.add_argument(
        "--report-json",
        type=_path,
        default=Path("experiments") / "rl" / "phase5_bc_ppo_train_report.json",
    )
    rl_train_phase5_bc_ppo.add_argument("--epochs", type=int, default=1)
    rl_train_phase5_bc_ppo.add_argument("--batch-size", type=int, default=64)
    rl_train_phase5_bc_ppo.add_argument("--learning-rate", type=float, default=5.0e-5)
    rl_train_phase5_bc_ppo.add_argument("--clip-epsilon", type=float, default=0.2)
    rl_train_phase5_bc_ppo.add_argument("--bc-loss-weight", type=float, default=1.0)
    rl_train_phase5_bc_ppo.add_argument(
        "--ppo-policy-loss-weight",
        type=float,
        default=1.0,
    )
    rl_train_phase5_bc_ppo.add_argument(
        "--ppo-value-loss-weight",
        type=float,
        default=0.5,
    )
    rl_train_phase5_bc_ppo.add_argument("--entropy-weight", type=float, default=0.01)
    rl_train_phase5_bc_ppo.add_argument(
        "--update-schedule",
        choices=["balanced-max", "ppo-epoch"],
        default="balanced-max",
        help=(
            "balanced-max preserves the original equal-source schedule; ppo-epoch "
            "uses every PPO example once and only a bounded rule anchor."
        ),
    )
    rl_train_phase5_bc_ppo.add_argument(
        "--rule-anchor-fraction",
        type=float,
        default=0.5,
        help=(
            "Rule-row fraction for ppo-epoch batches (0 to 0.5). This controls "
            "sampling; --bc-loss-weight controls objective strength."
        ),
    )
    rl_train_phase5_bc_ppo.add_argument(
        "--gradient-diagnostic-batches",
        type=int,
        default=0,
        help="Log weighted BC, PPO policy/value, and entropy gradients for the first N batches.",
    )
    rl_train_phase5_bc_ppo.add_argument(
        "--deck-index-filter",
        type=int,
        default=None,
        help="Only train from trajectories whose reward metadata deck_index matches.",
    )
    rl_train_phase5_bc_ppo.add_argument(
        "--rule-step-limit",
        type=int,
        default=0,
        help="Maximum raw rule trajectory steps to scan. Use 0 for all input.",
    )
    rl_train_phase5_bc_ppo.add_argument(
        "--on-policy-step-limit",
        type=int,
        default=0,
        help="Maximum raw on-policy trajectory steps to scan. Use 0 for all input.",
    )
    rl_train_phase5_bc_ppo.set_defaults(func=command_rl_train_phase5_bc_ppo)

    rl_evaluate = subparsers.add_parser(
        "rl-evaluate",
        help="Run the 9x4 required benchmark for rule, RL, hybrid, or Phase 5 symbolic agents.",
    )
    rl_evaluate.add_argument(
        "--sample-dir",
        type=_path,
        default=KAGGLE_INPUT_DIR / "sample_submission",
    )
    rl_evaluate.add_argument(
        "--agent",
        choices=["rule", "rl", "hybrid", "phase5-symbolic", "phase5-search", "phase5-full", "phase5-rl"],
        default="hybrid",
    )
    rl_evaluate.add_argument(
        "--model",
        type=_path,
        default=Path("models") / "rl" / "bc_model.json",
    )
    rl_evaluate.add_argument("--games-per-matchup", type=int, default=1)
    rl_evaluate.add_argument("--max-steps", type=int, default=120)
    _add_phase5_search_config_args(rl_evaluate)
    rl_evaluate.add_argument(
        "--report-json",
        type=_path,
        default=REPORTS_DIR / "phase4_required_benchmark.json",
    )
    rl_evaluate.add_argument(
        "--report-md",
        type=_path,
        default=REPORTS_DIR / "phase4_required_benchmark.md",
    )
    rl_evaluate.add_argument(
        "--search-trace-output",
        type=_path,
        default=None,
        help="Optional JSONL output for phase5-search root-search traces.",
    )
    rl_evaluate.set_defaults(func=command_rl_evaluate)

    rl_evaluate_league = subparsers.add_parser(
        "rl-evaluate-phase5-league",
        help="Run the 13x13 Phase 5 league benchmark against rule-agent opponents.",
    )
    rl_evaluate_league.add_argument(
        "--sample-dir",
        type=_path,
        default=KAGGLE_INPUT_DIR / "sample_submission",
    )
    rl_evaluate_league.add_argument(
        "--agent",
        choices=["rule", "rl", "hybrid", "phase5-symbolic", "phase5-search", "phase5-full", "phase5-rl"],
        default="phase5-search",
    )
    rl_evaluate_league.add_argument(
        "--model",
        type=_path,
        default=Path("models") / "rl" / "phase5_generalist_policy_13deck_10k.pt",
    )
    rl_evaluate_league.add_argument(
        "--specialist-model-dir",
        type=_path,
        default=None,
        help="Directory containing deck-01.pt through deck-13.pt specialist checkpoints.",
    )
    rl_evaluate_league.add_argument("--games-per-matchup", type=int, default=2)
    rl_evaluate_league.add_argument("--max-steps", type=int, default=600)
    _add_phase5_search_config_args(rl_evaluate_league)
    rl_evaluate_league.add_argument(
        "--report-json",
        type=_path,
        default=REPORTS_DIR / "phase5_league_benchmark.json",
    )
    rl_evaluate_league.add_argument(
        "--report-md",
        type=_path,
        default=REPORTS_DIR / "phase5_league_benchmark.md",
    )
    rl_evaluate_league.add_argument(
        "--search-trace-output",
        type=_path,
        default=None,
        help="Optional JSONL output for phase5-search root-search traces.",
    )
    rl_evaluate_league.set_defaults(func=command_rl_evaluate_phase5_league)

    rl_evaluate_public = subparsers.add_parser(
        "rl-evaluate-phase5-public-agents",
        help="Run Phase 5 specialists against locally available public/specialized Kaggle agents.",
    )
    rl_evaluate_public.add_argument(
        "--sample-dir",
        type=_path,
        default=KAGGLE_INPUT_DIR / "sample_submission",
    )
    rl_evaluate_public.add_argument(
        "--public-agent-root",
        type=_path,
        action="append",
        default=[],
        help="Directory containing downloaded/exported public agents. Repeatable.",
    )
    rl_evaluate_public.add_argument("--roster-notebook", type=_path, default=None)
    rl_evaluate_public.add_argument("--public-only", action="store_true")
    rl_evaluate_public.add_argument("--samples-only", action="store_true")
    rl_evaluate_public.add_argument("--no-builtin-samples", action="store_true")
    rl_evaluate_public.add_argument(
        "--public-agent-key",
        action="append",
        default=[],
        help="Restrict evaluation to a specific public-agent key. Repeatable.",
    )
    rl_evaluate_public.add_argument(
        "--controlled-public-agent-key",
        default=None,
        help="Use this public/sample agent's deck as the model-controlled deck.",
    )
    rl_evaluate_public.add_argument(
        "--controlled-deck-index",
        type=int,
        default=101,
        help="Synthetic deck index for --controlled-public-agent-key checkpoints/reports.",
    )
    rl_evaluate_public.add_argument("--require-min-opponents", type=int, default=1)
    rl_evaluate_public.add_argument(
        "--min-opponent-win-rate",
        type=float,
        default=0.5,
        help="Promotion gate threshold for public-opponent and controlled-deck aggregates.",
    )
    rl_evaluate_public.add_argument(
        "--fail-on-gate",
        action="store_true",
        help="Return a failing exit code when the public-agent gate is below threshold.",
    )
    rl_evaluate_public.add_argument(
        "--agent",
        choices=[
            "rule",
            "phase5-symbolic",
            "phase5-search",
            "phase5-full",
            "phase5-rl",
            "phase5-epsilon",
        ],
        default="phase5-full",
    )
    rl_evaluate_public.add_argument(
        "--policy-epsilon",
        type=float,
        default=0.0,
        help="Epsilon-greedy random legal-action probability for phase5-epsilon.",
    )
    rl_evaluate_public.add_argument(
        "--model",
        type=_path,
        default=Path("models") / "rl" / "phase5_generalist_policy_13deck_10k.pt",
    )
    rl_evaluate_public.add_argument(
        "--specialist-model-dir",
        type=_path,
        default=None,
        help="Directory containing deck-01.pt through deck-13.pt specialist checkpoints.",
    )
    rl_evaluate_public.add_argument(
        "--deck-index",
        type=int,
        action="append",
        default=[],
        help="Phase 5 league deck index to include. Repeat; defaults to all 13.",
    )
    rl_evaluate_public.add_argument("--games-per-matchup", type=int, default=2)
    rl_evaluate_public.add_argument("--max-steps", type=int, default=600)
    _add_phase5_search_config_args(rl_evaluate_public)
    rl_evaluate_public.add_argument(
        "--search-trace-output",
        type=_path,
        default=None,
        help="Optional JSONL output for phase5-full public-agent root-search traces.",
    )
    rl_evaluate_public.add_argument(
        "--search-trace-games",
        type=int,
        default=0,
        help="Number of games per matchup to trace; use 0 for all traced public-agent eval games.",
    )
    rl_evaluate_public.add_argument(
        "--replay-output-dir",
        type=_path,
        default=None,
        help="Optional directory for compact JSON and HTML replay views.",
    )
    rl_evaluate_public.add_argument("--saved-win-replays", type=int, default=0)
    rl_evaluate_public.add_argument("--saved-loss-replays", type=int, default=0)
    rl_evaluate_public.add_argument("--replay-trace-limit", type=int, default=120)
    rl_evaluate_public.add_argument(
        "--report-json",
        type=_path,
        default=REPORTS_DIR / "phase5_public_agent_benchmark.json",
    )
    rl_evaluate_public.add_argument(
        "--report-md",
        type=_path,
        default=REPORTS_DIR / "phase5_public_agent_benchmark.md",
    )
    rl_evaluate_public.add_argument(
        "--status-json",
        type=_path,
        default=REPORTS_DIR / "phase5_public_agent_status.json",
    )
    rl_evaluate_public.set_defaults(func=command_rl_evaluate_phase5_public_agents)

    rl_progression = subparsers.add_parser(
        "rl-image-progression",
        help="Run self-play/update/benchmark progression experiments by board image size.",
    )
    rl_progression.add_argument(
        "--sample-dir",
        type=_path,
        default=KAGGLE_INPUT_DIR / "sample_submission",
    )
    rl_progression.add_argument(
        "--image-size",
        type=int,
        action="append",
        default=[],
        help="Image dimension label to run. Repeat for multiple sizes.",
    )
    rl_progression.add_argument("--iterations", type=int, default=10)
    rl_progression.add_argument("--selfplay-games", type=int, default=1000)
    rl_progression.add_argument("--eval-games-per-matchup", type=int, default=100)
    rl_progression.add_argument("--max-steps", type=int, default=600)
    rl_progression.add_argument(
        "--deck-a-index",
        type=int,
        default=None,
        help="Optional fixed self-play deck A. If omitted with deck B, self-play rotates decks.",
    )
    rl_progression.add_argument(
        "--deck-b-index",
        type=int,
        default=None,
        help="Optional fixed self-play deck B. If omitted with deck A, self-play rotates decks.",
    )
    rl_progression.add_argument(
        "--selfplay-deck-index",
        type=int,
        action="append",
        default=[],
        help="Prepared deck index included in rotating self-play. Repeat to choose a subset.",
    )
    rl_progression.add_argument("--saved-replays-per-matchup", type=int, default=1)
    rl_progression.add_argument("--replay-trace-limit", type=int, default=60)
    rl_progression.add_argument("--update-epochs", type=int, default=1)
    rl_progression.add_argument(
        "--base-model",
        type=_path,
        default=None,
        help="Optional starting model JSON. If omitted, iteration 1 starts from rule fallback.",
    )
    rl_progression.add_argument(
        "--dataset-root",
        type=_path,
        default=Path("data") / "datasets" / "rl" / "image_progression",
    )
    rl_progression.add_argument(
        "--model-root",
        type=_path,
        default=Path("models") / "rl" / "image_progression",
    )
    rl_progression.add_argument(
        "--report-root",
        type=_path,
        default=Path("reports") / "image_progression",
    )
    rl_progression.add_argument(
        "--output-root",
        type=_path,
        default=Path("experiments") / "rl" / "image_progression",
    )
    rl_progression.set_defaults(func=command_rl_image_progression)

    rl_guidance = subparsers.add_parser(
        "rl-evaluate-guidance",
        help="Evaluate one Phase 4 rule-guidance intervention against the 9x4 grid.",
    )
    rl_guidance.add_argument(
        "--sample-dir",
        type=_path,
        default=KAGGLE_INPUT_DIR / "sample_submission",
    )
    rl_guidance.add_argument(
        "--model",
        type=_path,
        default=Path("models") / "rl" / "bc_model.json",
    )
    rl_guidance.add_argument("--guidance-rule", default="force_prize_attacks")
    rl_guidance.add_argument("--games-per-matchup", type=int, default=1)
    rl_guidance.add_argument("--max-steps", type=int, default=120)
    rl_guidance.add_argument("--baseline-json", type=_path)
    rl_guidance.add_argument(
        "--report-json",
        type=_path,
        default=REPORTS_DIR / "phase4_guidance_eval.json",
    )
    rl_guidance.add_argument(
        "--report-md",
        type=_path,
        default=REPORTS_DIR / "phase4_guidance_eval.md",
    )
    rl_guidance.set_defaults(func=command_rl_evaluate_guidance)

    rl_package = subparsers.add_parser(
        "rl-package",
        help="Build Phase 4 hybrid Kaggle submission bundles for selected Tournament 559 decks.",
    )
    rl_package.add_argument(
        "--sample-dir",
        type=_path,
        default=KAGGLE_INPUT_DIR / "sample_submission",
    )
    rl_package.add_argument(
        "--model",
        type=_path,
        default=Path("models") / "rl" / "bc_model.json",
    )
    rl_package.add_argument(
        "--output-dir",
        type=_path,
        default=Path("submissions") / "phase4",
    )
    rl_package.add_argument(
        "--deck-index",
        type=int,
        action="append",
        help="Tournament 559 prepared deck index to package. Repeat for multiple decks.",
    )
    rl_package.set_defaults(func=command_rl_package)

    phase5_package = subparsers.add_parser(
        "phase5-package",
        help="Build Phase 5 phase5-search Kaggle submission bundles.",
    )
    phase5_package.add_argument(
        "--sample-dir",
        type=_path,
        default=KAGGLE_INPUT_DIR / "sample_submission",
    )
    phase5_package.add_argument(
        "--model",
        type=_path,
        default=Path("models") / "rl" / "phase5_generalist_policy_10k.pt",
    )
    phase5_package.add_argument(
        "--model-dir",
        type=_path,
        default=None,
        help=(
            "Directory containing per-deck specialist checkpoints named "
            "deck-XX.pt. When set, each packaged deck uses its matching "
            "specialist and --model is ignored."
        ),
    )
    phase5_package.add_argument(
        "--output-dir",
        type=_path,
        default=Path("submissions") / "phase5_search_generalist_10k",
    )
    phase5_package.add_argument(
        "--deck-pool",
        choices=("tournament-9", "league-13"),
        default="tournament-9",
        help=(
            "Prepared deck pool to package. Use league-13 for the Phase 5 "
            "specialist decks 10-13."
        ),
    )
    phase5_package.add_argument(
        "--deck-index",
        type=int,
        action="append",
        help=(
            "Prepared deck index to package from --deck-pool. Defaults to "
            "decks 2 and 8."
        ),
    )
    phase5_package.set_defaults(func=command_phase5_package)

    phase5_compare = subparsers.add_parser(
        "phase5-compare-benchmarks",
        help="Compare two Phase 5 benchmark JSON reports for promotion review.",
    )
    phase5_compare.add_argument("--baseline", type=_path, required=True)
    phase5_compare.add_argument("--candidate", type=_path, required=True)
    phase5_compare.add_argument(
        "--report-json",
        type=_path,
        default=REPORTS_DIR / "phase5_benchmark_comparison.json",
    )
    phase5_compare.add_argument(
        "--report-md",
        type=_path,
        default=REPORTS_DIR / "phase5_benchmark_comparison.md",
    )
    phase5_compare.set_defaults(func=command_phase5_compare_benchmarks)

    rl_snapshots = subparsers.add_parser(
        "rl-board-snapshots",
        help="Run one rule-vs-benchmark game and write synthetic tabletop board snapshots.",
    )
    rl_snapshots.add_argument(
        "--sample-dir",
        type=_path,
        default=KAGGLE_INPUT_DIR / "sample_submission",
    )
    rl_snapshots.add_argument(
        "--output-dir",
        type=_path,
        default=REPORTS_DIR / "phase4_board_snapshots",
    )
    rl_snapshots.add_argument(
        "--our-deck-index",
        type=int,
        default=9,
        help="Prepared Tournament 559 deck index. Defaults to Ogerpon Box.",
    )
    rl_snapshots.add_argument(
        "--benchmark-index",
        type=int,
        default=4,
        help="Required benchmark deck index: 1 Crustle, 2 Mega Lucario, 3 Mega Abomasnow, 4 Iono.",
    )
    rl_snapshots.add_argument("--max-steps", type=int, default=120)
    rl_snapshots.add_argument("--image-limit", type=int, default=0)
    rl_snapshots.add_argument(
        "--turns-per-player",
        type=int,
        default=0,
        help="Only record decisions from the first N non-setup turns per player. 0 records all turns.",
    )
    rl_snapshots.add_argument(
        "--card-art-pdf",
        type=_path,
        default=KAGGLE_INPUT_DIR / "Card_ID List_EN.pdf",
        help="Kaggle Card ID PDF used to extract face-up card art.",
    )
    rl_snapshots.add_argument(
        "--card-art-dir",
        type=_path,
        default=PROCESSED_DIR / "card_art" / "en",
        help="Directory for cached card art PNGs extracted from the Card ID PDF.",
    )
    rl_snapshots.add_argument(
        "--record-player",
        choices=["ours", "benchmark", "both"],
        default="both",
    )
    rl_snapshots.set_defaults(func=command_rl_board_snapshots)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
