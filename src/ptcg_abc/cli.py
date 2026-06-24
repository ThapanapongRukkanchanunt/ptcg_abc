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
from ptcg_abc.limitless import (
    collect_limitless_decks,
    deck_collection_summary,
    write_deck_collection,
    write_missing_report,
)
from ptcg_abc.simulator import run_battle_smoke
from ptcg_abc.submission import build_hybrid_rl_submission_bundle, build_submission_bundle
from ptcg_abc.rl.workflow import (
    collect_bc_demonstrations,
    rollout_games,
    run_image_progression_experiment,
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
from ptcg_abc.rl.phase5_diagnostics import diagnose_search_distillation
from ptcg_abc.rl.phase5_policy import Phase5PolicyUnavailable
from ptcg_abc.rl.snapshots import run_rule_vs_benchmark_snapshots
from ptcg_abc.rl.phase5_symbolic_diagnostics import diagnose_phase5_symbolic_policy
from ptcg_abc.rl.phase5_symbolic_training import (
    build_phase5_symbolic_dataset,
    train_phase5_symbolic_policy_from_decisions,
)
from ptcg_abc.rl.torch_backend import TorchBackendUnavailable


def _path(value: str) -> Path:
    return Path(value)


def _add_common_limitless_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--snapshot-date", default=date.today().isoformat())
    parser.add_argument("--top-archetypes", type=int, default=10)
    parser.add_argument("--lists-per-variant", type=int, default=2)
    parser.add_argument("--candidate-limit", type=int, default=250)
    parser.add_argument("--delay-seconds", type=float, default=0.2)
    parser.add_argument("--limitless-format", default=LIMITLESS_FORMAT)
    parser.add_argument("--refresh", action="store_true")


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


def command_rl_rollout(args: argparse.Namespace) -> int:
    if not args.sample_dir.exists():
        print(
            f"Kaggle sample submission not found at {args.sample_dir}. "
            "Run `python -m ptcg_abc kaggle-setup` first.",
            file=sys.stderr,
        )
        return 2
    model_path = args.model if args.model and args.model.exists() else None
    if args.agent == "phase5-symbolic" and model_path is None:
        print(
            f"Phase 5 symbolic checkpoint not found at {args.model}.",
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


def command_rl_evaluate(args: argparse.Namespace) -> int:
    if not args.sample_dir.exists():
        print(
            f"Kaggle sample submission not found at {args.sample_dir}. "
            "Run `python -m ptcg_abc kaggle-setup` first.",
            file=sys.stderr,
        )
        return 2
    model_path = args.model if args.model and args.model.exists() else None
    if args.agent == "phase5-symbolic" and model_path is None:
        print(
            f"Phase 5 symbolic checkpoint not found at {args.model}.",
            file=sys.stderr,
        )
        return 2
    result = run_phase4_required_benchmark(
        sample_dir=args.sample_dir,
        agent_kind=args.agent,
        model_path=model_path,
        games_per_matchup=args.games_per_matchup,
        max_steps=args.max_steps,
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
        choices=["rule", "rl", "hybrid", "phase5-symbolic"],
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
    rl_search.add_argument("--rollout-steps", type=int, default=18)
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
        choices=["rule", "rl", "hybrid", "phase5-symbolic"],
        default="hybrid",
    )
    rl_evaluate.add_argument(
        "--model",
        type=_path,
        default=Path("models") / "rl" / "bc_model.json",
    )
    rl_evaluate.add_argument("--games-per-matchup", type=int, default=1)
    rl_evaluate.add_argument("--max-steps", type=int, default=120)
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
    rl_evaluate.set_defaults(func=command_rl_evaluate)

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
