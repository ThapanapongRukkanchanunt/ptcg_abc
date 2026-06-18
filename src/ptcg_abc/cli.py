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
    prepare_decks,
    run_archetype_sweep,
    run_random_evaluation,
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
from ptcg_abc.submission import build_submission_bundle


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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
