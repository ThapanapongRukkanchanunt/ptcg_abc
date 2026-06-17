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
    REPORTS_DIR,
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


def _path(value: str) -> Path:
    return Path(value)


def _add_common_limitless_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--snapshot-date", default=date.today().isoformat())
    parser.add_argument("--top-archetypes", type=int, default=10)
    parser.add_argument("--lists-per-variant", type=int, default=2)
    parser.add_argument("--candidate-limit", type=int, default=250)
    parser.add_argument("--delay-seconds", type=float, default=0.2)
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
    )
    write_deck_collection(collection, snapshot_date=args.snapshot_date)
    report_path = write_missing_report(collection, legal_cards, args.output)
    print(deck_collection_summary(collection))
    print(f"Wrote missing-card report to {report_path}.")
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
