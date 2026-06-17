from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ptcg_abc.card_db import CardIdLookup
from ptcg_abc.models import Archetype, CardLine, Decklist, TournamentResult, Variant


def _archetype_from_dict(value: dict) -> Archetype:
    return Archetype(
        rank=int(value["rank"]),
        name=str(value["name"]),
        deck_id=str(value["deck_id"]),
        points=value.get("points"),
        share=value.get("share"),
        source_url=str(value["source_url"]),
    )


def _variant_from_dict(value: dict) -> Variant:
    return Variant(
        name=str(value["name"]),
        value=value.get("value"),
        source_url=str(value["source_url"]),
    )


def _result_from_dict(value: dict) -> TournamentResult:
    return TournamentResult(
        event_name=str(value["event_name"]),
        event_date=str(value["event_date"]),
        placement=str(value["placement"]),
        placement_rank=int(value["placement_rank"]),
        player=str(value["player"]),
        decklist_url=str(value["decklist_url"]),
        source_url=str(value["source_url"]),
        page_order=int(value["page_order"]),
    )


def deck_from_record(record: dict) -> Decklist:
    return Decklist(
        archetype=_archetype_from_dict(record["archetype"]),
        variant=_variant_from_dict(record["variant"]),
        result=_result_from_dict(record["result"]),
        title=str(record["title"]),
        cards=[
            CardLine(
                count=int(card["count"]),
                name=str(card["name"]),
                section=str(card.get("section", "")),
            )
            for card in record["cards"]
        ],
        total_cards=int(record["total_cards"]),
        fingerprint=str(record["fingerprint"]),
        source_url=str(record["source_url"]),
    )


def load_deck_corpus(path: Path) -> list[Decklist]:
    decks: list[Decklist] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            decks.append(deck_from_record(record))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read deck corpus record {line_number} from {path}.") from exc
    return decks


def deck_card_names(deck: Decklist) -> Iterable[str]:
    for card in deck.cards:
        yield card.name


def deck_to_card_ids(deck: Decklist, lookup: CardIdLookup) -> list[int]:
    missing = lookup.missing_names(deck_card_names(deck))
    if missing:
        joined = ", ".join(missing[:12])
        suffix = "" if len(missing) <= 12 else f", and {len(missing) - 12} more"
        raise KeyError(f"Missing Kaggle card IDs for {joined}{suffix}.")

    card_ids: list[int] = []
    for card in deck.cards:
        card_ids.extend([lookup.preferred_id(card.name)] * card.count)

    if len(card_ids) != 60:
        raise ValueError(f"Deck {deck.title} resolves to {len(card_ids)} cards, expected 60.")
    return card_ids


def get_deck_by_index(decks: list[Decklist], deck_index: int) -> Decklist:
    if deck_index < 1 or deck_index > len(decks):
        raise IndexError(f"Deck index must be between 1 and {len(decks)}.")
    return decks[deck_index - 1]


def write_deck_csv(card_ids: list[int], path: Path) -> None:
    if len(card_ids) != 60:
        raise ValueError(f"Expected 60 card IDs, got {len(card_ids)}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(card_id) for card_id in card_ids) + "\n", encoding="utf-8")


def deck_label(deck: Decklist) -> str:
    return (
        f"{deck.archetype.name} / {deck.variant.name} / "
        f"{deck.result.player} {deck.result.placement}"
    )
