from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ptcg_abc.normalize import clean_text, normalize_card_name


CARD_ID_COLUMNS = {"card id", "card_id", "cardid", "id"}
CARD_NAME_COLUMNS = {"card name", "card_name", "cardname", "name"}


@dataclass(frozen=True)
class CardIdLookup:
    name_to_ids: dict[str, tuple[int, ...]]
    display_names: dict[str, str]

    def ids_for_name(self, name: str) -> tuple[int, ...]:
        return self.name_to_ids.get(normalize_card_name(name), ())

    def preferred_id(self, name: str) -> int:
        ids = self.ids_for_name(name)
        if not ids:
            raise KeyError(name)
        return ids[0]

    def missing_names(self, names: Iterable[str]) -> list[str]:
        missing = {
            clean_text(name)
            for name in names
            if clean_text(name) and not self.ids_for_name(name)
        }
        return sorted(missing, key=str.casefold)

    def ambiguous_names(self, names: Iterable[str]) -> dict[str, tuple[int, ...]]:
        ambiguous: dict[str, tuple[int, ...]] = {}
        display_names: dict[str, str] = {}
        for name in names:
            normalized = normalize_card_name(name)
            ids = self.name_to_ids.get(normalized, ())
            if len(ids) > 1:
                display_names.setdefault(normalized, clean_text(name))
                ambiguous[display_names[normalized]] = ids
        return dict(sorted(ambiguous.items(), key=lambda item: item[0].casefold()))


def _column_by_normalized_name(fieldnames: list[str], candidates: set[str]) -> str:
    normalized = {clean_text(field).casefold(): field for field in fieldnames}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(f"Could not find one of {sorted(candidates)} in CSV columns.")


def load_card_id_lookup(path: Path) -> CardIdLookup:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row.")
        id_column = _column_by_normalized_name(reader.fieldnames, CARD_ID_COLUMNS)
        name_column = _column_by_normalized_name(reader.fieldnames, CARD_NAME_COLUMNS)

        ids_by_name: dict[str, set[int]] = {}
        display_names: dict[str, str] = {}
        for row in reader:
            raw_id = clean_text(row.get(id_column, ""))
            raw_name = clean_text(row.get(name_column, ""))
            if not raw_id or not raw_name:
                continue
            try:
                card_id = int(raw_id)
            except ValueError:
                continue
            normalized = normalize_card_name(raw_name)
            ids_by_name.setdefault(normalized, set()).add(card_id)
            display_names.setdefault(normalized, raw_name)

    return CardIdLookup(
        name_to_ids={name: tuple(sorted(ids)) for name, ids in ids_by_name.items()},
        display_names=display_names,
    )
