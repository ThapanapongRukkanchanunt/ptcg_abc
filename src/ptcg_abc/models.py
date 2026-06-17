from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class CardLine:
    count: int
    name: str
    section: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Archetype:
    rank: int
    name: str
    deck_id: str
    points: int | None
    share: str | None
    source_url: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Variant:
    name: str
    value: str | None
    source_url: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TournamentResult:
    event_name: str
    event_date: str
    placement: str
    placement_rank: int
    player: str
    decklist_url: str
    source_url: str
    page_order: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Decklist:
    archetype: Archetype
    variant: Variant
    result: TournamentResult
    title: str
    cards: list[CardLine]
    total_cards: int
    fingerprint: str
    source_url: str

    def to_dict(self) -> dict:
        return {
            "archetype": self.archetype.to_dict(),
            "variant": self.variant.to_dict(),
            "result": self.result.to_dict(),
            "title": self.title,
            "cards": [card.to_dict() for card in self.cards],
            "total_cards": self.total_cards,
            "fingerprint": self.fingerprint,
            "source_url": self.source_url,
        }


@dataclass
class CollectionResult:
    decks: list[Decklist] = field(default_factory=list)
    skips: list[dict] = field(default_factory=list)

    def add_skip(self, reason: str, **details: object) -> None:
        self.skips.append({"reason": reason, **details})
