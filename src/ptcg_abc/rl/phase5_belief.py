from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from ptcg_abc.evaluation import PreparedDeck, phase5_league_prepared_decks


@dataclass(frozen=True)
class OpponentDeckPrior:
    index: int
    label: str
    card_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["card_ids"] = list(self.card_ids)
        return payload


@dataclass(frozen=True)
class OpponentDeckInference:
    index: int
    label: str
    card_ids: tuple[int, ...]
    visible_ids: tuple[int, ...]
    overlap: int
    coverage: int
    score: tuple[int, int]
    used_default: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["card_ids"] = list(self.card_ids)
        payload["visible_ids"] = list(self.visible_ids)
        payload["score"] = list(self.score)
        return payload


def phase5_league_opponent_priors() -> list[OpponentDeckPrior]:
    return [
        opponent_prior_from_prepared_deck(deck)
        for deck in phase5_league_prepared_decks()
    ]


def opponent_prior_from_prepared_deck(deck: PreparedDeck) -> OpponentDeckPrior:
    return OpponentDeckPrior(
        index=deck.index,
        label=deck.label,
        card_ids=tuple(int(card_id) for card_id in deck.card_ids),
    )


def infer_opponent_deck(
    observation: Any,
    priors: Sequence[OpponentDeckPrior],
    *,
    default_label_prefix: str = "Crustle / Required benchmark sample",
) -> OpponentDeckInference:
    if not priors:
        raise ValueError("At least one opponent deck prior is required.")
    visible = tuple(visible_opponent_card_ids(observation))
    if not visible:
        default = _default_prior(priors, default_label_prefix=default_label_prefix)
        return OpponentDeckInference(
            index=default.index,
            label=default.label,
            card_ids=default.card_ids,
            visible_ids=(),
            overlap=0,
            coverage=0,
            score=(0, 0),
            used_default=True,
        )

    visible_counts = Counter(visible)
    best_prior = priors[0]
    best_overlap = 0
    best_coverage = 0
    best_score: tuple[int, int, int] | None = None
    for prior in priors:
        deck_counts = Counter(prior.card_ids)
        overlap = sum(
            min(count, deck_counts.get(card_id, 0))
            for card_id, count in visible_counts.items()
        )
        coverage = sum(1 for card_id in visible_counts if card_id in deck_counts)
        score = (overlap, coverage, -prior.index)
        if best_score is None or score > best_score:
            best_score = score
            best_prior = prior
            best_overlap = overlap
            best_coverage = coverage
    return OpponentDeckInference(
        index=best_prior.index,
        label=best_prior.label,
        card_ids=best_prior.card_ids,
        visible_ids=visible,
        overlap=best_overlap,
        coverage=best_coverage,
        score=(best_overlap, best_coverage),
    )


def visible_opponent_card_ids(observation: Any) -> list[int]:
    current = _get(observation, "current")
    players = list(_get(current, "players", []) or [])
    your_index = int(_get(current, "yourIndex", 0) or 0)
    opponent_index = 1 - your_index
    opponent = players[opponent_index] if 0 <= opponent_index < len(players) else None
    ids: list[int] = []
    for zone_name in ("active", "bench", "discard", "lostZone"):
        ids.extend(card_ids_from_value(_get(opponent, zone_name)))
    return ids


def card_ids_from_value(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, dict):
        output: list[int] = []
        for key in ("id", "cardID", "cardId", "card_ids", "cardIds", "cards"):
            if key in value:
                output.extend(card_ids_from_value(value[key]))
        return output
    if isinstance(value, (list, tuple)):
        output: list[int] = []
        for item in value:
            output.extend(card_ids_from_value(item))
        return output
    for name in ("id", "cardID", "cardId"):
        if hasattr(value, name):
            try:
                return [int(getattr(value, name))]
            except (TypeError, ValueError):
                return []
    for name in ("cards", "card_ids", "cardIds"):
        if hasattr(value, name):
            return card_ids_from_value(getattr(value, name))
    return []


def _default_prior(
    priors: Sequence[OpponentDeckPrior],
    *,
    default_label_prefix: str,
) -> OpponentDeckPrior:
    for prior in priors:
        if prior.label.startswith(default_label_prefix):
            return prior
    return priors[0]


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)
