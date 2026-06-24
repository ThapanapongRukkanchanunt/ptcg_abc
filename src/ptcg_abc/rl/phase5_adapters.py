from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from ptcg_abc.agent.rule_based import (
    _area_name,
    _card_id,
    _card_name,
    _card_type_name,
    _get,
    _int_or_none,
)
from ptcg_abc.rl.featurizer import card_lookup, make_decision_frame


@dataclass(frozen=True)
class CardEntity:
    card_id: int | None
    name: str
    owner: int | None
    zone: str
    slot: int | None = None
    known: bool = True
    card_type: str = ""
    hp: int = 0
    max_hp: int = 0
    energy_count: int = 0
    tool_count: int = 0
    pre_evolution_count: int = 0
    is_ex: bool = False
    is_mega_ex: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlayerState:
    player_index: int
    is_us: bool
    hand_count: int
    deck_count: int
    prize_count: int
    active: tuple[CardEntity, ...] = ()
    bench: tuple[CardEntity, ...] = ()
    hand: tuple[CardEntity, ...] = ()
    discard: tuple[CardEntity, ...] = ()
    prize: tuple[CardEntity, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("active", "bench", "hand", "discard", "prize"):
            payload[key] = [card.to_dict() for card in getattr(self, key)]
        return payload


@dataclass(frozen=True)
class GameState:
    turn: int
    your_index: int
    energy_attached: bool
    supporter_played: bool
    stadium_count: int
    looking_count: int
    players: tuple[PlayerState, ...]
    stadium: tuple[CardEntity, ...] = ()
    looking: tuple[CardEntity, ...] = ()

    @property
    def entities(self) -> tuple[CardEntity, ...]:
        output: list[CardEntity] = []
        for player in self.players:
            output.extend(player.active)
            output.extend(player.bench)
            output.extend(player.hand)
            output.extend(player.discard)
            output.extend(player.prize)
        output.extend(self.stadium)
        output.extend(self.looking)
        return tuple(output)

    @property
    def us(self) -> PlayerState | None:
        return self.player(self.your_index)

    @property
    def opponent(self) -> PlayerState | None:
        return self.player(1 - self.your_index)

    def player(self, index: int) -> PlayerState | None:
        for player in self.players:
            if player.player_index == index:
                return player
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "your_index": self.your_index,
            "energy_attached": self.energy_attached,
            "supporter_played": self.supporter_played,
            "stadium_count": self.stadium_count,
            "looking_count": self.looking_count,
            "players": [player.to_dict() for player in self.players],
            "stadium": [card.to_dict() for card in self.stadium],
            "looking": [card.to_dict() for card in self.looking],
        }


@dataclass(frozen=True)
class LegalAction:
    local_index: int
    selected_indices: tuple[int, ...]
    action_type: str
    select_type: str
    context: str
    min_count: int
    max_count: int
    target_count: int
    rule_score: float = 0.0
    rule_rank: int = 0
    card_id: int | None = None
    card_name: str = ""
    target_card_id: int | None = None
    target_name: str = ""
    area: str = ""
    area_index: int | None = None
    target_area: str = ""
    target_index: int | None = None
    attack_id: int | None = None
    ends_turn: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selected_indices"] = list(self.selected_indices)
        return payload


@dataclass(frozen=True)
class BeliefState:
    own_unknown_prize_count: int
    own_unknown_deck_count: int
    opponent_unknown_hand_count: int
    opponent_unknown_deck_count: int
    opponent_unknown_prize_count: int
    observed_card_counts: dict[int, int] = field(default_factory=dict)
    own_observed_card_counts: dict[int, int] = field(default_factory=dict)
    opponent_observed_card_counts: dict[int, int] = field(default_factory=dict)
    own_deck_candidates: tuple[int, ...] = ()
    opponent_deck_candidates: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "own_unknown_prize_count": self.own_unknown_prize_count,
            "own_unknown_deck_count": self.own_unknown_deck_count,
            "opponent_unknown_hand_count": self.opponent_unknown_hand_count,
            "opponent_unknown_deck_count": self.opponent_unknown_deck_count,
            "opponent_unknown_prize_count": self.opponent_unknown_prize_count,
            "observed_card_counts": dict(self.observed_card_counts),
            "own_observed_card_counts": dict(self.own_observed_card_counts),
            "opponent_observed_card_counts": dict(self.opponent_observed_card_counts),
            "own_deck_candidates": list(self.own_deck_candidates),
            "opponent_deck_candidates": list(self.opponent_deck_candidates),
        }

    def sample_hidden_state(self) -> dict[str, list[int]]:
        own_candidates = list(self.own_deck_candidates)
        own_deck_end = self.own_unknown_deck_count
        own_prize_end = own_deck_end + self.own_unknown_prize_count
        opponent_candidates = list(self.opponent_deck_candidates)
        opponent_hand_end = self.opponent_unknown_hand_count
        opponent_deck_end = opponent_hand_end + self.opponent_unknown_deck_count
        opponent_prize_end = opponent_deck_end + self.opponent_unknown_prize_count
        return {
            "your_deck": own_candidates[:own_deck_end],
            "your_prize": own_candidates[own_deck_end:own_prize_end],
            "opponent_hand": opponent_candidates[:opponent_hand_end],
            "opponent_deck": opponent_candidates[opponent_hand_end:opponent_deck_end],
            "opponent_prize": opponent_candidates[opponent_deck_end:opponent_prize_end],
        }


@dataclass
class GameMemory:
    observed_card_counts: Counter[int] = field(default_factory=Counter)
    observed_card_counts_by_owner: dict[int, Counter[int]] = field(default_factory=dict)
    last_turn: int = 0
    observations: int = 0

    def observe(self, state: GameState) -> None:
        self.last_turn = state.turn
        self.observations += 1
        counts: Counter[int] = Counter()
        owner_counts: dict[int, Counter[int]] = {}
        for entity in state.entities:
            if entity.known and entity.card_id is not None:
                counts[int(entity.card_id)] += 1
                if entity.owner is not None:
                    owner_counts.setdefault(int(entity.owner), Counter())[
                        int(entity.card_id)
                    ] += 1
        self.observed_card_counts = counts
        self.observed_card_counts_by_owner = owner_counts

    def belief_state(
        self,
        state: GameState,
        *,
        own_deck_ids: Sequence[int] = (),
        opponent_deck_ids: Sequence[int] = (),
    ) -> BeliefState:
        us = state.us
        opponent = state.opponent
        own_prizes = us.prize_count if us is not None else 0
        opponent_hand = opponent.hand_count if opponent is not None else 0
        opponent_deck = opponent.deck_count if opponent is not None else 0
        opponent_prizes = opponent.prize_count if opponent is not None else 0
        own_observed = self.observed_card_counts_by_owner.get(state.your_index, Counter())
        opponent_observed = self.observed_card_counts_by_owner.get(
            1 - state.your_index,
            Counter(),
        )
        return BeliefState(
            own_unknown_prize_count=max(
                0,
                own_prizes - _known_count(us.prize if us else ()),
            ),
            own_unknown_deck_count=max(0, us.deck_count if us is not None else 0),
            opponent_unknown_hand_count=max(
                0,
                opponent_hand - _known_count(opponent.hand if opponent else ()),
            ),
            opponent_unknown_deck_count=max(0, opponent_deck),
            opponent_unknown_prize_count=max(
                0,
                opponent_prizes - _known_count(opponent.prize if opponent else ()),
            ),
            observed_card_counts=dict(self.observed_card_counts),
            own_observed_card_counts=dict(own_observed),
            opponent_observed_card_counts=dict(opponent_observed),
            own_deck_candidates=tuple(_remaining_candidates(own_deck_ids, own_observed)),
            opponent_deck_candidates=tuple(
                _remaining_candidates(opponent_deck_ids, opponent_observed)
            ),
        )


class StateAdapter:
    def __init__(self, *, card_data: Sequence[Any] = ()) -> None:
        self.card_by_id = card_lookup(card_data)

    def parse(self, observation: Any) -> GameState:
        current = _get(observation, "current")
        players = list(_get(current, "players", []) or [])
        your_index = int(_get(current, "yourIndex", 0) or 0)
        return GameState(
            turn=int(_get(current, "turn", 0) or 0),
            your_index=your_index,
            energy_attached=bool(_get(current, "energyAttached", False)),
            supporter_played=bool(_get(current, "supporterPlayed", False)),
            stadium_count=len(list(_get(current, "stadium", []) or [])),
            looking_count=len(list(_get(current, "looking", []) or [])),
            players=tuple(
                self._player_state(player, index, your_index=your_index)
                for index, player in enumerate(players)
            ),
            stadium=tuple(
                self._card_entity(card, owner=None, zone="STADIUM", slot=index)
                for index, card in enumerate(list(_get(current, "stadium", []) or []))
                if card is not None
            ),
            looking=tuple(
                self._card_entity(card, owner=your_index, zone="LOOKING", slot=index)
                for index, card in enumerate(list(_get(current, "looking", []) or []))
                if card is not None
            ),
        )

    def _player_state(self, player: Any, player_index: int, *, your_index: int) -> PlayerState:
        active = list(_get(player, "active", []) or [])
        bench = list(_get(player, "bench", []) or [])
        hand = list(_get(player, "hand", []) or [])
        discard = list(_get(player, "discard", []) or [])
        prize = list(_get(player, "prize", []) or [])
        hand_count = int(_get(player, "handCount", len(hand)) or len(hand))
        return PlayerState(
            player_index=player_index,
            is_us=player_index == your_index,
            hand_count=hand_count,
            deck_count=int(_get(player, "deckCount", 0) or 0),
            prize_count=len(prize),
            active=tuple(
                self._card_entity(card, owner=player_index, zone="ACTIVE", slot=index)
                for index, card in enumerate(active)
                if card is not None
            ),
            bench=tuple(
                self._card_entity(card, owner=player_index, zone="BENCH", slot=index)
                for index, card in enumerate(bench)
                if card is not None
            ),
            hand=tuple(
                self._card_entity(card, owner=player_index, zone="HAND", slot=index)
                for index, card in enumerate(hand)
                if card is not None
            ),
            discard=tuple(
                self._card_entity(card, owner=player_index, zone="DISCARD", slot=index)
                for index, card in enumerate(discard)
                if card is not None
            ),
            prize=tuple(
                self._card_entity(card, owner=player_index, zone="PRIZE", slot=index)
                for index, card in enumerate(prize)
                if card is not None
            ),
        )

    def _card_entity(
        self,
        card: Any,
        *,
        owner: int | None,
        zone: str,
        slot: int | None,
    ) -> CardEntity:
        card_id = _card_id(card)
        data = self.card_by_id.get(card_id)
        hp = int(_get(card, "hp", _get(data, "hp", 0)) or 0)
        max_hp = int(_get(card, "maxHp", _get(data, "hp", hp)) or hp or 0)
        energy_cards = list(_get(card, "energyCards", []) or [])
        energies = list(_get(card, "energies", []) or [])
        tools = list(_get(card, "tools", []) or [])
        pre_evolution = list(_get(card, "preEvolution", []) or [])
        return CardEntity(
            card_id=card_id,
            name=_card_name(card_id, self.card_by_id),
            owner=owner,
            zone=zone,
            slot=slot,
            known=card_id is not None,
            card_type=_card_type_name(data) if data is not None else "",
            hp=hp,
            max_hp=max_hp,
            energy_count=len(energy_cards) or len(energies),
            tool_count=len(tools),
            pre_evolution_count=len(pre_evolution),
            is_ex=bool(_get(data, "ex", False)),
            is_mega_ex=bool(_get(data, "megaEx", False)),
        )


class LegalOptionAdapter:
    def __init__(
        self,
        *,
        deck_ids: Sequence[int] = (),
        card_data: Sequence[Any] = (),
        attack_data: Sequence[Any] = (),
    ) -> None:
        self.deck_ids = list(deck_ids)
        self.card_data = card_data
        self.attack_data = attack_data

    def parse(self, observation: Any) -> list[LegalAction]:
        frame = make_decision_frame(
            observation,
            deck_ids=self.deck_ids,
            card_data=self.card_data,
            attack_data=self.attack_data,
        )
        if frame is None:
            return []
        return [
            LegalAction(
                local_index=action.index,
                selected_indices=(action.index,),
                action_type=action.option_type,
                select_type=frame.select_type,
                context=frame.context,
                min_count=frame.min_count,
                max_count=frame.max_count,
                target_count=frame.target_count,
                rule_score=action.rule_score,
                rule_rank=action.rule_rank,
                card_id=action.card_id,
                card_name=action.card_name,
                target_card_id=action.target_card_id,
                target_name=action.target_name,
                area=action.area,
                area_index=action.area_index,
                target_area=action.target_area,
                target_index=action.target_index,
                attack_id=action.attack_id,
                ends_turn=_ends_turn(action.option_type),
                raw=action.raw,
            )
            for action in frame.legal_options
        ]


def _ends_turn(action_type: str) -> bool:
    return action_type in {"ATTACK", "END"}


def _known_count(entities: Sequence[CardEntity]) -> int:
    return sum(1 for entity in entities if entity.known and entity.card_id is not None)


def _remaining_candidates(deck_ids: Sequence[int], observed: Counter[int]) -> list[int]:
    remaining = Counter(int(card_id) for card_id in deck_ids)
    remaining.subtract(observed)
    output: list[int] = []
    for card_id, count in sorted(remaining.items()):
        if count > 0:
            output.extend([card_id] * count)
    return output


def option_area_name(option: Any) -> str:
    return _area_name(_get(option, "area")) if _get(option, "area") is not None else ""


def option_area_index(option: Any) -> int | None:
    return _int_or_none(_get(option, "index"))
