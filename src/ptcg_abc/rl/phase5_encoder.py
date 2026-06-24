from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

from ptcg_abc.rl.phase5_adapters import BeliefState, CardEntity, GameState, LegalAction


ACTION_TYPES = (
    "PLAY",
    "ATTACH",
    "EVOLVE",
    "ABILITY",
    "ATTACK",
    "RETREAT",
    "END",
    "CARD",
    "ENERGY",
    "ENERGY_CARD",
    "TOOL_CARD",
    "DISCARD",
    "YES",
    "NO",
    "NUMBER",
)
ENTITY_ZONES = ("ACTIVE", "BENCH", "HAND", "DISCARD", "PRIZE", "STADIUM", "LOOKING")
CARD_TYPES = (
    "POKEMON",
    "ITEM",
    "TOOL",
    "SUPPORTER",
    "STADIUM",
    "BASIC_ENERGY",
    "SPECIAL_ENERGY",
)
ACTION_AREAS = ("DECK", "HAND", "DISCARD", "ACTIVE", "BENCH", "PRIZE", "STADIUM")


GLOBAL_FEATURE_NAMES = (
    "turn",
    "your_index",
    "energy_attached",
    "supporter_played",
    "stadium_count",
    "looking_count",
    "my_hand_count",
    "my_deck_count",
    "my_prize_count",
    "opponent_hand_count",
    "opponent_deck_count",
    "opponent_prize_count",
    "own_unknown_deck_count",
    "own_unknown_prize_count",
    "opponent_unknown_hand_count",
    "opponent_unknown_deck_count",
    "opponent_unknown_prize_count",
)


ENTITY_FEATURE_NAMES = (
    "known",
    "owner_is_us",
    "owner_is_opponent",
    "owner_missing",
    "card_id_hash",
    "slot",
    "hp",
    "max_hp",
    "damage_ratio",
    "energy_count",
    "tool_count",
    "pre_evolution_count",
    "is_ex",
    "is_mega_ex",
) + tuple(f"zone_{zone.casefold()}" for zone in ENTITY_ZONES) + tuple(
    f"type_{card_type.casefold()}" for card_type in CARD_TYPES
)


ACTION_FEATURE_NAMES = (
    "rule_score",
    "rule_rank_inv",
    "card_id_hash",
    "target_id_hash",
    "attack_id_hash",
    "selected_count",
    "min_count",
    "max_count",
    "target_count",
    "area_index",
    "target_index",
    "ends_turn",
) + tuple(f"action_{action_type.casefold()}" for action_type in ACTION_TYPES) + tuple(
    f"area_{area.casefold()}" for area in ACTION_AREAS
) + tuple(f"target_area_{area.casefold()}" for area in ACTION_AREAS)


@dataclass(frozen=True)
class EncodedPhase5Turn:
    global_features: list[float]
    entity_features: list[list[float]]
    legal_action_features: list[list[float]]
    entity_mask: list[float]
    legal_action_mask: list[float]
    legal_action_indices: list[int]
    global_feature_names: tuple[str, ...] = GLOBAL_FEATURE_NAMES
    entity_feature_names: tuple[str, ...] = ENTITY_FEATURE_NAMES
    action_feature_names: tuple[str, ...] = ACTION_FEATURE_NAMES

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_features": self.global_features,
            "entity_features": self.entity_features,
            "legal_action_features": self.legal_action_features,
            "entity_mask": self.entity_mask,
            "legal_action_mask": self.legal_action_mask,
            "legal_action_indices": self.legal_action_indices,
            "global_feature_names": list(self.global_feature_names),
            "entity_feature_names": list(self.entity_feature_names),
            "action_feature_names": list(self.action_feature_names),
        }


class Phase5SymbolicEncoder:
    def __init__(self, *, max_entities: int = 96, max_actions: int = 128) -> None:
        self.max_entities = max_entities
        self.max_actions = max_actions

    def encode(
        self,
        state: GameState,
        legal_actions: Sequence[LegalAction],
        belief: BeliefState | None = None,
    ) -> EncodedPhase5Turn:
        global_features = _global_features(state, belief)
        entity_rows = [
            _entity_features(entity, state)
            for entity in state.entities[: self.max_entities]
        ]
        action_rows = [
            _action_features(action)
            for action in list(legal_actions)[: self.max_actions]
        ]
        entity_width = len(ENTITY_FEATURE_NAMES)
        action_width = len(ACTION_FEATURE_NAMES)
        entity_mask = [1.0] * len(entity_rows)
        action_mask = [1.0] * len(action_rows)
        entity_rows.extend(
            [[0.0] * entity_width for _ in range(self.max_entities - len(entity_rows))]
        )
        action_rows.extend(
            [[0.0] * action_width for _ in range(self.max_actions - len(action_rows))]
        )
        entity_mask.extend([0.0] * (self.max_entities - len(entity_mask)))
        action_mask.extend([0.0] * (self.max_actions - len(action_mask)))
        action_indices = [action.local_index for action in list(legal_actions)[: self.max_actions]]
        action_indices.extend([-1] * (self.max_actions - len(action_indices)))
        return EncodedPhase5Turn(
            global_features=global_features,
            entity_features=entity_rows,
            legal_action_features=action_rows,
            entity_mask=entity_mask,
            legal_action_mask=action_mask,
            legal_action_indices=action_indices,
        )


def _global_features(state: GameState, belief: BeliefState | None) -> list[float]:
    us = state.us
    opponent = state.opponent
    return [
        _scaled(state.turn, 30.0),
        float(state.your_index),
        float(state.energy_attached),
        float(state.supporter_played),
        _scaled(state.stadium_count, 2.0),
        _scaled(state.looking_count, 20.0),
        _scaled(us.hand_count if us else 0, 20.0),
        _scaled(us.deck_count if us else 0, 60.0),
        _scaled(us.prize_count if us else 0, 6.0),
        _scaled(opponent.hand_count if opponent else 0, 20.0),
        _scaled(opponent.deck_count if opponent else 0, 60.0),
        _scaled(opponent.prize_count if opponent else 0, 6.0),
        _scaled(belief.own_unknown_deck_count if belief else 0, 60.0),
        _scaled(belief.own_unknown_prize_count if belief else 0, 6.0),
        _scaled(belief.opponent_unknown_hand_count if belief else 0, 20.0),
        _scaled(belief.opponent_unknown_deck_count if belief else 0, 60.0),
        _scaled(belief.opponent_unknown_prize_count if belief else 0, 6.0),
    ]


def _entity_features(entity: CardEntity, state: GameState) -> list[float]:
    owner_is_us = entity.owner == state.your_index
    owner_is_opponent = entity.owner == 1 - state.your_index
    base = [
        float(entity.known),
        float(owner_is_us),
        float(owner_is_opponent),
        float(entity.owner is None),
        _hash_unit(entity.card_id),
        _scaled(entity.slot if entity.slot is not None else -1, 10.0),
        _scaled(entity.hp, 400.0),
        _scaled(entity.max_hp, 400.0),
        1.0 - (entity.hp / entity.max_hp) if entity.max_hp else 0.0,
        _scaled(entity.energy_count, 12.0),
        _scaled(entity.tool_count, 4.0),
        _scaled(entity.pre_evolution_count, 3.0),
        float(entity.is_ex),
        float(entity.is_mega_ex),
    ]
    base.extend(_one_hot(entity.zone, ENTITY_ZONES))
    base.extend(_one_hot(entity.card_type, CARD_TYPES))
    return base


def _action_features(action: LegalAction) -> list[float]:
    rule_rank_inv = 1.0 / float(action.rule_rank) if action.rule_rank > 0 else 0.0
    base = [
        _scaled(action.rule_score, 100000.0),
        rule_rank_inv,
        _hash_unit(action.card_id),
        _hash_unit(action.target_card_id),
        _hash_unit(action.attack_id),
        _scaled(len(action.selected_indices), 8.0),
        _scaled(action.min_count, 8.0),
        _scaled(action.max_count, 8.0),
        _scaled(action.target_count, 8.0),
        _scaled(action.area_index if action.area_index is not None else -1, 10.0),
        _scaled(action.target_index if action.target_index is not None else -1, 10.0),
        float(action.ends_turn),
    ]
    base.extend(_one_hot(action.action_type, ACTION_TYPES))
    base.extend(_one_hot(action.area, ACTION_AREAS))
    base.extend(_one_hot(action.target_area, ACTION_AREAS))
    return base


def _one_hot(value: str, choices: Sequence[str]) -> list[float]:
    return [float(value == choice) for choice in choices]


def _scaled(value: Any, denominator: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if denominator <= 0:
        return numeric
    return max(-1.0, min(1.0, numeric / denominator))


def _hash_unit(value: Any) -> float:
    if value is None:
        return 0.0
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") / float(2**32 - 1)
