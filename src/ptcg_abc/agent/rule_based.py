from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


OPTION_TYPE_NAMES = {
    0: "NUMBER",
    1: "YES",
    2: "NO",
    3: "CARD",
    4: "TOOL_CARD",
    5: "ENERGY_CARD",
    6: "ENERGY",
    7: "PLAY",
    8: "ATTACH",
    9: "EVOLVE",
    10: "ABILITY",
    11: "DISCARD",
    12: "RETREAT",
    13: "ATTACK",
    14: "END",
    15: "SKILL",
    16: "SPECIAL_CONDITION",
}
SELECT_TYPE_NAMES = {
    0: "MAIN",
    1: "CARD",
    2: "ATTACHED_CARD",
    3: "CARD_OR_ATTACHED_CARD",
    4: "ENERGY",
    5: "SKILL",
    6: "ATTACK",
    7: "EVOLVE",
    8: "COUNT",
    9: "YES_NO",
    10: "SPECIAL_CONDITION",
}
SELECT_CONTEXT_NAMES = {
    0: "MAIN",
    1: "SETUP_ACTIVE_POKEMON",
    2: "SETUP_BENCH_POKEMON",
    3: "SWITCH",
    4: "TO_ACTIVE",
    5: "TO_BENCH",
    6: "TO_FIELD",
    7: "TO_HAND",
    8: "DISCARD",
    9: "TO_DECK",
    10: "TO_DECK_BOTTOM",
    11: "TO_PRIZE",
    12: "NOT_MOVE",
    13: "DAMAGE_COUNTER",
    14: "DAMAGE_COUNTER_ANY",
    15: "DAMAGE",
    16: "REMOVE_DAMAGE_COUNTER",
    17: "HEAL",
    18: "EVOLVES_FROM",
    19: "EVOLVES_TO",
    20: "DEVOLVE",
    21: "ATTACH_FROM",
    22: "ATTACH_TO",
    23: "DETACH_FROM",
    24: "LOOK",
    25: "EFFECT_TARGET",
    26: "DISCARD_ENERGY_CARD",
    27: "DISCARD_TOOL_CARD",
    28: "SWITCH_ENERGY_CARD",
    29: "DISCARD_CARD_OR_ATTACHED_CARD",
    30: "DISCARD_ENERGY",
    31: "TO_HAND_ENERGY",
    32: "TO_DECK_ENERGY",
    33: "SWITCH_ENERGY",
    34: "SKILL_ORDER",
    35: "ATTACK",
    36: "DISABLE_ATTACK",
    37: "EVOLVE",
    38: "DRAW_COUNT",
    39: "DAMAGE_COUNTER_COUNT",
    40: "REMOVE_DAMAGE_COUNTER_COUNT",
    41: "IS_FIRST",
    42: "MULLIGAN",
    43: "ACTIVATE",
    44: "FIRST_EFFECT",
    45: "MORE_DEVOLVE",
    46: "COIN_HEAD",
    47: "AFFECT_SPECIAL_CONDITION",
    48: "RECOVER_SPECIAL_CONDITION",
}
AREA_NAMES = {
    1: "DECK",
    2: "HAND",
    3: "DISCARD",
    4: "ACTIVE",
    5: "BENCH",
    6: "PRIZE",
    7: "STADIUM",
    8: "ENERGY",
    9: "TOOL",
    10: "PRE_EVOLUTION",
    11: "PLAYER",
    12: "LOOKING",
}

MAIN_ACTION_PRIORITY = {
    "ATTACH": 0,
    "EVOLVE": 10,
    "ABILITY": 20,
    "PLAY": 30,
    "ATTACK": 80,
    "RETREAT": 300,
    "DISCARD": 500,
    "END": 1000,
}
GENERAL_OPTION_PRIORITY = {
    "ATTACK": 0,
    "EVOLVE": 10,
    "ABILITY": 20,
    "PLAY": 30,
    "ATTACH": 40,
    "CARD": 50,
    "ENERGY": 60,
    "ENERGY_CARD": 65,
    "TOOL_CARD": 70,
    "SKILL": 80,
    "SPECIAL_CONDITION": 90,
    "YES": 100,
    "NO": 110,
    "RETREAT": 300,
    "DISCARD": 500,
    "END": 1000,
    "NUMBER": 1200,
}
MINIMUM_ONLY_CONTEXTS = {
    "DISCARD",
    "TO_DECK",
    "TO_DECK_BOTTOM",
    "TO_PRIZE",
    "DETACH_FROM",
    "DISCARD_ENERGY_CARD",
    "DISCARD_TOOL_CARD",
    "DISCARD_CARD_OR_ATTACHED_CARD",
    "DISCARD_ENERGY",
    "DISABLE_ATTACK",
}
OPPONENT_TARGET_CONTEXTS = {"DAMAGE", "DAMAGE_COUNTER", "DAMAGE_COUNTER_ANY"}
SELF_TARGET_CONTEXTS = {"HEAL", "REMOVE_DAMAGE_COUNTER", "RECOVER_SPECIAL_CONDITION"}


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _enum_name(value: Any, names: dict[int, str]) -> str:
    if hasattr(value, "name"):
        return str(value.name)
    if isinstance(value, str):
        return value
    try:
        return names[int(value)]
    except (KeyError, TypeError, ValueError):
        return str(value)


def _option_type_name(option: Any) -> str:
    return _enum_name(_get(option, "type"), OPTION_TYPE_NAMES)


def _select_type_name(select: Any) -> str:
    return _enum_name(_get(select, "type"), SELECT_TYPE_NAMES)


def _select_context_name(select: Any) -> str:
    return _enum_name(_get(select, "context"), SELECT_CONTEXT_NAMES)


def _area_name(value: Any) -> str:
    return _enum_name(value, AREA_NAMES)


def _your_index(current: Any) -> int | None:
    value = _get(current, "yourIndex")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_yes(option: Any) -> bool:
    return _option_type_name(option) == "YES"


def _yes_no_preference(select: Any, options: Sequence[Any]) -> int:
    context_name = _select_context_name(select)
    prefer_yes = context_name != "IS_FIRST"
    preferred_type = "YES" if prefer_yes else "NO"
    for index, option in enumerate(options):
        if _option_type_name(option) == preferred_type:
            return index
    for index, option in enumerate(options):
        if _is_yes(option):
            return index
    return 0


def _target_count(select: Any, options: Sequence[Any]) -> int:
    min_count = int(_get(select, "minCount", 0) or 0)
    max_count = int(_get(select, "maxCount", min_count) or 0)
    max_count = min(max_count, len(options))
    if max_count <= 0:
        return 0
    if _select_type_name(select) in {"MAIN", "COUNT", "YES_NO"}:
        return max(1, min_count)
    if _select_context_name(select) in MINIMUM_ONLY_CONTEXTS:
        return min_count
    return max_count


def _numeric_sort_key(option: Any) -> int:
    value = _get(option, "number")
    try:
        return -int(value)
    except (TypeError, ValueError):
        return 0


def _target_side_bonus(option: Any, context_name: str, current: Any) -> int:
    your_index = _your_index(current)
    if your_index is None:
        return 0
    player_index = _get(option, "playerIndex")
    try:
        player_index = int(player_index)
    except (TypeError, ValueError):
        return 0
    if context_name in OPPONENT_TARGET_CONTEXTS:
        return 0 if player_index != your_index else 50
    if context_name in SELF_TARGET_CONTEXTS:
        return 0 if player_index == your_index else 50
    return 0


def _option_sort_key(index: int, option: Any, select: Any, current: Any) -> tuple:
    option_name = _option_type_name(option)
    select_name = _select_type_name(select)
    context_name = _select_context_name(select)
    priority = MAIN_ACTION_PRIORITY.get(option_name, 900) if select_name == "MAIN" else (
        GENERAL_OPTION_PRIORITY.get(option_name, 900)
    )
    active_bonus = 0 if _area_name(_get(option, "area")) == "ACTIVE" else 10
    return (
        priority,
        _target_side_bonus(option, context_name, current),
        _numeric_sort_key(option) if select_name == "COUNT" else 0,
        active_bonus,
        index,
    )


def select_option_indices(select: Any, *, current: Any = None) -> list[int]:
    options = list(_get(select, "option", []) or [])
    if not options:
        return []

    if _select_type_name(select) == "YES_NO":
        return [_yes_no_preference(select, options)]

    target_count = _target_count(select, options)
    if target_count <= 0:
        return []

    ranked = sorted(
        range(len(options)),
        key=lambda index: _option_sort_key(index, options[index], select, current),
    )
    return ranked[:target_count]


@dataclass(frozen=True)
class RuleBasedAgent:
    deck_ids: Sequence[int]

    def __post_init__(self) -> None:
        if len(self.deck_ids) != 60:
            raise ValueError(f"RuleBasedAgent needs a 60-card deck, got {len(self.deck_ids)}.")

    def act(self, observation: Any) -> list[int]:
        select = _get(observation, "select")
        if select is None:
            return list(self.deck_ids)
        return select_option_indices(select, current=_get(observation, "current"))
