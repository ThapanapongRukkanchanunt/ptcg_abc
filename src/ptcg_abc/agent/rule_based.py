from __future__ import annotations

import itertools
import re
from collections import Counter
from dataclasses import dataclass, field
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
CARD_TYPE_NAMES = {
    0: "POKEMON",
    1: "ITEM",
    2: "TOOL",
    3: "SUPPORTER",
    4: "STADIUM",
    5: "BASIC_ENERGY",
    6: "SPECIAL_ENERGY",
}

MAIN_ACTION_PRIORITY = {
    "ATTACH": 0,
    "ABILITY": 10,
    "EVOLVE": 20,
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
OPTIONAL_POSITIVE_CONTEXTS = {
    "SETUP_BENCH_POKEMON",
    "TO_BENCH",
    "TO_FIELD",
    "NOT_MOVE",
}
OPPONENT_TARGET_CONTEXTS = {"DAMAGE", "DAMAGE_COUNTER", "DAMAGE_COUNTER_ANY"}
SELF_TARGET_CONTEXTS = {"HEAL", "REMOVE_DAMAGE_COUNTER", "RECOVER_SPECIAL_CONDITION"}
ACTIVE_CONTEXTS = {"SWITCH", "TO_ACTIVE", "SETUP_ACTIVE_POKEMON"}
SEARCH_CONTEXTS = {"TO_HAND", "TO_BENCH", "TO_FIELD", "LOOK", "EFFECT_TARGET"}
DRAW_WORDS = ("draw", "shuffle your hand", "put them into your hand", "search your deck")
RECOVERY_WORDS = ("discard", "from your discard", "put up to", "energy from your discard")


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


def _card_type_name(card_data: Any) -> str:
    return _enum_name(_get(card_data, "cardType"), CARD_TYPE_NAMES)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _card_id(card: Any) -> int | None:
    return _int_or_none(_get(card, "id"))


def _your_index(current: Any) -> int | None:
    return _int_or_none(_get(current, "yourIndex"))


def _players(current: Any) -> list[Any]:
    return list(_get(current, "players", []) or [])


def _player_state(current: Any, player_index: int | None) -> Any | None:
    if player_index is None:
        return None
    players = _players(current)
    if 0 <= player_index < len(players):
        return players[player_index]
    return None


def _iter_non_null(values: Any) -> list[Any]:
    return [value for value in list(values or []) if value is not None]


def _cards_in_play(player_state: Any) -> list[Any]:
    return _iter_non_null(_get(player_state, "active", [])) + _iter_non_null(
        _get(player_state, "bench", [])
    )


def _first_active(player_state: Any) -> Any | None:
    active = _iter_non_null(_get(player_state, "active", []))
    return active[0] if active else None


def _card_name(card_id: int | None, card_by_id: dict[int, Any]) -> str:
    if card_id is None:
        return ""
    return str(_get(card_by_id.get(card_id), "name", ""))


def _card_data_for(card: Any, card_by_id: dict[int, Any]) -> Any | None:
    card_id = _card_id(card)
    return card_by_id.get(card_id) if card_id is not None else None


def _is_pokemon_card(card_id: int | None, card_by_id: dict[int, Any]) -> bool:
    data = card_by_id.get(card_id) if card_id is not None else None
    return _card_type_name(data) == "POKEMON"


def _is_energy_card(card_id: int | None, card_by_id: dict[int, Any]) -> bool:
    data = card_by_id.get(card_id) if card_id is not None else None
    return _card_type_name(data) in {"BASIC_ENERGY", "SPECIAL_ENERGY"}


def _energy_count(pokemon: Any) -> int:
    return len(_get(pokemon, "energies", []) or [])


def _hp(pokemon: Any) -> int:
    return int(_get(pokemon, "hp", 0) or 0)


def _max_hp(pokemon: Any) -> int:
    return int(_get(pokemon, "maxHp", _hp(pokemon)) or 0)


def _attached_cards(pokemon: Any, name: str) -> list[Any]:
    return list(_get(pokemon, name, []) or [])


def _attack_cost(attack: Any) -> int:
    return len(_get(attack, "energies", []) or [])


def _attack_damage(attack: Any) -> int:
    return int(_get(attack, "damage", 0) or 0)


def _attacks_for_pokemon(pokemon: Any, card_by_id: dict[int, Any], attack_by_id: dict[int, Any]) -> list[Any]:
    data = _card_data_for(pokemon, card_by_id)
    attacks = []
    for attack_id in list(_get(data, "attacks", []) or []):
        attack = attack_by_id.get(int(attack_id))
        if attack is not None:
            attacks.append(attack)
    return attacks


def get_missing_energies(attached: list[int], cost: list[int]) -> list[int]:
    attached_pool = list(attached)
    colored_reqs = [r for r in cost if r > 0]
    colorless_reqs = [r for r in cost if r == 0]
    unsatisfied = []
    for req in colored_reqs:
        if req in attached_pool:
            attached_pool.remove(req)
        elif 10 in attached_pool:
            attached_pool.remove(10)
        else:
            unsatisfied.append(req)
    for req in colorless_reqs:
        colorless_candidates = [e for e in attached_pool if e not in {1, 2, 3, 4, 5, 6, 7, 8}]
        if colorless_candidates:
            attached_pool.remove(colorless_candidates[0])
        elif attached_pool:
            attached_pool.remove(attached_pool[0])
        else:
            unsatisfied.append(0)
    return unsatisfied


def _get_missing_for_attacks(pokemon: Any, card_by_id: dict[int, Any], attack_by_id: dict[int, Any]) -> list[tuple[Any, list[int]]]:
    attached = list(_get(pokemon, "energies", []) or [])
    res = []
    for attack in _attacks_for_pokemon(pokemon, card_by_id, attack_by_id):
        cost = list(_get(attack, "energies", []) or [])
        missing = get_missing_energies(attached, cost)
        res.append((attack, missing))
    return res


def _min_attack_cost(pokemon: Any, card_by_id: dict[int, Any], attack_by_id: dict[int, Any]) -> int:
    costs = [_attack_cost(attack) for attack in _attacks_for_pokemon(pokemon, card_by_id, attack_by_id)]
    return min(costs) if costs else 99


def _best_attack_damage(pokemon: Any, card_by_id: dict[int, Any], attack_by_id: dict[int, Any]) -> int:
    damages = [_attack_damage(attack) for attack in _attacks_for_pokemon(pokemon, card_by_id, attack_by_id)]
    return max(damages) if damages else 0


def _can_attack_soon(
    pokemon: Any,
    card_by_id: dict[int, Any],
    attack_by_id: dict[int, Any],
    extra_energy: int = 0,
) -> bool:
    if pokemon is None:
        return False
    attached = list(_get(pokemon, "energies", []) or [])
    if extra_energy > 0:
        attached.extend([10] * extra_energy)
    attacks = _attacks_for_pokemon(pokemon, card_by_id, attack_by_id)
    if not attacks:
        return False
    for attack in attacks:
        cost = list(_get(attack, "energies", []) or [])
        if not get_missing_energies(attached, cost):
            return True
    return False


def _prize_count(pokemon: Any, card_by_id: dict[int, Any], *, attack_damage: bool = True) -> int:
    data = _card_data_for(pokemon, card_by_id)
    if data is None:
        return 1
    count = 3 if bool(_get(data, "megaEx", False)) else 2 if bool(_get(data, "ex", False)) else 1
    if attack_damage:
        for card in _attached_cards(pokemon, "energyCards"):
            if _card_id(card) == 12:
                count -= 1
        for card in _attached_cards(pokemon, "tools"):
            card_name = _card_name(_card_id(card), card_by_id)
            if _card_id(card) == 1172 and "Lillie" in _card_name(_card_id(pokemon), card_by_id):
                count -= 1
            elif "pearl" in card_name.casefold() and "lillie" in card_name.casefold():
                count -= 1
    return max(0, count)


def _pokemon_target_value(pokemon: Any, card_by_id: dict[int, Any], *, attack_damage: bool = True) -> float:
    data = _card_data_for(pokemon, card_by_id)
    score = _prize_count(pokemon, card_by_id, attack_damage=attack_damage) * 1000
    score += _energy_count(pokemon) * 150
    score += len(_attached_cards(pokemon, "tools")) * 100
    if bool(_get(data, "stage2", False)):
        score += 250
    elif bool(_get(data, "stage1", False)):
        score += 130
    if _card_type_name(data) == "POKEMON":
        score += _hp(pokemon)
    return score


def _effective_damage(attacker: Any, target: Any, attack: Any, card_by_id: dict[int, Any]) -> int:
    damage = _attack_damage(attack)
    if damage <= 0:
        return damage
    attacker_data = _card_data_for(attacker, card_by_id)
    target_data = _card_data_for(target, card_by_id)
    attacker_type = _get(attacker_data, "energyType")
    if attacker_type is not None and _get(target_data, "weakness") == attacker_type:
        damage *= 2
    if attacker_type is not None and _get(target_data, "resistance") == attacker_type:
        damage = max(0, damage - 30)
    return damage


def _get_card(select: Any, current: Any, area: Any, index: Any, player_index: Any) -> Any | None:
    area_name = _area_name(area)
    index_int = _int_or_none(index)
    player_int = _int_or_none(player_index)
    if index_int is None:
        return None
    player_state = _player_state(current, player_int)
    try:
        if area_name == "DECK":
            return list(_get(select, "deck", []) or [])[index_int]
        if area_name == "HAND":
            return list(_get(player_state, "hand", []) or [])[index_int]
        if area_name == "DISCARD":
            return list(_get(player_state, "discard", []) or [])[index_int]
        if area_name == "ACTIVE":
            return list(_get(player_state, "active", []) or [])[index_int]
        if area_name == "BENCH":
            return list(_get(player_state, "bench", []) or [])[index_int]
        if area_name == "PRIZE":
            return list(_get(player_state, "prize", []) or [])[index_int]
        if area_name == "STADIUM":
            return list(_get(current, "stadium", []) or [])[index_int]
        if area_name == "LOOKING":
            return list(_get(current, "looking", []) or [])[index_int]
    except IndexError:
        return None
    return None


def _attached_option_card(select: Any, current: Any, option: Any) -> Any | None:
    pokemon = _get_card(select, current, _get(option, "area"), _get(option, "index"), _get(option, "playerIndex"))
    if pokemon is None:
        return None
    option_name = _option_type_name(option)
    try:
        if option_name == "ENERGY_CARD":
            return _attached_cards(pokemon, "energyCards")[int(_get(option, "energyIndex"))]
        if option_name == "TOOL_CARD":
            return _attached_cards(pokemon, "tools")[int(_get(option, "toolIndex"))]
    except (IndexError, TypeError, ValueError):
        return None
    return pokemon


@dataclass
class AttackPlan:
    attacker_area: str = ""
    attacker_index: int = -1
    target_area: str = ""
    target_index: int = -1
    attack_id: int = -1
    needs_energy: bool = False
    score: float = -1.0


@dataclass
class PrizeMapStep:
    attacker_card_id: int
    attacker_area: str
    attacker_index: int
    attack_id: int
    target_area: str
    target_index: int
    target_card_id: int
    prizes_taken: int
    damage: int
    needs_boss: bool
    needs_switch: bool
    needs_energy: bool
    setup_cost: float
    score: float = 0.0
    target_damages: dict[tuple[str, int], int] = field(default_factory=dict)


@dataclass
class PrizeMap:
    steps: list[PrizeMapStep] = field(default_factory=list)
    total_prizes: int = 0
    attack_count: int = 0
    setup_cost: float = 0.0
    score: float = -1.0


@dataclass
class TargetState:
    area: str
    index: int
    pokemon: Any
    card_id: int
    hp: int
    prizes: int
    value: float


@dataclass
class CandidateAttack:
    attacker_card_id: int
    attacker_area: str
    attacker_index: int
    attacker: Any
    attack_id: int
    attack: Any
    missing_energy: list[int]
    needs_switch: bool
    needs_energy: bool
    setup_cost: float


@dataclass
class RuleDecisionTrace:
    turn: int
    player_index: int
    select_type: str
    context: str
    selected_options: list[dict[str, Any]]
    prize_map: dict[str, Any]
    key_attackers: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "player_index": self.player_index,
            "select_type": self.select_type,
            "context": self.context,
            "selected_options": self.selected_options,
            "prize_map": self.prize_map,
            "key_attackers": self.key_attackers,
        }


@dataclass
class BoardFeatures:
    select: Any
    current: Any
    card_by_id: dict[int, Any]
    attack_by_id: dict[int, Any]
    deck_ids: Sequence[int] = ()
    my_index: int = 0
    opponent_index: int = 1
    my_state: Any | None = None
    op_state: Any | None = None
    field_counts: Counter = field(default_factory=Counter)
    hand_counts: Counter = field(default_factory=Counter)
    discard_counts: Counter = field(default_factory=Counter)
    field_hand_counts: Counter = field(default_factory=Counter)
    active_ready: bool = False
    bench_ready_indices: list[int] = field(default_factory=list)
    plan: AttackPlan = field(default_factory=AttackPlan)
    prize_map: PrizeMap = field(default_factory=PrizeMap)
    key_attackers: set[int] = field(default_factory=set)
    can_switch: bool = False
    can_attack: bool = False
    can_boss: bool = False
    no_draw: bool = False
    hand_energy_count: int = 0


def _make_features(
    select: Any,
    current: Any,
    card_by_id: dict[int, Any],
    attack_by_id: dict[int, Any],
    deck_ids: Sequence[int] = (),
) -> BoardFeatures:
    my_index = _your_index(current)
    if my_index is None:
        my_index = 0
    features = BoardFeatures(
        select=select,
        current=current,
        card_by_id=card_by_id,
        attack_by_id=attack_by_id,
        deck_ids=deck_ids,
        my_index=my_index,
        opponent_index=1 - my_index,
        my_state=_player_state(current, my_index),
        op_state=_player_state(current, 1 - my_index),
    )
    features.no_draw = int(_get(features.my_state, "deckCount", 99) or 99) <= 6

    for pokemon in _cards_in_play(features.my_state):
        card_id = _card_id(pokemon)
        if card_id is not None:
            features.field_counts[card_id] += 1
            features.field_hand_counts[card_id] += 1
    for card in list(_get(features.my_state, "hand", []) or []):
        card_id = _card_id(card)
        if card_id is not None:
            features.hand_counts[card_id] += 1
            features.field_hand_counts[card_id] += 1
            if _is_energy_card(card_id, card_by_id):
                features.hand_energy_count += 1
    for card in list(_get(features.my_state, "discard", []) or []):
        card_id = _card_id(card)
        if card_id is not None:
            features.discard_counts[card_id] += 1

    active = _first_active(features.my_state)
    features.active_ready = _can_attack_soon(active, card_by_id, attack_by_id)
    for index, pokemon in enumerate(_get(features.my_state, "bench", []) or []):
        if _can_attack_soon(pokemon, card_by_id, attack_by_id):
            features.bench_ready_indices.append(index)

    for option in list(_get(select, "option", []) or []):
        option_name = _option_type_name(option)
        if option_name == "RETREAT":
            features.can_switch = True
        elif option_name == "ATTACK":
            features.can_attack = True
        elif option_name == "PLAY":
            card = _get_card(select, current, "HAND", _get(option, "index"), my_index)
            name = _card_name(_card_id(card), card_by_id).casefold()
            if "boss" in name or "catcher" in name or "counter catcher" in name:
                features.can_boss = True

    features.prize_map = _build_prize_map(features)
    features.key_attackers = _identify_key_attackers(features)
    features.plan = _build_attack_plan(features)
    return features


def _get_max_attachments_for_pokemon(pokemon: Any, features: BoardFeatures) -> tuple[int, list[int]]:
    allowed = 0 if bool(_get(features.current, "energyAttached", False)) else 1
    virtual_types = []
    
    # Check for Crispin in hand
    has_supporter_played = bool(_get(features.current, "supporterPlayed", False))
    if not has_supporter_played:
        hand_cards = list(_get(features.my_state, "hand", []) or [])
        hand_card_ids = [_card_id(hc) for hc in hand_cards]
        if 1198 in hand_card_ids:
            allowed += 1
            virtual_types.extend([2, 5])  # Fire (2) and Psychic (5)
            
    # Check for active energy-attaching abilities on the board
    for option in list(_get(features.select, "option", []) or []):
        if _option_type_name(option) == "ABILITY":
            ability_card = _get_card(features.select, features.current, _get(option, "area"), _get(option, "index"), features.my_index)
            cdata = _card_data_for(ability_card, features.card_by_id)
            if cdata is not None:
                skills = " ".join(f"{_get(s, 'name', '')} {_get(s, 'text', '')}" for s in list(_get(cdata, "skills", []) or [])).casefold()
                if "attach" in skills:
                    can_use = True
                    if "to this pok" in skills or "to itself" in skills or "teal dance" in skills:
                        if ability_card is not pokemon and _card_id(ability_card) != _card_id(pokemon):
                            can_use = False
                    elif "to your benched" in skills or "to 1 of your benched" in skills:
                        is_bench = False
                        bench = list(_get(features.my_state, "bench", []) or [])
                        if any(b is pokemon for b in bench):
                            is_bench = True
                        if not is_bench:
                            can_use = False
                    
                    if can_use:
                        allowed += 1
                        
    return allowed, virtual_types


def _get_hand_energy_types(features: BoardFeatures) -> list[int]:
    hand_cards = list(_get(features.my_state, "hand", []) or [])
    types = []
    for hc in hand_cards:
        hc_id = _card_id(hc)
        if hc_id is not None and _is_energy_card(hc_id, features.card_by_id):
            hc_data = features.card_by_id.get(hc_id)
            if hc_data is not None:
                types.append(hc_data.energyType)
    return types


def _remaining_prizes_needed(features: BoardFeatures) -> int:
    prizes = len(list(_get(features.my_state, "prize", []) or []))
    return min(6, prizes) if prizes > 0 else 6


def _effective_damage_by_id(attacker_card_id: int, target: Any, attack: Any, card_by_id: dict[int, Any]) -> int:
    damage = _attack_damage(attack)
    if damage <= 0:
        return damage
    attacker_data = card_by_id.get(attacker_card_id)
    target_data = _card_data_for(target, card_by_id)
    attacker_type = _get(attacker_data, "energyType")
    if attacker_type is not None and _get(target_data, "weakness") == attacker_type:
        damage *= 2
    if attacker_type is not None and _get(target_data, "resistance") == attacker_type:
        damage = max(0, damage - 30)
    return damage


def _build_target_states(features: BoardFeatures) -> list[TargetState]:
    if features.op_state is None:
        return []
    targets: list[TargetState] = []
    active = _first_active(features.op_state)
    if active is not None:
        card_id = _card_id(active)
        targets.append(
            TargetState(
                area="ACTIVE",
                index=0,
                pokemon=active,
                card_id=card_id if card_id is not None else -1,
                hp=max(1, _hp(active)),
                prizes=_prize_count(active, features.card_by_id),
                value=_pokemon_target_value(active, features.card_by_id),
            )
        )
    for index, pokemon in enumerate(_get(features.op_state, "bench", []) or []):
        card_id = _card_id(pokemon)
        targets.append(
            TargetState(
                area="BENCH",
                index=index,
                pokemon=pokemon,
                card_id=card_id if card_id is not None else -1,
                hp=max(1, _hp(pokemon)),
                prizes=_prize_count(pokemon, features.card_by_id),
                value=_pokemon_target_value(pokemon, features.card_by_id),
            )
        )
    return targets


def _can_pay_missing_energy(
    pokemon: Any,
    missing: list[int],
    retreat_attachment_cost: int,
    features: BoardFeatures,
) -> bool:
    max_allowed, virtual_types = _get_max_attachments_for_pokemon(pokemon, features)
    total_needed = len(missing) + retreat_attachment_cost
    if total_needed == 0:
        return True
    if total_needed > max_allowed:
        return False

    hand_energy_types = _get_hand_energy_types(features) + virtual_types
    if len(hand_energy_types) < total_needed:
        return False

    temp_missing = list(missing)
    temp_hand = list(hand_energy_types)
    for req in list(temp_missing):
        matched = False
        for he_type in list(temp_hand):
            if req == 0 or he_type == req or he_type == 10:
                temp_missing.remove(req)
                temp_hand.remove(he_type)
                matched = True
                break
        if not matched:
            return False
    return len(temp_hand) >= retreat_attachment_cost


def _build_candidate_attacks(features: BoardFeatures, *, legal_active_only: bool = False) -> list[CandidateAttack]:
    if features.my_state is None:
        return []
    legal_active_attack_ids = {
        int(_get(option, "attackId"))
        for option in list(_get(features.select, "option", []) or [])
        if _option_type_name(option) == "ATTACK" and _get(option, "attackId") is not None
    }
    own_pokemon = [("ACTIVE", 0, _first_active(features.my_state))]
    own_pokemon.extend(
        ("BENCH", index, pokemon) for index, pokemon in enumerate(_get(features.my_state, "bench", []) or [])
    )

    candidates: list[CandidateAttack] = []
    for area, attacker_index, attacker in own_pokemon:
        if attacker is None:
            continue
        retreat_attachment_cost = 0
        needs_switch = area == "BENCH"
        if needs_switch and not features.can_switch:
            active_pk = _first_active(features.my_state)
            if active_pk is not None:
                active_data = _card_data_for(active_pk, features.card_by_id)
                retreat_cost = int(_get(active_data, "retreatCost", 0) or 0)
                attached_energy_count = len(list(_get(active_pk, "energies", []) or []))
                retreat_attachment_cost = max(0, retreat_cost - attached_energy_count)
            if retreat_attachment_cost <= 0:
                continue

        for attack in _attacks_for_pokemon(attacker, features.card_by_id, features.attack_by_id):
            attack_id = int(_get(attack, "attackId", -1))
            if legal_active_only and area == "ACTIVE" and legal_active_attack_ids and attack_id not in legal_active_attack_ids:
                continue
            attached = list(_get(attacker, "energies", []) or [])
            cost = list(_get(attack, "energies", []) or [])
            missing = get_missing_energies(attached, cost)
            if not _can_pay_missing_energy(attacker, missing, retreat_attachment_cost, features):
                continue
            setup_cost = len(missing) * 450.0 + retreat_attachment_cost * 350.0
            if needs_switch:
                setup_cost += 500.0
            candidates.append(
                CandidateAttack(
                    attacker_card_id=_card_id(attacker) or -1,
                    attacker_area=area,
                    attacker_index=attacker_index,
                    attacker=attacker,
                    attack_id=attack_id,
                    attack=attack,
                    missing_energy=missing,
                    needs_switch=needs_switch,
                    needs_energy=bool(missing or retreat_attachment_cost),
                    setup_cost=setup_cost,
                )
            )
    return candidates


def _attack_text(attack: Any) -> str:
    return f"{_get(attack, 'name', '')} {_get(attack, 'text', '')}".casefold()


def _damage_counter_pool(attack: Any) -> int:
    text = _attack_text(attack)
    if "damage counter" not in text or "opponent" not in text or "benched" not in text:
        return 0
    match = re.search(r"(?:put|place)\s+(\d+)\s+damage counters?.*benched.*in any way", text)
    return int(match.group(1)) * 10 if match else 0


def _single_bench_counter_damage(attack: Any) -> int:
    text = _attack_text(attack)
    if "damage counter" not in text or "opponent" not in text or "benched" not in text:
        return 0
    match = re.search(r"(?:put|place)\s+(\d+)\s+damage counters?.*(?:1|one) of .*benched", text)
    return int(match.group(1)) * 10 if match else 0


def _bench_damage_spec(attack: Any) -> tuple[int, int | None]:
    text = _attack_text(attack)
    if "damage" not in text or "opponent" not in text or "benched" not in text:
        return (0, None)
    each = re.search(r"(?:also )?does\s+(\d+)\s+damage to each of your opponent.?s benched", text)
    if each:
        return (int(each.group(1)), None)
    numbered = re.search(
        r"(?:also )?does\s+(\d+)\s+damage to\s+(\d+|one|two|three)\s+of your opponent.?s benched",
        text,
    )
    if numbered:
        count_raw = numbered.group(2)
        count = {"one": 1, "two": 2, "three": 3}.get(count_raw, int(count_raw) if count_raw.isdigit() else 1)
        return (int(numbered.group(1)), count)
    return (0, None)


def _multi_pokemon_damage_spec(attack: Any) -> tuple[int, int]:
    text = _attack_text(attack)
    if _attack_damage(attack) > 0 or "opponent" not in text or "pok" not in text:
        return (0, 0)
    match = re.search(
        r"does\s+(\d+)\s+damage to\s+(\d+|one|two|three)\s+of your opponent.?s pok",
        text,
    )
    if not match:
        match = re.search(r"does\s+(\d+)\s+damage to\s+(?:1|one) of your opponent.?s pok", text)
        if not match:
            return (0, 0)
        return (int(match.group(1)), 1)
    count_raw = match.group(2)
    count = {"one": 1, "two": 2, "three": 3}.get(count_raw, int(count_raw) if count_raw.isdigit() else 1)
    return (int(match.group(1)), count)


def _target_key(target: TargetState) -> tuple[str, int]:
    return (target.area, target.index)


def _target_by_key(targets: list[TargetState]) -> dict[tuple[str, int], TargetState]:
    return {_target_key(target): target for target in targets}


def _best_counter_distributions(
    base_damages: dict[tuple[str, int], int],
    bench_targets: list[TargetState],
    pool_damage: int,
    *,
    limit: int = 10,
) -> list[dict[tuple[str, int], int]]:
    if pool_damage <= 0 or not bench_targets:
        return [dict(base_damages)]

    variants: list[tuple[float, dict[tuple[str, int], int]]] = []
    for size in range(0, min(3, len(bench_targets)) + 1):
        for combo in itertools.combinations(bench_targets, size):
            needed = sum(target.hp for target in combo)
            if needed > pool_damage:
                continue
            damages = dict(base_damages)
            for target in combo:
                damages[_target_key(target)] = damages.get(_target_key(target), 0) + target.hp
            leftover = pool_damage - needed
            if leftover > 0:
                remaining = [target for target in bench_targets if target not in combo]
                if remaining:
                    chip_target = max(remaining, key=lambda target: (target.value, -target.hp))
                    damages[_target_key(chip_target)] = damages.get(_target_key(chip_target), 0) + min(
                        leftover,
                        chip_target.hp,
                    )
            prizes = sum(target.prizes for target in combo)
            score = prizes * 10000.0 + sum(target.value for target in combo)
            if leftover > 0 and bench_targets:
                score += leftover * 5.0
            variants.append((score, damages))

    if not variants:
        chip_target = max(bench_targets, key=lambda target: (target.value, -target.hp))
        damages = dict(base_damages)
        damages[_target_key(chip_target)] = min(pool_damage, chip_target.hp)
        variants.append((pool_damage * 5.0, damages))

    dedup: dict[tuple[tuple[tuple[str, int], int], ...], tuple[float, dict[tuple[str, int], int]]] = {}
    for score, damages in variants:
        key = tuple(sorted(damages.items()))
        if key not in dedup or score > dedup[key][0]:
            dedup[key] = (score, damages)
    return [damages for _, damages in sorted(dedup.values(), key=lambda item: item[0], reverse=True)[:limit]]


def _spread_damage_variants(
    base_damages: dict[tuple[str, int], int],
    targets: list[TargetState],
    *,
    damage: int,
    target_count: int | None,
    bench_only: bool,
    limit: int = 12,
) -> list[dict[tuple[str, int], int]]:
    if damage <= 0:
        return [dict(base_damages)]
    candidate_targets = [target for target in targets if not bench_only or target.area == "BENCH"]
    if not candidate_targets:
        return [dict(base_damages)]
    if target_count is None:
        damages = dict(base_damages)
        for target in candidate_targets:
            damages[_target_key(target)] = damages.get(_target_key(target), 0) + min(damage, target.hp)
        return [damages]

    variants: list[tuple[float, dict[tuple[str, int], int]]] = []
    for combo in itertools.combinations(candidate_targets, min(target_count, len(candidate_targets))):
        damages = dict(base_damages)
        prizes = 0
        value = 0.0
        for target in combo:
            dealt = min(damage, target.hp)
            damages[_target_key(target)] = damages.get(_target_key(target), 0) + dealt
            if damage >= target.hp:
                prizes += target.prizes
            value += target.value * min(1.0, damage / target.hp)
        variants.append((prizes * 10000.0 + value, damages))
    return [damages for _, damages in sorted(variants, key=lambda item: item[0], reverse=True)[:limit]]


def _make_prize_map_step_from_damages(
    candidate: CandidateAttack,
    target: TargetState,
    features: BoardFeatures,
    *,
    target_damages: dict[tuple[str, int], int],
    prizes_taken: int = 0,
    score: float = 0.0,
) -> PrizeMapStep | None:
    if not target_damages:
        return None
    needs_boss = target.area == "BENCH"
    setup_cost = candidate.setup_cost + (850.0 if needs_boss else 0.0)
    return PrizeMapStep(
        attacker_card_id=candidate.attacker_card_id,
        attacker_area=candidate.attacker_area,
        attacker_index=candidate.attacker_index,
        attack_id=candidate.attack_id,
        target_area=target.area,
        target_index=target.index,
        target_card_id=target.card_id,
        prizes_taken=prizes_taken,
        damage=target_damages.get(_target_key(target), 0),
        needs_boss=needs_boss,
        needs_switch=candidate.needs_switch,
        needs_energy=candidate.needs_energy,
        setup_cost=setup_cost,
        score=score,
        target_damages=dict(target_damages),
    )


def _make_prize_map_steps(
    candidate: CandidateAttack,
    target: TargetState,
    features: BoardFeatures,
    targets: list[TargetState],
) -> list[PrizeMapStep]:
    text = _attack_text(candidate.attack)
    can_target_any = "opponent" in text and "pok" in text and "(don" in text
    if target.area == "BENCH" and not (features.can_boss or can_target_any or "benched" in text):
        return []

    target_damages: dict[tuple[str, int], int] = {}
    direct_damage = _effective_damage(candidate.attacker, target.pokemon, candidate.attack, features.card_by_id)
    if direct_damage > 0:
        target_damages[_target_key(target)] = direct_damage

    multi_damage, multi_count = _multi_pokemon_damage_spec(candidate.attack)
    if multi_damage > 0 and multi_count > 0:
        if target != targets[0]:
            return []
        variants = _spread_damage_variants(
            {},
            targets,
            damage=multi_damage,
            target_count=multi_count,
            bench_only=False,
        )
    else:
        variants = [target_damages]

    counter_pool = _damage_counter_pool(candidate.attack)
    single_counter = _single_bench_counter_damage(candidate.attack)
    bench_damage, bench_count = _bench_damage_spec(candidate.attack)
    bench_targets = [bench_target for bench_target in targets if bench_target.area == "BENCH"]

    next_variants: list[dict[tuple[str, int], int]] = []
    for damages in variants:
        if counter_pool:
            next_variants.extend(_best_counter_distributions(damages, bench_targets, counter_pool))
        elif single_counter:
            next_variants.extend(
                _spread_damage_variants(
                    damages,
                    targets,
                    damage=single_counter,
                    target_count=1,
                    bench_only=True,
                )
            )
        elif bench_damage:
            next_variants.extend(
                _spread_damage_variants(
                    damages,
                    targets,
                    damage=bench_damage,
                    target_count=bench_count,
                    bench_only=True,
                )
            )
        else:
            next_variants.append(damages)

    target_by_key = _target_by_key(targets)
    steps: list[PrizeMapStep] = []
    for damages in next_variants:
        capped = {
            key: min(damage, target_by_key[key].hp)
            for key, damage in damages.items()
            if key in target_by_key and damage > 0
        }
        if not capped:
            continue
        step = _make_prize_map_step_from_damages(
            candidate,
            target,
            features,
            target_damages=capped,
        )
        if step is not None:
            steps.append(step)
    return steps


def _step_with_result(step: PrizeMapStep, *, prizes_taken: int, score: float) -> PrizeMapStep:
    return PrizeMapStep(
        attacker_card_id=step.attacker_card_id,
        attacker_area=step.attacker_area,
        attacker_index=step.attacker_index,
        attack_id=step.attack_id,
        target_area=step.target_area,
        target_index=step.target_index,
        target_card_id=step.target_card_id,
        prizes_taken=prizes_taken,
        damage=step.damage,
        needs_boss=step.needs_boss,
        needs_switch=step.needs_switch,
        needs_energy=step.needs_energy,
        setup_cost=step.setup_cost,
        score=score,
        target_damages=dict(step.target_damages),
    )


def _build_prize_map(features: BoardFeatures) -> PrizeMap:
    if features.my_state is None or features.op_state is None:
        return PrizeMap()

    targets = _build_target_states(features)
    candidates = _build_candidate_attacks(features)
    if not targets or not candidates:
        return PrizeMap()

    target_by_key = {(target.area, target.index): target for target in targets}
    step_options: list[PrizeMapStep] = []
    for candidate in candidates:
        for target in targets:
            for step in _make_prize_map_steps(candidate, target, features, targets):
                immediate_prizes = 0
                progress_score = 0.0
                overkill = 0
                for key, damage in step.target_damages.items():
                    damage_target = target_by_key.get(key)
                    if damage_target is None:
                        continue
                    progress_score += damage_target.value * min(1.0, damage / damage_target.hp)
                    if damage >= damage_target.hp:
                        immediate_prizes += damage_target.prizes
                    overkill += max(0, damage - damage_target.hp)
                step.score = (
                    immediate_prizes * 10000.0
                    + progress_score
                    - step.setup_cost
                    - overkill * 2.0
                )
                step.prizes_taken = immediate_prizes
                step_options.append(step)

    if not step_options:
        return PrizeMap()

    step_options.sort(key=lambda step: (-step.score, step.setup_cost, step.attack_id))
    step_options = step_options[:28]
    prizes_needed = _remaining_prizes_needed(features)
    initial_hp = {key: target.hp for key, target in target_by_key.items()}
    beam = [([], initial_hp, frozenset(), 0, 0.0, 0.0)]
    best_partial: tuple[list[PrizeMapStep], dict[tuple[str, int], int], frozenset, int, float, float] | None = None

    for _depth in range(1, min(6, prizes_needed + 2) + 1):
        expanded = []
        for route, hp_state, knocked_out, prizes, setup_cost, route_score in beam:
            for step in step_options:
                gained = 0
                overkill = 0
                progress_score = 0.0
                next_hp = dict(hp_state)
                next_knocked_out = set(knocked_out)
                affected = False
                for key, damage in step.target_damages.items():
                    if key in knocked_out:
                        continue
                    target = target_by_key.get(key)
                    if target is None:
                        continue
                    hp_before = hp_state.get(key, target.hp)
                    if hp_before <= 0:
                        continue
                    affected = True
                    hp_after = max(0, hp_before - damage)
                    next_hp[key] = hp_after
                    if hp_after == 0:
                        gained += target.prizes
                        next_knocked_out.add(key)
                    overkill += max(0, damage - hp_before)
                    progress_score += target.value * (min(damage, hp_before) / max(1, target.hp))
                if not affected:
                    continue
                step_score = gained * 10000.0 + progress_score - step.setup_cost - overkill * 2.0
                result_step = _step_with_result(step, prizes_taken=gained, score=step_score)
                next_route = route + [result_step]
                next_state = (
                    next_route,
                    next_hp,
                    frozenset(next_knocked_out),
                    prizes + gained,
                    setup_cost + step.setup_cost,
                    route_score + step_score,
                )
                expanded.append(next_state)

        if not expanded:
            break

        expanded.sort(key=lambda state: (-state[3], state[4], -state[5], len(state[0])))
        beam = expanded[:96]
        complete = [state for state in expanded if state[3] >= prizes_needed]
        if complete:
            complete.sort(key=lambda state: (len(state[0]), state[4], -state[5]))
            best = complete[0]
            return PrizeMap(
                steps=best[0],
                total_prizes=best[3],
                attack_count=len(best[0]),
                setup_cost=best[4],
                score=best[5],
            )
        if best_partial is None or (beam[0][3], beam[0][5]) > (best_partial[3], best_partial[5]):
            best_partial = beam[0]

    if best_partial is None:
        return PrizeMap()
    return PrizeMap(
        steps=best_partial[0],
        total_prizes=best_partial[3],
        attack_count=len(best_partial[0]),
        setup_cost=best_partial[4],
        score=best_partial[5],
    )


def _stage_setup_cost(card_data: Any) -> float:
    if bool(_get(card_data, "stage2", False)):
        return 4200.0
    if bool(_get(card_data, "stage1", False)):
        return 2200.0
    return 600.0


def _is_primary_attacker_card(card_id: int | None, features: BoardFeatures) -> bool:
    if card_id is None:
        return False
    data = features.card_by_id.get(card_id)
    if _card_type_name(data) != "POKEMON":
        return False
    attacks = [
        features.attack_by_id.get(int(attack_id))
        for attack_id in list(_get(data, "attacks", []) or [])
        if features.attack_by_id.get(int(attack_id)) is not None
    ]
    if not attacks:
        return False
    best_damage = max((_attack_damage(attack) for attack in attacks), default=0)
    best_spread = 0
    for attack in attacks:
        best_spread = max(
            best_spread,
            _damage_counter_pool(attack),
            _single_bench_counter_damage(attack),
            _bench_damage_spec(attack)[0],
            _multi_pokemon_damage_spec(attack)[0],
        )
    if bool(_get(data, "ex", False)) or bool(_get(data, "megaEx", False)):
        return max(best_damage, best_spread) >= 80
    if bool(_get(data, "stage1", False)) or bool(_get(data, "stage2", False)):
        return max(best_damage, best_spread) >= 80
    return max(best_damage, best_spread) >= 130


def _route_is_actionable(features: BoardFeatures) -> bool:
    if not features.prize_map.steps:
        return False
    if features.prize_map.total_prizes <= 0 or features.prize_map.score <= 0:
        return False
    return any(
        step.prizes_taken > 0 and _is_primary_attacker_card(step.attacker_card_id, features)
        for step in features.prize_map.steps
    )


def _identify_key_attackers(features: BoardFeatures) -> set[int]:
    key_attackers = {
        step.attacker_card_id
        for step in features.prize_map.steps
        if _route_is_actionable(features) and _is_primary_attacker_card(step.attacker_card_id, features)
    }
    targets = _build_target_states(features)
    if not targets:
        return key_attackers

    source_ids = set(features.deck_ids)
    source_ids.update(features.field_hand_counts)
    scored: list[tuple[float, int]] = []
    for card_id in source_ids:
        data = features.card_by_id.get(card_id)
        if _card_type_name(data) != "POKEMON":
            continue
        attacks = [features.attack_by_id.get(int(attack_id)) for attack_id in list(_get(data, "attacks", []) or [])]
        attacks = [attack for attack in attacks if attack is not None]
        if not attacks:
            continue
        best_score = -1.0
        for attack in attacks:
            cost = _attack_cost(attack)
            best_attack_damage = max(
                _attack_damage(attack),
                _damage_counter_pool(attack),
                _single_bench_counter_damage(attack),
                _bench_damage_spec(attack)[0],
                _multi_pokemon_damage_spec(attack)[0],
            )
            if not _is_primary_attacker_card(card_id, features):
                continue
            for target in targets:
                damage = _effective_damage_by_id(card_id, target.pokemon, attack, features.card_by_id)
                damage = max(damage, best_attack_damage)
                if damage <= 0:
                    continue
                progress = min(1.0, damage / target.hp)
                prize_bonus = target.prizes * 3500.0 if damage >= target.hp else 0.0
                score = target.value * progress + prize_bonus - cost * 350.0 - _stage_setup_cost(data)
                best_score = max(best_score, score)
        if best_score >= 2200:
            scored.append((best_score, card_id))

    scored.sort(reverse=True)
    key_attackers.update(card_id for _, card_id in scored[:3])
    return key_attackers


def _build_attack_plan(features: BoardFeatures) -> AttackPlan:
    plan = AttackPlan()
    if features.my_state is None or features.op_state is None:
        return plan

    if _route_is_actionable(features):
        step = features.prize_map.steps[0]
        return AttackPlan(
            attacker_area=step.attacker_area,
            attacker_index=step.attacker_index,
            target_area=step.target_area,
            target_index=step.target_index,
            attack_id=step.attack_id,
            needs_energy=step.needs_energy,
            score=features.prize_map.score,
        )

    legal_active_attack_ids = {
        int(_get(option, "attackId"))
        for option in list(_get(features.select, "option", []) or [])
        if _option_type_name(option) == "ATTACK" and _get(option, "attackId") is not None
    }
    targets = [("ACTIVE", 0, _first_active(features.op_state))]
    if features.can_boss:
        targets.extend(("BENCH", index, pokemon) for index, pokemon in enumerate(_get(features.op_state, "bench", []) or []))

    for candidate in _build_candidate_attacks(features, legal_active_only=True):
        area = candidate.attacker_area
        attacker_index = candidate.attacker_index
        attacker = candidate.attacker
        attack = candidate.attack
        attack_id = candidate.attack_id
        if area == "ACTIVE" and legal_active_attack_ids and attack_id not in legal_active_attack_ids:
            continue
        for target_area, target_index, target in targets:
            if target is None:
                continue
            damage = _effective_damage(attacker, target, attack, features.card_by_id)
            hp = max(1, _hp(target))
            prize = _prize_count(target, features.card_by_id)
            score = _pokemon_target_value(target, features.card_by_id)
            score *= min(1.0, damage / hp) if damage > 0 else 0.15
            if damage >= hp:
                score += 5000 + prize * 2500
            if target_area == "ACTIVE":
                score += 300
            if area == "ACTIVE":
                score += 200
            if candidate.needs_energy:
                score -= 150
            if score > plan.score:
                plan = AttackPlan(
                    attacker_area=area,
                    attacker_index=attacker_index,
                    target_area=target_area,
                    target_index=target_index if target_area == "BENCH" else 0,
                    attack_id=attack_id,
                    needs_energy=candidate.needs_energy,
                    score=score,
                )
    return plan


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


def _numeric_score(option: Any) -> float:
    value = _get(option, "number")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _base_priority_score(option: Any, select: Any) -> float:
    option_name = _option_type_name(option)
    select_name = _select_type_name(select)
    priority = MAIN_ACTION_PRIORITY.get(option_name, 900) if select_name == "MAIN" else (
        GENERAL_OPTION_PRIORITY.get(option_name, 900)
    )
    return 10000.0 - priority


def _first_prize_step(features: BoardFeatures) -> PrizeMapStep | None:
    return features.prize_map.steps[0] if _route_is_actionable(features) else None


def _route_needs_boss(features: BoardFeatures) -> bool:
    route_needs_boss = _route_is_actionable(features) and any(step.needs_boss for step in features.prize_map.steps)
    return route_needs_boss or features.plan.target_area == "BENCH"


def _is_first_step_attacker(card: Any, features: BoardFeatures) -> bool:
    step = _first_prize_step(features)
    if step is None:
        return False
    if step.attacker_area == "ACTIVE":
        return _first_active(features.my_state) is card
    if step.attacker_area == "BENCH":
        bench = list(_get(features.my_state, "bench", []) or [])
        return 0 <= step.attacker_index < len(bench) and bench[step.attacker_index] is card
    return False


def _is_key_preevolution(card_id: int | None, features: BoardFeatures) -> bool:
    if card_id is None:
        return False
    name = _card_name(card_id, features.card_by_id).casefold()
    if not name:
        return False
    checked: set[int] = set()

    def evolves_from_key(attacker_id: int) -> bool:
        if attacker_id in checked:
            return False
        checked.add(attacker_id)
        data = features.card_by_id.get(attacker_id)
        evolves_from = str(_get(data, "evolvesFrom", "") or "").casefold()
        if not evolves_from:
            return False
        if evolves_from == name:
            return True
        for candidate_id, candidate_data in features.card_by_id.items():
            if _card_name(candidate_id, features.card_by_id).casefold() == evolves_from:
                if evolves_from_key(candidate_id):
                    return True
        return False

    for attacker_id in features.key_attackers:
        if evolves_from_key(attacker_id):
            return True
    return False


def _key_stage2_exists(features: BoardFeatures) -> bool:
    return any(bool(_get(features.card_by_id.get(card_id), "stage2", False)) for card_id in features.key_attackers)


def _active_candidate_score(card: Any, features: BoardFeatures, *, own: bool) -> float:
    if card is None:
        return -10000
    if not own:
        return _pokemon_target_value(card, features.card_by_id)
    data = _card_data_for(card, features.card_by_id)
    card_id = _card_id(card)
    score = _hp(card) + _energy_count(card) * 800
    if _can_attack_soon(card, features.card_by_id, features.attack_by_id):
        score += 12000
    score += _best_attack_damage(card, features.card_by_id, features.attack_by_id) * 8
    if card_id in features.key_attackers:
        score += 10000
    if _route_is_actionable(features) and _is_first_step_attacker(card, features):
        score += 20000
    if bool(_get(data, "basic", False)):
        score += 1000
    if bool(_get(data, "ex", False)) or bool(_get(data, "megaEx", False)):
        score -= 250
    score -= int(_get(data, "retreatCost", 0) or 0) * 80
    if _route_is_actionable(features) and features.plan.attacker_area == "BENCH":
        bench = list(_get(features.my_state, "bench", []) or [])
        if 0 <= features.plan.attacker_index < len(bench) and bench[features.plan.attacker_index] is card:
            score += 20000
    return score


def _bench_candidate_score(card: Any, features: BoardFeatures) -> float:
    card_id = _card_id(card)
    data = features.card_by_id.get(card_id) if card_id is not None else None
    if _card_type_name(data) != "POKEMON":
        return -1000
    copies_in_play = features.field_counts[card_id]
    if copies_in_play >= 2:
        return -1000
    score = 5000 - copies_in_play * 1200
    if bool(_get(data, "basic", False)):
        score += 3000
    if bool(_get(data, "stage1", False)) or bool(_get(data, "stage2", False)):
        evolves_from = str(_get(data, "evolvesFrom", "") or "").casefold()
        if any(_card_name(existing_id, features.card_by_id).casefold() == evolves_from for existing_id in features.field_counts):
            score += 2500
    if card_id in features.key_attackers:
        score += 9000
    elif _is_key_preevolution(card_id, features):
        score += 5000
    score += int(_get(data, "hp", 0) or 0)
    if bool(_get(data, "ex", False)) or bool(_get(data, "megaEx", False)):
        score -= 300
    return score


def _search_candidate_score(card: Any, features: BoardFeatures) -> float:
    card_id = _card_id(card)
    data = features.card_by_id.get(card_id) if card_id is not None else None
    card_type = _card_type_name(data)
    if card_type == "POKEMON":
        score = _bench_candidate_score(card, features)
        if bool(_get(data, "stage1", False)) or bool(_get(data, "stage2", False)):
            score += 1000
        if card_id in features.key_attackers:
            score += 15000
        elif _is_key_preevolution(card_id, features):
            score += 8000
        if features.field_hand_counts[card_id] >= 2:
            score -= 3000
        return score
    if card_type in {"BASIC_ENERGY", "SPECIAL_ENERGY"}:
        if features.plan.needs_energy and features.plan.attack_id != -1:
            attacker = None
            plan = features.plan
            if plan.attacker_area == "ACTIVE":
                attacker = _first_active(features.my_state)
            elif plan.attacker_area == "BENCH":
                bench = list(_get(features.my_state, "bench", []) or [])
                if 0 <= plan.attacker_index < len(bench):
                    attacker = bench[plan.attacker_index]
            if attacker is not None:
                attack = features.attack_by_id.get(plan.attack_id)
                if attack is not None:
                    cost = list(_get(attack, "energies", []) or [])
                    attached = list(_get(attacker, "energies", []) or [])
                    missing = get_missing_energies(attached, cost)
                    energy_type = getattr(data, "energyType", 0)
                    if any(req == 0 or energy_type == req or energy_type == 10 for req in missing):
                        return 7000
        return 800 - features.hand_counts[card_id] * 200
    name = _card_name(card_id, features.card_by_id).casefold()
    if ("boss" in name or "catcher" in name) and _route_needs_boss(features):
        return 11000
    if "rare candy" in name and _key_stage2_exists(features):
        return 8500
    if "ball" in name or "poffin" in name or "search" in name:
        return 5500
    if any(word in name for word in ("rod", "stretcher", "retrieval")):
        return 5000 if sum(features.discard_counts.values()) > 0 else -1000
    return 1000 - features.hand_counts[card_id] * 500


def _discard_candidate_score(card: Any, features: BoardFeatures) -> float:
    card_id = _card_id(card)
    if card_id is None:
        return 0
    data = features.card_by_id.get(card_id)
    score = 0.0
    if features.hand_counts[card_id] >= 2:
        score += 5000
    card_type = _card_type_name(data)
    if card_type in {"BASIC_ENERGY", "SPECIAL_ENERGY"}:
        score += 800
        if features.plan.needs_energy and features.plan.attack_id != -1:
            attacker = None
            plan = features.plan
            if plan.attacker_area == "ACTIVE":
                attacker = _first_active(features.my_state)
            elif plan.attacker_area == "BENCH":
                bench = list(_get(features.my_state, "bench", []) or [])
                if 0 <= plan.attacker_index < len(bench):
                    attacker = bench[plan.attacker_index]
            if attacker is not None:
                attack = features.attack_by_id.get(plan.attack_id)
                if attack is not None:
                    cost = list(_get(attack, "energies", []) or [])
                    attached = list(_get(attacker, "energies", []) or [])
                    missing = get_missing_energies(attached, cost)
                    energy_type = getattr(data, "energyType", 0)
                    if any(req == 0 or energy_type == req or energy_type == 10 for req in missing):
                        score -= 5000
    elif card_type == "POKEMON":
        if card_id in features.key_attackers:
            score -= 5000
        elif _is_key_preevolution(card_id, features):
            score -= 3500
        if features.field_counts[card_id] == 0:
            score -= 2500
        else:
            score += 500
    elif card_type == "SUPPORTER":
        name = _card_name(card_id, features.card_by_id).casefold()
        if ("boss" in name or "catcher" in name) and _route_needs_boss(features):
            score -= 4000
        score -= 1200
    elif card_type == "ITEM":
        name = _card_name(card_id, features.card_by_id).casefold()
        if "rare candy" in name and _key_stage2_exists(features):
            score -= 3500
        score += 1000
    return score


def _is_energy_useful(energy_card: Any, pokemon: Any, features: BoardFeatures) -> bool:
    card_id = _card_id(energy_card)
    card_data = features.card_by_id.get(card_id) if card_id is not None else None
    if card_data is None:
        return False
    energy_type = getattr(card_data, "energyType", 0)
    
    is_planned_attacker = False
    plan = features.plan
    if plan.attacker_area == "ACTIVE":
        active = _first_active(features.my_state)
        if active is not None and active is pokemon:
            is_planned_attacker = True
    elif plan.attacker_area == "BENCH":
        bench = list(_get(features.my_state, "bench", []) or [])
        if 0 <= plan.attacker_index < len(bench) and bench[plan.attacker_index] is pokemon:
            is_planned_attacker = True
            
    if is_planned_attacker and plan.attack_id != -1:
        attack = features.attack_by_id.get(plan.attack_id)
        if attack is not None:
            cost = list(_get(attack, "energies", []) or [])
            attached = list(_get(pokemon, "energies", []) or [])
            missing = get_missing_energies(attached, cost)
            for req in missing:
                if req == 0 or energy_type == req or energy_type == 10:
                    return True
            return False
            
    for attack, missing in _get_missing_for_attacks(pokemon, features.card_by_id, features.attack_by_id):
        for req in missing:
            if req == 0 or energy_type == req or energy_type == 10:
                return True
                
    return False


def _attach_score(card: Any, pokemon: Any, features: BoardFeatures, *, active: bool) -> float:
    if pokemon is None:
        return -10000
    card_id = _card_id(card)
    card_data = features.card_by_id.get(card_id) if card_id is not None else None
    target_data = _card_data_for(pokemon, features.card_by_id)
    if _card_type_name(card_data) == "TOOL":
        score = 32000 + _hp(pokemon)
        if active:
            score += 1000
        if bool(_get(target_data, "ex", False)) or bool(_get(target_data, "megaEx", False)):
            score += 1200
        return score

    if not _is_energy_card(card_id, features.card_by_id):
        return 8000

    if not _is_energy_useful(card, pokemon, features):
        return 100.0

    missing_list = [len(missing) for attack, missing in _get_missing_for_attacks(pokemon, features.card_by_id, features.attack_by_id)]
    if not missing_list:
        return 500.0
    min_missing = min(missing_list)

    score = 25000
    if active:
        score += 1000
    if features.plan.needs_energy:
        if features.plan.attacker_area == "ACTIVE" and active:
            score += 30000
        elif features.plan.attacker_area == "BENCH" and not active:
            bench = list(_get(features.my_state, "bench", []) or [])
            try:
                if bench[features.plan.attacker_index] is pokemon:
                    score += 30000
            except IndexError:
                pass
    if min_missing > 0:
        score += min_missing * 3000
    else:
        score -= 12000
    if not features.active_ready and active:
        score += 5000
    if features.active_ready and not active:
        score += 1500
    if bool(_get(target_data, "ex", False)) or bool(_get(target_data, "megaEx", False)):
        score += 600
    return score


def _play_score(option: Any, features: BoardFeatures) -> float:
    card = _get_card(features.select, features.current, "HAND", _get(option, "index"), features.my_index)
    card_id = _card_id(card)
    data = features.card_by_id.get(card_id) if card_id is not None else None
    card_type = _card_type_name(data)
    name = _card_name(card_id, features.card_by_id).casefold()

    if card_type == "POKEMON":
        score = 52000 + _bench_candidate_score(card, features)
        if card_id in features.key_attackers:
            score += 12000
        elif _is_key_preevolution(card_id, features):
            score += 7000
        if features.field_counts[card_id] >= 2:
            score = -1
        return score
    if card_type == "SUPPORTER":
        if bool(_get(features.current, "supporterPlayed", False)):
            return -1
        if "boss" in name:
            return 59000 if _route_needs_boss(features) else -1
        if features.no_draw and any(word in name for word in DRAW_WORDS):
            return -1
        return 28000 + (1000 if any(word in name for word in ("lillie", "iono", "carmine")) else 0)
    if card_type == "STADIUM":
        current_stadium = _get(features.current, "stadium", []) or []
        if current_stadium and _card_id(current_stadium[0]) == card_id:
            return -1
        return 20000
    if card_type == "ITEM":
        if features.no_draw and any(word in name for word in ("ball", "poffin", "pad", "search")):
            return -1
        if "ultra ball" in name:
            discardable = sum(1 for score_id, count in features.hand_counts.items() if count >= 2 and score_id != card_id)
            return 47000 if discardable >= 1 else 12000
        if "rare candy" in name and _key_stage2_exists(features):
            return 52000
        if "poffin" in name or "ball" in name or "pad" in name:
            return 47000 if features.key_attackers else 44000
        if any(word in name for word in ("rod", "stretcher", "retrieval")):
            useful_discard = sum(
                count
                for discard_id, count in features.discard_counts.items()
                if _card_type_name(features.card_by_id.get(discard_id)) in {"POKEMON", "BASIC_ENERGY", "SPECIAL_ENERGY"}
            )
            return 42000 if useful_discard > 0 else -1
        return 24000
    if card_type == "TOOL":
        return 18000
    return 1000


def _ability_score(option: Any, features: BoardFeatures) -> float:
    card = _get_card(features.select, features.current, _get(option, "area"), _get(option, "index"), features.my_index)
    data = _card_data_for(card, features.card_by_id)
    skills = " ".join(
        f"{_get(skill, 'name', '')} {_get(skill, 'text', '')}" for skill in list(_get(data, "skills", []) or [])
    ).casefold()
    if features.no_draw and any(word in skills for word in DRAW_WORDS):
        return -1
    if any(word in skills for word in RECOVERY_WORDS) and not features.discard_counts:
        return -1
    score = 43000
    if features.plan.needs_energy and "attach" in skills:
        score += 12000
    if "draw" in skills:
        score += 3000
    return score


def _option_involves_pokemon(option: Any, select: Any, features: BoardFeatures, check_fn) -> bool:
    option_name = _option_type_name(option)
    
    # 1. Check direct card referenced by area/index
    area = _get(option, "area")
    index = _get(option, "index")
    if area is not None and index is not None:
        player_index = _get(option, "playerIndex")
        if player_index is None:
            player_index = features.my_index
        card = _get_card(select, features.current, area, index, player_index)
        if card is not None:
            card_data = _card_data_for(card, features.card_by_id)
            if card_data is not None and check_fn(card_data):
                return True
                
    # 2. Check inPlayArea/inPlayIndex (for ATTACH, EVOLVE)
    in_play_area = _get(option, "inPlayArea")
    in_play_index = _get(option, "inPlayIndex")
    if in_play_area is not None and in_play_index is not None:
        card = _get_card(select, features.current, in_play_area, in_play_index, features.my_index)
        if card is not None:
            card_data = _card_data_for(card, features.card_by_id)
            if card_data is not None and check_fn(card_data):
                return True
                
    # 3. Check Active Pokémon for ATTACK or RETREAT
    if option_name in {"ATTACK", "RETREAT"}:
        active = _first_active(features.my_state)
        if active is not None:
            card_data = _card_data_for(active, features.card_by_id)
            if card_data is not None and check_fn(card_data):
                return True

    # 4. Check contextCard on select
    context_card = _get(select, "contextCard")
    if context_card is not None:
        card_data = _card_data_for(context_card, features.card_by_id)
        if card_data is not None and check_fn(card_data):
            return True

    return False


def _score_option(index: int, option: Any, select: Any, features: BoardFeatures) -> float:
    option_name = _option_type_name(option)
    
    # Original scoring logic first
    score = _score_option_base(index, option, select, features)
    
    # Prevention rules check
    if option_name not in {"END", "RETREAT", "EVOLVE", "DISCARD"}:
        op_active = _first_active(features.op_state) if features.op_state is not None else None
        if op_active is not None:
            op_active_id = _card_id(op_active)
            if op_active_id is not None:
                # Helper predicates
                def check_basic_ex(c):
                    return _card_type_name(c) == "POKEMON" and bool(_get(c, "ex")) and bool(_get(c, "basic"))
                
                def check_ex(c):
                    return _card_type_name(c) == "POKEMON" and bool(_get(c, "ex"))
                    
                def check_ability(c):
                    return _card_type_name(c) == "POKEMON" and bool(_get(c, "skills"))

                # Rule 1: Farigiraf ex (ID=83)
                if op_active_id == 83:
                    if _option_involves_pokemon(option, select, features, check_basic_ex):
                        return 1.0

                # Rule 2: Sylveon (ID=330) or Crustle (ID=345)
                elif op_active_id in {330, 345}:
                    if _option_involves_pokemon(option, select, features, check_ex):
                        return 1.0

                # Rule 3: Cornerstone Mask Ogerpon ex (ID=117)
                elif op_active_id == 117:
                    if _option_involves_pokemon(option, select, features, check_ability):
                        return 1.0
                        
    return score


def _score_option_base(index: int, option: Any, select: Any, features: BoardFeatures) -> float:
    option_name = _option_type_name(option)
    context_name = _select_context_name(select)

    if option_name == "NUMBER":
        return _numeric_score(option)
    if option_name in {"YES", "NO"}:
        preferred = _yes_no_preference(select, list(_get(select, "option", []) or []))
        return 2.0 if index == preferred else 1.0
    if option_name == "END":
        return -1000

    if option_name == "CARD":
        card = _get_card(select, features.current, _get(option, "area"), _get(option, "index"), _get(option, "playerIndex"))
        player_index = _int_or_none(_get(option, "playerIndex"))
        own = player_index == features.my_index
        if context_name in ACTIVE_CONTEXTS:
            score = _active_candidate_score(card, features, own=own)
            step = _first_prize_step(features)
            if (
                step is not None
                and not own
                and player_index == features.opponent_index
                and _area_name(_get(option, "area")) == step.target_area
                and _int_or_none(_get(option, "index")) == step.target_index
            ):
                score += 15000
            return score
        if context_name == "SETUP_BENCH_POKEMON":
            return _bench_candidate_score(card, features)
        if context_name in SEARCH_CONTEXTS:
            return _search_candidate_score(card, features)
        if context_name == "DISCARD":
            return _discard_candidate_score(card, features)
        if context_name in OPPONENT_TARGET_CONTEXTS:
            score = _pokemon_target_value(card, features.card_by_id, attack_damage=False)
            remain_damage = int(_get(select, "remainDamageCounter", 0) or 0) * 10
            if remain_damage and _hp(card) <= remain_damage:
                score += 4000
            if player_index == features.my_index:
                score -= 10000
            step = _first_prize_step(features)
            if (
                step is not None
                and player_index == features.opponent_index
                and _area_name(_get(option, "area")) == step.target_area
                and _int_or_none(_get(option, "index")) == step.target_index
            ):
                score += 15000
            return score
        if context_name in SELF_TARGET_CONTEXTS:
            score = max(0, _max_hp(card) - _hp(card)) * 20
            if player_index != features.my_index:
                score -= 10000
            return score
        if context_name == "ATTACH_FROM":
            return _attach_score(_get(select, "contextCard"), card, features, active=_area_name(_get(option, "area")) == "ACTIVE")
        return _search_candidate_score(card, features)

    if option_name in {"ENERGY_CARD", "TOOL_CARD", "ENERGY"}:
        attached = _attached_option_card(select, features.current, option)
        player_index = _int_or_none(_get(option, "playerIndex"))
        score = 1000
        if player_index != features.my_index:
            score += 2000
        if attached is not None and _is_energy_card(_card_id(attached), features.card_by_id):
            score += 500
        if _area_name(_get(option, "area")) == "BENCH":
            score += 200
        return score

    if option_name == "PLAY":
        return _play_score(option, features)
    if option_name == "ATTACH":
        card = _get_card(select, features.current, _get(option, "area"), _get(option, "index"), features.my_index)
        pokemon = _get_card(
            select,
            features.current,
            _get(option, "inPlayArea"),
            _get(option, "inPlayIndex"),
            features.my_index,
        )
        return _attach_score(card, pokemon, features, active=_area_name(_get(option, "inPlayArea")) == "ACTIVE")
    if option_name == "EVOLVE":
        target = _get_card(
            select,
            features.current,
            _get(option, "inPlayArea"),
            _get(option, "inPlayIndex"),
            features.my_index,
        )
        evolved = _get_card(select, features.current, _get(option, "area"), _get(option, "index"), features.my_index)
        evolved_data = _card_data_for(evolved, features.card_by_id)
        evolved_id = _card_id(evolved)
        score = 65000 + _energy_count(target) * 1000 + int(_get(evolved_data, "hp", 0) or 0)
        if features.plan.attacker_area == _area_name(_get(option, "inPlayArea")):
            score += 5000
        if evolved_id in features.key_attackers:
            score += 16000
        return score
    if option_name == "ABILITY":
        return _ability_score(option, features)
    if option_name == "RETREAT":
        if features.plan.attacker_area == "BENCH" or (not features.active_ready and features.bench_ready_indices):
            return 16000
        return -1
    if option_name == "ATTACK":
        attack_id = _int_or_none(_get(option, "attackId"))
        attack = features.attack_by_id.get(attack_id) if attack_id is not None else None
        score = 5000 + _attack_damage(attack) * 20
        if attack_id == features.plan.attack_id:
            score += 18000
        active = _first_active(features.my_state)
        op_active = _first_active(features.op_state)
        if active is not None and op_active is not None and attack is not None:
            damage = _effective_damage(active, op_active, attack, features.card_by_id)
            if damage >= _hp(op_active):
                score += 8000 + _prize_count(op_active, features.card_by_id) * 3000
        return score
    if option_name == "DISCARD":
        return -1

    return _base_priority_score(option, select)


def _fallback_sort_key(index: int, option: Any, select: Any, current: Any) -> tuple:
    option_name = _option_type_name(option)
    select_name = _select_type_name(select)
    context_name = _select_context_name(select)
    priority = MAIN_ACTION_PRIORITY.get(option_name, 900) if select_name == "MAIN" else (
        GENERAL_OPTION_PRIORITY.get(option_name, 900)
    )
    your_index = _your_index(current)
    player_index = _int_or_none(_get(option, "playerIndex"))
    side_bonus = 0
    if your_index is not None and player_index is not None:
        if context_name in OPPONENT_TARGET_CONTEXTS and player_index == your_index:
            side_bonus = 50
        elif context_name in SELF_TARGET_CONTEXTS and player_index != your_index:
            side_bonus = 50
    active_bonus = 0 if _area_name(_get(option, "area")) == "ACTIVE" else 10
    numeric = -_numeric_score(option) if select_name == "COUNT" else 0
    return (priority, side_bonus, numeric, active_bonus, index)


def _card_trace(card: Any, features: BoardFeatures) -> dict[str, Any] | None:
    card_id = _card_id(card)
    if card_id is None:
        return None
    return {"id": card_id, "name": _card_name(card_id, features.card_by_id)}


def _option_trace(index: int, option: Any, select: Any, features: BoardFeatures, score: float) -> dict[str, Any]:
    option_name = _option_type_name(option)
    trace: dict[str, Any] = {
        "index": index,
        "type": option_name,
        "score": round(float(score), 3),
    }
    card = None
    if option_name == "PLAY":
        card = _get_card(select, features.current, "HAND", _get(option, "index"), features.my_index)
    elif option_name == "ATTACH":
        card = _get_card(select, features.current, _get(option, "area"), _get(option, "index"), features.my_index)
        target = _get_card(
            select,
            features.current,
            _get(option, "inPlayArea"),
            _get(option, "inPlayIndex"),
            features.my_index,
        )
        target_trace = _card_trace(target, features)
        if target_trace is not None:
            trace["target"] = target_trace
    elif option_name == "EVOLVE":
        card = _get_card(select, features.current, _get(option, "area"), _get(option, "index"), features.my_index)
        target = _get_card(
            select,
            features.current,
            _get(option, "inPlayArea"),
            _get(option, "inPlayIndex"),
            features.my_index,
        )
        target_trace = _card_trace(target, features)
        if target_trace is not None:
            trace["target"] = target_trace
    elif option_name == "CARD":
        card = _get_card(
            select,
            features.current,
            _get(option, "area"),
            _get(option, "index"),
            _get(option, "playerIndex"),
        )
        trace["area"] = _area_name(_get(option, "area"))
        player_index = _int_or_none(_get(option, "playerIndex"))
        if player_index is not None:
            trace["player_index"] = player_index
    elif option_name == "ATTACK":
        attack_id = _int_or_none(_get(option, "attackId"))
        trace["attack_id"] = attack_id
        attack = features.attack_by_id.get(attack_id) if attack_id is not None else None
        if attack is not None:
            trace["attack_damage"] = _attack_damage(attack)
    elif option_name == "NUMBER":
        trace["number"] = _get(option, "number")

    card_trace = _card_trace(card, features)
    if card_trace is not None:
        trace["card"] = card_trace
    return trace


def _prize_step_trace(step: PrizeMapStep, features: BoardFeatures) -> dict[str, Any]:
    target_lookup = _target_by_key(_build_target_states(features))
    return {
        "attacker": {
            "id": step.attacker_card_id,
            "name": _card_name(step.attacker_card_id, features.card_by_id),
            "area": step.attacker_area,
            "index": step.attacker_index,
        },
        "attack_id": step.attack_id,
        "target": {
            "id": step.target_card_id,
            "name": _card_name(step.target_card_id, features.card_by_id),
            "area": step.target_area,
            "index": step.target_index,
        },
        "prizes_taken": step.prizes_taken,
        "damage": step.damage,
        "needs_boss": step.needs_boss,
        "needs_switch": step.needs_switch,
        "needs_energy": step.needs_energy,
        "setup_cost": round(step.setup_cost, 3),
        "score": round(step.score, 3),
        "target_damages": [
            {
                "area": area,
                "index": index,
                "id": target_lookup.get((area, index)).card_id if target_lookup.get((area, index)) else -1,
                "name": _card_name(target_lookup.get((area, index)).card_id, features.card_by_id)
                if target_lookup.get((area, index))
                else "",
                "damage": damage,
            }
            for (area, index), damage in sorted(step.target_damages.items())
        ],
    }


def _decision_trace(
    select: Any,
    features: BoardFeatures,
    scores: list[float],
    output: list[int],
) -> RuleDecisionTrace:
    selected_options = [
        _option_trace(index, list(_get(select, "option", []) or [])[index], select, features, scores[index])
        for index in output
        if 0 <= index < len(scores)
    ]
    prize_map = {
        "total_prizes": features.prize_map.total_prizes,
        "attack_count": features.prize_map.attack_count,
        "setup_cost": round(features.prize_map.setup_cost, 3),
        "score": round(features.prize_map.score, 3),
        "steps": [_prize_step_trace(step, features) for step in features.prize_map.steps[:3]],
    }
    key_attackers = [
        {"id": card_id, "name": _card_name(card_id, features.card_by_id)}
        for card_id in sorted(features.key_attackers)
    ]
    return RuleDecisionTrace(
        turn=int(_get(features.current, "turn", 0) or 0),
        player_index=features.my_index,
        select_type=_select_type_name(select),
        context=_select_context_name(select),
        selected_options=selected_options,
        prize_map=prize_map,
        key_attackers=key_attackers,
    )


def select_option_indices(
    select: Any,
    *,
    current: Any = None,
    card_by_id: dict[int, Any] | None = None,
    attack_by_id: dict[int, Any] | None = None,
    deck_ids: Sequence[int] = (),
    trace_sink: list[RuleDecisionTrace] | None = None,
    trace_limit: int = 0,
) -> list[int]:
    options = list(_get(select, "option", []) or [])
    if not options:
        return []

    if _select_type_name(select) == "YES_NO":
        return [_yes_no_preference(select, options)]

    target_count = _target_count(select, options)
    if target_count <= 0:
        return []

    card_by_id = card_by_id or {}
    attack_by_id = attack_by_id or {}
    context_name = _select_context_name(select)

    if card_by_id or attack_by_id:
        features = _make_features(select, current, card_by_id, attack_by_id, deck_ids=deck_ids)
        scores = [_score_option(index, option, select, features) for index, option in enumerate(options)]
        ranked = sorted(range(len(options)), key=lambda index: (-scores[index], index))
        output: list[int] = []
        min_count = int(_get(select, "minCount", 0) or 0)
        for index in ranked:
            if len(output) >= target_count:
                break
            if len(output) < min_count or context_name not in OPTIONAL_POSITIVE_CONTEXTS or scores[index] > 0:
                output.append(index)
        if trace_sink is not None and (trace_limit <= 0 or len(trace_sink) < trace_limit):
            trace_sink.append(_decision_trace(select, features, scores, output))
        return output

    ranked = sorted(
        range(len(options)),
        key=lambda index: _fallback_sort_key(index, options[index], select, current),
    )
    return ranked[:target_count]


@dataclass
class RuleBasedAgent:
    deck_ids: Sequence[int]
    card_data: Sequence[Any] = ()
    attack_data: Sequence[Any] = ()
    trace: list[RuleDecisionTrace] | None = None
    trace_limit: int = 0
    card_by_id: dict[int, Any] = field(init=False, default_factory=dict)
    attack_by_id: dict[int, Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.deck_ids) != 60:
            raise ValueError(f"RuleBasedAgent needs a 60-card deck, got {len(self.deck_ids)}.")
        self.card_by_id = {
            int(_get(card, "cardId")): card for card in self.card_data if _get(card, "cardId") is not None
        }
        self.attack_by_id = {
            int(_get(attack, "attackId")): attack
            for attack in self.attack_data
            if _get(attack, "attackId") is not None
        }

        # Identify all types of attackers in the deck at the start of the game
        basic_ex_attackers = []
        ex_attackers = []
        ability_attackers = []
        for card_id in self.deck_ids:
            card = self.card_by_id.get(card_id)
            if card is not None and _card_type_name(card) == "POKEMON":
                name = str(_get(card, "name", ""))
                is_ex = bool(_get(card, "ex"))
                is_basic = bool(_get(card, "basic"))
                has_ability = bool(_get(card, "skills"))
                if is_ex and is_basic:
                    basic_ex_attackers.append(name)
                if is_ex:
                    ex_attackers.append(name)
                if has_ability:
                    ability_attackers.append(name)
                    
        print("Deck attackers identified:")
        print(f"  Basic Pokemon ex: {sorted(list(set(basic_ex_attackers)))}")
        print(f"  Pokemon ex: {sorted(list(set(ex_attackers)))}")
        print(f"  Pokemon with Ability: {sorted(list(set(ability_attackers)))}")

    def act(self, observation: Any) -> list[int]:
        select = _get(observation, "select")
        if select is None:
            return list(self.deck_ids)
        return select_option_indices(
            select,
            current=_get(observation, "current"),
            card_by_id=self.card_by_id,
            attack_by_id=self.attack_by_id,
            deck_ids=self.deck_ids,
            trace_sink=self.trace,
            trace_limit=self.trace_limit,
        )
