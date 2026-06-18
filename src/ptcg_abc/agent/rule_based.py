from __future__ import annotations

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
    target_index: int = -1
    attack_id: int = -1
    needs_energy: bool = False
    score: float = -1.0


@dataclass
class BoardFeatures:
    select: Any
    current: Any
    card_by_id: dict[int, Any]
    attack_by_id: dict[int, Any]
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
) -> BoardFeatures:
    my_index = _your_index(current)
    if my_index is None:
        my_index = 0
    features = BoardFeatures(
        select=select,
        current=current,
        card_by_id=card_by_id,
        attack_by_id=attack_by_id,
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


def _build_attack_plan(features: BoardFeatures) -> AttackPlan:
    plan = AttackPlan()
    if features.my_state is None or features.op_state is None:
        return plan

    legal_active_attack_ids = {
        int(_get(option, "attackId"))
        for option in list(_get(features.select, "option", []) or [])
        if _option_type_name(option) == "ATTACK" and _get(option, "attackId") is not None
    }
    own_pokemon = [("ACTIVE", 0, _first_active(features.my_state))]
    own_pokemon.extend(("BENCH", index, pokemon) for index, pokemon in enumerate(_get(features.my_state, "bench", []) or []))
    targets = [("ACTIVE", 0, _first_active(features.op_state))]
    if features.can_boss:
        targets.extend(("BENCH", index, pokemon) for index, pokemon in enumerate(_get(features.op_state, "bench", []) or []))

    for area, attacker_index, attacker in own_pokemon:
        if attacker is None:
            continue
        retreat_attachment_cost = 0
        if area == "BENCH":
            if not features.can_switch:
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
            cost = list(_get(attack, "energies", []) or [])
            attached = list(_get(attacker, "energies", []) or [])
            
            missing = get_missing_energies(attached, cost)
            if not missing:
                needs_energy = False
                if retreat_attachment_cost > 0:
                    max_allowed, virtual_types = _get_max_attachments_for_pokemon(attacker, features)
                    if retreat_attachment_cost > max_allowed or features.hand_energy_count < retreat_attachment_cost:
                        continue
                    needs_energy = True
            else:
                max_allowed, virtual_types = _get_max_attachments_for_pokemon(attacker, features)
                total_needed = len(missing) + retreat_attachment_cost
                if total_needed > max_allowed:
                    continue
                
                hand_energy_types = _get_hand_energy_types(features) + virtual_types
                if len(hand_energy_types) < total_needed:
                    continue
                
                temp_missing = list(missing)
                temp_hand = list(hand_energy_types)
                possible = True
                for req in list(temp_missing):
                    matched = False
                    for he_type in temp_hand:
                        if req == 0 or he_type == req or he_type == 10:
                            temp_missing.remove(req)
                            temp_hand.remove(he_type)
                            matched = True
                            break
                    if not matched:
                        possible = False
                        break
                if not possible or len(temp_hand) < retreat_attachment_cost:
                    continue
                
                needs_energy = True

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
                if needs_energy:
                    score -= 150
                if score > plan.score:
                    plan = AttackPlan(
                        attacker_area=area,
                        attacker_index=attacker_index,
                        target_index=target_index if target_area == "BENCH" else 0,
                        attack_id=attack_id,
                        needs_energy=needs_energy,
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


def _active_candidate_score(card: Any, features: BoardFeatures, *, own: bool) -> float:
    if card is None:
        return -10000
    if not own:
        return _pokemon_target_value(card, features.card_by_id)
    data = _card_data_for(card, features.card_by_id)
    score = _hp(card) + _energy_count(card) * 800
    if _can_attack_soon(card, features.card_by_id, features.attack_by_id):
        score += 12000
    score += _best_attack_damage(card, features.card_by_id, features.attack_by_id) * 8
    if bool(_get(data, "basic", False)):
        score += 1000
    if bool(_get(data, "ex", False)) or bool(_get(data, "megaEx", False)):
        score -= 250
    score -= int(_get(data, "retreatCost", 0) or 0) * 80
    if features.plan.attacker_area == "BENCH":
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
    if "boss" in name and features.plan.target_index > 0:
        return 9000
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
                        score -= 3000
    elif card_type == "POKEMON":
        if features.field_counts[card_id] == 0:
            score -= 2500
        else:
            score += 500
    elif card_type == "SUPPORTER":
        score -= 1200
    elif card_type == "ITEM":
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
        if features.field_counts[card_id] >= 2:
            score = -1
        return score
    if card_type == "SUPPORTER":
        if bool(_get(features.current, "supporterPlayed", False)):
            return -1
        if "boss" in name:
            return 56000 if features.plan.target_index > 0 else -1
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
        if "poffin" in name or "ball" in name or "pad" in name:
            return 44000
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
            return _active_candidate_score(card, features, own=own)
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
        score = 65000 + _energy_count(target) * 1000 + int(_get(evolved_data, "hp", 0) or 0)
        if features.plan.attacker_area == _area_name(_get(option, "inPlayArea")):
            score += 5000
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
            score += 10000
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


def select_option_indices(
    select: Any,
    *,
    current: Any = None,
    card_by_id: dict[int, Any] | None = None,
    attack_by_id: dict[int, Any] | None = None,
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
        features = _make_features(select, current, card_by_id, attack_by_id)
        scores = [_score_option(index, option, select, features) for index, option in enumerate(options)]
        ranked = sorted(range(len(options)), key=lambda index: (-scores[index], index))
        output: list[int] = []
        min_count = int(_get(select, "minCount", 0) or 0)
        for index in ranked:
            if len(output) >= target_count:
                break
            if len(output) < min_count or context_name not in OPTIONAL_POSITIVE_CONTEXTS or scores[index] > 0:
                output.append(index)
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
        )
