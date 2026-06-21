from __future__ import annotations

import hashlib
from typing import Any, Sequence

from ptcg_abc.agent.rule_based import (
    _area_name,
    _card_id,
    _card_name,
    _card_type_name,
    _get,
    _get_card,
    _int_or_none,
    _option_type_name,
    _select_context_name,
    _select_type_name,
    _target_count,
    score_legal_options,
    select_option_indices,
)
from ptcg_abc.rl.records import ActionFrame, DecisionFrame


BOARD_IMAGE_HEIGHT = 64
BOARD_IMAGE_WIDTH = 64
CARD_SLOT_WIDTH = 6
CARD_SLOT_HEIGHT = 8


def card_lookup(card_data: Sequence[Any]) -> dict[int, Any]:
    return {
        int(_get(card, "cardId")): card
        for card in card_data
        if _get(card, "cardId") is not None
    }


def attack_lookup(attack_data: Sequence[Any]) -> dict[int, Any]:
    return {
        int(_get(attack, "attackId")): attack
        for attack in attack_data
        if _get(attack, "attackId") is not None
    }


def make_decision_frame(
    observation: Any,
    *,
    deck_ids: Sequence[int] = (),
    card_data: Sequence[Any] = (),
    attack_data: Sequence[Any] = (),
    card_by_id: dict[int, Any] | None = None,
    attack_by_id: dict[int, Any] | None = None,
    selected_indices: Sequence[int] | None = None,
    reward_metadata: dict[str, Any] | None = None,
) -> DecisionFrame | None:
    select = _get(observation, "select")
    current = _get(observation, "current")
    if select is None:
        return None

    options = list(_get(select, "option", []) or [])
    card_by_id = card_by_id or card_lookup(card_data)
    attack_by_id = attack_by_id or attack_lookup(attack_data)
    rule_scores = score_legal_options(
        select,
        current=current,
        card_by_id=card_by_id,
        attack_by_id=attack_by_id,
        deck_ids=deck_ids,
    )
    if selected_indices is None:
        selected_indices = select_option_indices(
            select,
            current=current,
            card_by_id=card_by_id,
            attack_by_id=attack_by_id,
            deck_ids=deck_ids,
        )

    ranked_indices = sorted(range(len(options)), key=lambda index: (-rule_scores[index], index))
    rule_ranks = {index: rank + 1 for rank, index in enumerate(ranked_indices)}
    board = summarize_board(current, card_by_id=card_by_id)
    target_count = _target_count(select, options)

    actions = [
        make_action_frame(
            index=index,
            option=option,
            select=select,
            current=current,
            board=board,
            card_by_id=card_by_id,
            rule_score=rule_scores[index] if index < len(rule_scores) else 0.0,
            rule_rank=rule_ranks.get(index, len(options)),
        )
        for index, option in enumerate(options)
    ]

    return DecisionFrame(
        select_type=_select_type_name(select),
        context=_select_context_name(select),
        min_count=int(_get(select, "minCount", 0) or 0),
        max_count=int(_get(select, "maxCount", 0) or 0),
        target_count=target_count,
        legal_options=actions,
        rule_selected_indices=[int(index) for index in selected_indices],
        board=board,
        board_image=render_board_image(board),
        reward_metadata=dict(reward_metadata or {}),
    )


def make_action_frame(
    *,
    index: int,
    option: Any,
    select: Any,
    current: Any,
    board: dict[str, Any],
    card_by_id: dict[int, Any],
    rule_score: float,
    rule_rank: int,
) -> ActionFrame:
    option_type = _option_type_name(option)
    card = _option_card(option, select, current, option_type)
    target = _option_target(option, select, current, option_type)
    card_id = _card_id(card)
    target_id = _card_id(target)
    area = _area_name(_get(option, "area")) if _get(option, "area") is not None else ""
    area_index = _int_or_none(_get(option, "index"))
    player_index = _int_or_none(_get(option, "playerIndex"))
    target_area = (
        _area_name(_get(option, "inPlayArea"))
        if _get(option, "inPlayArea") is not None
        else ""
    )
    target_index = _int_or_none(_get(option, "inPlayIndex"))
    attack_id = _int_or_none(_get(option, "attackId"))
    features = option_feature_vector(
        option_type=option_type,
        rule_score=rule_score,
        rule_rank=rule_rank,
        card=card,
        target=target,
        card_by_id=card_by_id,
        option=option,
        board=board,
    )
    return ActionFrame(
        index=index,
        option_type=option_type,
        features=features,
        rule_score=float(rule_score),
        rule_rank=rule_rank,
        card_id=card_id,
        card_name=_card_name(card_id, card_by_id),
        area=area,
        area_index=area_index,
        player_index=player_index,
        attack_id=attack_id,
        target_card_id=target_id,
        target_name=_card_name(target_id, card_by_id),
        target_area=target_area,
        target_index=target_index,
        raw=_raw_option(option),
    )


def option_feature_vector(
    *,
    option_type: str,
    rule_score: float,
    rule_rank: int,
    card: Any,
    target: Any,
    card_by_id: dict[int, Any],
    option: Any,
    board: dict[str, Any],
) -> dict[str, float]:
    card_id = _card_id(card)
    target_id = _card_id(target)
    card_data = card_by_id.get(card_id) if card_id is not None else None
    target_data = card_by_id.get(target_id) if target_id is not None else None
    target_hp = float(_get(target, "hp", 0) or 0)
    target_max_hp = float(_get(target, "maxHp", target_hp) or 0)
    own_player = _int_or_none(_get(option, "playerIndex")) == int(board.get("your_index", 0))
    features = {
        "bias": 1.0,
        "rule_score": _scaled(rule_score, 100000.0),
        "rule_rank_inv": 1.0 / max(1.0, float(rule_rank)),
        "is_main": float(option_type in {"PLAY", "ATTACH", "EVOLVE", "ABILITY", "ATTACK", "RETREAT", "END"}),
        "is_attack": float(option_type == "ATTACK"),
        "is_attach": float(option_type == "ATTACH"),
        "is_evolve": float(option_type == "EVOLVE"),
        "is_play": float(option_type == "PLAY"),
        "is_ability": float(option_type == "ABILITY"),
        "is_retreat": float(option_type == "RETREAT"),
        "is_end": float(option_type == "END"),
        "is_card": float(option_type == "CARD"),
        "is_discard": float(option_type == "DISCARD"),
        "is_number": float(option_type == "NUMBER"),
        "own_target": float(own_player),
        "card_id_hash": _hash_unit(card_id),
        "target_id_hash": _hash_unit(target_id),
        "card_hp": _scaled(_get(card, "hp", 0) or _get(card_data, "hp", 0) or 0, 400.0),
        "target_hp": _scaled(target_hp, 400.0),
        "target_damage_ratio": 1.0 - (target_hp / target_max_hp) if target_max_hp else 0.0,
        "card_is_ex": float(bool(_get(card_data, "ex", False))),
        "card_is_mega_ex": float(bool(_get(card_data, "megaEx", False))),
        "target_is_ex": float(bool(_get(target_data, "ex", False))),
        "target_is_mega_ex": float(bool(_get(target_data, "megaEx", False))),
        "my_prizes_remaining": _scaled(board.get("my_prizes", 0), 6.0),
        "op_prizes_remaining": _scaled(board.get("opponent_prizes", 0), 6.0),
        "my_hand_count": _scaled(board.get("my_hand_count", 0), 20.0),
        "my_deck_count": _scaled(board.get("my_deck_count", 0), 60.0),
        "turn": _scaled(board.get("turn", 0), 30.0),
    }
    number = _get(option, "number")
    if number is not None:
        features["number"] = _scaled(number, 20.0)
    attack_id = _int_or_none(_get(option, "attackId"))
    if attack_id is not None:
        features["attack_id_hash"] = _hash_unit(attack_id)
    return features


def summarize_board(current: Any, *, card_by_id: dict[int, Any] | None = None) -> dict[str, Any]:
    card_by_id = card_by_id or {}
    players = list(_get(current, "players", []) or [])
    your_index = int(_get(current, "yourIndex", 0) or 0)
    opponent_index = 1 - your_index
    mine = players[your_index] if 0 <= your_index < len(players) else None
    opponent = players[opponent_index] if 0 <= opponent_index < len(players) else None
    board = {
        "turn": int(_get(current, "turn", 0) or 0),
        "your_index": your_index,
        "energy_attached": bool(_get(current, "energyAttached", False)),
        "supporter_played": bool(_get(current, "supporterPlayed", False)),
        "stadium_count": len(list(_get(current, "stadium", []) or [])),
        "looking_count": len(list(_get(current, "looking", []) or [])),
    }
    board["stadium_card"] = _card_state(
        _first_or_none(_get(current, "stadium", [])),
        card_by_id=card_by_id,
    )
    board.update(_player_summary(mine, prefix="my", card_by_id=card_by_id))
    board.update(_player_summary(opponent, prefix="opponent", card_by_id=card_by_id))
    return board


def render_board_image(
    board: dict[str, Any],
    *,
    height: int = BOARD_IMAGE_HEIGHT,
    width: int = BOARD_IMAGE_WIDTH,
) -> list[list[float]]:
    image = [[0.0 for _ in range(width)] for _ in range(height)]
    if height < BOARD_IMAGE_HEIGHT or width < BOARD_IMAGE_WIDTH:
        return _legacy_board_image(board, height=height, width=width)

    _draw_global_state(image, board)
    _draw_hidden_count(image, 4, 1, int(board.get("opponent_hand_count", 0) or 0), top=True)
    _draw_hidden_count(image, 36, 62, int(board.get("my_hand_count", 0) or 0), top=False)

    _draw_count_zone(image, 2, 24, board.get("my_deck_count", 0), 60, 0.22)
    _draw_count_zone(image, 2, 36, board.get("my_discard_count", 0), 30, 0.32)
    _draw_count_zone(image, 2, 48, board.get("my_prizes", 0), 6, 0.42)
    _draw_count_zone(image, 56, 8, board.get("opponent_prizes", 0), 6, 0.42)
    _draw_count_zone(image, 56, 20, board.get("opponent_discard_count", 0), 30, 0.32)
    _draw_count_zone(image, 56, 32, board.get("opponent_deck_count", 0), 60, 0.22)

    _draw_card_tensor(
        image,
        29,
        20,
        board.get("opponent_active_card"),
        attachment_direction="right",
    )
    _draw_card_tensor(image, 29, 36, board.get("my_active_card"), attachment_direction="right")

    for index, state in enumerate(list(board.get("opponent_bench_cards", []) or [])[:5]):
        _draw_card_tensor(image, 9 + index * 10, 8, state, attachment_direction="up")
    for index, state in enumerate(list(board.get("my_bench_cards", []) or [])[:5]):
        _draw_card_tensor(image, 9 + index * 10, 50, state, attachment_direction="up")

    _draw_card_tensor(image, 45, 28, board.get("stadium_card"), width=8, height=5)
    return image


def _legacy_board_image(
    board: dict[str, Any],
    *,
    height: int,
    width: int,
) -> list[list[float]]:
    image = [[0.0 for _ in range(width)] for _ in range(height)]
    numeric_items = [
        (key, value)
        for key, value in sorted(board.items())
        if isinstance(value, (int, float, bool))
    ]
    for index, (key, value) in enumerate(numeric_items):
        row = index % height
        col = (index // height) % width
        denom = _feature_denominator(key)
        image[row][col] = _scaled(float(value), denom)
    for key in ("my_active_id", "opponent_active_id"):
        value = board.get(key)
        if value is None:
            continue
        row = min(height - 1, 12 if key.startswith("my") else 13)
        image[row][0] = _hash_unit(value)
    return image


def _draw_global_state(image: list[list[float]], board: dict[str, Any]) -> None:
    values = [
        _scaled(board.get("turn", 0), 30.0),
        float(bool(board.get("energy_attached", False))),
        float(bool(board.get("supporter_played", False))),
        _scaled(board.get("stadium_count", 0), 1.0),
        _scaled(board.get("looking_count", 0), 10.0),
    ]
    for index, value in enumerate(values):
        _set_cell(image, index, 0, value)


def _draw_hidden_count(
    image: list[list[float]],
    x: int,
    y: int,
    count: int,
    *,
    top: bool,
) -> None:
    count = max(0, count)
    for index in range(min(count, 12)):
        card_x = x + index * 2
        card_y = y if top else y - 2
        _fill_rect(image, card_x, card_y, 1, 2, 0.18)
    _draw_count_bar(image, x, y + (3 if top else -4), 24, count, 20, 0.36)


def _draw_count_zone(
    image: list[list[float]],
    x: int,
    y: int,
    count: Any,
    max_count: int,
    value: float,
) -> None:
    _fill_rect(image, x, y, CARD_SLOT_WIDTH, CARD_SLOT_HEIGHT, 0.05)
    _draw_border(image, x, y, CARD_SLOT_WIDTH, CARD_SLOT_HEIGHT, value)
    _draw_count_bar(image, x + 1, y + CARD_SLOT_HEIGHT - 2, CARD_SLOT_WIDTH - 2, count, max_count, value)


def _draw_card_tensor(
    image: list[list[float]],
    x: int,
    y: int,
    state: Any,
    *,
    width: int = CARD_SLOT_WIDTH,
    height: int = CARD_SLOT_HEIGHT,
    attachment_direction: str = "up",
) -> None:
    if not isinstance(state, dict) or not state:
        _fill_rect(image, x, y, width, height, 0.03)
        _draw_border(image, x, y, width, height, 0.12)
        return

    identity = float(state.get("id_hash", 0.0) or 0.0)
    type_value = float(state.get("type_hash", 0.0) or 0.0)
    hp_ratio = float(state.get("hp_ratio", 0.0) or 0.0)
    damage_ratio = float(state.get("damage_ratio", 0.0) or 0.0)
    _fill_rect(image, x, y, width, height, 0.08 + 0.5 * identity)
    _draw_border(image, x, y, width, height, 0.2 + 0.5 * type_value)
    _draw_count_bar(image, x + 1, y + height - 2, width - 2, hp_ratio, 1, 0.82)
    _draw_count_bar(image, x + 1, y + height - 1, width - 2, damage_ratio, 1, 0.68)

    _set_cell(image, x + 1, y + 1, _scaled(state.get("energy_count", 0), 8.0))
    _set_cell(image, x + 2, y + 1, _scaled(state.get("tool_count", 0), 2.0))
    _set_cell(image, x + 3, y + 1, _scaled(state.get("stage", 0), 2.0))
    _set_cell(image, x + 4, y + 1, 0.9 if state.get("is_mega_ex") else 0.65 if state.get("is_ex") else 0.0)
    _draw_condition_marks(image, x, y, width, height, state)
    _draw_attachment_tabs(image, x, y, width, height, state, direction=attachment_direction)


def _draw_condition_marks(
    image: list[list[float]],
    x: int,
    y: int,
    width: int,
    height: int,
    state: dict[str, Any],
) -> None:
    marks = [
        ("poisoned", 0.74),
        ("burned", 0.78),
        ("asleep", 0.58),
        ("paralyzed", 0.62),
        ("confused", 0.66),
    ]
    for index, (name, value) in enumerate(marks):
        if state.get(name):
            _set_cell(image, x + width - 1, y + min(height - 1, 2 + index), value)


def _draw_attachment_tabs(
    image: list[list[float]],
    x: int,
    y: int,
    width: int,
    height: int,
    state: dict[str, Any],
    *,
    direction: str,
) -> None:
    tabs = (
        [0.52] * min(2, int(state.get("tool_count", 0) or 0))
        + [0.62] * min(4, int(state.get("special_energy_count", 0) or 0))
        + [0.58] * min(4, int(state.get("other_energy_count", 0) or 0))
        + [0.72] * min(4, int(state.get("basic_energy_count", 0) or 0))
    )[:4]
    for index, value in enumerate(tabs):
        if direction == "right":
            _fill_rect(image, x + width + index, y + 1, 1, max(1, height - 2), value)
        else:
            _fill_rect(image, x + 1, y - 1 - index, max(1, width - 2), 1, value)


def _draw_count_bar(
    image: list[list[float]],
    x: int,
    y: int,
    width: int,
    count: Any,
    max_count: int,
    value: float,
) -> None:
    try:
        ratio = float(count) / float(max_count)
    except (TypeError, ValueError, ZeroDivisionError):
        ratio = 0.0
    filled = max(0, min(width, int(round(width * max(0.0, min(1.0, ratio))))))
    _fill_rect(image, x, y, width, 1, 0.04)
    _fill_rect(image, x, y, filled, 1, value)


def _draw_border(
    image: list[list[float]],
    x: int,
    y: int,
    width: int,
    height: int,
    value: float,
) -> None:
    _fill_rect(image, x, y, width, 1, value)
    _fill_rect(image, x, y + height - 1, width, 1, value)
    _fill_rect(image, x, y, 1, height, value)
    _fill_rect(image, x + width - 1, y, 1, height, value)


def _fill_rect(
    image: list[list[float]],
    x: int,
    y: int,
    width: int,
    height: int,
    value: float,
) -> None:
    for row in range(y, y + height):
        for col in range(x, x + width):
            _set_cell(image, col, row, value)


def _set_cell(image: list[list[float]], x: int, y: int, value: float) -> None:
    if y < 0 or y >= len(image):
        return
    if x < 0 or x >= len(image[y]):
        return
    image[y][x] = max(0.0, min(1.0, float(value)))


def _player_summary(
    player: Any,
    *,
    prefix: str,
    card_by_id: dict[int, Any],
) -> dict[str, Any]:
    active = _first_or_none(_get(player, "active", []))
    bench = [card for card in list(_get(player, "bench", []) or []) if card is not None]
    hand = list(_get(player, "hand", []) or [])
    hand_count_value = _get(player, "handCount", None)
    hand_count = int(hand_count_value) if hand_count_value is not None else len(hand)
    discard = list(_get(player, "discard", []) or [])
    prize = list(_get(player, "prize", []) or [])
    active_id = _card_id(active)
    active_card = _card_state(active, card_by_id=card_by_id)
    active_card.update(_active_condition_state(player, active))
    bench_cards = [_card_state(pokemon, card_by_id=card_by_id) for pokemon in bench[:5]]
    summary = {
        f"{prefix}_active_id": active_id,
        f"{prefix}_active_hp": int(_get(active, "hp", 0) or 0),
        f"{prefix}_active_max_hp": int(_get(active, "maxHp", 0) or 0),
        f"{prefix}_active_energy_count": len(list(_get(active, "energies", []) or [])),
        f"{prefix}_bench_count": len(bench),
        f"{prefix}_hand_count": hand_count,
        f"{prefix}_discard_count": len(discard),
        f"{prefix}_prizes": len(prize),
        f"{prefix}_deck_count": int(_get(player, "deckCount", 0) or 0),
        f"{prefix}_active_is_ex": bool(_get(card_by_id.get(active_id), "ex", False)),
        f"{prefix}_active_is_mega_ex": bool(_get(card_by_id.get(active_id), "megaEx", False)),
        f"{prefix}_active_card": active_card,
        f"{prefix}_bench_cards": bench_cards,
        f"{prefix}_discard_top_card": _card_state(discard[-1], card_by_id=card_by_id)
        if discard
        else {},
    }
    for index, pokemon in enumerate(bench[:5]):
        card_id = _card_id(pokemon)
        summary[f"{prefix}_bench_{index}_id"] = card_id
        summary[f"{prefix}_bench_{index}_hp"] = int(_get(pokemon, "hp", 0) or 0)
        summary[f"{prefix}_bench_{index}_energy_count"] = len(
            list(_get(pokemon, "energies", []) or [])
        )
    return summary


def _card_state(card: Any, *, card_by_id: dict[int, Any]) -> dict[str, Any]:
    if card is None:
        return {}
    card_id = _card_id(card)
    data = card_by_id.get(card_id)
    hp = int(_get(card, "hp", _get(data, "hp", 0)) or 0)
    max_hp = int(_get(card, "maxHp", _get(data, "hp", hp)) or hp or 0)
    energy_cards = list(_get(card, "energyCards", []) or [])
    energies = list(_get(card, "energies", []) or [])
    tools = list(_get(card, "tools", []) or [])
    special_energy = basic_energy = other_energy = 0
    for energy_card in energy_cards:
        energy_type = _card_type_name(card_by_id.get(_card_id(energy_card)))
        if energy_type == "SPECIAL_ENERGY":
            special_energy += 1
        elif energy_type == "BASIC_ENERGY":
            basic_energy += 1
        else:
            other_energy += 1
    if not energy_cards and energies:
        basic_energy = len(energies)
    card_type = _card_type_name(data) if data is not None else ""
    stage = 0
    if bool(_get(data, "stage2", False)):
        stage = 2
    elif bool(_get(data, "stage1", False)):
        stage = 1
    return {
        "id": card_id,
        "id_hash": _hash_unit(card_id),
        "type_hash": _hash_unit(card_type),
        "energy_type_hash": _hash_unit(_get(data, "energyType")),
        "hp": hp,
        "max_hp": max_hp,
        "hp_ratio": hp / max_hp if max_hp else 0.0,
        "damage_ratio": max(0, max_hp - hp) / max_hp if max_hp else 0.0,
        "energy_count": len(energies) or len(energy_cards),
        "tool_count": len(tools),
        "special_energy_count": special_energy,
        "basic_energy_count": basic_energy,
        "other_energy_count": other_energy,
        "is_ex": bool(_get(data, "ex", False)),
        "is_mega_ex": bool(_get(data, "megaEx", False)),
        "stage": stage,
        "poisoned": bool(_get(card, "poisoned", False)),
        "burned": bool(_get(card, "burned", False)),
        "asleep": bool(_get(card, "asleep", False)),
        "paralyzed": bool(_get(card, "paralyzed", False)),
        "confused": bool(_get(card, "confused", False)),
    }


def _active_condition_state(player: Any, card: Any) -> dict[str, bool]:
    return {
        "poisoned": bool(_get(player, "poisoned", False)) or bool(_get(card, "poisoned", False)),
        "burned": bool(_get(player, "burned", False)) or bool(_get(card, "burned", False)),
        "asleep": bool(_get(player, "asleep", False)) or bool(_get(card, "asleep", False)),
        "paralyzed": bool(_get(player, "paralyzed", False)) or bool(_get(card, "paralyzed", False)),
        "confused": bool(_get(player, "confused", False)) or bool(_get(card, "confused", False)),
    }


def _option_card(option: Any, select: Any, current: Any, option_type: str) -> Any | None:
    if option_type == "PLAY":
        return _get_card(select, current, "HAND", _get(option, "index"), _get(current, "yourIndex", 0))
    if option_type in {"CARD", "ENERGY_CARD", "TOOL_CARD"}:
        return _get_card(
            select,
            current,
            _get(option, "area"),
            _get(option, "index"),
            _get(option, "playerIndex"),
        )
    if option_type in {"ATTACH", "EVOLVE"}:
        return _get_card(
            select,
            current,
            _get(option, "area"),
            _get(option, "index"),
            _get(current, "yourIndex", 0),
        )
    return None


def _option_target(option: Any, select: Any, current: Any, option_type: str) -> Any | None:
    if option_type in {"ATTACH", "EVOLVE"}:
        return _get_card(
            select,
            current,
            _get(option, "inPlayArea"),
            _get(option, "inPlayIndex"),
            _get(current, "yourIndex", 0),
        )
    if option_type == "CARD":
        return _option_card(option, select, current, option_type)
    return None


def _raw_option(option: Any) -> dict[str, Any]:
    keys = (
        "type",
        "number",
        "area",
        "index",
        "playerIndex",
        "inPlayArea",
        "inPlayIndex",
        "attackId",
        "energyIndex",
        "toolIndex",
    )
    return {
        key: _jsonable(_get(option, key))
        for key in keys
        if _get(option, key) is not None
    }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "value"):
        return _jsonable(value.value)
    if hasattr(value, "name") and not isinstance(value, str):
        return str(value.name)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _first_or_none(values: Any) -> Any | None:
    for value in list(values or []):
        if value is not None:
            return value
    return None


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
    return int.from_bytes(digest, "big") / 0xFFFFFFFF


def _feature_denominator(key: str) -> float:
    if key.endswith("_hp") or key.endswith("_max_hp"):
        return 400.0
    if key.endswith("_count"):
        return 60.0
    if key.endswith("_prizes"):
        return 6.0
    if key == "turn":
        return 30.0
    return 1.0
