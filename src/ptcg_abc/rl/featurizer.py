from __future__ import annotations

import hashlib
from typing import Any, Sequence

from ptcg_abc.agent.rule_based import (
    _area_name,
    _card_id,
    _card_name,
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


BOARD_IMAGE_HEIGHT = 16
BOARD_IMAGE_WIDTH = 16


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
        row = 12 if key.startswith("my") else 13
        image[row][0] = _hash_unit(value)
    return image


def _player_summary(
    player: Any,
    *,
    prefix: str,
    card_by_id: dict[int, Any],
) -> dict[str, Any]:
    active = _first_or_none(_get(player, "active", []))
    bench = [card for card in list(_get(player, "bench", []) or []) if card is not None]
    hand = list(_get(player, "hand", []) or [])
    discard = list(_get(player, "discard", []) or [])
    prize = list(_get(player, "prize", []) or [])
    active_id = _card_id(active)
    summary = {
        f"{prefix}_active_id": active_id,
        f"{prefix}_active_hp": int(_get(active, "hp", 0) or 0),
        f"{prefix}_active_max_hp": int(_get(active, "maxHp", 0) or 0),
        f"{prefix}_active_energy_count": len(list(_get(active, "energies", []) or [])),
        f"{prefix}_bench_count": len(bench),
        f"{prefix}_hand_count": len(hand),
        f"{prefix}_discard_count": len(discard),
        f"{prefix}_prizes": len(prize),
        f"{prefix}_deck_count": int(_get(player, "deckCount", 0) or 0),
        f"{prefix}_active_is_ex": bool(_get(card_by_id.get(active_id), "ex", False)),
        f"{prefix}_active_is_mega_ex": bool(_get(card_by_id.get(active_id), "megaEx", False)),
    }
    for index, pokemon in enumerate(bench[:5]):
        card_id = _card_id(pokemon)
        summary[f"{prefix}_bench_{index}_id"] = card_id
        summary[f"{prefix}_bench_{index}_hp"] = int(_get(pokemon, "hp", 0) or 0)
        summary[f"{prefix}_bench_{index}_energy_count"] = len(
            list(_get(pokemon, "energies", []) or [])
        )
    return summary


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
