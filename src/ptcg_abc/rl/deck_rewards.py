from __future__ import annotations

from collections import Counter
from typing import Any


def deck_reward_potential_components(
    board: dict[str, Any],
    *,
    deck_index: int,
) -> dict[str, float]:
    if deck_index == 1:
        return alakazam_reward_potential_components(board)
    if deck_index == 3:
        return dragapult_reward_potential_components(board)
    raise ValueError(f"No deck reward potential is defined for deck index {deck_index}.")


def deck_reward_potential(board: dict[str, Any], *, deck_index: int) -> float:
    return sum(deck_reward_potential_components(board, deck_index=deck_index).values())


def deck_shaped_transition_value(
    root_board: dict[str, Any],
    end_board: dict[str, Any],
    *,
    deck_index: int,
    gamma: float,
    terminal: bool = False,
) -> float:
    root_prizes = optional_board_int(root_board.get("my_prizes"))
    end_prizes = optional_board_int(end_board.get("my_prizes"))
    prizes_taken = (
        max(0, root_prizes - end_prizes)
        if root_prizes is not None and end_prizes is not None
        else 0
    )
    root_potential = deck_reward_potential(root_board, deck_index=deck_index)
    end_potential = (
        0.0 if terminal else deck_reward_potential(end_board, deck_index=deck_index)
    )
    return 10.0 * prizes_taken + float(gamma) * end_potential - root_potential


def in_play_card_states(board: dict[str, Any], *, prefix: str) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    active = board.get(f"{prefix}_active_card")
    if isinstance(active, dict) and active:
        states.append(active)
    states.extend(
        state
        for state in list(board.get(f"{prefix}_bench_cards", []) or [])
        if isinstance(state, dict) and state
    )
    return states


def state_card_id(state: dict[str, Any]) -> int | None:
    return optional_board_int(state.get("id"))


def energy_ids(state: dict[str, Any]) -> set[int]:
    return {
        card_id
        for value in list(state.get("energy_card_ids", []) or [])
        if (card_id := optional_board_int(value)) is not None
    }


def alakazam_reward_potential_components(board: dict[str, Any]) -> dict[str, float]:
    mine = in_play_card_states(board, prefix="my")
    opponent = in_play_card_states(board, prefix="opponent")
    play_counts = Counter(state_card_id(state) for state in mine)
    hand_counts = Counter(
        card_id
        for value in list(board.get("my_hand_card_ids", []) or [])
        if (card_id := optional_board_int(value)) is not None
    )

    abra_count = play_counts[109]
    kadabra_count = play_counts[742]
    line_energy_count = sum(
        1
        for state in mine
        if state_card_id(state) in {109, 742, 245}
        and bool(energy_ids(state) & {5, 19})
    )
    genesect_ready = any(
        state_card_id(state) == 142 and int(state.get("tool_count", 0) or 0) > 0
        for state in mine
    ) and not bool(board.get("opponent_ace_spec_seen", False))
    psyduck_guard = play_counts[858] > 0 and any(
        state_card_id(state) in {131, 132, 133} for state in opponent
    )

    return {
        "abra_in_play": float(abra_count),
        "kadabra_in_play": 4.0 * kadabra_count,
        "alakazam_in_play_binary": 10.0 if play_counts[245] else 0.0,
        "psychic_energy_on_alakazam_line": float(line_energy_count),
        "kadabra_hand_abra_play_pairs": 2.0 * min(hand_counts[742], abra_count),
        "alakazam_candy_abra_sets": 3.0
        * min(hand_counts[245], hand_counts[1079], abra_count),
        "alakazam_hand_kadabra_play_pairs": 4.0
        * min(hand_counts[245], kadabra_count),
        "genesect_tool_before_opponent_ace_spec": 1.0 if genesect_ready else 0.0,
        "psyduck_into_dusknoir_line": 1.0 if psyduck_guard else 0.0,
        "dudunsparce_or_fezandipiti": min(
            2.0,
            0.5 * (play_counts[66] + play_counts[140]),
        ),
    }


def dragapult_reward_potential_components(board: dict[str, Any]) -> dict[str, float]:
    mine = in_play_card_states(board, prefix="my")
    play_counts = Counter(state_card_id(state) for state in mine)
    hand_counts = Counter(
        card_id
        for value in list(board.get("my_hand_card_ids", []) or [])
        if (card_id := optional_board_int(value)) is not None
    )

    line_energy_value = 0.0
    powered_dragapult = False
    for state in mine:
        if state_card_id(state) not in {119, 120, 121}:
            continue
        attached_energy_ids = energy_ids(state)
        has_fire = 2 in attached_energy_ids
        has_psychic = 5 in attached_energy_ids
        line_energy_value += (
            3.0
            if has_fire and has_psychic
            else 1.0
            if has_fire or has_psychic
            else 0.0
        )
        powered_dragapult = powered_dragapult or (
            state_card_id(state) == 121 and has_fire and has_psychic
        )

    munkidori_ready = any(
        state_card_id(state) == 112 and 7 in energy_ids(state) for state in mine
    )
    moltres_ready = any(
        state_card_id(state) == 791 and 2 in energy_ids(state) for state in mine
    )
    opponent_active = board.get("opponent_active_card")
    moltres_target = (
        isinstance(opponent_active, dict)
        and bool(opponent_active.get("is_ex", False))
        and bool(opponent_active.get("weak_to_fire", False))
        and int(opponent_active.get("hp", 0) or 0) <= 220
    )
    budew_active = optional_board_int(board.get("my_active_id")) == 235

    return {
        "dreepy_in_play": float(play_counts[119]),
        "drakloak_in_play": 4.0 * play_counts[120],
        "dragapult_in_play_binary": 10.0 if play_counts[121] else 0.0,
        "drakloak_hand_dreepy_play_pairs": 2.0
        * min(hand_counts[120], play_counts[119]),
        "dragapult_hand_drakloak_play_pairs": 4.0
        * min(hand_counts[121], play_counts[120]),
        "fire_psychic_on_dragapult_line": line_energy_value,
        "dusknoir_line_in_play": 0.5
        * (play_counts[131] + play_counts[132] + play_counts[133]),
        "munkidori_with_darkness": 0.5 if munkidori_ready else 0.0,
        "fezandipiti_in_play": 0.5 * play_counts[140],
        "budew_active_before_powered_dragapult": 2.0
        if budew_active and not powered_dragapult
        else 0.0,
        "moltres_fire_ko_window": 5.0 if moltres_ready and moltres_target else 0.0,
    }


def optional_board_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


# Compatibility aliases for the existing collector and its focused tests.
_deck_reward_potential_components = deck_reward_potential_components
_energy_ids = energy_ids
_optional_board_int = optional_board_int
