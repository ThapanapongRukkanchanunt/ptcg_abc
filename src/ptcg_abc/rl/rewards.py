from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RewardConfig:
    win: float = 1.0
    loss: float = -1.0
    draw: float = 0.0
    prize_gained: float = 0.2
    opponent_prize_gained: float = -0.15
    step_cost: float = -0.001
    illegal_recovery: float = -0.05

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def terminal_reward(
    *,
    winner: int | None,
    acting_player: int,
    leader: int | None = None,
    finished: bool = True,
    config: RewardConfig = RewardConfig(),
) -> float:
    effective = winner if winner is not None else leader
    if effective is None:
        return config.draw if finished else config.step_cost
    return config.win if effective == acting_player else config.loss


def shaped_step_reward(
    *,
    previous_prizes: tuple[int, int] | None,
    current_prizes: tuple[int, int] | None,
    acting_player: int,
    illegal_recovery: bool = False,
    config: RewardConfig = RewardConfig(),
) -> float:
    reward = config.step_cost
    if previous_prizes is not None and current_prizes is not None:
        mine_before = previous_prizes[acting_player]
        mine_after = current_prizes[acting_player]
        op_before = previous_prizes[1 - acting_player]
        op_after = current_prizes[1 - acting_player]
        reward += max(0, mine_before - mine_after) * config.prize_gained
        reward += max(0, op_before - op_after) * config.opponent_prize_gained
    if illegal_recovery:
        reward += config.illegal_recovery
    return reward


def reward_from_result_metadata(metadata: dict[str, Any], config: RewardConfig = RewardConfig()) -> float:
    player = int(metadata.get("player_index", 0) or 0)
    reward = terminal_reward(
        winner=_optional_int(metadata.get("winner")),
        acting_player=player,
        leader=_optional_int(metadata.get("leader")),
        finished=bool(metadata.get("finished", True)),
        config=config,
    )
    reward += shaped_step_reward(
        previous_prizes=_optional_prizes(metadata.get("previous_prizes")),
        current_prizes=_optional_prizes(metadata.get("prize_counts")),
        acting_player=player,
        illegal_recovery=bool(metadata.get("illegal_recovery", False)),
        config=config,
    )
    return reward


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_prizes(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    try:
        first, second = value
        return (int(first), int(second))
    except (TypeError, ValueError):
        return None
