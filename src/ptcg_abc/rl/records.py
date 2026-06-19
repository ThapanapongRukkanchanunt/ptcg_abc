from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ActionFrame:
    index: int
    option_type: str
    features: dict[str, float]
    rule_score: float = 0.0
    rule_rank: int = 0
    legal_mask: bool = True
    card_id: int | None = None
    card_name: str = ""
    area: str = ""
    area_index: int | None = None
    player_index: int | None = None
    attack_id: int | None = None
    target_card_id: int | None = None
    target_name: str = ""
    target_area: str = ""
    target_index: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionFrame:
        return cls(
            index=int(data["index"]),
            option_type=str(data["option_type"]),
            features={str(key): float(value) for key, value in data.get("features", {}).items()},
            rule_score=float(data.get("rule_score", 0.0)),
            rule_rank=int(data.get("rule_rank", 0)),
            legal_mask=bool(data.get("legal_mask", True)),
            card_id=_optional_int(data.get("card_id")),
            card_name=str(data.get("card_name", "")),
            area=str(data.get("area", "")),
            area_index=_optional_int(data.get("area_index")),
            player_index=_optional_int(data.get("player_index")),
            attack_id=_optional_int(data.get("attack_id")),
            target_card_id=_optional_int(data.get("target_card_id")),
            target_name=str(data.get("target_name", "")),
            target_area=str(data.get("target_area", "")),
            target_index=_optional_int(data.get("target_index")),
            raw=dict(data.get("raw", {})),
        )


@dataclass(frozen=True)
class DecisionFrame:
    select_type: str
    context: str
    min_count: int
    max_count: int
    target_count: int
    legal_options: list[ActionFrame]
    rule_selected_indices: list[int]
    board: dict[str, Any]
    board_image: list[list[float]]
    reward_metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "select_type": self.select_type,
            "context": self.context,
            "min_count": self.min_count,
            "max_count": self.max_count,
            "target_count": self.target_count,
            "legal_options": [action.to_dict() for action in self.legal_options],
            "rule_selected_indices": list(self.rule_selected_indices),
            "board": dict(self.board),
            "board_image": [list(row) for row in self.board_image],
            "reward_metadata": dict(self.reward_metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionFrame:
        return cls(
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            select_type=str(data["select_type"]),
            context=str(data["context"]),
            min_count=int(data.get("min_count", 0)),
            max_count=int(data.get("max_count", 0)),
            target_count=int(data.get("target_count", 0)),
            legal_options=[
                ActionFrame.from_dict(action) for action in data.get("legal_options", [])
            ],
            rule_selected_indices=[
                int(index) for index in data.get("rule_selected_indices", [])
            ],
            board=dict(data.get("board", {})),
            board_image=[
                [float(value) for value in row] for row in data.get("board_image", [])
            ],
            reward_metadata=dict(data.get("reward_metadata", {})),
        )


@dataclass(frozen=True)
class TrajectoryStep:
    decision: DecisionFrame
    chosen_indices: list[int]
    logprob: float = 0.0
    value: float = 0.0
    reward: float = 0.0
    terminal: bool = False
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.to_dict(),
            "chosen_indices": list(self.chosen_indices),
            "logprob": self.logprob,
            "value": self.value,
            "reward": self.reward,
            "terminal": self.terminal,
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrajectoryStep:
        return cls(
            decision=DecisionFrame.from_dict(data["decision"]),
            chosen_indices=[int(index) for index in data.get("chosen_indices", [])],
            logprob=float(data.get("logprob", 0.0)),
            value=float(data.get("value", 0.0)),
            reward=float(data.get("reward", 0.0)),
            terminal=bool(data.get("terminal", False)),
            truncated=bool(data.get("truncated", False)),
        )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
