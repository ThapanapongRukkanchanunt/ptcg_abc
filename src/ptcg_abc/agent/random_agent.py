from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


@dataclass
class RandomAgent:
    deck_ids: Sequence[int]
    seed: int | None = None
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        if len(self.deck_ids) != 60:
            raise ValueError(f"RandomAgent needs a 60-card deck, got {len(self.deck_ids)}.")
        self.rng = random.Random(self.seed)

    def act(self, observation: Any) -> list[int]:
        select = _get(observation, "select")
        if select is None:
            return list(self.deck_ids)
        options = list(_get(select, "option", []) or [])
        min_count = int(_get(select, "minCount", 0) or 0)
        max_count = int(_get(select, "maxCount", min_count) or 0)
        max_count = min(max_count, len(options))
        if max_count <= 0:
            return []
        count = max_count
        if min_count < max_count:
            count = self.rng.randint(min_count, max_count)
        return self.rng.sample(range(len(options)), count)
