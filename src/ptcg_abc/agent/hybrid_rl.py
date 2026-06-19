from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ptcg_abc.agent.rule_based import RuleBasedAgent
from ptcg_abc.rl.featurizer import attack_lookup, card_lookup, make_decision_frame
from ptcg_abc.rl.guidance import default_guidance_rules, evaluate_guidance
from ptcg_abc.rl.model import LinearOptionModel


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


@dataclass
class HybridRlAgent:
    deck_ids: Sequence[int]
    card_data: Sequence[Any] = ()
    attack_data: Sequence[Any] = ()
    model: LinearOptionModel | None = None
    model_path: Path | None = None
    model_weight: float = 1.0
    rule_weight: float = 0.35
    guidance_rules: Sequence[str] = field(default_factory=default_guidance_rules)
    fallback_agent: RuleBasedAgent = field(init=False)
    card_by_id: dict[int, Any] = field(init=False, default_factory=dict)
    attack_by_id: dict[int, Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.deck_ids) != 60:
            raise ValueError(f"HybridRlAgent needs a 60-card deck, got {len(self.deck_ids)}.")
        self.card_by_id = card_lookup(self.card_data)
        self.attack_by_id = attack_lookup(self.attack_data)
        if self.model_path is not None:
            self.model_path = Path(self.model_path)
        if self.model is None and self.model_path is not None and self.model_path.exists():
            self.model = LinearOptionModel.load(self.model_path)
        with contextlib.redirect_stdout(io.StringIO()):
            self.fallback_agent = RuleBasedAgent(
                self.deck_ids,
                card_data=self.card_data,
                attack_data=self.attack_data,
            )

    def act(self, observation: Any) -> list[int]:
        select = _get(observation, "select")
        if select is None:
            return list(self.deck_ids)
        if self.model is None:
            return self.fallback_agent.act(observation)

        frame = make_decision_frame(
            observation,
            deck_ids=self.deck_ids,
            card_by_id=self.card_by_id,
            attack_by_id=self.attack_by_id,
        )
        if frame is None or not frame.legal_options:
            return self.fallback_agent.act(observation)

        target_count = max(0, min(frame.target_count, len(frame.legal_options)))
        if target_count <= 0:
            return []

        guidance = evaluate_guidance(frame, enabled_rules=self.guidance_rules)
        blocked = set(guidance.blocked_indices)
        selected: list[int] = [
            index
            for index in sorted(guidance.forced_indices)
            if index not in blocked and 0 <= index < len(frame.legal_options)
        ][:target_count]

        model_scores = self.model.score_frame(frame)
        rule_scores = _normalize([action.rule_score for action in frame.legal_options])
        combined = [
            self.model_weight * model_scores[index] + self.rule_weight * rule_scores[index]
            for index in range(len(frame.legal_options))
        ]
        ranked = sorted(
            range(len(frame.legal_options)),
            key=lambda index: (combined[index], frame.legal_options[index].rule_score, -index),
            reverse=True,
        )
        for index in ranked:
            if len(selected) >= target_count:
                break
            if index in blocked or index in selected:
                continue
            selected.append(index)

        if len(selected) < frame.min_count:
            fallback = self.fallback_agent.act(observation)
            return _valid_indices(fallback, len(frame.legal_options))
        if not selected and frame.rule_selected_indices:
            return _valid_indices(frame.rule_selected_indices, len(frame.legal_options))
        return _valid_indices(selected, len(frame.legal_options))


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _valid_indices(values: Sequence[int], size: int) -> list[int]:
    output = []
    for value in values:
        index = int(value)
        if 0 <= index < size and index not in output:
            output.append(index)
    return output
