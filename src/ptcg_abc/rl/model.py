from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ptcg_abc.rl.records import ActionFrame, DecisionFrame, TrajectoryStep


MODEL_FORMAT = "ptcg_abc_linear_option_ranker_v1"


@dataclass
class LinearOptionModel:
    weights: dict[str, float] = field(default_factory=dict)
    bias: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def score_action(self, action: ActionFrame) -> float:
        score = self.bias
        for name, value in action.features.items():
            score += self.weights.get(name, 0.0) * float(value)
        return score

    def score_frame(self, frame: DecisionFrame) -> list[float]:
        return [self.score_action(action) for action in frame.legal_options]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": MODEL_FORMAT,
            "weights": dict(sorted(self.weights.items())),
            "bias": self.bias,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LinearOptionModel:
        if data.get("format") != MODEL_FORMAT:
            raise ValueError(f"Unsupported model format: {data.get('format')}")
        return cls(
            weights={str(key): float(value) for key, value in data.get("weights", {}).items()},
            bias=float(data.get("bias", 0.0)),
            metadata=dict(data.get("metadata", {})),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> LinearOptionModel:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class TrainingSummary:
    frames: int
    actions: int
    positives: int
    negatives: int
    epochs: int
    model_path: str
    accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def train_behavior_cloning_model(
    frames: Iterable[DecisionFrame],
    *,
    epochs: int = 1,
    learning_rate: float = 0.4,
    changed_weight: float = 1.0,
    unchanged_weight: float = 1.0,
    excluded_features: Sequence[str] = (),
    pairwise_changed: bool = False,
    pairwise_margin: float = 0.0,
) -> tuple[LinearOptionModel, TrainingSummary]:
    model = LinearOptionModel()
    frame_list = list(frames)
    excluded = set(excluded_features)
    positives = negatives = actions = 0
    pairwise_pairs = 0
    for _ in range(max(1, epochs)):
        for frame in frame_list:
            selected = set(frame.rule_selected_indices)
            frame_weight = _frame_training_weight(
                frame,
                changed_weight=changed_weight,
                unchanged_weight=unchanged_weight,
            )
            if pairwise_changed and _phase5_search_changed(frame):
                pairs = _search_baseline_pairs(frame)
                if pairs:
                    for search_index, baseline_index in pairs:
                        search_action = frame.legal_options[search_index]
                        baseline_action = frame.legal_options[baseline_index]
                        if not search_action.legal_mask or not baseline_action.legal_mask:
                            continue
                        actions += 2
                        positives += 1
                        negatives += 1
                        pairwise_pairs += 1
                        diff = model.score_action(search_action) - model.score_action(baseline_action)
                        error = _sigmoid(float(pairwise_margin) - diff) * frame_weight
                        _apply_pairwise_update(
                            model,
                            search_action,
                            baseline_action,
                            learning_rate=learning_rate,
                            error=error,
                            excluded_features=excluded,
                        )
                    continue
            for action in frame.legal_options:
                if not action.legal_mask:
                    continue
                actions += 1
                target = 1.0 if action.index in selected else 0.0
                positives += int(target == 1.0)
                negatives += int(target == 0.0)
                prediction = _sigmoid(model.score_action(action))
                error = (target - prediction) * frame_weight
                model.bias += learning_rate * error
                for name, value in action.features.items():
                    if name in excluded:
                        continue
                    model.weights[name] = model.weights.get(name, 0.0) + learning_rate * error * value
    model.metadata.update(
        {
            "training": "behavior_cloning",
            "frames": len(frame_list),
            "actions": actions,
            "positives": positives,
            "negatives": negatives,
            "epochs": max(1, epochs),
            "changed_weight": changed_weight,
            "unchanged_weight": unchanged_weight,
            "excluded_features": sorted(excluded),
            "pairwise_changed": pairwise_changed,
            "pairwise_margin": pairwise_margin,
            "pairwise_pairs": pairwise_pairs,
        }
    )
    summary = TrainingSummary(
        frames=len(frame_list),
        actions=actions,
        positives=positives,
        negatives=negatives,
        epochs=max(1, epochs),
        model_path="",
        accuracy=evaluate_topk_accuracy(model, frame_list),
    )
    return model, summary


def train_reward_weighted_model(
    steps: Iterable[TrajectoryStep],
    *,
    base_model: LinearOptionModel | None = None,
    epochs: int = 1,
    learning_rate: float = 0.2,
) -> tuple[LinearOptionModel, TrainingSummary]:
    model = base_model or LinearOptionModel()
    step_list = list(steps)
    frames = [step.decision for step in step_list]
    positives = negatives = actions = 0
    for _ in range(max(1, epochs)):
        for step in step_list:
            selected = set(step.chosen_indices or step.decision.rule_selected_indices)
            reward_weight = max(-1.0, min(1.0, step.reward))
            for action in step.decision.legal_options:
                if not action.legal_mask:
                    continue
                actions += 1
                chosen = action.index in selected
                target = 1.0 if chosen and reward_weight >= 0 else 0.0
                if chosen and reward_weight < 0:
                    target = 0.0
                elif not chosen and reward_weight < 0:
                    target = 1.0 / max(1, len(step.decision.legal_options) - len(selected))
                positives += int(target > 0.5)
                negatives += int(target <= 0.5)
                prediction = _sigmoid(model.score_action(action))
                error = (target - prediction) * max(0.1, abs(reward_weight))
                model.bias += learning_rate * error
                for name, value in action.features.items():
                    model.weights[name] = model.weights.get(name, 0.0) + learning_rate * error * value
    model.metadata.update(
        {
            "training": "reward_weighted_rollout_update",
            "frames": len(frames),
            "actions": actions,
            "positives": positives,
            "negatives": negatives,
            "epochs": max(1, epochs),
        }
    )
    summary = TrainingSummary(
        frames=len(frames),
        actions=actions,
        positives=positives,
        negatives=negatives,
        epochs=max(1, epochs),
        model_path="",
        accuracy=evaluate_topk_accuracy(model, frames),
    )
    return model, summary


def evaluate_topk_accuracy(model: LinearOptionModel, frames: Iterable[DecisionFrame]) -> float:
    total = correct = 0
    for frame in frames:
        selected = set(frame.rule_selected_indices)
        if not selected or not frame.legal_options:
            continue
        target_count = max(1, min(frame.target_count, len(frame.legal_options)))
        ranked = sorted(
            frame.legal_options,
            key=lambda action: (model.score_action(action), action.rule_score, -action.index),
            reverse=True,
        )
        predicted = {action.index for action in ranked[:target_count]}
        correct += int(bool(predicted & selected))
        total += 1
    return correct / total if total else 0.0


def _frame_training_weight(
    frame: DecisionFrame,
    *,
    changed_weight: float,
    unchanged_weight: float,
) -> float:
    metadata = frame.reward_metadata
    if bool(metadata.get("phase5_search_changed", False)):
        return max(0.0, float(changed_weight))
    return max(0.0, float(unchanged_weight))


def _phase5_search_changed(frame: DecisionFrame) -> bool:
    return bool(frame.reward_metadata.get("phase5_search_changed", False))


def _search_baseline_pairs(frame: DecisionFrame) -> list[tuple[int, int]]:
    size = len(frame.legal_options)
    metadata = frame.reward_metadata
    search = _valid_indices(metadata.get("phase5_search_indices", frame.rule_selected_indices), size)
    baseline = _valid_indices(metadata.get("phase5_baseline_indices", frame.rule_selected_indices), size)
    return [
        (search_index, baseline_index)
        for search_index in search
        for baseline_index in baseline
        if search_index != baseline_index
    ]


def _valid_indices(values: Any, size: int) -> list[int]:
    output: list[int] = []
    for value in values or []:
        index = int(value)
        if 0 <= index < size and index not in output:
            output.append(index)
    return output


def _apply_pairwise_update(
    model: LinearOptionModel,
    search_action: ActionFrame,
    baseline_action: ActionFrame,
    *,
    learning_rate: float,
    error: float,
    excluded_features: set[str],
) -> None:
    feature_names = (set(search_action.features) | set(baseline_action.features)) - excluded_features
    for name in feature_names:
        delta = float(search_action.features.get(name, 0.0)) - float(
            baseline_action.features.get(name, 0.0)
        )
        if delta:
            model.weights[name] = model.weights.get(name, 0.0) + learning_rate * error * delta


def _sigmoid(value: float) -> float:
    if value >= 50:
        return 1.0
    if value <= -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))
