from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ptcg_abc.rl.model import (
    LinearOptionModel,
    TrainingSummary,
    evaluate_topk_accuracy,
    train_behavior_cloning_model,
)
from ptcg_abc.rl.records import DecisionFrame
from ptcg_abc.rl.rewards import reward_from_result_metadata


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
LINEAR_CHECKPOINT_FORMAT = "ptcg_abc_torch_actor_value_linear_v1"
CHECKPOINT_FORMAT = "ptcg_abc_torch_actor_value_cnn_v1"


class TorchBackendUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class TorchTrainingSummary:
    frames: int
    actions: int
    positives: int
    negatives: int
    epochs: int
    checkpoint_path: str
    export_model_path: str
    accuracy: float
    final_loss: float
    device: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def train_torch_bc_model(
    frames: Iterable[DecisionFrame],
    *,
    checkpoint_path: Path,
    export_model_path: Path,
    epochs: int = 1,
    learning_rate: float = 0.02,
    changed_weight: float = 1.0,
    unchanged_weight: float = 1.0,
    excluded_features: Sequence[str] = (),
    pairwise_changed: bool = False,
    pairwise_margin: float = 0.0,
    pairwise_negatives: str = "baseline",
) -> TorchTrainingSummary:
    torch, nn = _require_torch()
    frame_list = list(frames)
    excluded = set(excluded_features)
    _validate_pairwise_negatives(pairwise_negatives)
    feature_names = collect_feature_names(frame_list, excluded_features=excluded)
    board_shape = _board_shape(frame_list)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = _make_network(nn, len(feature_names), board_shape).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    positives = negatives = actions = 0
    final_loss = 0.0

    for _ in range(max(1, epochs)):
        for frame in frame_list:
            if not frame.legal_options:
                continue
            action_x = torch.tensor(
                action_matrix(frame, feature_names),
                dtype=torch.float32,
                device=device,
            )
            board_x = torch.tensor(
                [board_image_matrix(frame)],
                dtype=torch.float32,
                device=device,
            )
            labels = torch.tensor(
                [
                    1.0 if action.index in set(frame.rule_selected_indices) else 0.0
                    for action in frame.legal_options
                ],
                dtype=torch.float32,
                device=device,
            )
            reward = torch.tensor(
                [reward_from_result_metadata(frame.reward_metadata)],
                dtype=torch.float32,
                device=device,
            )
            frame_weight = _frame_training_weight(
                frame,
                changed_weight=changed_weight,
                unchanged_weight=unchanged_weight,
            )
            logits, value = network(action_x, board_x)
            pairs = (
                _search_pairwise_pairs(frame, negative_mode=pairwise_negatives)
                if pairwise_changed
                else []
            )
            if pairs:
                pair_losses = [
                    nn.functional.softplus(
                        torch.tensor(float(pairwise_margin), dtype=torch.float32, device=device)
                        - (logits[search_index] - logits[baseline_index])
                    )
                    for search_index, baseline_index in pairs
                ]
                positives += len(pairs)
                negatives += len(pairs)
                actions += 2 * len(pairs)
                actor_loss = torch.stack(pair_losses).mean() * frame_weight
            else:
                positives += int(labels.sum().item())
                negatives += int(labels.numel() - labels.sum().item())
                actions += int(labels.numel())
                actor_loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
                actor_loss = actor_loss * frame_weight
            value_loss = nn.functional.mse_loss(value.reshape(1), reward)
            loss = actor_loss + 0.2 * value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().item())

    export_model, export_summary = train_behavior_cloning_model(
        frame_list,
        epochs=max(1, epochs),
        changed_weight=changed_weight,
        unchanged_weight=unchanged_weight,
        excluded_features=excluded,
        pairwise_changed=pairwise_changed,
        pairwise_margin=pairwise_margin,
        pairwise_negatives=pairwise_negatives,
    )
    export_model.metadata.update(
        {
            "source_format": CHECKPOINT_FORMAT,
            "export_role": "linear_kaggle_fallback",
            "board_shape": list(board_shape),
            "torch_device": str(device),
            "torch_final_loss": final_loss,
            "torch_actions": actions,
            "changed_weight": changed_weight,
            "unchanged_weight": unchanged_weight,
            "excluded_features": sorted(excluded),
            "pairwise_changed": pairwise_changed,
            "pairwise_margin": pairwise_margin,
            "pairwise_negatives": pairwise_negatives,
        }
    )
    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "feature_names": feature_names,
        "board_shape": list(board_shape),
        "board_size": board_shape[0] * board_shape[1],
        "state_dict": {key: value.detach().cpu() for key, value in network.state_dict().items()},
        "linear_export": export_model.to_dict(),
        "metadata": {
            "frames": len(frame_list),
            "actions": actions,
            "positives": positives,
            "negatives": negatives,
            "epochs": max(1, epochs),
            "final_loss": final_loss,
            "device": str(device),
            "linear_export_accuracy": export_summary.accuracy,
            "changed_weight": changed_weight,
            "unchanged_weight": unchanged_weight,
            "excluded_features": sorted(excluded),
            "pairwise_changed": pairwise_changed,
            "pairwise_margin": pairwise_margin,
            "pairwise_negatives": pairwise_negatives,
        },
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    exported = export_torch_checkpoint(checkpoint_path, export_model_path)
    return TorchTrainingSummary(
        frames=len(frame_list),
        actions=actions,
        positives=positives,
        negatives=negatives,
        epochs=max(1, epochs),
        checkpoint_path=str(checkpoint_path.as_posix()),
        export_model_path=str(export_model_path.as_posix()),
        accuracy=evaluate_topk_accuracy(exported, frame_list),
        final_loss=final_loss,
        device=str(device),
    )


def export_torch_checkpoint(checkpoint_path: Path, export_model_path: Path) -> LinearOptionModel:
    torch, _ = _require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_format = checkpoint.get("format")
    if checkpoint_format == LINEAR_CHECKPOINT_FORMAT:
        state = checkpoint["state_dict"]
        feature_names = [str(name) for name in checkpoint["feature_names"]]
        actor_weight = state["actor.weight"].detach().reshape(-1).tolist()
        actor_bias = float(state["actor.bias"].detach().reshape(-1)[0].item())
        model = linear_model_from_actor_params(
            feature_names,
            actor_weight,
            actor_bias,
            metadata={
                "source_format": LINEAR_CHECKPOINT_FORMAT,
                "checkpoint_path": str(checkpoint_path.as_posix()),
                "training": checkpoint.get("metadata", {}),
            },
        )
        model.save(export_model_path)
        return model
    if checkpoint_format != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported torch checkpoint format: {checkpoint.get('format')}")
    model = LinearOptionModel.from_dict(checkpoint["linear_export"])
    model.metadata.update(
        {
            "source_format": CHECKPOINT_FORMAT,
            "checkpoint_path": str(checkpoint_path.as_posix()),
            "training": checkpoint.get("metadata", {}),
        }
    )
    model.save(export_model_path)
    return model


def score_frame_with_torch_checkpoint(checkpoint_path: Path, frame: DecisionFrame) -> list[float]:
    torch, nn = _require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported torch checkpoint format: {checkpoint.get('format')}")
    feature_names = [str(name) for name in checkpoint["feature_names"]]
    board_shape = _checkpoint_board_shape(checkpoint)
    network = _make_network(nn, len(feature_names), board_shape)
    network.load_state_dict(checkpoint["state_dict"])
    network.eval()
    with torch.no_grad():
        logits, _ = network(
            torch.tensor(action_matrix(frame, feature_names), dtype=torch.float32),
            torch.tensor([board_image_matrix(frame)], dtype=torch.float32),
        )
    return [float(value) for value in logits.tolist()]


def collect_feature_names(
    frames: Iterable[DecisionFrame],
    *,
    excluded_features: Iterable[str] = (),
) -> list[str]:
    excluded = set(excluded_features)
    names = set()
    for frame in frames:
        for action in frame.legal_options:
            names.update(name for name in action.features if name not in excluded)
    return sorted(names)


def action_matrix(frame: DecisionFrame, feature_names: list[str]) -> list[list[float]]:
    return [
        [float(action.features.get(name, 0.0)) for name in feature_names]
        for action in frame.legal_options
    ]


def flatten_board_image(frame: DecisionFrame) -> list[float]:
    return [float(value) for row in frame.board_image for value in row]


def board_image_matrix(frame: DecisionFrame) -> list[list[float]]:
    return [[float(value) for value in row] for row in frame.board_image]


def linear_model_from_actor_params(
    feature_names: list[str],
    weights: list[float],
    bias: float,
    *,
    metadata: dict[str, Any] | None = None,
) -> LinearOptionModel:
    if len(feature_names) != len(weights):
        raise ValueError("Feature-name and weight lengths must match.")
    return LinearOptionModel(
        weights={name: float(weight) for name, weight in zip(feature_names, weights, strict=True)},
        bias=float(bias),
        metadata=dict(metadata or {}),
    )


def write_torch_report(summary: TorchTrainingSummary | TrainingSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = summary.to_dict()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _board_shape(frames: list[DecisionFrame]) -> tuple[int, int]:
    for frame in frames:
        height = len(frame.board_image)
        width = len(frame.board_image[0]) if height and frame.board_image[0] else 0
        if height and width:
            return height, width
    return 64, 64


def _checkpoint_board_shape(checkpoint: dict[str, Any]) -> tuple[int, int]:
    raw_shape = checkpoint.get("board_shape")
    if isinstance(raw_shape, (list, tuple)) and len(raw_shape) == 2:
        return int(raw_shape[0]), int(raw_shape[1])
    board_size = int(checkpoint.get("board_size", 64 * 64) or (64 * 64))
    side = int(board_size**0.5)
    if side * side == board_size:
        return side, side
    return 1, board_size


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


def _search_pairwise_pairs(
    frame: DecisionFrame,
    *,
    negative_mode: str = "baseline",
) -> list[tuple[int, int]]:
    _validate_pairwise_negatives(negative_mode)
    if not bool(frame.reward_metadata.get("phase5_search_changed", False)):
        return []
    size = len(frame.legal_options)
    metadata = frame.reward_metadata
    search = _valid_indices(metadata.get("phase5_search_indices", frame.rule_selected_indices), size)
    if negative_mode == "baseline":
        negatives = _valid_indices(
            metadata.get("phase5_baseline_indices", frame.rule_selected_indices), size
        )
    else:
        negatives = list(range(size))
    return [
        (search_index, negative_index)
        for search_index in search
        for negative_index in negatives
        if search_index != negative_index
    ]


def _validate_pairwise_negatives(mode: str) -> None:
    if mode not in {"baseline", "all"}:
        raise ValueError(f"Unsupported pairwise negative mode: {mode}")


def _valid_indices(values: Any, size: int) -> list[int]:
    output: list[int] = []
    for value in values or []:
        index = int(value)
        if 0 <= index < size and index not in output:
            output.append(index)
    return output


def _make_network(nn: Any, feature_count: int, board_shape: tuple[int, int]) -> Any:
    torch = __import__("torch")

    class ActorValueNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.board_shape = board_shape
            self.board_encoder = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(16, 32),
                nn.ReLU(),
            )
            self.option_encoder = nn.Sequential(
                nn.Linear(feature_count, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
            )
            self.actor = nn.Linear(64, 1)
            self.value = nn.Linear(32, 1)

        def forward(self, action_x: Any, board_x: Any) -> tuple[Any, Any]:
            if board_x.dim() == 2:
                board_x = board_x.reshape(1, 1, self.board_shape[0], self.board_shape[1])
            elif board_x.dim() == 3:
                board_x = board_x.unsqueeze(1)
            board_embedding = self.board_encoder(board_x)
            option_embedding = self.option_encoder(action_x)
            if board_embedding.shape[0] == 1 and option_embedding.shape[0] != 1:
                option_board = board_embedding.expand(option_embedding.shape[0], -1)
            elif board_embedding.shape[0] == option_embedding.shape[0]:
                option_board = board_embedding
            else:
                option_board = board_embedding[:1].expand(option_embedding.shape[0], -1)
            logits = self.actor(torch.cat([option_embedding, option_board], dim=1)).squeeze(-1)
            value = self.value(board_embedding).squeeze(-1)
            return logits, value

    return ActorValueNetwork()


def _require_torch() -> tuple[Any, Any]:
    if not TORCH_AVAILABLE:
        raise TorchBackendUnavailable(
            "PyTorch is not installed. Install the `rl` extra or run this command "
            "on the ERAWAN training environment."
        )
    import torch
    from torch import nn

    return torch, nn
