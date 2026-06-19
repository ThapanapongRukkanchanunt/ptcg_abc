from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from ptcg_abc.rl.model import LinearOptionModel, TrainingSummary, evaluate_topk_accuracy
from ptcg_abc.rl.records import DecisionFrame
from ptcg_abc.rl.rewards import reward_from_result_metadata


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
CHECKPOINT_FORMAT = "ptcg_abc_torch_actor_value_linear_v1"


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
) -> TorchTrainingSummary:
    torch, nn = _require_torch()
    frame_list = list(frames)
    feature_names = collect_feature_names(frame_list)
    board_size = _board_size(frame_list)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = _make_network(nn, len(feature_names), board_size).to(device)
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
                [flatten_board_image(frame)],
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
            positives += int(labels.sum().item())
            negatives += int(labels.numel() - labels.sum().item())
            actions += int(labels.numel())
            logits, value = network(action_x, board_x)
            actor_loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            value_loss = nn.functional.mse_loss(value.reshape(1), reward)
            loss = actor_loss + 0.2 * value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().item())

    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "feature_names": feature_names,
        "board_size": board_size,
        "state_dict": {key: value.detach().cpu() for key, value in network.state_dict().items()},
        "metadata": {
            "frames": len(frame_list),
            "actions": actions,
            "positives": positives,
            "negatives": negatives,
            "epochs": max(1, epochs),
            "final_loss": final_loss,
            "device": str(device),
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
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported torch checkpoint format: {checkpoint.get('format')}")
    state = checkpoint["state_dict"]
    feature_names = [str(name) for name in checkpoint["feature_names"]]
    actor_weight = state["actor.weight"].detach().reshape(-1).tolist()
    actor_bias = float(state["actor.bias"].detach().reshape(-1)[0].item())
    model = linear_model_from_actor_params(
        feature_names,
        actor_weight,
        actor_bias,
        metadata={
            "source_format": CHECKPOINT_FORMAT,
            "checkpoint_path": str(checkpoint_path.as_posix()),
            "training": checkpoint.get("metadata", {}),
        },
    )
    model.save(export_model_path)
    return model


def score_frame_with_torch_checkpoint(checkpoint_path: Path, frame: DecisionFrame) -> list[float]:
    torch, nn = _require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported torch checkpoint format: {checkpoint.get('format')}")
    feature_names = [str(name) for name in checkpoint["feature_names"]]
    network = _make_network(nn, len(feature_names), int(checkpoint["board_size"]))
    network.load_state_dict(checkpoint["state_dict"])
    network.eval()
    with torch.no_grad():
        logits, _ = network(
            torch.tensor(action_matrix(frame, feature_names), dtype=torch.float32),
            torch.tensor([flatten_board_image(frame)], dtype=torch.float32),
        )
    return [float(value) for value in logits.tolist()]


def collect_feature_names(frames: Iterable[DecisionFrame]) -> list[str]:
    names = set()
    for frame in frames:
        for action in frame.legal_options:
            names.update(action.features)
    return sorted(names)


def action_matrix(frame: DecisionFrame, feature_names: list[str]) -> list[list[float]]:
    return [
        [float(action.features.get(name, 0.0)) for name in feature_names]
        for action in frame.legal_options
    ]


def flatten_board_image(frame: DecisionFrame) -> list[float]:
    return [float(value) for row in frame.board_image for value in row]


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


def _board_size(frames: list[DecisionFrame]) -> int:
    for frame in frames:
        size = len(flatten_board_image(frame))
        if size:
            return size
    return 16 * 16


def _make_network(nn: Any, feature_count: int, board_size: int) -> Any:
    class ActorValueNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.actor = nn.Linear(feature_count, 1)
            self.value = nn.Linear(board_size, 1)

        def forward(self, action_x: Any, board_x: Any) -> tuple[Any, Any]:
            logits = self.actor(action_x).squeeze(-1)
            value = self.value(board_x).squeeze(-1)
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
