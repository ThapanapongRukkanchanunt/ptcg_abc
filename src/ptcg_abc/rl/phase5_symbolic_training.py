from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Sequence

from ptcg_abc.rl.phase5_adapters import CardEntity, GameState, LegalAction, PlayerState
from ptcg_abc.rl.phase5_encoder import EncodedPhase5Turn, Phase5SymbolicEncoder
from ptcg_abc.rl.phase5_policy import (
    AlphaStarTurnPolicy,
    PHASE5_POLICY_CHECKPOINT_FORMAT,
    Phase5PolicyConfig,
    Phase5PolicyUnavailable,
    TORCH_AVAILABLE,
    make_phase5_policy_config,
)
from ptcg_abc.rl.records import ActionFrame, DecisionFrame, TrajectoryStep
from ptcg_abc.rl.rewards import reward_from_result_metadata


PHASE5_SYMBOLIC_DATASET_FORMAT = "ptcg_abc_phase5_symbolic_decision_v1"
PHASE5_SYMBOLIC_PAIRWISE_NEGATIVES = ("all", "baseline")
PHASE5_DIFFERENTIABLE_POLICY_MODES = ("epsilon_mixture", "sample")
PHASE5_BC_PPO_UPDATE_SCHEDULES = ("balanced-max", "ppo-epoch")
PHASE5_ADVANTAGE_NORMALIZATION_MODES = ("batch", "global", "none")
PHASE5_VALUE_BACKPROP_SCOPES = ("shared", "head-only")


@dataclass(frozen=True)
class Phase5SymbolicDecisionRecord:
    encoded: EncodedPhase5Turn
    target_indices: list[int]
    target_positions: list[int]
    previous_action_features: list[list[float]]
    previous_action_mask: list[float]
    weight: float
    changed: bool
    metadata: dict[str, Any]
    value_target: float | None = None
    action_value_targets: list[float] | None = None
    action_value_mask: list[float] | None = None
    tactical_targets: list[float] | None = None
    tactical_mask: list[float] | None = None
    source_format: str = PHASE5_SYMBOLIC_DATASET_FORMAT

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "source_format": self.source_format,
            "encoded": self.encoded.to_dict(),
            "target_indices": list(self.target_indices),
            "target_positions": list(self.target_positions),
            "previous_action_features": [
                list(row) for row in self.previous_action_features
            ],
            "previous_action_mask": list(self.previous_action_mask),
            "weight": self.weight,
            "changed": self.changed,
            "metadata": dict(self.metadata),
        }
        if self.value_target is not None:
            payload["value_target"] = float(self.value_target)
        if self.action_value_targets is not None:
            payload["action_value_targets"] = list(self.action_value_targets)
        if self.action_value_mask is not None:
            payload["action_value_mask"] = list(self.action_value_mask)
        if self.tactical_targets is not None:
            payload["tactical_targets"] = list(self.tactical_targets)
        if self.tactical_mask is not None:
            payload["tactical_mask"] = list(self.tactical_mask)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Phase5SymbolicDecisionRecord:
        if data.get("source_format") != PHASE5_SYMBOLIC_DATASET_FORMAT:
            raise ValueError(f"Unsupported symbolic record format: {data.get('source_format')}")
        encoded_payload = data["encoded"]
        return cls(
            encoded=EncodedPhase5Turn(
                global_features=[float(value) for value in encoded_payload["global_features"]],
                entity_features=[
                    [float(value) for value in row]
                    for row in encoded_payload["entity_features"]
                ],
                legal_action_features=[
                    [float(value) for value in row]
                    for row in encoded_payload["legal_action_features"]
                ],
                entity_mask=[float(value) for value in encoded_payload["entity_mask"]],
                legal_action_mask=[
                    float(value) for value in encoded_payload["legal_action_mask"]
                ],
                legal_action_indices=[
                    int(value) for value in encoded_payload["legal_action_indices"]
                ],
            ),
            target_indices=[int(value) for value in data.get("target_indices", [])],
            target_positions=[int(value) for value in data.get("target_positions", [])],
            previous_action_features=[
                [float(value) for value in row]
                for row in data.get("previous_action_features", [])
            ],
            previous_action_mask=[
                float(value) for value in data.get("previous_action_mask", [])
            ],
            weight=float(data.get("weight", 1.0)),
            changed=bool(data.get("changed", False)),
            metadata=dict(data.get("metadata", {})),
            value_target=(
                None if data.get("value_target") is None else float(data["value_target"])
            ),
            action_value_targets=_optional_float_list(data.get("action_value_targets")),
            action_value_mask=_optional_float_list(data.get("action_value_mask")),
            tactical_targets=_optional_float_list(data.get("tactical_targets")),
            tactical_mask=_optional_float_list(data.get("tactical_mask")),
        )


@dataclass(frozen=True)
class Phase5SymbolicBuildSummary:
    input_path: str
    output_path: str
    frames_seen: int
    records_written: int
    skipped_no_target: int
    changed_records: int
    target_source: str
    max_entities: int
    max_actions: int
    max_previous_actions: int
    limit: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase5SymbolicTrainingSummary:
    frames_seen: int
    examples: int
    skipped_no_target: int
    changed_examples: int
    actions: int
    epochs: int
    checkpoint_path: str
    accuracy: float
    final_loss: float
    device: str
    target_source: str
    changed_weight: float
    unchanged_weight: float
    pairwise_changed: bool
    pairwise_weight: float
    pairwise_margin: float
    pairwise_negatives: str
    max_entities: int
    max_actions: int
    max_previous_actions: int
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase5GeneralistTrainingSummary:
    decision_frames_seen: int
    decision_examples: int
    rule_examples: int
    selfplay_steps_seen: int
    selfplay_examples: int
    skipped_no_target: int
    changed_examples: int
    actions: int
    value_examples: int
    action_value_examples: int
    tactical_examples: int
    epochs: int
    checkpoint_path: str
    accuracy: float
    final_loss: float
    device: str
    decision_dataset_path: str | None
    selfplay_dataset_paths: list[str]
    deck_index_filter: int | None
    search_decision_weight: float
    rule_demo_weight: float
    selfplay_weight: float
    value_loss_weight: float
    action_value_loss_weight: float
    tactical_loss_weight: float
    max_entities: int
    max_actions: int
    max_previous_actions: int
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase5PPOTrainingSummary:
    trajectory_dataset_paths: list[str]
    steps_seen: int
    examples: int
    skipped_no_target: int
    skipped_off_policy: int
    epochs: int
    checkpoint_path: str
    output_checkpoint_path: str
    final_loss: float
    mean_advantage: float
    device: str
    deck_index_filter: int | None
    require_on_policy: bool
    clip_epsilon: float
    policy_loss_weight: float
    value_loss_weight: float
    entropy_weight: float
    max_entities: int
    max_actions: int
    max_previous_actions: int
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase5BCPPOTrainingSummary:
    rule_trajectory_dataset_paths: list[str]
    on_policy_trajectory_dataset_paths: list[str]
    rule_steps_seen: int
    rule_examples_available: int
    rule_skipped_no_target: int
    on_policy_steps_seen: int
    on_policy_examples_available: int
    on_policy_skipped_no_target: int
    on_policy_skipped_off_policy: int
    on_policy_skipped_policy_mode: int
    on_policy_skipped_nonfinite: int
    accepted_policy_modes: list[str]
    update_schedule: str
    rule_anchor_fraction: float
    advantage_normalization: str
    advantage_normalization_mean: float
    advantage_normalization_std: float
    value_backprop_scope: str
    balanced_examples_per_source_per_epoch: int
    rule_anchor_examples_per_epoch: int
    on_policy_examples_per_epoch: int
    rule_examples_used: int
    on_policy_examples_used: int
    rule_reuse_factor: float
    on_policy_reuse_factor: float
    optimizer_steps: int
    epochs: int
    checkpoint_path: str
    output_checkpoint_path: str
    final_loss: float
    average_bc_loss: float
    average_bc_accuracy: float
    average_ppo_policy_loss: float
    average_ppo_value_loss: float
    average_entropy: float
    average_ratio: float
    average_clip_fraction: float
    gradient_diagnostic_batches_requested: int
    gradient_diagnostic_batches_recorded: int
    average_bc_gradient_norm: float
    average_ppo_policy_gradient_norm: float
    average_ppo_policy_shared_gradient_norm: float
    average_ppo_value_gradient_norm: float
    average_ppo_value_shared_gradient_norm: float
    average_entropy_gradient_norm: float
    average_bc_ppo_policy_gradient_cosine: float
    average_ppo_policy_value_gradient_cosine: float
    average_ppo_policy_value_shared_gradient_cosine: float
    mean_advantage: float
    device: str
    deck_index_filter: int | None
    bc_loss_weight: float
    ppo_policy_loss_weight: float
    ppo_value_loss_weight: float
    entropy_weight: float
    clip_epsilon: float
    max_entities: int
    max_actions: int
    max_previous_actions: int
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _TrajectorySourceStats:
    steps_seen: int = 0
    examples: int = 0
    skipped_no_target: int = 0
    skipped_off_policy: int = 0
    skipped_policy_mode: int = 0
    skipped_nonfinite: int = 0
    advantage_sum: float = 0.0
    advantage_square_sum: float = 0.0


@dataclass(frozen=True)
class _BCPPOBatchMetrics:
    loss: float
    bc_loss: float
    bc_accuracy: float
    ppo_policy_loss: float
    ppo_value_loss: float
    entropy: float
    ratio: float
    clip_fraction: float
    bc_gradient_norm: float = 0.0
    ppo_policy_gradient_norm: float = 0.0
    ppo_policy_shared_gradient_norm: float = 0.0
    ppo_value_gradient_norm: float = 0.0
    ppo_value_shared_gradient_norm: float = 0.0
    entropy_gradient_norm: float = 0.0
    bc_ppo_policy_gradient_cosine: float = 0.0
    ppo_policy_value_gradient_cosine: float = 0.0
    ppo_policy_value_shared_gradient_cosine: float = 0.0


@dataclass(frozen=True)
class Phase5TrajectoryBCTrainingSummary:
    rule_trajectory_dataset_paths: list[str]
    steps_seen: int
    examples_available: int
    skipped_no_target: int
    examples_used: int
    optimizer_steps: int
    epochs: int
    checkpoint_path: str
    output_checkpoint_path: str
    final_loss: float
    average_bc_loss: float
    average_bc_accuracy: float
    device: str
    deck_index_filter: int | None
    max_entities: int
    max_actions: int
    max_previous_actions: int
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase5PolicyInitSummary:
    checkpoint_path: str
    max_entities: int
    max_actions: int
    max_previous_actions: int
    config: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Phase5TurnContext:
    def __init__(self, *, max_previous_actions: int) -> None:
        self.max_previous_actions = max_previous_actions
        self._key: tuple[Any, ...] | None = None
        self._history: list[list[float]] = []

    def previous_rows(self, frame: DecisionFrame) -> list[list[float]]:
        key = _turn_context_key(frame)
        if key != self._key:
            self._key = key
            self._history = []
        return [list(row) for row in self._history[-self.max_previous_actions :]]

    def observe(
        self,
        record: Phase5SymbolicDecisionRecord,
    ) -> None:
        self.observe_positions(record, record.target_positions)

    def observe_indices(
        self,
        record: Phase5SymbolicDecisionRecord,
        action_indices: Sequence[int],
    ) -> None:
        position_by_index = {
            action_index: position
            for position, action_index in enumerate(record.encoded.legal_action_indices)
            if action_index >= 0
        }
        positions: list[int] = []
        for action_index in action_indices:
            try:
                index = int(action_index)
            except (TypeError, ValueError):
                continue
            if index in position_by_index:
                positions.append(position_by_index[index])
        self.observe_positions(record, positions)

    def observe_positions(
        self,
        record: Phase5SymbolicDecisionRecord,
        positions: Sequence[int],
    ) -> None:
        action_rows = record.encoded.legal_action_features
        for position in positions:
            if 0 <= position < len(action_rows):
                self._history.append(list(action_rows[position]))
        if len(self._history) > self.max_previous_actions:
            self._history = self._history[-self.max_previous_actions :]


def iter_decision_jsonl(path: Path) -> Iterable[DecisionFrame]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield DecisionFrame.from_dict(json.loads(stripped))


def iter_trajectory_jsonl(path: Path) -> Iterable[TrajectoryStep]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield TrajectoryStep.from_dict(json.loads(stripped))


def _matches_deck_index_filter(
    metadata: dict[str, Any],
    deck_index_filter: int | None,
) -> bool:
    if deck_index_filter is None:
        return True
    try:
        return int(metadata.get("deck_index", -1)) == int(deck_index_filter)
    except (TypeError, ValueError):
        return False


def decision_frame_to_state(frame: DecisionFrame) -> GameState:
    board = frame.board
    your_index = _int(board.get("your_index"), 0)
    opponent_index = 1 - your_index
    return GameState(
        turn=_int(board.get("turn"), 0),
        your_index=your_index,
        energy_attached=bool(board.get("energy_attached", False)),
        supporter_played=bool(board.get("supporter_played", False)),
        stadium_count=_int(board.get("stadium_count"), 0),
        looking_count=_int(board.get("looking_count"), 0),
        players=(
            _player_from_board(board, prefix="my", player_index=your_index, is_us=True),
            _player_from_board(
                board,
                prefix="opponent",
                player_index=opponent_index,
                is_us=False,
            ),
        ),
        stadium=_stadium_entities(board),
    )


def decision_frame_to_legal_actions(frame: DecisionFrame) -> list[LegalAction]:
    return [
        LegalAction(
            local_index=action.index,
            selected_indices=(action.index,),
            action_type=action.option_type,
            select_type=frame.select_type,
            context=frame.context,
            min_count=frame.min_count,
            max_count=frame.max_count,
            target_count=frame.target_count,
            rule_score=action.rule_score,
            rule_rank=action.rule_rank,
            card_id=action.card_id,
            card_name=action.card_name,
            target_card_id=action.target_card_id,
            target_name=action.target_name,
            area=action.area,
            area_index=action.area_index,
            target_area=action.target_area,
            target_index=action.target_index,
            attack_id=action.attack_id,
            ends_turn=action.option_type in {"ATTACK", "END"},
            raw=action.raw,
        )
        for action in frame.legal_options
        if action.legal_mask
    ]


def phase5_target_indices(frame: DecisionFrame, *, target_source: str = "search") -> list[int]:
    if target_source == "search":
        raw = frame.reward_metadata.get("phase5_search_indices", frame.rule_selected_indices)
    elif target_source == "baseline":
        raw = frame.reward_metadata.get("phase5_baseline_indices", frame.rule_selected_indices)
    elif target_source == "rule":
        raw = frame.rule_selected_indices
    else:
        raise ValueError(f"Unsupported Phase 5 target source: {target_source}")
    size = len(frame.legal_options)
    output: list[int] = []
    for value in raw or []:
        index = int(value)
        if 0 <= index < size and index not in output:
            output.append(index)
    return output


def phase5_symbolic_record_from_decision(
    frame: DecisionFrame,
    *,
    encoder: Phase5SymbolicEncoder,
    previous_action_features: Sequence[Sequence[float]] = (),
    max_previous_actions: int = 16,
    target_source: str = "search",
    changed_weight: float = 1.0,
    unchanged_weight: float = 1.0,
    target_indices_override: Sequence[int] | None = None,
    weight_override: float | None = None,
    value_target: float | None = None,
    action_value_target: float | None = None,
    include_tactical_targets: bool = True,
) -> Phase5SymbolicDecisionRecord | None:
    state = decision_frame_to_state(frame)
    legal_actions = decision_frame_to_legal_actions(frame)
    encoded = encoder.encode(state, legal_actions)
    target_indices = (
        _valid_indices(target_indices_override, len(frame.legal_options))
        if target_indices_override is not None
        else phase5_target_indices(frame, target_source=target_source)
    )
    position_by_index = {
        action_index: position
        for position, action_index in enumerate(encoded.legal_action_indices)
        if action_index >= 0
    }
    target_positions = [
        position_by_index[index]
        for index in target_indices
        if index in position_by_index
    ]
    if not target_positions:
        return None
    previous_rows, previous_mask = _pad_previous_actions(
        previous_action_features,
        max_previous_actions=max_previous_actions,
        action_dim=len(encoded.legal_action_features[0]) if encoded.legal_action_features else 0,
    )
    changed = bool(frame.reward_metadata.get("phase5_search_changed", False))
    record_weight = (
        float(weight_override)
        if weight_override is not None
        else float(changed_weight if changed else unchanged_weight)
    )
    action_value_targets, action_value_mask = _action_value_targets(
        encoded,
        target_positions,
        action_value_target,
    )
    tactical_targets, tactical_mask = (
        _tactical_targets(encoded) if include_tactical_targets else (None, None)
    )
    return Phase5SymbolicDecisionRecord(
        encoded=encoded,
        target_indices=target_indices,
        target_positions=target_positions,
        previous_action_features=previous_rows,
        previous_action_mask=previous_mask,
        weight=record_weight,
        changed=changed,
        metadata=dict(frame.reward_metadata)
        | {
            "select_type": frame.select_type,
            "context": frame.context,
            "target_source": target_source,
        },
        value_target=value_target,
        action_value_targets=action_value_targets,
        action_value_mask=action_value_mask,
        tactical_targets=tactical_targets,
        tactical_mask=tactical_mask,
    )


def phase5_symbolic_record_from_trajectory(
    step: TrajectoryStep,
    *,
    encoder: Phase5SymbolicEncoder,
    previous_action_features: Sequence[Sequence[float]] = (),
    max_previous_actions: int = 16,
    weight: float = 1.0,
) -> Phase5SymbolicDecisionRecord | None:
    reward = float(step.reward)
    return phase5_symbolic_record_from_decision(
        step.decision,
        encoder=encoder,
        previous_action_features=previous_action_features,
        max_previous_actions=max_previous_actions,
        target_source="selfplay",
        target_indices_override=step.chosen_indices,
        weight_override=weight,
        value_target=reward,
        action_value_target=reward,
    )


def _trajectory_behavior_indices(step: TrajectoryStep) -> list[int] | None:
    metadata = step.decision.reward_metadata
    if not bool(metadata.get("teacher_forced_target", False)):
        return None
    raw_indices = metadata.get("behavior_selected_indices")
    if not isinstance(raw_indices, Sequence) or isinstance(raw_indices, (str, bytes)):
        return None
    indices: list[int] = []
    for value in raw_indices:
        try:
            indices.append(int(value))
        except (TypeError, ValueError):
            continue
    return indices


def build_phase5_symbolic_dataset(
    *,
    dataset_path: Path,
    output_path: Path,
    limit: int | None = None,
    max_entities: int = 96,
    max_actions: int = 128,
    max_previous_actions: int = 16,
    target_source: str = "search",
    changed_weight: float = 1.0,
    unchanged_weight: float = 1.0,
) -> Phase5SymbolicBuildSummary:
    encoder = Phase5SymbolicEncoder(max_entities=max_entities, max_actions=max_actions)
    context = Phase5TurnContext(max_previous_actions=max_previous_actions)
    frames_seen = records_written = skipped_no_target = changed_records = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for frame in iter_decision_jsonl(dataset_path):
            frames_seen += 1
            previous_rows = context.previous_rows(frame)
            record = phase5_symbolic_record_from_decision(
                frame,
                encoder=encoder,
                previous_action_features=previous_rows,
                max_previous_actions=max_previous_actions,
                target_source=target_source,
                changed_weight=changed_weight,
                unchanged_weight=unchanged_weight,
            )
            if record is None:
                skipped_no_target += 1
                continue
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
            records_written += 1
            changed_records += int(record.changed)
            context.observe(record)
            if limit is not None and records_written >= limit:
                break
    return Phase5SymbolicBuildSummary(
        input_path=str(dataset_path.as_posix()),
        output_path=str(output_path.as_posix()),
        frames_seen=frames_seen,
        records_written=records_written,
        skipped_no_target=skipped_no_target,
        changed_records=changed_records,
        target_source=target_source,
        max_entities=max_entities,
        max_actions=max_actions,
        max_previous_actions=max_previous_actions,
        limit=limit,
    )


def train_phase5_symbolic_policy_from_decisions(
    *,
    dataset_path: Path,
    checkpoint_path: Path,
    report_path: Path | None = None,
    epochs: int = 1,
    batch_size: int = 64,
    learning_rate: float = 1.0e-4,
    max_entities: int = 96,
    max_actions: int = 128,
    max_previous_actions: int = 16,
    d_model: int = 128,
    target_source: str = "search",
    changed_weight: float = 1.0,
    unchanged_weight: float = 1.0,
    pairwise_changed: bool = False,
    pairwise_weight: float = 1.0,
    pairwise_margin: float = 1.0,
    pairwise_negatives: str = "all",
    value_loss_weight: float = 0.0,
    limit: int | None = None,
    changed_only: bool = False,
) -> Phase5SymbolicTrainingSummary:
    torch, nn = _require_torch()
    if pairwise_negatives not in PHASE5_SYMBOLIC_PAIRWISE_NEGATIVES:
        raise ValueError(
            "Unsupported Phase 5 pairwise negative mode: "
            f"{pairwise_negatives!r}. Expected one of "
            f"{', '.join(PHASE5_SYMBOLIC_PAIRWISE_NEGATIVES)}."
        )
    encoder = Phase5SymbolicEncoder(max_entities=max_entities, max_actions=max_actions)
    empty_encoded = encoder.encode(decision_frame_to_state(_empty_frame()), [])
    config = make_phase5_policy_config(
        global_dim=len(empty_encoded.global_features),
        entity_dim=_entity_dim(encoder),
        action_dim=_action_dim(encoder),
        d_model=d_model,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaStarTurnPolicy(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    final_loss = 0.0
    final_frames_seen = 0
    final_examples = 0
    final_skipped = 0
    final_changed = 0
    final_actions = 0
    final_correct = 0

    for _ in range(max(1, epochs)):
        context = Phase5TurnContext(max_previous_actions=max_previous_actions)
        batch: list[Phase5SymbolicDecisionRecord] = []
        frames_seen = examples = skipped = changed_examples = actions = correct = 0
        for frame in iter_decision_jsonl(dataset_path):
            frames_seen += 1
            if limit is not None and frames_seen > limit:
                break
            if changed_only and not bool(
                frame.reward_metadata.get("phase5_search_changed", False)
            ):
                continue
            previous_rows = context.previous_rows(frame)
            record = phase5_symbolic_record_from_decision(
                frame,
                encoder=encoder,
                previous_action_features=previous_rows,
                max_previous_actions=max_previous_actions,
                target_source=target_source,
                changed_weight=changed_weight,
                unchanged_weight=unchanged_weight,
            )
            if record is None:
                skipped += 1
                continue
            batch.append(record)
            context.observe(record)
            examples += 1
            changed_examples += int(record.changed)
            actions += int(sum(record.encoded.legal_action_mask))
            if len(batch) >= max(1, batch_size):
                loss, batch_correct = _train_symbolic_batch(
                    batch,
                    model=model,
                    optimizer=optimizer,
                    torch=torch,
                    nn=nn,
                    device=device,
                    value_loss_weight=value_loss_weight,
                    pairwise_changed=pairwise_changed,
                    pairwise_weight=pairwise_weight,
                    pairwise_margin=pairwise_margin,
                    pairwise_negatives=pairwise_negatives,
                )
                final_loss = loss
                correct += batch_correct
                batch = []
        if batch:
            loss, batch_correct = _train_symbolic_batch(
                batch,
                model=model,
                optimizer=optimizer,
                torch=torch,
                nn=nn,
                device=device,
                value_loss_weight=value_loss_weight,
                pairwise_changed=pairwise_changed,
                pairwise_weight=pairwise_weight,
                pairwise_margin=pairwise_margin,
                pairwise_negatives=pairwise_negatives,
            )
            final_loss = loss
            correct += batch_correct
        final_frames_seen = frames_seen
        final_examples = examples
        final_skipped = skipped
        final_changed = changed_examples
        final_actions = actions
        final_correct = correct

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = model.checkpoint_payload()
    checkpoint["format"] = PHASE5_POLICY_CHECKPOINT_FORMAT
    checkpoint["encoder"] = {
        "max_entities": max_entities,
        "max_actions": max_actions,
        "max_previous_actions": max_previous_actions,
    }
    checkpoint["metadata"] = {
        "training": "phase5_symbolic_supervised",
        "dataset_path": str(dataset_path.as_posix()),
        "epochs": max(1, epochs),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "target_source": target_source,
        "changed_weight": changed_weight,
        "unchanged_weight": unchanged_weight,
        "pairwise_changed": pairwise_changed,
        "pairwise_weight": pairwise_weight,
        "pairwise_margin": pairwise_margin,
        "pairwise_negatives": pairwise_negatives,
        "value_loss_weight": value_loss_weight,
        "limit": limit,
        "changed_only": changed_only,
        "examples": final_examples,
        "changed_examples": final_changed,
        "actions": final_actions,
        "accuracy": final_correct / final_examples if final_examples else 0.0,
        "final_loss": final_loss,
        "device": str(device),
    }
    torch.save(checkpoint, checkpoint_path)
    summary = Phase5SymbolicTrainingSummary(
        frames_seen=final_frames_seen,
        examples=final_examples,
        skipped_no_target=final_skipped,
        changed_examples=final_changed,
        actions=final_actions,
        epochs=max(1, epochs),
        checkpoint_path=str(checkpoint_path.as_posix()),
        accuracy=final_correct / final_examples if final_examples else 0.0,
        final_loss=final_loss,
        device=str(device),
        target_source=target_source,
        changed_weight=changed_weight,
        unchanged_weight=unchanged_weight,
        pairwise_changed=pairwise_changed,
        pairwise_weight=pairwise_weight,
        pairwise_margin=pairwise_margin,
        pairwise_negatives=pairwise_negatives,
        max_entities=max_entities,
        max_actions=max_actions,
        max_previous_actions=max_previous_actions,
        config=config.to_dict(),
    )
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(summary.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return summary


def train_phase5_generalist_policy(
    *,
    decision_dataset_path: Path | None,
    selfplay_dataset_paths: Sequence[Path] = (),
    checkpoint_path: Path,
    initial_checkpoint_path: Path | None = None,
    report_path: Path | None = None,
    epochs: int = 1,
    batch_size: int = 64,
    learning_rate: float = 1.0e-4,
    max_entities: int = 96,
    max_actions: int = 128,
    max_previous_actions: int = 16,
    d_model: int = 128,
    target_source: str = "search",
    rule_demo_target_source: str = "baseline",
    changed_weight: float = 2.0,
    unchanged_weight: float = 0.5,
    search_decision_weight: float = 1.0,
    rule_demo_weight: float = 0.25,
    selfplay_weight: float = 1.0,
    pairwise_changed: bool = False,
    pairwise_weight: float = 0.25,
    pairwise_margin: float = 1.0,
    pairwise_negatives: str = "baseline",
    value_loss_weight: float = 0.25,
    action_value_loss_weight: float = 0.25,
    tactical_loss_weight: float = 0.05,
    decision_limit: int | None = None,
    selfplay_limit: int | None = None,
    deck_index_filter: int | None = None,
) -> Phase5GeneralistTrainingSummary:
    if decision_dataset_path is None and not selfplay_dataset_paths:
        raise ValueError("Provide at least one decision or self-play dataset.")
    torch, nn = _require_torch()
    if pairwise_negatives not in PHASE5_SYMBOLIC_PAIRWISE_NEGATIVES:
        raise ValueError(
            "Unsupported Phase 5 pairwise negative mode: "
            f"{pairwise_negatives!r}. Expected one of "
            f"{', '.join(PHASE5_SYMBOLIC_PAIRWISE_NEGATIVES)}."
        )
    initial_checkpoint: dict[str, Any] | None = None
    if initial_checkpoint_path is not None:
        if not initial_checkpoint_path.exists():
            raise ValueError(f"Initial Phase 5 checkpoint not found at {initial_checkpoint_path}.")
        initial_checkpoint = torch.load(
            initial_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if initial_checkpoint.get("format") != PHASE5_POLICY_CHECKPOINT_FORMAT:
            raise ValueError(
                "Unsupported initial Phase 5 checkpoint format: "
                f"{initial_checkpoint.get('format')}"
            )
        encoder_config = initial_checkpoint.get("encoder", {})
        max_entities = int(encoder_config.get("max_entities", max_entities))
        max_actions = int(encoder_config.get("max_actions", max_actions))
        max_previous_actions = int(
            encoder_config.get("max_previous_actions", max_previous_actions)
        )
        config = Phase5PolicyConfig(**initial_checkpoint["config"])
    else:
        encoder = Phase5SymbolicEncoder(max_entities=max_entities, max_actions=max_actions)
        empty_encoded = encoder.encode(decision_frame_to_state(_empty_frame()), [])
        config = make_phase5_policy_config(
            global_dim=len(empty_encoded.global_features),
            entity_dim=_entity_dim(encoder),
            action_dim=_action_dim(encoder),
            d_model=d_model,
        )
    encoder = Phase5SymbolicEncoder(max_entities=max_entities, max_actions=max_actions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaStarTurnPolicy(config).to(device)
    if initial_checkpoint is not None:
        model.load_state_dict(initial_checkpoint["state_dict"], strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    final_loss = 0.0
    final_decision_frames_seen = 0
    final_decision_examples = 0
    final_rule_examples = 0
    final_selfplay_steps_seen = 0
    final_selfplay_examples = 0
    final_skipped = 0
    final_changed = 0
    final_actions = 0
    final_value_examples = 0
    final_action_value_examples = 0
    final_tactical_examples = 0
    final_correct = 0

    for _ in range(max(1, epochs)):
        batch: list[Phase5SymbolicDecisionRecord] = []
        decision_context = Phase5TurnContext(max_previous_actions=max_previous_actions)
        selfplay_context = Phase5TurnContext(max_previous_actions=max_previous_actions)
        decision_frames_seen = 0
        decision_examples = 0
        rule_examples = 0
        selfplay_steps_seen = 0
        selfplay_examples = 0
        skipped = 0
        changed_examples = 0
        actions = 0
        value_examples = 0
        action_value_examples = 0
        tactical_examples = 0
        correct = 0

        def add_record(record: Phase5SymbolicDecisionRecord | None) -> None:
            nonlocal actions
            nonlocal action_value_examples
            nonlocal tactical_examples
            nonlocal value_examples
            nonlocal changed_examples
            if record is None:
                return
            batch.append(record)
            actions += int(sum(record.encoded.legal_action_mask))
            changed_examples += int(record.changed)
            value_examples += int(_record_value_target(record) is not None)
            action_value_examples += int(sum(record.action_value_mask or []))
            tactical_examples += int(sum(record.tactical_mask or []))

        def flush_batch() -> None:
            nonlocal batch, final_loss, correct
            if not batch:
                return
            loss, batch_correct = _train_symbolic_batch(
                batch,
                model=model,
                optimizer=optimizer,
                torch=torch,
                nn=nn,
                device=device,
                value_loss_weight=value_loss_weight,
                pairwise_changed=pairwise_changed,
                pairwise_weight=pairwise_weight,
                pairwise_margin=pairwise_margin,
                pairwise_negatives=pairwise_negatives,
                action_value_loss_weight=action_value_loss_weight,
                tactical_loss_weight=tactical_loss_weight,
            )
            final_loss = loss
            correct += batch_correct
            batch = []

        if decision_dataset_path is not None and (
            search_decision_weight > 0.0 or rule_demo_weight > 0.0
        ):
            for frame in iter_decision_jsonl(decision_dataset_path):
                decision_frames_seen += 1
                if decision_limit is not None and decision_frames_seen > decision_limit:
                    break
                if not _matches_deck_index_filter(frame.reward_metadata, deck_index_filter):
                    continue
                previous_rows = decision_context.previous_rows(frame)
                if search_decision_weight > 0.0:
                    search_record = phase5_symbolic_record_from_decision(
                        frame,
                        encoder=encoder,
                        previous_action_features=previous_rows,
                        max_previous_actions=max_previous_actions,
                        target_source=target_source,
                        changed_weight=changed_weight * search_decision_weight,
                        unchanged_weight=unchanged_weight * search_decision_weight,
                    )
                    if search_record is None:
                        skipped += 1
                    else:
                        add_record(search_record)
                        decision_examples += 1
                        decision_context.observe(search_record)
                        if len(batch) >= max(1, batch_size):
                            flush_batch()
                if rule_demo_weight > 0.0:
                    rule_record = phase5_symbolic_record_from_decision(
                        frame,
                        encoder=encoder,
                        previous_action_features=previous_rows,
                        max_previous_actions=max_previous_actions,
                        target_source=rule_demo_target_source,
                        weight_override=rule_demo_weight,
                    )
                    if rule_record is None:
                        skipped += 1
                    else:
                        add_record(rule_record)
                        rule_examples += 1
                        if len(batch) >= max(1, batch_size):
                            flush_batch()

        for selfplay_path in selfplay_dataset_paths:
            for step in iter_trajectory_jsonl(selfplay_path):
                if selfplay_limit is not None and selfplay_steps_seen >= selfplay_limit:
                    break
                selfplay_steps_seen += 1
                if not _matches_deck_index_filter(
                    step.decision.reward_metadata,
                    deck_index_filter,
                ):
                    continue
                previous_rows = selfplay_context.previous_rows(step.decision)
                record = phase5_symbolic_record_from_trajectory(
                    step,
                    encoder=encoder,
                    previous_action_features=previous_rows,
                    max_previous_actions=max_previous_actions,
                    weight=selfplay_weight,
                )
                if record is None:
                    skipped += 1
                    continue
                add_record(record)
                selfplay_examples += 1
                behavior_indices = _trajectory_behavior_indices(step)
                if behavior_indices is None:
                    selfplay_context.observe(record)
                else:
                    selfplay_context.observe_indices(record, behavior_indices)
                if len(batch) >= max(1, batch_size):
                    flush_batch()
            if selfplay_limit is not None and selfplay_steps_seen >= selfplay_limit:
                break
        flush_batch()

        final_decision_frames_seen = decision_frames_seen
        final_decision_examples = decision_examples
        final_rule_examples = rule_examples
        final_selfplay_steps_seen = selfplay_steps_seen
        final_selfplay_examples = selfplay_examples
        final_skipped = skipped
        final_changed = changed_examples
        final_actions = actions
        final_value_examples = value_examples
        final_action_value_examples = action_value_examples
        final_tactical_examples = tactical_examples
        final_correct = correct

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = model.checkpoint_payload()
    checkpoint["format"] = PHASE5_POLICY_CHECKPOINT_FORMAT
    checkpoint["encoder"] = {
        "max_entities": max_entities,
        "max_actions": max_actions,
        "max_previous_actions": max_previous_actions,
    }
    checkpoint["metadata"] = {
        "training": "phase5_generalist_mixed",
        "initial_checkpoint_path": (
            initial_checkpoint_path.as_posix() if initial_checkpoint_path else None
        ),
        "decision_dataset_path": (
            str(decision_dataset_path.as_posix()) if decision_dataset_path else None
        ),
        "selfplay_dataset_paths": [
            str(path.as_posix()) for path in selfplay_dataset_paths
        ],
        "deck_index_filter": deck_index_filter,
        "epochs": max(1, epochs),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "target_source": target_source,
        "rule_demo_target_source": rule_demo_target_source,
        "changed_weight": changed_weight,
        "unchanged_weight": unchanged_weight,
        "search_decision_weight": search_decision_weight,
        "rule_demo_weight": rule_demo_weight,
        "selfplay_weight": selfplay_weight,
        "pairwise_changed": pairwise_changed,
        "pairwise_weight": pairwise_weight,
        "pairwise_margin": pairwise_margin,
        "pairwise_negatives": pairwise_negatives,
        "value_loss_weight": value_loss_weight,
        "action_value_loss_weight": action_value_loss_weight,
        "tactical_loss_weight": tactical_loss_weight,
        "decision_limit": decision_limit,
        "selfplay_limit": selfplay_limit,
        "decision_examples": final_decision_examples,
        "rule_examples": final_rule_examples,
        "selfplay_examples": final_selfplay_examples,
        "value_examples": final_value_examples,
        "action_value_examples": final_action_value_examples,
        "tactical_examples": final_tactical_examples,
        "accuracy": (
            final_correct
            / (final_decision_examples + final_rule_examples + final_selfplay_examples)
            if (final_decision_examples + final_rule_examples + final_selfplay_examples)
            else 0.0
        ),
        "final_loss": final_loss,
        "device": str(device),
    }
    torch.save(checkpoint, checkpoint_path)
    total_examples = final_decision_examples + final_rule_examples + final_selfplay_examples
    summary = Phase5GeneralistTrainingSummary(
        decision_frames_seen=final_decision_frames_seen,
        decision_examples=final_decision_examples,
        rule_examples=final_rule_examples,
        selfplay_steps_seen=final_selfplay_steps_seen,
        selfplay_examples=final_selfplay_examples,
        skipped_no_target=final_skipped,
        changed_examples=final_changed,
        actions=final_actions,
        value_examples=final_value_examples,
        action_value_examples=final_action_value_examples,
        tactical_examples=final_tactical_examples,
        epochs=max(1, epochs),
        checkpoint_path=str(checkpoint_path.as_posix()),
        accuracy=final_correct / total_examples if total_examples else 0.0,
        final_loss=final_loss,
        device=str(device),
        decision_dataset_path=(
            str(decision_dataset_path.as_posix()) if decision_dataset_path else None
        ),
        selfplay_dataset_paths=[str(path.as_posix()) for path in selfplay_dataset_paths],
        deck_index_filter=deck_index_filter,
        search_decision_weight=search_decision_weight,
        rule_demo_weight=rule_demo_weight,
        selfplay_weight=selfplay_weight,
        value_loss_weight=value_loss_weight,
        action_value_loss_weight=action_value_loss_weight,
        tactical_loss_weight=tactical_loss_weight,
        max_entities=max_entities,
        max_actions=max_actions,
        max_previous_actions=max_previous_actions,
        config=config.to_dict(),
    )
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(summary.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return summary


def train_phase5_ppo_policy_from_trajectories(
    *,
    trajectory_dataset_paths: Sequence[Path],
    checkpoint_path: Path,
    output_checkpoint_path: Path,
    report_path: Path | None = None,
    epochs: int = 1,
    batch_size: int = 64,
    learning_rate: float = 5.0e-5,
    clip_epsilon: float = 0.2,
    policy_loss_weight: float = 1.0,
    value_loss_weight: float = 0.5,
    entropy_weight: float = 0.01,
    selfplay_limit: int | None = None,
    deck_index_filter: int | None = None,
    require_on_policy: bool = False,
) -> Phase5PPOTrainingSummary:
    if not trajectory_dataset_paths:
        raise ValueError("Provide at least one trajectory dataset.")
    torch, nn = _require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != PHASE5_POLICY_CHECKPOINT_FORMAT:
        raise ValueError(
            f"Unsupported Phase 5 checkpoint format: {checkpoint.get('format')}"
        )
    config = Phase5PolicyConfig(**checkpoint["config"])
    encoder_config = checkpoint.get("encoder", {})
    max_entities = int(encoder_config.get("max_entities", 96))
    max_actions = int(encoder_config.get("max_actions", 128))
    max_previous_actions = int(encoder_config.get("max_previous_actions", 16))
    encoder = Phase5SymbolicEncoder(max_entities=max_entities, max_actions=max_actions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaStarTurnPolicy(config).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    final_loss = 0.0
    final_steps_seen = 0
    final_examples = 0
    final_skipped = 0
    final_skipped_off_policy = 0
    final_advantage_sum = 0.0

    for _ in range(max(1, epochs)):
        batch: list[tuple[Phase5SymbolicDecisionRecord, float, float]] = []
        context = Phase5TurnContext(max_previous_actions=max_previous_actions)
        steps_seen = 0
        examples = 0
        skipped = 0
        skipped_off_policy = 0
        advantage_sum = 0.0

        def flush_batch() -> None:
            nonlocal batch, final_loss
            if not batch:
                return
            final_loss = _train_ppo_batch(
                batch,
                model=model,
                optimizer=optimizer,
                torch=torch,
                nn=nn,
                device=device,
                clip_epsilon=clip_epsilon,
                policy_loss_weight=policy_loss_weight,
                value_loss_weight=value_loss_weight,
                entropy_weight=entropy_weight,
            )
            batch = []

        for trajectory_path in trajectory_dataset_paths:
            for step in iter_trajectory_jsonl(trajectory_path):
                if selfplay_limit is not None and steps_seen >= selfplay_limit:
                    break
                steps_seen += 1
                if not _matches_deck_index_filter(
                    step.decision.reward_metadata,
                    deck_index_filter,
                ):
                    continue
                if require_on_policy and not bool(
                    step.decision.reward_metadata.get("policy_on_policy", False)
                ):
                    skipped_off_policy += 1
                    continue
                previous_rows = context.previous_rows(step.decision)
                record = phase5_symbolic_record_from_trajectory(
                    step,
                    encoder=encoder,
                    previous_action_features=previous_rows,
                    max_previous_actions=max_previous_actions,
                    weight=1.0,
                )
                if record is None:
                    skipped += 1
                    continue
                advantage = float(step.reward) - float(step.value)
                batch.append((record, float(step.logprob), advantage))
                examples += 1
                advantage_sum += advantage
                behavior_indices = _trajectory_behavior_indices(step)
                if behavior_indices is None:
                    context.observe(record)
                else:
                    context.observe_indices(record, behavior_indices)
                if len(batch) >= max(1, batch_size):
                    flush_batch()
            if selfplay_limit is not None and steps_seen >= selfplay_limit:
                break
        flush_batch()
        final_steps_seen = steps_seen
        final_examples = examples
        final_skipped = skipped
        final_skipped_off_policy = skipped_off_policy
        final_advantage_sum = advantage_sum

    output_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    output = model.checkpoint_payload()
    output["format"] = PHASE5_POLICY_CHECKPOINT_FORMAT
    output["encoder"] = {
        "max_entities": max_entities,
        "max_actions": max_actions,
        "max_previous_actions": max_previous_actions,
    }
    output["metadata"] = checkpoint.get("metadata", {}) | {
        "training": "phase5_ppo_trajectory_update",
        "source_checkpoint": checkpoint_path.as_posix(),
        "trajectory_dataset_paths": [
            path.as_posix() for path in trajectory_dataset_paths
        ],
        "epochs": max(1, epochs),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "clip_epsilon": clip_epsilon,
        "policy_loss_weight": policy_loss_weight,
        "value_loss_weight": value_loss_weight,
        "entropy_weight": entropy_weight,
        "selfplay_limit": selfplay_limit,
        "deck_index_filter": deck_index_filter,
        "require_on_policy": require_on_policy,
        "examples": final_examples,
        "mean_advantage": final_advantage_sum / final_examples
        if final_examples
        else 0.0,
    }
    torch.save(output, output_checkpoint_path)
    summary = Phase5PPOTrainingSummary(
        trajectory_dataset_paths=[path.as_posix() for path in trajectory_dataset_paths],
        steps_seen=final_steps_seen,
        examples=final_examples,
        skipped_no_target=final_skipped,
        skipped_off_policy=final_skipped_off_policy,
        epochs=max(1, epochs),
        checkpoint_path=checkpoint_path.as_posix(),
        output_checkpoint_path=output_checkpoint_path.as_posix(),
        final_loss=final_loss,
        mean_advantage=final_advantage_sum / final_examples if final_examples else 0.0,
        device=str(device),
        deck_index_filter=deck_index_filter,
        require_on_policy=require_on_policy,
        clip_epsilon=clip_epsilon,
        policy_loss_weight=policy_loss_weight,
        value_loss_weight=value_loss_weight,
        entropy_weight=entropy_weight,
        max_entities=max_entities,
        max_actions=max_actions,
        max_previous_actions=max_previous_actions,
        config=config.to_dict(),
    )
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(summary.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return summary


def train_phase5_bc_policy_from_trajectories(
    *,
    rule_trajectory_dataset_paths: Sequence[Path],
    checkpoint_path: Path,
    output_checkpoint_path: Path,
    report_path: Path | None = None,
    epochs: int = 1,
    batch_size: int = 64,
    learning_rate: float = 5.0e-5,
    rule_step_limit: int | None = None,
    deck_index_filter: int | None = None,
) -> Phase5TrajectoryBCTrainingSummary:
    if not rule_trajectory_dataset_paths:
        raise ValueError("Provide at least one rule trajectory dataset.")
    for path in rule_trajectory_dataset_paths:
        if not path.exists():
            raise ValueError(f"Phase 5 trajectory dataset not found at {path}.")
    if batch_size <= 0:
        raise ValueError("Phase 5 trajectory BC batch_size must be positive.")

    torch, nn = _require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != PHASE5_POLICY_CHECKPOINT_FORMAT:
        raise ValueError(
            f"Unsupported Phase 5 checkpoint format: {checkpoint.get('format')}"
        )
    config = Phase5PolicyConfig(**checkpoint["config"])
    encoder_config = checkpoint.get("encoder", {})
    max_entities = int(encoder_config.get("max_entities", 96))
    max_actions = int(encoder_config.get("max_actions", 128))
    max_previous_actions = int(encoder_config.get("max_previous_actions", 16))
    encoder = Phase5SymbolicEncoder(max_entities=max_entities, max_actions=max_actions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaStarTurnPolicy(config).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    stats = _TrajectorySourceStats()
    for _record, _step in _iter_phase5_hybrid_source(
        rule_trajectory_dataset_paths,
        encoder=encoder,
        max_previous_actions=max_previous_actions,
        deck_index_filter=deck_index_filter,
        step_limit=rule_step_limit,
        require_on_policy=False,
        accepted_policy_modes=(),
        stats=stats,
    ):
        pass
    if stats.examples <= 0:
        raise ValueError("Rule trajectory datasets produced zero BC examples.")

    def rule_factory() -> Iterable[Phase5SymbolicDecisionRecord]:
        for record, _step in _iter_phase5_hybrid_source(
            rule_trajectory_dataset_paths,
            encoder=encoder,
            max_previous_actions=max_previous_actions,
            deck_index_filter=deck_index_filter,
            step_limit=rule_step_limit,
            require_on_policy=False,
            accepted_policy_modes=(),
        ):
            yield record

    epochs_value = max(1, epochs)
    optimizer_steps = 0
    examples_used = 0
    final_loss = 0.0
    loss_sum = 0.0
    accuracy_sum = 0.0
    for _epoch in range(epochs_value):
        records = iter(rule_factory())
        while True:
            batch = list(islice(records, batch_size))
            if not batch:
                break
            loss, accuracy = _train_bc_only_batch(
                batch,
                model=model,
                optimizer=optimizer,
                torch=torch,
                nn=nn,
                device=device,
            )
            final_loss = loss
            loss_sum += loss
            accuracy_sum += accuracy
            optimizer_steps += 1
            examples_used += len(batch)

    output_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    output = model.checkpoint_payload()
    output["format"] = PHASE5_POLICY_CHECKPOINT_FORMAT
    output["encoder"] = {
        "max_entities": max_entities,
        "max_actions": max_actions,
        "max_previous_actions": max_previous_actions,
    }
    output["metadata"] = checkpoint.get("metadata", {}) | {
        "training": "phase5_rule_trajectory_behavior_cloning",
        "source_checkpoint": checkpoint_path.as_posix(),
        "rule_trajectory_dataset_paths": [
            path.as_posix() for path in rule_trajectory_dataset_paths
        ],
        "epochs": epochs_value,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "deck_index_filter": deck_index_filter,
    }
    torch.save(output, output_checkpoint_path)

    divisor = max(1, optimizer_steps)
    summary = Phase5TrajectoryBCTrainingSummary(
        rule_trajectory_dataset_paths=[
            path.as_posix() for path in rule_trajectory_dataset_paths
        ],
        steps_seen=stats.steps_seen,
        examples_available=stats.examples,
        skipped_no_target=stats.skipped_no_target,
        examples_used=examples_used,
        optimizer_steps=optimizer_steps,
        epochs=epochs_value,
        checkpoint_path=checkpoint_path.as_posix(),
        output_checkpoint_path=output_checkpoint_path.as_posix(),
        final_loss=final_loss,
        average_bc_loss=loss_sum / divisor,
        average_bc_accuracy=accuracy_sum / divisor,
        device=str(device),
        deck_index_filter=deck_index_filter,
        max_entities=max_entities,
        max_actions=max_actions,
        max_previous_actions=max_previous_actions,
        config=config.to_dict(),
    )
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(summary.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return summary


def train_phase5_bc_ppo_policy_from_trajectories(
    *,
    rule_trajectory_dataset_paths: Sequence[Path],
    on_policy_trajectory_dataset_paths: Sequence[Path],
    checkpoint_path: Path,
    output_checkpoint_path: Path,
    report_path: Path | None = None,
    epochs: int = 1,
    batch_size: int = 64,
    learning_rate: float = 5.0e-5,
    clip_epsilon: float = 0.2,
    bc_loss_weight: float = 1.0,
    ppo_policy_loss_weight: float = 1.0,
    ppo_value_loss_weight: float = 0.5,
    entropy_weight: float = 0.01,
    rule_step_limit: int | None = None,
    on_policy_step_limit: int | None = None,
    deck_index_filter: int | None = None,
    accepted_policy_modes: Sequence[str] = PHASE5_DIFFERENTIABLE_POLICY_MODES,
    update_schedule: str = "balanced-max",
    rule_anchor_fraction: float = 0.5,
    gradient_diagnostic_batches: int = 0,
    advantage_normalization: str = "batch",
    value_backprop_scope: str = "shared",
) -> Phase5BCPPOTrainingSummary:
    if not on_policy_trajectory_dataset_paths:
        raise ValueError("Provide at least one on-policy trajectory dataset.")
    for path in [*rule_trajectory_dataset_paths, *on_policy_trajectory_dataset_paths]:
        if not path.exists():
            raise ValueError(f"Phase 5 trajectory dataset not found at {path}.")
    if batch_size <= 0:
        raise ValueError("Phase 5 BC+PPO batch_size must be positive.")
    if update_schedule not in PHASE5_BC_PPO_UPDATE_SCHEDULES:
        raise ValueError(
            "Phase 5 BC+PPO update_schedule must be one of "
            + ", ".join(PHASE5_BC_PPO_UPDATE_SCHEDULES)
            + "."
        )
    if not 0.0 <= rule_anchor_fraction <= 0.5:
        raise ValueError("Phase 5 BC+PPO rule_anchor_fraction must be between 0 and 0.5.")
    if update_schedule == "balanced-max" and rule_anchor_fraction != 0.5:
        raise ValueError(
            "balanced-max requires rule_anchor_fraction=0.5; use ppo-epoch for a small anchor."
        )
    if gradient_diagnostic_batches < 0:
        raise ValueError("gradient_diagnostic_batches must be non-negative.")
    if advantage_normalization not in PHASE5_ADVANTAGE_NORMALIZATION_MODES:
        raise ValueError(
            "Phase 5 BC+PPO advantage_normalization must be one of "
            + ", ".join(PHASE5_ADVANTAGE_NORMALIZATION_MODES)
            + "."
        )
    if value_backprop_scope not in PHASE5_VALUE_BACKPROP_SCOPES:
        raise ValueError(
            "Phase 5 BC+PPO value_backprop_scope must be one of "
            + ", ".join(PHASE5_VALUE_BACKPROP_SCOPES)
            + "."
        )
    needs_rule_examples = update_schedule == "balanced-max" or rule_anchor_fraction > 0.0
    if needs_rule_examples and not rule_trajectory_dataset_paths:
        raise ValueError("Provide a rule trajectory dataset for the requested BC anchor.")
    for name, value in (
        ("bc_loss_weight", bc_loss_weight),
        ("ppo_policy_loss_weight", ppo_policy_loss_weight),
        ("ppo_value_loss_weight", ppo_value_loss_weight),
        ("entropy_weight", entropy_weight),
    ):
        if value < 0.0:
            raise ValueError(f"Phase 5 BC+PPO {name} must be non-negative.")
    normalized_policy_modes = tuple(dict.fromkeys(str(mode) for mode in accepted_policy_modes))
    unsupported_modes = [
        mode for mode in normalized_policy_modes if mode not in PHASE5_DIFFERENTIABLE_POLICY_MODES
    ]
    if unsupported_modes:
        raise ValueError(
            "Unsupported differentiable policy mode(s): " + ", ".join(unsupported_modes)
        )

    torch, nn = _require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != PHASE5_POLICY_CHECKPOINT_FORMAT:
        raise ValueError(
            f"Unsupported Phase 5 checkpoint format: {checkpoint.get('format')}"
        )
    config = Phase5PolicyConfig(**checkpoint["config"])
    encoder_config = checkpoint.get("encoder", {})
    max_entities = int(encoder_config.get("max_entities", 96))
    max_actions = int(encoder_config.get("max_actions", 128))
    max_previous_actions = int(encoder_config.get("max_previous_actions", 16))
    encoder = Phase5SymbolicEncoder(max_entities=max_entities, max_actions=max_actions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaStarTurnPolicy(config).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    rule_stats = _TrajectorySourceStats()
    if needs_rule_examples:
        for _record, _step in _iter_phase5_hybrid_source(
            rule_trajectory_dataset_paths,
            encoder=encoder,
            max_previous_actions=max_previous_actions,
            deck_index_filter=deck_index_filter,
            step_limit=rule_step_limit,
            require_on_policy=False,
            accepted_policy_modes=(),
            stats=rule_stats,
        ):
            pass
    on_policy_stats = _TrajectorySourceStats()
    for _record, _step in _iter_phase5_hybrid_source(
        on_policy_trajectory_dataset_paths,
        encoder=encoder,
        max_previous_actions=max_previous_actions,
        deck_index_filter=deck_index_filter,
        step_limit=on_policy_step_limit,
        require_on_policy=True,
        accepted_policy_modes=normalized_policy_modes,
        stats=on_policy_stats,
    ):
        pass
    if needs_rule_examples and rule_stats.examples <= 0:
        raise ValueError("Rule trajectory datasets produced zero BC examples.")
    if on_policy_stats.examples <= 0:
        raise ValueError(
            "On-policy trajectory datasets produced zero valid PPO examples. "
            "Use policy_mode=sample or epsilon_mixture with policy_on_policy=true."
        )
    advantage_normalization_mean = (
        on_policy_stats.advantage_sum / on_policy_stats.examples
    )
    advantage_variance = max(
        0.0,
        on_policy_stats.advantage_square_sum / on_policy_stats.examples
        - advantage_normalization_mean**2,
    )
    advantage_normalization_std = max(math.sqrt(advantage_variance), 1.0e-6)

    balanced_examples = (
        max(rule_stats.examples, on_policy_stats.examples)
        if update_schedule == "balanced-max"
        else 0
    )
    on_policy_examples_per_epoch = (
        balanced_examples
        if update_schedule == "balanced-max"
        else on_policy_stats.examples
    )
    rule_anchor_examples_per_epoch = (
        balanced_examples
        if update_schedule == "balanced-max"
        else round(
            on_policy_stats.examples
            * rule_anchor_fraction
            / max(1.0e-12, 1.0 - rule_anchor_fraction)
        )
    )

    def rule_factory() -> Iterable[Phase5SymbolicDecisionRecord]:
        for record, _step in _iter_phase5_hybrid_source(
            rule_trajectory_dataset_paths,
            encoder=encoder,
            max_previous_actions=max_previous_actions,
            deck_index_filter=deck_index_filter,
            step_limit=rule_step_limit,
            require_on_policy=False,
            accepted_policy_modes=(),
        ):
            yield record

    def on_policy_factory() -> Iterable[
        tuple[Phase5SymbolicDecisionRecord, float, float]
    ]:
        for record, step in _iter_phase5_hybrid_source(
            on_policy_trajectory_dataset_paths,
            encoder=encoder,
            max_previous_actions=max_previous_actions,
            deck_index_filter=deck_index_filter,
            step_limit=on_policy_step_limit,
            require_on_policy=True,
            accepted_policy_modes=normalized_policy_modes,
        ):
            yield record, float(step.logprob), float(step.reward) - float(step.value)

    metric_sums = {
        "loss": 0.0,
        "bc_loss": 0.0,
        "bc_accuracy": 0.0,
        "ppo_policy_loss": 0.0,
        "ppo_value_loss": 0.0,
        "entropy": 0.0,
        "ratio": 0.0,
        "clip_fraction": 0.0,
    }
    gradient_metric_sums = {
        "bc_gradient_norm": 0.0,
        "ppo_policy_gradient_norm": 0.0,
        "ppo_policy_shared_gradient_norm": 0.0,
        "ppo_value_gradient_norm": 0.0,
        "ppo_value_shared_gradient_norm": 0.0,
        "entropy_gradient_norm": 0.0,
        "bc_ppo_policy_gradient_cosine": 0.0,
        "ppo_policy_value_gradient_cosine": 0.0,
        "ppo_policy_value_shared_gradient_cosine": 0.0,
    }
    optimizer_steps = 0
    gradient_diagnostic_batches_recorded = 0
    rule_examples_used = 0
    on_policy_examples_used = 0
    final_loss = 0.0
    epochs_value = max(1, epochs)
    for _epoch in range(epochs_value):
        if update_schedule == "balanced-max":
            rule_records = _cycle_phase5_records(rule_factory)
            on_policy_examples = _cycle_phase5_records(on_policy_factory)
            batches: Iterable[tuple[list[Any], list[Any]]] = (
                (
                    list(islice(rule_records, current_batch_size)),
                    list(islice(on_policy_examples, current_batch_size)),
                )
                for current_batch_size in _phase5_batch_sizes(
                    balanced_examples,
                    batch_size,
                )
            )
        else:
            batches = _iter_phase5_ppo_epoch_batches(
                rule_factory=rule_factory,
                on_policy_factory=on_policy_factory,
                rule_examples_available=rule_stats.examples,
                on_policy_examples_available=on_policy_stats.examples,
                rule_examples_target=rule_anchor_examples_per_epoch,
                batch_size=batch_size,
            )
        for rule_batch, on_policy_batch in batches:
            if not on_policy_batch:
                raise ValueError("BC+PPO update unexpectedly produced an empty PPO batch.")
            if update_schedule == "balanced-max" and (
                len(rule_batch) != len(on_policy_batch)
            ):
                raise ValueError("Balanced BC+PPO source unexpectedly produced too few examples.")
            collect_gradient_diagnostics = (
                gradient_diagnostic_batches_recorded < gradient_diagnostic_batches
            )
            metrics = _train_bc_ppo_batch(
                rule_batch,
                on_policy_batch,
                model=model,
                optimizer=optimizer,
                torch=torch,
                nn=nn,
                device=device,
                clip_epsilon=clip_epsilon,
                bc_loss_weight=bc_loss_weight,
                ppo_policy_loss_weight=ppo_policy_loss_weight,
                ppo_value_loss_weight=ppo_value_loss_weight,
                entropy_weight=entropy_weight,
                collect_gradient_diagnostics=collect_gradient_diagnostics,
                advantage_normalization=advantage_normalization,
                advantage_normalization_mean=advantage_normalization_mean,
                advantage_normalization_std=advantage_normalization_std,
                value_backprop_scope=value_backprop_scope,
            )
            final_loss = metrics.loss
            for field in metric_sums:
                metric_sums[field] += float(getattr(metrics, field))
            if collect_gradient_diagnostics:
                for field in gradient_metric_sums:
                    gradient_metric_sums[field] += float(getattr(metrics, field))
                gradient_diagnostic_batches_recorded += 1
            optimizer_steps += 1
            rule_examples_used += len(rule_batch)
            on_policy_examples_used += len(on_policy_batch)

    output_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    output = model.checkpoint_payload()
    output["format"] = PHASE5_POLICY_CHECKPOINT_FORMAT
    output["encoder"] = {
        "max_entities": max_entities,
        "max_actions": max_actions,
        "max_previous_actions": max_previous_actions,
    }
    output["metadata"] = checkpoint.get("metadata", {}) | {
        "training": (
            "phase5_balanced_bc_ppo_trajectory_update"
            if update_schedule == "balanced-max"
            else "phase5_ppo_epoch_bc_anchor_trajectory_update"
        ),
        "source_checkpoint": checkpoint_path.as_posix(),
        "rule_trajectory_dataset_paths": [
            path.as_posix() for path in rule_trajectory_dataset_paths
        ],
        "on_policy_trajectory_dataset_paths": [
            path.as_posix() for path in on_policy_trajectory_dataset_paths
        ],
        "accepted_policy_modes": list(normalized_policy_modes),
        "update_schedule": update_schedule,
        "rule_anchor_fraction": rule_anchor_fraction,
        "advantage_normalization": advantage_normalization,
        "advantage_normalization_mean": advantage_normalization_mean,
        "advantage_normalization_std": advantage_normalization_std,
        "value_backprop_scope": value_backprop_scope,
        "balanced_examples_per_source_per_epoch": balanced_examples,
        "rule_anchor_examples_per_epoch": rule_anchor_examples_per_epoch,
        "on_policy_examples_per_epoch": on_policy_examples_per_epoch,
        "epochs": max(1, epochs),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "clip_epsilon": clip_epsilon,
        "bc_loss_weight": bc_loss_weight,
        "ppo_policy_loss_weight": ppo_policy_loss_weight,
        "ppo_value_loss_weight": ppo_value_loss_weight,
        "entropy_weight": entropy_weight,
        "gradient_diagnostic_batches": gradient_diagnostic_batches,
        "deck_index_filter": deck_index_filter,
    }
    torch.save(output, output_checkpoint_path)

    divisor = max(1, optimizer_steps)
    gradient_divisor = max(1, gradient_diagnostic_batches_recorded)
    summary = Phase5BCPPOTrainingSummary(
        rule_trajectory_dataset_paths=[
            path.as_posix() for path in rule_trajectory_dataset_paths
        ],
        on_policy_trajectory_dataset_paths=[
            path.as_posix() for path in on_policy_trajectory_dataset_paths
        ],
        rule_steps_seen=rule_stats.steps_seen,
        rule_examples_available=rule_stats.examples,
        rule_skipped_no_target=rule_stats.skipped_no_target,
        on_policy_steps_seen=on_policy_stats.steps_seen,
        on_policy_examples_available=on_policy_stats.examples,
        on_policy_skipped_no_target=on_policy_stats.skipped_no_target,
        on_policy_skipped_off_policy=on_policy_stats.skipped_off_policy,
        on_policy_skipped_policy_mode=on_policy_stats.skipped_policy_mode,
        on_policy_skipped_nonfinite=on_policy_stats.skipped_nonfinite,
        accepted_policy_modes=list(normalized_policy_modes),
        update_schedule=update_schedule,
        rule_anchor_fraction=rule_anchor_fraction,
        advantage_normalization=advantage_normalization,
        advantage_normalization_mean=advantage_normalization_mean,
        advantage_normalization_std=advantage_normalization_std,
        value_backprop_scope=value_backprop_scope,
        balanced_examples_per_source_per_epoch=balanced_examples,
        rule_anchor_examples_per_epoch=rule_anchor_examples_per_epoch,
        on_policy_examples_per_epoch=on_policy_examples_per_epoch,
        rule_examples_used=rule_examples_used,
        on_policy_examples_used=on_policy_examples_used,
        rule_reuse_factor=(
            rule_examples_used / rule_stats.examples
            if rule_stats.examples
            else 0.0
        ),
        on_policy_reuse_factor=(
            on_policy_examples_used / on_policy_stats.examples
        ),
        optimizer_steps=optimizer_steps,
        epochs=epochs_value,
        checkpoint_path=checkpoint_path.as_posix(),
        output_checkpoint_path=output_checkpoint_path.as_posix(),
        final_loss=final_loss,
        average_bc_loss=metric_sums["bc_loss"] / divisor,
        average_bc_accuracy=metric_sums["bc_accuracy"] / divisor,
        average_ppo_policy_loss=metric_sums["ppo_policy_loss"] / divisor,
        average_ppo_value_loss=metric_sums["ppo_value_loss"] / divisor,
        average_entropy=metric_sums["entropy"] / divisor,
        average_ratio=metric_sums["ratio"] / divisor,
        average_clip_fraction=metric_sums["clip_fraction"] / divisor,
        gradient_diagnostic_batches_requested=gradient_diagnostic_batches,
        gradient_diagnostic_batches_recorded=gradient_diagnostic_batches_recorded,
        average_bc_gradient_norm=(
            gradient_metric_sums["bc_gradient_norm"] / gradient_divisor
        ),
        average_ppo_policy_gradient_norm=(
            gradient_metric_sums["ppo_policy_gradient_norm"] / gradient_divisor
        ),
        average_ppo_policy_shared_gradient_norm=(
            gradient_metric_sums["ppo_policy_shared_gradient_norm"]
            / gradient_divisor
        ),
        average_ppo_value_gradient_norm=(
            gradient_metric_sums["ppo_value_gradient_norm"] / gradient_divisor
        ),
        average_ppo_value_shared_gradient_norm=(
            gradient_metric_sums["ppo_value_shared_gradient_norm"]
            / gradient_divisor
        ),
        average_entropy_gradient_norm=(
            gradient_metric_sums["entropy_gradient_norm"] / gradient_divisor
        ),
        average_bc_ppo_policy_gradient_cosine=(
            gradient_metric_sums["bc_ppo_policy_gradient_cosine"] / gradient_divisor
        ),
        average_ppo_policy_value_gradient_cosine=(
            gradient_metric_sums["ppo_policy_value_gradient_cosine"] / gradient_divisor
        ),
        average_ppo_policy_value_shared_gradient_cosine=(
            gradient_metric_sums["ppo_policy_value_shared_gradient_cosine"]
            / gradient_divisor
        ),
        mean_advantage=(
            on_policy_stats.advantage_sum / on_policy_stats.examples
            if on_policy_stats.examples
            else 0.0
        ),
        device=str(device),
        deck_index_filter=deck_index_filter,
        bc_loss_weight=bc_loss_weight,
        ppo_policy_loss_weight=ppo_policy_loss_weight,
        ppo_value_loss_weight=ppo_value_loss_weight,
        entropy_weight=entropy_weight,
        clip_epsilon=clip_epsilon,
        max_entities=max_entities,
        max_actions=max_actions,
        max_previous_actions=max_previous_actions,
        config=config.to_dict(),
    )
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(summary.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return summary


def _iter_phase5_hybrid_source(
    trajectory_dataset_paths: Sequence[Path],
    *,
    encoder: Phase5SymbolicEncoder,
    max_previous_actions: int,
    deck_index_filter: int | None,
    step_limit: int | None,
    require_on_policy: bool,
    accepted_policy_modes: Sequence[str],
    stats: _TrajectorySourceStats | None = None,
) -> Iterable[tuple[Phase5SymbolicDecisionRecord, TrajectoryStep]]:
    steps_seen = 0
    accepted_modes = set(accepted_policy_modes)
    for trajectory_path in trajectory_dataset_paths:
        context = Phase5TurnContext(max_previous_actions=max_previous_actions)
        for step in iter_trajectory_jsonl(trajectory_path):
            if step_limit is not None and steps_seen >= step_limit:
                return
            steps_seen += 1
            if stats is not None:
                stats.steps_seen += 1
            if not _matches_deck_index_filter(
                step.decision.reward_metadata,
                deck_index_filter,
            ):
                continue
            previous_rows = context.previous_rows(step.decision)
            record = phase5_symbolic_record_from_trajectory(
                step,
                encoder=encoder,
                previous_action_features=previous_rows,
                max_previous_actions=max_previous_actions,
                weight=1.0,
            )
            if record is None:
                if stats is not None:
                    stats.skipped_no_target += 1
                continue
            behavior_indices = _trajectory_behavior_indices(step)
            if behavior_indices is None:
                context.observe(record)
            else:
                context.observe_indices(record, behavior_indices)
            if require_on_policy:
                metadata = step.decision.reward_metadata
                if not bool(metadata.get("policy_on_policy", False)):
                    if stats is not None:
                        stats.skipped_off_policy += 1
                    continue
                policy_mode = str(metadata.get("policy_mode", ""))
                if policy_mode not in accepted_modes:
                    if stats is not None:
                        stats.skipped_policy_mode += 1
                    continue
                if not math.isfinite(float(step.logprob)) or not math.isfinite(float(step.value)):
                    if stats is not None:
                        stats.skipped_nonfinite += 1
                    continue
            if stats is not None:
                stats.examples += 1
                advantage = float(step.reward) - float(step.value)
                stats.advantage_sum += advantage
                stats.advantage_square_sum += advantage * advantage
            yield record, step


def _cycle_phase5_records(factory: Any) -> Iterable[Any]:
    while True:
        yielded = False
        for value in factory():
            yielded = True
            yield value
        if not yielded:
            return


def _phase5_batch_sizes(example_count: int, batch_size: int) -> Iterable[int]:
    remaining = example_count
    while remaining > 0:
        current = min(batch_size, remaining)
        yield current
        remaining -= current


def _sample_phase5_records_evenly(
    factory: Any,
    *,
    examples_available: int,
    examples_target: int,
) -> Iterable[Any]:
    if examples_target <= 0:
        return
    if examples_target > examples_available:
        yield from islice(_cycle_phase5_records(factory), examples_target)
        return
    selected = 0
    for index, value in enumerate(factory(), start=1):
        expected_selected = index * examples_target // examples_available
        if expected_selected > selected:
            yield value
            selected += 1
        if selected >= examples_target:
            return


def _iter_phase5_ppo_epoch_batches(
    *,
    rule_factory: Any,
    on_policy_factory: Any,
    rule_examples_available: int,
    on_policy_examples_available: int,
    rule_examples_target: int,
    batch_size: int,
) -> Iterable[tuple[list[Any], list[Any]]]:
    on_policy_records = iter(on_policy_factory())
    rule_records = iter(
        _sample_phase5_records_evenly(
            rule_factory,
            examples_available=rule_examples_available,
            examples_target=rule_examples_target,
        )
    )
    batches_remaining = math.ceil(on_policy_examples_available / batch_size)
    rule_examples_remaining = rule_examples_target
    for current_batch_size in _phase5_batch_sizes(
        on_policy_examples_available,
        batch_size,
    ):
        on_policy_batch = list(islice(on_policy_records, current_batch_size))
        if len(on_policy_batch) != current_batch_size:
            raise ValueError("PPO epoch source unexpectedly produced too few examples.")
        current_rule_count = (
            round(rule_examples_remaining / batches_remaining)
            if batches_remaining > 0
            else 0
        )
        rule_batch = list(islice(rule_records, current_rule_count))
        if len(rule_batch) != current_rule_count:
            raise ValueError("Rule anchor source unexpectedly produced too few examples.")
        yield rule_batch, on_policy_batch
        rule_examples_remaining -= current_rule_count
        batches_remaining -= 1


def initialize_phase5_policy_checkpoint(
    *,
    checkpoint_path: Path,
    report_path: Path | None = None,
    max_entities: int = 96,
    max_actions: int = 128,
    max_previous_actions: int = 16,
    d_model: int = 128,
    seed: int | None = None,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> Phase5PolicyInitSummary:
    if checkpoint_path.exists() and not overwrite:
        raise ValueError(f"Scratch checkpoint already exists at {checkpoint_path}.")
    torch, _ = _require_torch()
    if seed is not None:
        torch.manual_seed(int(seed))
    encoder = Phase5SymbolicEncoder(max_entities=max_entities, max_actions=max_actions)
    empty_encoded = encoder.encode(decision_frame_to_state(_empty_frame()), [])
    config = make_phase5_policy_config(
        global_dim=len(empty_encoded.global_features),
        entity_dim=_entity_dim(encoder),
        action_dim=_action_dim(encoder),
        d_model=d_model,
    )
    model = AlphaStarTurnPolicy(config)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = model.checkpoint_payload()
    payload["format"] = PHASE5_POLICY_CHECKPOINT_FORMAT
    payload["encoder"] = {
        "max_entities": max_entities,
        "max_actions": max_actions,
        "max_previous_actions": max_previous_actions,
    }
    payload["metadata"] = {
        "training": "phase5_scratch_init",
        "checkpoint_path": checkpoint_path.as_posix(),
        "seed": seed,
    } | dict(metadata or {})
    torch.save(payload, checkpoint_path)
    summary = Phase5PolicyInitSummary(
        checkpoint_path=checkpoint_path.as_posix(),
        max_entities=max_entities,
        max_actions=max_actions,
        max_previous_actions=max_previous_actions,
        config=config.to_dict(),
        metadata=dict(payload["metadata"]),
    )
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(summary.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return summary


def read_phase5_symbolic_jsonl(path: Path) -> list[Phase5SymbolicDecisionRecord]:
    records: list[Phase5SymbolicDecisionRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(Phase5SymbolicDecisionRecord.from_dict(json.loads(stripped)))
    return records


def phase5_symbolic_pairwise_positions(
    record: Phase5SymbolicDecisionRecord,
    *,
    negative_mode: str = "all",
) -> list[tuple[int, int]]:
    if negative_mode not in PHASE5_SYMBOLIC_PAIRWISE_NEGATIVES:
        raise ValueError(
            "Unsupported Phase 5 pairwise negative mode: "
            f"{negative_mode!r}. Expected one of "
            f"{', '.join(PHASE5_SYMBOLIC_PAIRWISE_NEGATIVES)}."
        )
    legal_positions = [
        position
        for position, mask_value in enumerate(record.encoded.legal_action_mask)
        if mask_value > 0.0
    ]
    legal_set = set(legal_positions)
    positives = [
        position
        for position in record.target_positions
        if position in legal_set
    ]
    if not positives:
        return []
    positive_set = set(positives)
    if negative_mode == "baseline":
        negative_positions = _baseline_target_positions(record)
    else:
        negative_positions = legal_positions
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for positive in positives:
        for negative in negative_positions:
            if negative not in legal_set or negative in positive_set:
                continue
            pair = (positive, negative)
            if pair not in seen:
                pairs.append(pair)
                seen.add(pair)
    return pairs


def _train_symbolic_batch(
    records: Sequence[Phase5SymbolicDecisionRecord],
    *,
    model: Any,
    optimizer: Any,
    torch: Any,
    nn: Any,
    device: Any,
    value_loss_weight: float,
    pairwise_changed: bool,
    pairwise_weight: float,
    pairwise_margin: float,
    pairwise_negatives: str,
    action_value_loss_weight: float = 0.0,
    tactical_loss_weight: float = 0.0,
) -> tuple[float, int]:
    global_x = torch.tensor(
        [record.encoded.global_features for record in records],
        dtype=torch.float32,
        device=device,
    )
    entity_x = torch.tensor(
        [record.encoded.entity_features for record in records],
        dtype=torch.float32,
        device=device,
    )
    entity_mask = torch.tensor(
        [record.encoded.entity_mask for record in records],
        dtype=torch.float32,
        device=device,
    )
    action_x = torch.tensor(
        [record.encoded.legal_action_features for record in records],
        dtype=torch.float32,
        device=device,
    )
    action_mask = torch.tensor(
        [record.encoded.legal_action_mask for record in records],
        dtype=torch.float32,
        device=device,
    )
    previous_action_x = torch.tensor(
        [record.previous_action_features for record in records],
        dtype=torch.float32,
        device=device,
    )
    previous_action_mask = torch.tensor(
        [record.previous_action_mask for record in records],
        dtype=torch.float32,
        device=device,
    )
    targets = torch.tensor(
        [record.target_positions[0] for record in records],
        dtype=torch.long,
        device=device,
    )
    weights = torch.tensor(
        [record.weight for record in records],
        dtype=torch.float32,
        device=device,
    )
    output = model(
        global_x,
        entity_x,
        entity_mask,
        action_x,
        action_mask,
        previous_action_x,
        previous_action_mask,
    )
    logits = output["action_logits"]
    actor_loss = nn.functional.cross_entropy(logits, targets, reduction="none")
    loss = (actor_loss * weights).mean()
    if pairwise_changed and pairwise_weight > 0.0:
        pairwise_loss = _changed_pairwise_loss(
            records,
            logits=logits,
            torch=torch,
            nn=nn,
            device=device,
            margin=pairwise_margin,
            negative_mode=pairwise_negatives,
        )
        if pairwise_loss is not None:
            loss = loss + float(pairwise_weight) * pairwise_loss
    if value_loss_weight > 0:
        value_rows: list[int] = []
        value_targets: list[float] = []
        value_weights: list[float] = []
        for row_index, record in enumerate(records):
            target = _record_value_target(record)
            if target is None:
                continue
            value_rows.append(row_index)
            value_targets.append(float(target))
            value_weights.append(float(record.weight))
        if value_rows:
            rows = torch.tensor(value_rows, dtype=torch.long, device=device)
            targets_value = torch.tensor(value_targets, dtype=torch.float32, device=device)
            value_weights_tensor = torch.tensor(value_weights, dtype=torch.float32, device=device)
            value_losses = nn.functional.mse_loss(
                output["state_value"][rows],
                targets_value,
                reduction="none",
            )
            loss = loss + float(value_loss_weight) * (
                value_losses * value_weights_tensor
            ).sum() / value_weights_tensor.sum().clamp(min=1.0e-6)
    if action_value_loss_weight > 0.0:
        action_value_targets = torch.tensor(
            [
                record.action_value_targets
                if record.action_value_targets is not None
                else [0.0] * len(record.encoded.legal_action_mask)
                for record in records
            ],
            dtype=torch.float32,
            device=device,
        )
        action_value_mask = torch.tensor(
            [
                record.action_value_mask
                if record.action_value_mask is not None
                else [0.0] * len(record.encoded.legal_action_mask)
                for record in records
            ],
            dtype=torch.float32,
            device=device,
        )
        weighted_mask = action_value_mask * weights.unsqueeze(1)
        if float(weighted_mask.sum().detach().item()) > 0.0:
            action_value_losses = nn.functional.mse_loss(
                output["action_q"],
                action_value_targets,
                reduction="none",
            )
            loss = loss + float(action_value_loss_weight) * (
                action_value_losses * weighted_mask
            ).sum() / weighted_mask.sum().clamp(min=1.0e-6)
    if tactical_loss_weight > 0.0:
        tactical_targets = torch.tensor(
            [
                record.tactical_targets
                if record.tactical_targets is not None
                else [0.0] * len(record.encoded.legal_action_mask)
                for record in records
            ],
            dtype=torch.float32,
            device=device,
        )
        tactical_mask = torch.tensor(
            [
                record.tactical_mask
                if record.tactical_mask is not None
                else [0.0] * len(record.encoded.legal_action_mask)
                for record in records
            ],
            dtype=torch.float32,
            device=device,
        )
        weighted_mask = tactical_mask * weights.unsqueeze(1)
        if float(weighted_mask.sum().detach().item()) > 0.0:
            tactical_losses = nn.functional.mse_loss(
                output["tactical_score"],
                tactical_targets,
                reduction="none",
            )
            loss = loss + float(tactical_loss_weight) * (
                tactical_losses * weighted_mask
            ).sum() / weighted_mask.sum().clamp(min=1.0e-6)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    predictions = logits.detach().argmax(dim=1)
    correct = int((predictions == targets).sum().item())
    return float(loss.detach().item()), correct


def _train_ppo_batch(
    examples: Sequence[tuple[Phase5SymbolicDecisionRecord, float, float]],
    *,
    model: Any,
    optimizer: Any,
    torch: Any,
    nn: Any,
    device: Any,
    clip_epsilon: float,
    policy_loss_weight: float,
    value_loss_weight: float,
    entropy_weight: float,
) -> float:
    records = [example[0] for example in examples]
    global_x = torch.tensor(
        [record.encoded.global_features for record in records],
        dtype=torch.float32,
        device=device,
    )
    entity_x = torch.tensor(
        [record.encoded.entity_features for record in records],
        dtype=torch.float32,
        device=device,
    )
    entity_mask = torch.tensor(
        [record.encoded.entity_mask for record in records],
        dtype=torch.float32,
        device=device,
    )
    action_x = torch.tensor(
        [record.encoded.legal_action_features for record in records],
        dtype=torch.float32,
        device=device,
    )
    action_mask = torch.tensor(
        [record.encoded.legal_action_mask for record in records],
        dtype=torch.float32,
        device=device,
    )
    previous_action_x = torch.tensor(
        [record.previous_action_features for record in records],
        dtype=torch.float32,
        device=device,
    )
    previous_action_mask = torch.tensor(
        [record.previous_action_mask for record in records],
        dtype=torch.float32,
        device=device,
    )
    old_logprobs = torch.tensor(
        [example[1] for example in examples],
        dtype=torch.float32,
        device=device,
    )
    advantages = torch.tensor(
        [example[2] for example in examples],
        dtype=torch.float32,
        device=device,
    )
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / advantages.std().clamp(min=1.0e-6)
    output = model(
        global_x,
        entity_x,
        entity_mask,
        action_x,
        action_mask,
        previous_action_x,
        previous_action_mask,
    )
    logits = output["action_logits"]
    masked_logits = logits.masked_fill(action_mask <= 0, -1.0e9)
    log_probs = nn.functional.log_softmax(masked_logits, dim=1)
    selected_log_probs = torch.zeros(len(records), dtype=torch.float32, device=device)
    row_indices: list[int] = []
    target_indices: list[int] = []
    for row_index, record in enumerate(records):
        for position in record.target_positions:
            row_indices.append(row_index)
            target_indices.append(position)
    if row_indices:
        rows = torch.tensor(row_indices, dtype=torch.long, device=device)
        targets = torch.tensor(target_indices, dtype=torch.long, device=device)
        selected_log_probs = selected_log_probs.index_add(
            0,
            rows,
            log_probs[rows, targets],
        )
    ratios = torch.exp(selected_log_probs - old_logprobs)
    clipped = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    policy_loss = -torch.minimum(ratios * advantages, clipped * advantages).mean()
    value_target_values: list[float] = []
    for record, _, advantage in examples:
        target = _record_value_target(record)
        value_target_values.append(float(target) if target is not None else float(advantage))
    value_targets = torch.tensor(value_target_values, dtype=torch.float32, device=device)
    value_loss = nn.functional.mse_loss(output["state_value"], value_targets)
    probs = torch.exp(log_probs) * action_mask
    entropy = -(probs * log_probs.masked_fill(action_mask <= 0, 0.0)).sum(dim=1).mean()
    loss = (
        float(policy_loss_weight) * policy_loss
        + float(value_loss_weight) * value_loss
        - float(entropy_weight) * entropy
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def _train_bc_only_batch(
    records: Sequence[Phase5SymbolicDecisionRecord],
    *,
    model: Any,
    optimizer: Any,
    torch: Any,
    nn: Any,
    device: Any,
) -> tuple[float, float]:
    output, action_mask = _forward_phase5_records(
        records,
        model=model,
        torch=torch,
        device=device,
    )
    selected_logprobs = _sequential_selected_logprobs(
        records,
        logits=output["action_logits"],
        action_mask=action_mask,
        torch=torch,
        nn=nn,
        use_record_policy=False,
    )
    weights = torch.tensor(
        [record.weight for record in records],
        dtype=torch.float32,
        device=device,
    )
    loss = -(selected_logprobs * weights).sum() / weights.sum().clamp(min=1.0e-6)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    masked_logits = output["action_logits"].masked_fill(action_mask <= 0, -1.0e9)
    predictions = masked_logits.detach().argmax(dim=1).tolist()
    correct = sum(
        int(int(prediction) in set(record.target_positions))
        for prediction, record in zip(predictions, records, strict=True)
    )
    return float(loss.detach().item()), correct / max(1, len(records))


def _train_bc_ppo_batch(
    rule_records: Sequence[Phase5SymbolicDecisionRecord],
    on_policy_examples: Sequence[tuple[Phase5SymbolicDecisionRecord, float, float]],
    *,
    model: Any,
    optimizer: Any,
    torch: Any,
    nn: Any,
    device: Any,
    clip_epsilon: float,
    bc_loss_weight: float,
    ppo_policy_loss_weight: float,
    ppo_value_loss_weight: float,
    entropy_weight: float,
    collect_gradient_diagnostics: bool = False,
    advantage_normalization: str = "batch",
    advantage_normalization_mean: float = 0.0,
    advantage_normalization_std: float = 1.0,
    value_backprop_scope: str = "shared",
) -> _BCPPOBatchMetrics:
    on_policy_records = [example[0] for example in on_policy_examples]
    on_policy_output, on_policy_action_mask = _forward_phase5_records(
        on_policy_records,
        model=model,
        torch=torch,
        device=device,
    )

    if rule_records:
        rule_output, rule_action_mask = _forward_phase5_records(
            rule_records,
            model=model,
            torch=torch,
            device=device,
        )
        rule_logprobs = _sequential_selected_logprobs(
            rule_records,
            logits=rule_output["action_logits"],
            action_mask=rule_action_mask,
            torch=torch,
            nn=nn,
            use_record_policy=False,
        )
        rule_weights = torch.tensor(
            [record.weight for record in rule_records],
            dtype=torch.float32,
            device=device,
        )
        bc_loss = -(rule_logprobs * rule_weights).sum() / rule_weights.sum().clamp(
            min=1.0e-6
        )
        masked_rule_logits = rule_output["action_logits"].masked_fill(
            rule_action_mask <= 0,
            -1.0e9,
        )
        rule_predictions = masked_rule_logits.detach().argmax(dim=1).tolist()
        bc_correct = sum(
            int(int(prediction) in set(record.target_positions))
            for prediction, record in zip(rule_predictions, rule_records, strict=True)
        )
        bc_accuracy = bc_correct / len(rule_records)
    else:
        bc_loss = on_policy_output["action_logits"].sum() * 0.0
        bc_accuracy = 0.0

    selected_logprobs = _sequential_selected_logprobs(
        on_policy_records,
        logits=on_policy_output["action_logits"],
        action_mask=on_policy_action_mask,
        torch=torch,
        nn=nn,
        use_record_policy=True,
    )
    old_logprobs = torch.tensor(
        [example[1] for example in on_policy_examples],
        dtype=torch.float32,
        device=device,
    )
    advantages = torch.tensor(
        [example[2] for example in on_policy_examples],
        dtype=torch.float32,
        device=device,
    )
    if advantage_normalization == "batch" and advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / advantages.std().clamp(
            min=1.0e-6
        )
    elif advantage_normalization == "global":
        advantages = (
            advantages - float(advantage_normalization_mean)
        ) / max(1.0e-6, float(advantage_normalization_std))
    ratios = torch.exp(selected_logprobs - old_logprobs)
    clipped_ratios = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    ppo_policy_loss = -torch.minimum(
        ratios * advantages,
        clipped_ratios * advantages,
    ).mean()
    value_targets = torch.tensor(
        [
            float(target) if (target := _record_value_target(record)) is not None else advantage
            for record, advantage in zip(on_policy_records, advantages.tolist(), strict=True)
        ],
        dtype=torch.float32,
        device=device,
    )
    value_predictions = (
        model.value_head(on_policy_output["state_embedding"].detach()).squeeze(-1)
        if value_backprop_scope == "head-only"
        else on_policy_output["state_value"]
    )
    ppo_value_loss = nn.functional.mse_loss(value_predictions, value_targets)
    entropy = _first_action_policy_entropy(
        on_policy_records,
        logits=on_policy_output["action_logits"],
        action_mask=on_policy_action_mask,
        torch=torch,
        nn=nn,
    )
    loss = (
        float(bc_loss_weight) * bc_loss
        + float(ppo_policy_loss_weight) * ppo_policy_loss
        + float(ppo_value_loss_weight) * ppo_value_loss
        - float(entropy_weight) * entropy
    )
    gradient_metrics = {
        "bc_gradient_norm": 0.0,
        "ppo_policy_gradient_norm": 0.0,
        "ppo_policy_shared_gradient_norm": 0.0,
        "ppo_value_gradient_norm": 0.0,
        "ppo_value_shared_gradient_norm": 0.0,
        "entropy_gradient_norm": 0.0,
        "bc_ppo_policy_gradient_cosine": 0.0,
        "ppo_policy_value_gradient_cosine": 0.0,
        "ppo_policy_value_shared_gradient_cosine": 0.0,
    }
    if collect_gradient_diagnostics:
        parameters = tuple(
            parameter for parameter in model.parameters() if parameter.requires_grad
        )
        bc_gradients, bc_gradient_norm = _phase5_objective_gradients(
            float(bc_loss_weight) * bc_loss,
            parameters=parameters,
            torch=torch,
        )
        policy_gradients, ppo_policy_gradient_norm = _phase5_objective_gradients(
            float(ppo_policy_loss_weight) * ppo_policy_loss,
            parameters=parameters,
            torch=torch,
        )
        value_gradients, ppo_value_gradient_norm = _phase5_objective_gradients(
            float(ppo_value_loss_weight) * ppo_value_loss,
            parameters=parameters,
            torch=torch,
        )
        entropy_gradients, entropy_gradient_norm = _phase5_objective_gradients(
            -float(entropy_weight) * entropy,
            parameters=parameters,
            torch=torch,
        )
        shared_parameter_ids = _phase5_shared_actor_parameter_ids(model)
        ppo_policy_shared_gradient_norm = _phase5_filtered_gradient_norm(
            policy_gradients,
            parameters=parameters,
            parameter_ids=shared_parameter_ids,
        )
        ppo_value_shared_gradient_norm = _phase5_filtered_gradient_norm(
            value_gradients,
            parameters=parameters,
            parameter_ids=shared_parameter_ids,
        )
        gradient_metrics = {
            "bc_gradient_norm": bc_gradient_norm,
            "ppo_policy_gradient_norm": ppo_policy_gradient_norm,
            "ppo_policy_shared_gradient_norm": ppo_policy_shared_gradient_norm,
            "ppo_value_gradient_norm": ppo_value_gradient_norm,
            "ppo_value_shared_gradient_norm": ppo_value_shared_gradient_norm,
            "entropy_gradient_norm": entropy_gradient_norm,
            "bc_ppo_policy_gradient_cosine": _phase5_gradient_cosine(
                bc_gradients,
                policy_gradients,
                first_norm=bc_gradient_norm,
                second_norm=ppo_policy_gradient_norm,
            ),
            "ppo_policy_value_gradient_cosine": _phase5_gradient_cosine(
                policy_gradients,
                value_gradients,
                first_norm=ppo_policy_gradient_norm,
                second_norm=ppo_value_gradient_norm,
            ),
            "ppo_policy_value_shared_gradient_cosine": _phase5_gradient_cosine(
                policy_gradients,
                value_gradients,
                first_norm=ppo_policy_shared_gradient_norm,
                second_norm=ppo_value_shared_gradient_norm,
                parameters=parameters,
                parameter_ids=shared_parameter_ids,
            ),
        }
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    clip_fraction = ((ratios - 1.0).abs() > clip_epsilon).float().mean()
    return _BCPPOBatchMetrics(
        loss=float(loss.detach().item()),
        bc_loss=float(bc_loss.detach().item()),
        bc_accuracy=float(bc_accuracy),
        ppo_policy_loss=float(ppo_policy_loss.detach().item()),
        ppo_value_loss=float(ppo_value_loss.detach().item()),
        entropy=float(entropy.detach().item()),
        ratio=float(ratios.detach().mean().item()),
        clip_fraction=float(clip_fraction.detach().item()),
        **gradient_metrics,
    )


def _phase5_objective_gradients(
    objective: Any,
    *,
    parameters: Sequence[Any],
    torch: Any,
) -> tuple[tuple[Any | None, ...], float]:
    gradients = torch.autograd.grad(
        objective,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    squared_norm = torch.zeros((), dtype=torch.float32, device=objective.device)
    for gradient in gradients:
        if gradient is not None:
            squared_norm = squared_norm + gradient.detach().float().pow(2).sum()
    return gradients, float(torch.sqrt(squared_norm).item())


def _phase5_gradient_cosine(
    first: Sequence[Any | None],
    second: Sequence[Any | None],
    *,
    first_norm: float,
    second_norm: float,
    parameters: Sequence[Any] | None = None,
    parameter_ids: set[int] | None = None,
) -> float:
    if first_norm <= 0.0 or second_norm <= 0.0:
        return 0.0
    dot = 0.0
    for index, (first_gradient, second_gradient) in enumerate(
        zip(first, second, strict=True)
    ):
        if (
            parameter_ids is not None
            and parameters is not None
            and id(parameters[index]) not in parameter_ids
        ):
            continue
        if first_gradient is not None and second_gradient is not None:
            dot += float(
                (first_gradient.detach().float() * second_gradient.detach().float())
                .sum()
                .item()
            )
    return dot / (first_norm * second_norm)


def _phase5_shared_actor_parameter_ids(model: Any) -> set[int]:
    modules = (
        model.global_encoder,
        model.entity_encoder,
        model.entity_core,
        model.action_encoder,
        model.turn_core,
    )
    return {
        id(parameter)
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    }


def _phase5_filtered_gradient_norm(
    gradients: Sequence[Any | None],
    *,
    parameters: Sequence[Any],
    parameter_ids: set[int],
) -> float:
    squared_norm = 0.0
    for parameter, gradient in zip(parameters, gradients, strict=True):
        if id(parameter) in parameter_ids and gradient is not None:
            squared_norm += float(gradient.detach().float().pow(2).sum().item())
    return math.sqrt(squared_norm)


def _forward_phase5_records(
    records: Sequence[Phase5SymbolicDecisionRecord],
    *,
    model: Any,
    torch: Any,
    device: Any,
) -> tuple[dict[str, Any], Any]:
    global_x = torch.tensor(
        [record.encoded.global_features for record in records],
        dtype=torch.float32,
        device=device,
    )
    entity_x = torch.tensor(
        [record.encoded.entity_features for record in records],
        dtype=torch.float32,
        device=device,
    )
    entity_mask = torch.tensor(
        [record.encoded.entity_mask for record in records],
        dtype=torch.float32,
        device=device,
    )
    action_x = torch.tensor(
        [record.encoded.legal_action_features for record in records],
        dtype=torch.float32,
        device=device,
    )
    action_mask = torch.tensor(
        [record.encoded.legal_action_mask for record in records],
        dtype=torch.float32,
        device=device,
    )
    previous_action_x = torch.tensor(
        [record.previous_action_features for record in records],
        dtype=torch.float32,
        device=device,
    )
    previous_action_mask = torch.tensor(
        [record.previous_action_mask for record in records],
        dtype=torch.float32,
        device=device,
    )
    return (
        model(
            global_x,
            entity_x,
            entity_mask,
            action_x,
            action_mask,
            previous_action_x,
            previous_action_mask,
        ),
        action_mask,
    )


def _sequential_selected_logprobs(
    records: Sequence[Phase5SymbolicDecisionRecord],
    *,
    logits: Any,
    action_mask: Any,
    torch: Any,
    nn: Any,
    use_record_policy: bool,
) -> Any:
    selected_logprobs: list[Any] = []
    for row_index, record in enumerate(records):
        epsilon, temperature = (
            _record_policy_parameters(record) if use_record_policy else (0.0, 1.0)
        )
        remaining = action_mask[row_index] > 0
        total = torch.zeros((), dtype=torch.float32, device=logits.device)
        for position in record.target_positions:
            if position < 0 or position >= remaining.numel() or not bool(remaining[position]):
                raise ValueError("Phase 5 target position is not available in policy mask.")
            remaining_count = remaining.float().sum().clamp(min=1.0)
            masked_logits = (logits[row_index] / temperature).masked_fill(
                ~remaining,
                -1.0e9,
            )
            policy_probabilities = nn.functional.softmax(masked_logits, dim=0)
            selected_probability = (
                (1.0 - epsilon) * policy_probabilities[position]
                + epsilon / remaining_count
            )
            total = total + torch.log(selected_probability.clamp(min=1.0e-12))
            remaining = remaining.clone()
            remaining[position] = False
        selected_logprobs.append(total)
    return torch.stack(selected_logprobs)


def _first_action_policy_entropy(
    records: Sequence[Phase5SymbolicDecisionRecord],
    *,
    logits: Any,
    action_mask: Any,
    torch: Any,
    nn: Any,
) -> Any:
    entropies: list[Any] = []
    for row_index, record in enumerate(records):
        epsilon, temperature = _record_policy_parameters(record)
        legal = action_mask[row_index] > 0
        legal_count = legal.float().sum().clamp(min=1.0)
        masked_logits = (logits[row_index] / temperature).masked_fill(legal <= 0, -1.0e9)
        policy_probabilities = nn.functional.softmax(masked_logits, dim=0)
        probabilities = (
            (1.0 - epsilon) * policy_probabilities
            + epsilon * legal.float() / legal_count
        )
        log_probabilities = torch.log(probabilities.clamp(min=1.0e-12))
        entropies.append(-(probabilities * log_probabilities * legal.float()).sum())
    return torch.stack(entropies).mean()


def _record_policy_parameters(
    record: Phase5SymbolicDecisionRecord,
) -> tuple[float, float]:
    metadata = record.metadata
    mode = str(metadata.get("policy_mode", ""))
    epsilon = (
        max(0.0, min(1.0, float(metadata.get("policy_epsilon", 0.0))))
        if mode == "epsilon_mixture"
        else 0.0
    )
    temperature = max(1.0e-6, float(metadata.get("policy_temperature", 1.0)))
    return epsilon, temperature


def _changed_pairwise_loss(
    records: Sequence[Phase5SymbolicDecisionRecord],
    *,
    logits: Any,
    torch: Any,
    nn: Any,
    device: Any,
    margin: float,
    negative_mode: str,
) -> Any | None:
    row_indices: list[int] = []
    positive_indices: list[int] = []
    negative_indices: list[int] = []
    weights: list[float] = []
    for row_index, record in enumerate(records):
        if not record.changed:
            continue
        for positive, negative in phase5_symbolic_pairwise_positions(
            record,
            negative_mode=negative_mode,
        ):
            row_indices.append(row_index)
            positive_indices.append(positive)
            negative_indices.append(negative)
            weights.append(record.weight)
    if not row_indices:
        return None
    rows = torch.tensor(row_indices, dtype=torch.long, device=device)
    positives = torch.tensor(positive_indices, dtype=torch.long, device=device)
    negatives = torch.tensor(negative_indices, dtype=torch.long, device=device)
    pair_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    positive_scores = logits[rows, positives]
    negative_scores = logits[rows, negatives]
    pair_losses = nn.functional.softplus(
        float(margin) - (positive_scores - negative_scores)
    )
    return (pair_losses * pair_weights).mean()


def _baseline_target_positions(record: Phase5SymbolicDecisionRecord) -> list[int]:
    baseline_indices = record.metadata.get("phase5_baseline_indices", [])
    index_to_position = {
        action_index: position
        for position, action_index in enumerate(record.encoded.legal_action_indices)
        if action_index >= 0
    }
    positions: list[int] = []
    for raw_index in baseline_indices or []:
        try:
            position = index_to_position[int(raw_index)]
        except (KeyError, TypeError, ValueError):
            continue
        if position not in positions:
            positions.append(position)
    return positions


def _valid_indices(values: Sequence[int] | None, size: int) -> list[int]:
    output: list[int] = []
    for value in values or []:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= index < size and index not in output:
            output.append(index)
    return output


def _action_value_targets(
    encoded: EncodedPhase5Turn,
    target_positions: Sequence[int],
    target: float | None,
) -> tuple[list[float] | None, list[float] | None]:
    if target is None:
        return None, None
    width = len(encoded.legal_action_mask)
    targets = [0.0] * width
    mask = [0.0] * width
    for position in target_positions:
        if 0 <= position < width and encoded.legal_action_mask[position] > 0.0:
            targets[position] = float(target)
            mask[position] = 1.0
    if not any(mask):
        return None, None
    return targets, mask


def _tactical_targets(
    encoded: EncodedPhase5Turn,
) -> tuple[list[float], list[float]]:
    targets: list[float] = []
    mask: list[float] = []
    for row, mask_value in zip(
        encoded.legal_action_features,
        encoded.legal_action_mask,
        strict=False,
    ):
        if mask_value > 0.0:
            targets.append(float(row[0]) if row else 0.0)
            mask.append(1.0)
        else:
            targets.append(0.0)
            mask.append(0.0)
    return targets, mask


def _record_value_target(record: Phase5SymbolicDecisionRecord) -> float | None:
    if record.value_target is not None:
        return float(record.value_target)
    metadata = record.metadata
    if not any(key in metadata for key in ("winner", "leader", "finished")):
        return None
    return reward_from_result_metadata(metadata)


def _player_from_board(
    board: dict[str, Any],
    *,
    prefix: str,
    player_index: int,
    is_us: bool,
) -> PlayerState:
    active = _entity_from_card_state(
        board.get(f"{prefix}_active_card"),
        owner=player_index,
        zone="ACTIVE",
        slot=0,
        fallback_id=board.get(f"{prefix}_active_id"),
        fallback_hp=board.get(f"{prefix}_active_hp"),
        fallback_max_hp=board.get(f"{prefix}_active_max_hp"),
        fallback_energy_count=board.get(f"{prefix}_active_energy_count"),
        fallback_is_ex=board.get(f"{prefix}_active_is_ex"),
        fallback_is_mega_ex=board.get(f"{prefix}_active_is_mega_ex"),
    )
    bench = []
    for index, card_state in enumerate(list(board.get(f"{prefix}_bench_cards", []) or [])):
        bench_entity = _entity_from_card_state(
            card_state,
            owner=player_index,
            zone="BENCH",
            slot=index,
        )
        if bench_entity is not None:
            bench.append(bench_entity)
    if not bench:
        for index in range(5):
            card_id = board.get(f"{prefix}_bench_{index}_id")
            if card_id is None:
                continue
            bench_entity = _entity_from_card_state(
                {},
                owner=player_index,
                zone="BENCH",
                slot=index,
                fallback_id=card_id,
                fallback_hp=board.get(f"{prefix}_bench_{index}_hp"),
                fallback_energy_count=board.get(f"{prefix}_bench_{index}_energy_count"),
            )
            if bench_entity is not None:
                bench.append(bench_entity)
    discard_top = _entity_from_card_state(
        board.get(f"{prefix}_discard_top_card"),
        owner=player_index,
        zone="DISCARD",
        slot=0,
    )
    return PlayerState(
        player_index=player_index,
        is_us=is_us,
        hand_count=_int(board.get(f"{prefix}_hand_count"), 0),
        deck_count=_int(board.get(f"{prefix}_deck_count"), 0),
        prize_count=_int(board.get(f"{prefix}_prizes"), 0),
        active=(active,) if active is not None else (),
        bench=tuple(bench),
        discard=(discard_top,) if discard_top is not None else (),
    )


def _stadium_entities(board: dict[str, Any]) -> tuple[CardEntity, ...]:
    stadium = _entity_from_card_state(
        board.get("stadium_card"),
        owner=None,
        zone="STADIUM",
        slot=0,
    )
    return (stadium,) if stadium is not None else ()


def _entity_from_card_state(
    card_state: Any,
    *,
    owner: int | None,
    zone: str,
    slot: int | None,
    fallback_id: Any = None,
    fallback_hp: Any = None,
    fallback_max_hp: Any = None,
    fallback_energy_count: Any = None,
    fallback_is_ex: Any = None,
    fallback_is_mega_ex: Any = None,
) -> CardEntity | None:
    payload = card_state if isinstance(card_state, dict) else {}
    card_id = _optional_int(payload.get("id", fallback_id))
    if card_id is None:
        return None
    max_hp = _int(payload.get("max_hp", fallback_max_hp), 0)
    hp = _int(payload.get("hp", fallback_hp), max_hp)
    return CardEntity(
        card_id=card_id,
        name="",
        owner=owner,
        zone=zone,
        slot=slot,
        known=True,
        hp=hp,
        max_hp=max_hp,
        energy_count=_int(payload.get("energy_count", fallback_energy_count), 0),
        tool_count=_int(payload.get("tool_count"), 0),
        is_ex=bool(payload.get("is_ex", fallback_is_ex or False)),
        is_mega_ex=bool(payload.get("is_mega_ex", fallback_is_mega_ex or False)),
    )


def _pad_previous_actions(
    rows: Sequence[Sequence[float]],
    *,
    max_previous_actions: int,
    action_dim: int,
) -> tuple[list[list[float]], list[float]]:
    clipped = [list(row) for row in rows[-max_previous_actions:]]
    mask = [1.0] * len(clipped)
    clipped.extend([[0.0] * action_dim for _ in range(max_previous_actions - len(clipped))])
    mask.extend([0.0] * (max_previous_actions - len(mask)))
    return clipped, mask


def _turn_context_key(frame: DecisionFrame) -> tuple[Any, ...]:
    metadata = frame.reward_metadata
    board = frame.board
    return (
        metadata.get("game_index"),
        metadata.get("player_index", board.get("your_index")),
        metadata.get("deck_index"),
        board.get("turn"),
    )


def _entity_dim(encoder: Phase5SymbolicEncoder) -> int:
    state = decision_frame_to_state(_empty_frame())
    encoded = encoder.encode(state, [])
    return len(encoded.entity_features[0])


def _action_dim(encoder: Phase5SymbolicEncoder) -> int:
    state = decision_frame_to_state(_empty_frame())
    encoded = encoder.encode(state, [])
    return len(encoded.legal_action_features[0])


def _empty_frame() -> DecisionFrame:
    return DecisionFrame(
        select_type="MAIN",
        context="MAIN",
        min_count=1,
        max_count=1,
        target_count=1,
        legal_options=[],
        rule_selected_indices=[],
        board={},
        board_image=[],
    )


def _require_torch() -> tuple[Any, Any]:
    if not TORCH_AVAILABLE:
        raise Phase5PolicyUnavailable(
            "PyTorch is not installed. Train Phase 5 symbolic policy models on "
            "ERAWAN or install the rl extra locally."
        )
    import torch
    from torch import nn

    return torch, nn


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None
