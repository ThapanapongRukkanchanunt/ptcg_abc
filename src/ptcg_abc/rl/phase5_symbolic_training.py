from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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
from ptcg_abc.rl.records import ActionFrame, DecisionFrame
from ptcg_abc.rl.rewards import reward_from_result_metadata


PHASE5_SYMBOLIC_DATASET_FORMAT = "ptcg_abc_phase5_symbolic_decision_v1"
PHASE5_SYMBOLIC_PAIRWISE_NEGATIVES = ("all", "baseline")


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
    source_format: str = PHASE5_SYMBOLIC_DATASET_FORMAT

    def to_dict(self) -> dict[str, Any]:
        return {
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
        action_rows = record.encoded.legal_action_features
        for position in record.target_positions:
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
) -> Phase5SymbolicDecisionRecord | None:
    state = decision_frame_to_state(frame)
    legal_actions = decision_frame_to_legal_actions(frame)
    encoded = encoder.encode(state, legal_actions)
    target_indices = phase5_target_indices(frame, target_source=target_source)
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
    return Phase5SymbolicDecisionRecord(
        encoded=encoded,
        target_indices=target_indices,
        target_positions=target_positions,
        previous_action_features=previous_rows,
        previous_action_mask=previous_mask,
        weight=float(changed_weight if changed else unchanged_weight),
        changed=changed,
        metadata=dict(frame.reward_metadata)
        | {
            "select_type": frame.select_type,
            "context": frame.context,
            "target_source": target_source,
        },
    )


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
        rewards = torch.tensor(
            [reward_from_result_metadata(record.metadata) for record in records],
            dtype=torch.float32,
            device=device,
        )
        value_loss = nn.functional.mse_loss(output["state_value"], rewards)
        loss = loss + float(value_loss_weight) * value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    predictions = logits.detach().argmax(dim=1)
    correct = int((predictions == targets).sum().item())
    return float(loss.detach().item()), correct


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
