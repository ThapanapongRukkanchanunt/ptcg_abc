from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from ptcg_abc.rl.phase5_symbolic_training import (
    Phase5GeneralistTrainingSummary,
    Phase5PPOTrainingSummary,
    train_phase5_generalist_policy,
    train_phase5_ppo_policy_from_trajectories,
)
from ptcg_abc.rl.workflow import (
    PHASE5_SELFPLAY_DECK_POOL_LEAGUE_13,
    SelfPlaySummary,
    phase5_selfplay_prepared_decks,
    rollout_selfplay_games,
)


PHASE5_ALPHA_DECK_INDICES = tuple(range(1, 14))
PHASE5_ALPHA_RAW_TRAIN_DIRNAME = "raw_train"
PHASE5_ALPHA_RUN_NAME = "phase5_league_alpha"


@dataclass(frozen=True)
class AlphaLeagueDataPolicy:
    capacity_gb: int = 400
    reference_shard_gb: int = 30
    raw_training_data_retention: str = "delete_after_successful_update"
    raw_evaluation_trajectories: str = "disabled_by_default"
    active_raw_windows: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AlphaRuleBootstrapSummary:
    iteration: int
    iteration_dir: str
    raw_train_dir: str
    trajectory_path: str
    report_path: str
    games_per_pair: int
    games_requested: int
    selfplay: dict[str, Any]
    data_policy: dict[str, Any]
    cleanup_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AlphaLeagueIterationSummary:
    iteration: int
    iteration_dir: str
    raw_train_dir: str
    trajectory_path: str
    report_path: str
    agent: str
    specialist_model_dir: str
    specialist_model_paths: dict[str, str]
    games_per_deck: int
    games_requested: int
    selfplay: dict[str, Any]
    data_policy: dict[str, Any]
    cleanup_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeckSpecialistTrainingSummary:
    iteration: int | None
    deck_indices: list[int]
    checkpoint_dir: str
    report_dir: str
    decision_dataset_path: str | None
    selfplay_dataset_paths: list[str]
    summaries: dict[str, dict[str, Any]]
    data_policy: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeckSpecialistPPOTrainingSummary:
    iteration: int | None
    deck_indices: list[int]
    source_checkpoint_dir: str
    output_checkpoint_dir: str
    report_dir: str
    trajectory_dataset_paths: list[str]
    summaries: dict[str, dict[str, Any]]
    data_policy: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AlphaRawCleanupSummary:
    iteration_dir: str
    raw_train_dir: str
    removed: bool
    files_removed: int
    bytes_removed: int
    cleanup_report_path: str
    data_policy: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def alpha_iteration_dir(root: Path, iteration: int) -> Path:
    if iteration < 0:
        raise ValueError("Alpha league iteration must be non-negative.")
    return root / "iterations" / f"iter-{iteration:04d}"


def generate_phase5_alpha_rule_bootstrap(
    *,
    sample_dir: Path,
    iteration_dir: Path,
    report_path: Path,
    games_per_pair: int = 2,
    max_steps: int = 600,
    deck_indices: Sequence[int] = PHASE5_ALPHA_DECK_INDICES,
    game_offset: int = 0,
    allow_existing_raw: bool = False,
) -> AlphaRuleBootstrapSummary:
    if games_per_pair <= 0:
        raise ValueError("games_per_pair must be positive.")
    selected_indices = _normal_deck_indices(deck_indices)
    raw_train_dir = iteration_dir / PHASE5_ALPHA_RAW_TRAIN_DIRNAME
    _ensure_raw_slot_available(raw_train_dir, allow_existing_raw=allow_existing_raw)
    raw_train_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = raw_train_dir / "phase5_alpha_rule_bootstrap.jsonl"
    pair_count = len(selected_indices) * len(selected_indices)
    games_requested = pair_count * games_per_pair
    selfplay = rollout_selfplay_games(
        sample_dir=sample_dir,
        output_path=trajectory_path,
        agent_kind="rule",
        model_path=None,
        games=games_requested,
        game_offset=game_offset,
        max_steps=max_steps,
        deck_pool=PHASE5_SELFPLAY_DECK_POOL_LEAGUE_13,
        selfplay_deck_indices=selected_indices,
    )
    summary = AlphaRuleBootstrapSummary(
        iteration=_iteration_number(iteration_dir),
        iteration_dir=iteration_dir.as_posix(),
        raw_train_dir=raw_train_dir.as_posix(),
        trajectory_path=trajectory_path.as_posix(),
        report_path=report_path.as_posix(),
        games_per_pair=games_per_pair,
        games_requested=games_requested,
        selfplay=_selfplay_payload(selfplay),
        data_policy=AlphaLeagueDataPolicy().to_dict(),
    )
    _write_json(report_path, summary.to_dict())
    return summary


def generate_phase5_alpha_league_iteration(
    *,
    sample_dir: Path,
    iteration_dir: Path,
    report_path: Path,
    specialist_model_dir: Path,
    games_per_deck: int = 100,
    max_steps: int = 600,
    deck_indices: Sequence[int] = PHASE5_ALPHA_DECK_INDICES,
    game_offset: int = 0,
    allow_existing_raw: bool = False,
    agent_kind: str = "phase5-full",
    search_trace_path: Path | None = None,
    search_trace_game_limit: int = 3,
    search_config: Any | None = None,
) -> AlphaLeagueIterationSummary:
    if games_per_deck <= 0:
        raise ValueError("games_per_deck must be positive.")
    selected_indices = _normal_deck_indices(deck_indices)
    specialist_paths = _validate_specialist_model_dir(
        specialist_model_dir,
        selected_indices,
    )
    raw_train_dir = iteration_dir / PHASE5_ALPHA_RAW_TRAIN_DIRNAME
    _ensure_raw_slot_available(raw_train_dir, allow_existing_raw=allow_existing_raw)
    raw_train_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = raw_train_dir / "phase5_alpha_league_selfplay.jsonl"
    games_requested = len(selected_indices) * games_per_deck
    selfplay = rollout_selfplay_games(
        sample_dir=sample_dir,
        output_path=trajectory_path,
        agent_kind=agent_kind,
        model_path=None,
        games=games_requested,
        game_offset=game_offset,
        max_steps=max_steps,
        deck_pool=PHASE5_SELFPLAY_DECK_POOL_LEAGUE_13,
        selfplay_deck_indices=selected_indices,
        specialist_model_dir=specialist_model_dir,
        search_trace_path=search_trace_path,
        search_trace_game_limit=search_trace_game_limit,
        search_config=search_config,
    )
    summary = AlphaLeagueIterationSummary(
        iteration=_iteration_number(iteration_dir),
        iteration_dir=iteration_dir.as_posix(),
        raw_train_dir=raw_train_dir.as_posix(),
        trajectory_path=trajectory_path.as_posix(),
        report_path=report_path.as_posix(),
        agent=agent_kind,
        specialist_model_dir=specialist_model_dir.as_posix(),
        specialist_model_paths={
            str(index): path.as_posix()
            for index, path in sorted(specialist_paths.items())
        },
        games_per_deck=games_per_deck,
        games_requested=games_requested,
        selfplay=_selfplay_payload(selfplay),
        data_policy=AlphaLeagueDataPolicy().to_dict(),
    )
    _write_json(report_path, summary.to_dict())
    return summary


def train_phase5_deck_specialists(
    *,
    decision_dataset_path: Path | None,
    selfplay_dataset_paths: Sequence[Path],
    checkpoint_dir: Path,
    report_dir: Path,
    aggregate_report_path: Path,
    iteration: int | None = None,
    deck_indices: Sequence[int] = PHASE5_ALPHA_DECK_INDICES,
    allow_empty_decks: bool = False,
    **trainer_kwargs: Any,
) -> DeckSpecialistTrainingSummary:
    if decision_dataset_path is None and not selfplay_dataset_paths:
        raise ValueError("Provide at least one decision or self-play dataset.")
    selected_indices = _normal_deck_indices(deck_indices)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}
    for deck_index in selected_indices:
        checkpoint_path = checkpoint_dir / f"deck-{deck_index:02d}.pt"
        report_path = report_dir / f"deck-{deck_index:02d}_train_report.json"
        summary: Phase5GeneralistTrainingSummary = train_phase5_generalist_policy(
            decision_dataset_path=decision_dataset_path,
            selfplay_dataset_paths=selfplay_dataset_paths,
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            deck_index_filter=deck_index,
            **trainer_kwargs,
        )
        payload = summary.to_dict()
        total_examples = (
            summary.decision_examples + summary.rule_examples + summary.selfplay_examples
        )
        if total_examples <= 0 and not allow_empty_decks:
            raise ValueError(f"Deck {deck_index} produced zero specialist examples.")
        summaries[str(deck_index)] = payload
    aggregate = DeckSpecialistTrainingSummary(
        iteration=iteration,
        deck_indices=selected_indices,
        checkpoint_dir=checkpoint_dir.as_posix(),
        report_dir=report_dir.as_posix(),
        decision_dataset_path=(
            decision_dataset_path.as_posix() if decision_dataset_path is not None else None
        ),
        selfplay_dataset_paths=[path.as_posix() for path in selfplay_dataset_paths],
        summaries=summaries,
        data_policy=AlphaLeagueDataPolicy().to_dict(),
    )
    _write_json(aggregate_report_path, aggregate.to_dict())
    return aggregate


def train_phase5_deck_specialists_ppo(
    *,
    trajectory_dataset_paths: Sequence[Path],
    source_checkpoint_dir: Path,
    output_checkpoint_dir: Path,
    report_dir: Path,
    aggregate_report_path: Path,
    iteration: int | None = None,
    deck_indices: Sequence[int] = PHASE5_ALPHA_DECK_INDICES,
    allow_empty_decks: bool = False,
    require_on_policy: bool = True,
    **trainer_kwargs: Any,
) -> DeckSpecialistPPOTrainingSummary:
    if not trajectory_dataset_paths:
        raise ValueError("Provide at least one trajectory dataset.")
    selected_indices = _normal_deck_indices(deck_indices)
    for path in trajectory_dataset_paths:
        if not path.exists():
            raise ValueError(f"Trajectory dataset not found at {path}.")
    output_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}
    for deck_index in selected_indices:
        source_checkpoint_path = source_checkpoint_dir / f"deck-{deck_index:02d}.pt"
        if not source_checkpoint_path.exists():
            raise ValueError(f"Missing source specialist checkpoint: {source_checkpoint_path}.")
        output_checkpoint_path = output_checkpoint_dir / f"deck-{deck_index:02d}.pt"
        report_path = report_dir / f"deck-{deck_index:02d}_ppo_report.json"
        summary: Phase5PPOTrainingSummary = train_phase5_ppo_policy_from_trajectories(
            trajectory_dataset_paths=trajectory_dataset_paths,
            checkpoint_path=source_checkpoint_path,
            output_checkpoint_path=output_checkpoint_path,
            report_path=report_path,
            deck_index_filter=deck_index,
            require_on_policy=require_on_policy,
            **trainer_kwargs,
        )
        if summary.examples <= 0 and not allow_empty_decks:
            raise ValueError(f"Deck {deck_index} produced zero PPO examples.")
        summaries[str(deck_index)] = summary.to_dict()
    aggregate = DeckSpecialistPPOTrainingSummary(
        iteration=iteration,
        deck_indices=selected_indices,
        source_checkpoint_dir=source_checkpoint_dir.as_posix(),
        output_checkpoint_dir=output_checkpoint_dir.as_posix(),
        report_dir=report_dir.as_posix(),
        trajectory_dataset_paths=[path.as_posix() for path in trajectory_dataset_paths],
        summaries=summaries,
        data_policy=AlphaLeagueDataPolicy().to_dict(),
    )
    _write_json(aggregate_report_path, aggregate.to_dict())
    return aggregate


def cleanup_phase5_alpha_raw_train(
    *,
    iteration_dir: Path,
    cleanup_report_path: Path,
    update_report_path: Path | None = None,
    require_update_report: bool = True,
) -> AlphaRawCleanupSummary:
    raw_train_dir = iteration_dir / PHASE5_ALPHA_RAW_TRAIN_DIRNAME
    if raw_train_dir.name != PHASE5_ALPHA_RAW_TRAIN_DIRNAME:
        raise ValueError("Refusing to clean a path that is not named raw_train.")
    if require_update_report and (update_report_path is None or not update_report_path.exists()):
        raise ValueError("Refusing to remove raw data before an update report exists.")
    files, bytes_total = _tree_size(raw_train_dir)
    removed = False
    if raw_train_dir.exists():
        shutil.rmtree(raw_train_dir)
        removed = True
    summary = AlphaRawCleanupSummary(
        iteration_dir=iteration_dir.as_posix(),
        raw_train_dir=raw_train_dir.as_posix(),
        removed=removed,
        files_removed=files,
        bytes_removed=bytes_total,
        cleanup_report_path=cleanup_report_path.as_posix(),
        data_policy=AlphaLeagueDataPolicy().to_dict(),
    )
    _write_json(cleanup_report_path, summary.to_dict())
    return summary


def _ensure_raw_slot_available(raw_train_dir: Path, *, allow_existing_raw: bool) -> None:
    if allow_existing_raw:
        return
    if raw_train_dir.exists() and any(raw_train_dir.iterdir()):
        raise ValueError(
            f"Raw training data already exists at {raw_train_dir}. "
            "Clean it after a successful update before launching another iteration, "
            "or pass the explicit resume/overwrite flag."
        )


def _normal_deck_indices(deck_indices: Sequence[int]) -> list[int]:
    valid = {deck.index for deck in phase5_selfplay_prepared_decks(PHASE5_SELFPLAY_DECK_POOL_LEAGUE_13)}
    output: list[int] = []
    for value in deck_indices:
        index = int(value)
        if index not in valid:
            raise ValueError(f"Unknown Phase 5 league deck index: {index}.")
        if index not in output:
            output.append(index)
    if not output:
        raise ValueError("At least one Phase 5 league deck index is required.")
    return output


def _validate_specialist_model_dir(
    specialist_model_dir: Path,
    deck_indices: Sequence[int],
) -> dict[int, Path]:
    missing: list[Path] = []
    paths: dict[int, Path] = {}
    for deck_index in deck_indices:
        path = specialist_model_dir / f"deck-{deck_index:02d}.pt"
        paths[deck_index] = path
        if not path.exists():
            missing.append(path)
    if missing:
        preview = ", ".join(path.as_posix() for path in missing[:3])
        suffix = "..." if len(missing) > 3 else ""
        raise ValueError(f"Missing Phase 5 specialist checkpoint(s): {preview}{suffix}")
    return paths


def _iteration_number(iteration_dir: Path) -> int:
    name = iteration_dir.name
    if name.startswith("iter-"):
        try:
            return int(name.removeprefix("iter-"))
        except ValueError:
            pass
    return 0


def _tree_size(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    files = 0
    bytes_total = 0
    for child in path.rglob("*"):
        if child.is_file():
            files += 1
            bytes_total += child.stat().st_size
    return files, bytes_total


def _selfplay_payload(summary: SelfPlaySummary) -> dict[str, Any]:
    return summary.to_dict()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
