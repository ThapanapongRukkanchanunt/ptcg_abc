from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ptcg_abc.rl.phase5_encoder import Phase5SymbolicEncoder
from ptcg_abc.rl.phase5_policy import (
    AlphaStarTurnPolicy,
    PHASE5_POLICY_CHECKPOINT_FORMAT,
    Phase5PolicyConfig,
    Phase5PolicyUnavailable,
    TORCH_AVAILABLE,
)
from ptcg_abc.rl.phase5_symbolic_training import (
    Phase5SymbolicDecisionRecord,
    Phase5TurnContext,
    iter_decision_jsonl,
    phase5_symbolic_record_from_decision,
    phase5_target_indices,
)
from ptcg_abc.rl.records import DecisionFrame


@dataclass
class SymbolicAgreementStats:
    frames: int = 0
    actions: int = 0
    changed: int = 0
    search_exact: int = 0
    search_hit: int = 0
    baseline_exact: int = 0
    baseline_hit: int = 0
    predicted_baseline_when_changed: int = 0
    predicted_third_when_changed: int = 0
    model_prefers_search: int = 0
    model_prefers_baseline: int = 0
    model_ties_search_baseline: int = 0
    model_search_margin_sum: float = 0.0
    rule_search_margin_sum: float = 0.0
    margin_frames: int = 0
    model_score_range_sum: float = 0.0
    model_score_flat_frames: int = 0

    def update(
        self,
        *,
        action_count: int,
        predicted: set[int],
        search: set[int],
        baseline: set[int],
        changed: bool,
        model_margin: float | None,
        rule_margin: float | None,
        model_score_range: float | None,
    ) -> None:
        self.frames += 1
        self.actions += action_count
        self.changed += int(changed)
        self.search_exact += int(bool(search) and predicted == search)
        self.search_hit += int(bool(predicted & search))
        self.baseline_exact += int(bool(baseline) and predicted == baseline)
        self.baseline_hit += int(bool(predicted & baseline))
        if changed:
            self.predicted_baseline_when_changed += int(bool(predicted & baseline))
            self.predicted_third_when_changed += int(not predicted & (search | baseline))
        if model_margin is not None:
            self.margin_frames += 1
            self.model_search_margin_sum += model_margin
            if model_margin > 1.0e-9:
                self.model_prefers_search += 1
            elif model_margin < -1.0e-9:
                self.model_prefers_baseline += 1
            else:
                self.model_ties_search_baseline += 1
        if rule_margin is not None:
            self.rule_search_margin_sum += rule_margin
        if model_score_range is not None:
            self.model_score_range_sum += model_score_range
            self.model_score_flat_frames += int(model_score_range <= 1.0e-9)

    def to_dict(self) -> dict[str, Any]:
        frames = max(1, self.frames)
        margin_frames = max(1, self.margin_frames)
        return {
            "frames": self.frames,
            "actions": self.actions,
            "avg_legal_actions": self.actions / frames if self.frames else 0.0,
            "changed": self.changed,
            "changed_rate": self.changed / frames if self.frames else 0.0,
            "search_exact": self.search_exact,
            "search_exact_rate": self.search_exact / frames if self.frames else 0.0,
            "search_hit": self.search_hit,
            "search_hit_rate": self.search_hit / frames if self.frames else 0.0,
            "baseline_exact": self.baseline_exact,
            "baseline_exact_rate": self.baseline_exact / frames if self.frames else 0.0,
            "baseline_hit": self.baseline_hit,
            "baseline_hit_rate": self.baseline_hit / frames if self.frames else 0.0,
            "predicted_baseline_when_changed": self.predicted_baseline_when_changed,
            "predicted_baseline_when_changed_rate": (
                self.predicted_baseline_when_changed / frames if self.frames else 0.0
            ),
            "predicted_third_when_changed": self.predicted_third_when_changed,
            "predicted_third_when_changed_rate": (
                self.predicted_third_when_changed / frames if self.frames else 0.0
            ),
            "model_prefers_search": self.model_prefers_search,
            "model_prefers_search_rate": (
                self.model_prefers_search / margin_frames if self.margin_frames else 0.0
            ),
            "model_prefers_baseline": self.model_prefers_baseline,
            "model_prefers_baseline_rate": (
                self.model_prefers_baseline / margin_frames if self.margin_frames else 0.0
            ),
            "model_ties_search_baseline": self.model_ties_search_baseline,
            "mean_model_search_minus_baseline_score": (
                self.model_search_margin_sum / margin_frames if self.margin_frames else 0.0
            ),
            "mean_rule_search_minus_baseline_score": (
                self.rule_search_margin_sum / margin_frames if self.margin_frames else 0.0
            ),
            "margin_frames": self.margin_frames,
            "mean_model_score_range": (
                self.model_score_range_sum / frames if self.frames else 0.0
            ),
            "model_score_flat_frames": self.model_score_flat_frames,
            "model_score_flat_rate": (
                self.model_score_flat_frames / frames if self.frames else 0.0
            ),
        }


@dataclass(frozen=True)
class Phase5SymbolicDiagnostics:
    dataset_path: str
    checkpoint_path: str
    model_metadata: dict[str, Any]
    frames_seen: int
    skipped_no_target: int
    limit: int | None
    device: str
    overall: dict[str, Any]
    search_changed: dict[str, Any]
    search_unchanged: dict[str, Any]
    by_select_context: dict[str, dict[str, Any]]
    by_deck: dict[str, dict[str, Any]]
    predicted_action_types: dict[str, int]
    search_action_types: dict[str, int]
    baseline_action_types: dict[str, int]
    examples: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagnose_phase5_symbolic_policy(
    *,
    dataset_path: Path,
    checkpoint_path: Path,
    report_json_path: Path | None = None,
    report_md_path: Path | None = None,
    limit: int | None = None,
    batch_size: int = 64,
    example_limit: int = 20,
) -> Phase5SymbolicDiagnostics:
    torch, model, encoder, max_previous_actions, metadata, device = _load_symbolic_model(
        checkpoint_path
    )
    context = Phase5TurnContext(max_previous_actions=max_previous_actions)
    overall = SymbolicAgreementStats()
    changed_stats = SymbolicAgreementStats()
    unchanged_stats = SymbolicAgreementStats()
    by_context: dict[str, SymbolicAgreementStats] = defaultdict(SymbolicAgreementStats)
    by_deck: dict[str, SymbolicAgreementStats] = defaultdict(SymbolicAgreementStats)
    predicted_types: Counter[str] = Counter()
    search_types: Counter[str] = Counter()
    baseline_types: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    frames_seen = skipped_no_target = 0
    pending: list[tuple[DecisionFrame, Phase5SymbolicDecisionRecord]] = []

    def flush() -> None:
        nonlocal examples
        if not pending:
            return
        scores = _score_records(torch, model, pending, device=device)
        for (frame, record), score_row in zip(pending, scores, strict=True):
            row = _diagnostic_row(frame, record, score_row)
            _update_symbolic_stats(overall, row)
            _update_symbolic_stats(changed_stats if row["changed"] else unchanged_stats, row)
            _update_symbolic_stats(
                by_context[f"{frame.select_type}/{frame.context}"],
                row,
            )
            _update_symbolic_stats(by_deck[_deck_key(frame)], row)
            for action_type in row["predicted_action_types"]:
                predicted_types[action_type] += 1
            for action_type in row["search_action_types"]:
                search_types[action_type] += 1
            for action_type in row["baseline_action_types"]:
                baseline_types[action_type] += 1
            if (
                row["changed"]
                and row["predicted"] != row["search"]
                and len(examples) < example_limit
            ):
                examples.append(_symbolic_example(frame, row))
        pending.clear()

    for frame in iter_decision_jsonl(dataset_path):
        frames_seen += 1
        if limit is not None and frames_seen > limit:
            break
        previous_rows = context.previous_rows(frame)
        record = phase5_symbolic_record_from_decision(
            frame,
            encoder=encoder,
            previous_action_features=previous_rows,
            max_previous_actions=max_previous_actions,
        )
        if record is None:
            skipped_no_target += 1
            continue
        context.observe(record)
        pending.append((frame, record))
        if len(pending) >= max(1, batch_size):
            flush()
    flush()

    diagnostics = Phase5SymbolicDiagnostics(
        dataset_path=str(dataset_path.as_posix()),
        checkpoint_path=str(checkpoint_path.as_posix()),
        model_metadata=metadata,
        frames_seen=frames_seen,
        skipped_no_target=skipped_no_target,
        limit=limit,
        device=str(device),
        overall=overall.to_dict(),
        search_changed=changed_stats.to_dict(),
        search_unchanged=unchanged_stats.to_dict(),
        by_select_context={
            key: stats.to_dict()
            for key, stats in sorted(
                by_context.items(),
                key=lambda item: (-item[1].frames, item[0]),
            )
        },
        by_deck={
            key: stats.to_dict()
            for key, stats in sorted(
                by_deck.items(),
                key=lambda item: (-item[1].frames, item[0]),
            )
        },
        predicted_action_types=dict(sorted(predicted_types.items())),
        search_action_types=dict(sorted(search_types.items())),
        baseline_action_types=dict(sorted(baseline_types.items())),
        examples=examples,
        recommendations=_symbolic_recommendations(overall.to_dict(), changed_stats.to_dict()),
    )
    if report_json_path is not None:
        report_json_path.parent.mkdir(parents=True, exist_ok=True)
        report_json_path.write_text(
            json.dumps(diagnostics.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if report_md_path is not None:
        write_phase5_symbolic_diagnostics_markdown(diagnostics, report_md_path)
    return diagnostics


def write_phase5_symbolic_diagnostics_markdown(
    diagnostics: Phase5SymbolicDiagnostics,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 5 Symbolic Policy Diagnostics",
        "",
        f"- Dataset: `{diagnostics.dataset_path}`",
        f"- Checkpoint: `{diagnostics.checkpoint_path}`",
        f"- Frames seen: {diagnostics.frames_seen}",
        f"- Skipped no target: {diagnostics.skipped_no_target}",
        f"- Device: `{diagnostics.device}`",
        "",
        "## Overall",
        "",
        _stats_line("Overall", diagnostics.overall),
        _stats_line("Search-changed", diagnostics.search_changed),
        _stats_line("Search-unchanged", diagnostics.search_unchanged),
        "",
        "## Changed-Decision Margins",
        "",
        (
            "- Model search-minus-baseline score: "
            f"{diagnostics.search_changed['mean_model_search_minus_baseline_score']:.6f}"
        ),
        (
            "- Rule search-minus-baseline score: "
            f"{diagnostics.search_changed['mean_rule_search_minus_baseline_score']:.6f}"
        ),
        (
            "- Changed third-action rate: "
            f"{diagnostics.search_changed['predicted_third_when_changed_rate']:.3f}"
        ),
        "",
        "## Action Types",
        "",
        f"- Predicted: `{diagnostics.predicted_action_types}`",
        f"- Search labels: `{diagnostics.search_action_types}`",
        f"- Baseline labels: `{diagnostics.baseline_action_types}`",
        "",
        "## By Deck",
        "",
        "| Deck | Frames | Search hit | Baseline hit | Changed | Third on changed |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, stats in list(diagnostics.by_deck.items())[:16]:
        lines.append(
            f"| {key} | {stats['frames']} | {stats['search_hit_rate']:.3f} | "
            f"{stats['baseline_hit_rate']:.3f} | {stats['changed_rate']:.3f} | "
            f"{stats['predicted_third_when_changed_rate']:.3f} |"
        )
    lines.extend(["", "## Recommendations", ""])
    for recommendation in diagnostics.recommendations:
        lines.append(f"- {recommendation}")
    if diagnostics.examples:
        lines.extend(["", "## Examples", ""])
        for example in diagnostics.examples:
            lines.append("```json")
            lines.append(json.dumps(example, indent=2, sort_keys=True))
            lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_symbolic_model(
    checkpoint_path: Path,
) -> tuple[Any, Any, Phase5SymbolicEncoder, int, dict[str, Any], Any]:
    if not TORCH_AVAILABLE:
        raise Phase5PolicyUnavailable(
            "PyTorch is not installed. Run symbolic diagnostics on ERAWAN or install the rl extra."
        )
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != PHASE5_POLICY_CHECKPOINT_FORMAT:
        raise ValueError(
            f"Unsupported Phase 5 symbolic checkpoint: {checkpoint.get('format')}"
        )
    config = Phase5PolicyConfig(**checkpoint["config"])
    encoder_config = checkpoint.get("encoder", {})
    encoder = Phase5SymbolicEncoder(
        max_entities=int(encoder_config.get("max_entities", 96)),
        max_actions=int(encoder_config.get("max_actions", 128)),
    )
    max_previous_actions = int(encoder_config.get("max_previous_actions", 16))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaStarTurnPolicy(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    metadata = dict(checkpoint.get("metadata", {}))
    metadata.update({"format": checkpoint.get("format"), "config": config.to_dict()})
    return torch, model, encoder, max_previous_actions, metadata, device


def _score_records(
    torch: Any,
    model: Any,
    records: Sequence[tuple[DecisionFrame, Phase5SymbolicDecisionRecord]],
    *,
    device: Any,
) -> list[list[float]]:
    with torch.no_grad():
        output = model(
            torch.tensor(
                [record.encoded.global_features for _, record in records],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                [record.encoded.entity_features for _, record in records],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                [record.encoded.entity_mask for _, record in records],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                [record.encoded.legal_action_features for _, record in records],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                [record.encoded.legal_action_mask for _, record in records],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                [record.previous_action_features for _, record in records],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                [record.previous_action_mask for _, record in records],
                dtype=torch.float32,
                device=device,
            ),
        )
    return [
        [float(value) for value in row]
        for row in output["action_logits"].detach().cpu().tolist()
    ]


def _diagnostic_row(
    frame: DecisionFrame,
    record: Phase5SymbolicDecisionRecord,
    scores: Sequence[float],
) -> dict[str, Any]:
    legal_positions = [
        pos
        for pos, index in enumerate(record.encoded.legal_action_indices)
        if index >= 0 and record.encoded.legal_action_mask[pos] > 0
    ]
    target_count = max(1, min(frame.target_count, len(legal_positions)))
    ranked = sorted(
        legal_positions,
        key=lambda pos: (
            scores[pos],
            frame.legal_options[record.encoded.legal_action_indices[pos]].rule_score,
            -pos,
        ),
        reverse=True,
    )
    predicted = [record.encoded.legal_action_indices[pos] for pos in ranked[:target_count]]
    search = phase5_target_indices(frame, target_source="search")
    baseline = phase5_target_indices(frame, target_source="baseline")
    return {
        "predicted": predicted,
        "search": search,
        "baseline": baseline,
        "changed": bool(frame.reward_metadata.get("phase5_search_changed", False)),
        "action_count": len(legal_positions),
        "model_margin": _score_margin(scores, record, search, baseline),
        "rule_margin": _rule_margin(frame, search, baseline),
        "model_score_range": _score_range(scores, legal_positions),
        "predicted_action_types": _action_types(frame, predicted),
        "search_action_types": _action_types(frame, search),
        "baseline_action_types": _action_types(frame, baseline),
        "top_indices": [
            record.encoded.legal_action_indices[position]
            for position in ranked[:8]
        ],
        "scores": scores,
        "score_by_index": {
            record.encoded.legal_action_indices[position]: scores[position]
            for position in legal_positions
        },
    }


def _update_symbolic_stats(stats: SymbolicAgreementStats, row: dict[str, Any]) -> None:
    stats.update(
        action_count=int(row["action_count"]),
        predicted=set(row["predicted"]),
        search=set(row["search"]),
        baseline=set(row["baseline"]),
        changed=bool(row["changed"]),
        model_margin=row["model_margin"],
        rule_margin=row["rule_margin"],
        model_score_range=row["model_score_range"],
    )


def _score_margin(
    scores: Sequence[float],
    record: Phase5SymbolicDecisionRecord,
    search: Sequence[int],
    baseline: Sequence[int],
) -> float | None:
    positions = {
        action_index: position
        for position, action_index in enumerate(record.encoded.legal_action_indices)
    }
    search_scores = [scores[positions[index]] for index in search if index in positions]
    baseline_scores = [scores[positions[index]] for index in baseline if index in positions]
    if not search_scores or not baseline_scores:
        return None
    return max(search_scores) - max(baseline_scores)


def _rule_margin(
    frame: DecisionFrame,
    search: Sequence[int],
    baseline: Sequence[int],
) -> float | None:
    search_scores = [
        frame.legal_options[index].rule_score
        for index in search
        if 0 <= index < len(frame.legal_options)
    ]
    baseline_scores = [
        frame.legal_options[index].rule_score
        for index in baseline
        if 0 <= index < len(frame.legal_options)
    ]
    if not search_scores or not baseline_scores:
        return None
    return max(search_scores) - max(baseline_scores)


def _score_range(scores: Sequence[float], positions: Sequence[int]) -> float:
    legal_scores = [scores[position] for position in positions]
    if not legal_scores:
        return 0.0
    return max(legal_scores) - min(legal_scores)


def _action_types(frame: DecisionFrame, indices: Sequence[int]) -> list[str]:
    return [
        frame.legal_options[index].option_type
        for index in indices
        if 0 <= index < len(frame.legal_options)
    ]


def _symbolic_example(frame: DecisionFrame, row: dict[str, Any]) -> dict[str, Any]:
    top_actions = []
    for action_index in row["top_indices"]:
        if action_index < 0 or action_index >= len(frame.legal_options):
            continue
        action = frame.legal_options[action_index]
        top_actions.append(
            {
                "index": action.index,
                "model_score": row["score_by_index"].get(action_index),
                "rule_score": action.rule_score,
                "option_type": action.option_type,
                "card_name": action.card_name,
                "attack_id": action.attack_id,
            }
        )
    return {
        "game_index": frame.reward_metadata.get("game_index"),
        "step_index": frame.reward_metadata.get("step_index"),
        "deck_index": frame.reward_metadata.get("deck_index"),
        "deck_label": frame.reward_metadata.get("deck_label"),
        "opponent": frame.reward_metadata.get("opponent"),
        "select_type": frame.select_type,
        "context": frame.context,
        "baseline": row["baseline"],
        "search": row["search"],
        "predicted": row["predicted"],
        "model_margin": row["model_margin"],
        "rule_margin": row["rule_margin"],
        "top_actions": top_actions,
    }


def _deck_key(frame: DecisionFrame) -> str:
    index = frame.reward_metadata.get("deck_index", "?")
    label = frame.reward_metadata.get("deck_label", "")
    return f"{index}: {label}"


def _stats_line(label: str, stats: dict[str, Any]) -> str:
    return (
        f"- {label}: frames={stats['frames']}, search_hit={stats['search_hit_rate']:.3f}, "
        f"baseline_hit={stats['baseline_hit_rate']:.3f}, changed={stats['changed_rate']:.3f}, "
        f"mean_margin={stats['mean_model_search_minus_baseline_score']:.6f}"
    )


def _symbolic_recommendations(
    overall: dict[str, Any],
    changed: dict[str, Any],
) -> list[str]:
    recommendations: list[str] = []
    if changed.get("search_hit_rate", 0.0) < overall.get("search_hit_rate", 0.0):
        recommendations.append(
            "Changed-decision agreement is much weaker than overall; train or sample changed "
            "frames separately before scaling data."
        )
    if changed.get("predicted_third_when_changed_rate", 0.0) > 0.25:
        recommendations.append(
            "The symbolic policy often picks neither search nor baseline on changed frames; "
            "add pairwise or contrastive loss against all non-search legal actions."
        )
    if changed.get("mean_model_search_minus_baseline_score", 0.0) < 0:
        recommendations.append(
            "The checkpoint scores baseline actions above search actions on changed frames; "
            "increase changed-decision weight or train changed-only warmup epochs."
        )
    if changed.get("model_score_flat_rate", 0.0) > 0.05:
        recommendations.append(
            "Some legal sets have flat symbolic logits; inspect feature coverage and action masks."
        )
    if not recommendations:
        recommendations.append(
            "Offline symbolic agreement is coherent; next bottleneck is likely battle-state "
            "distribution shift or search-label quality."
        )
    return recommendations
