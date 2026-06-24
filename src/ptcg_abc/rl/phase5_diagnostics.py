from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ptcg_abc.rl.model import LinearOptionModel
from ptcg_abc.rl.records import DecisionFrame
from ptcg_abc.rl.torch_backend import TorchCheckpointOptionModel


@dataclass
class AgreementStats:
    frames: int = 0
    actions: int = 0
    search_exact: int = 0
    search_hit: int = 0
    baseline_exact: int = 0
    baseline_hit: int = 0
    changed: int = 0
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
        if model_margin is not None:
            self.margin_frames += 1
            self.model_search_margin_sum += model_margin
            if model_margin > 1e-9:
                self.model_prefers_search += 1
            elif model_margin < -1e-9:
                self.model_prefers_baseline += 1
            else:
                self.model_ties_search_baseline += 1
        if rule_margin is not None:
            self.rule_search_margin_sum += rule_margin
        if model_score_range is not None:
            self.model_score_range_sum += model_score_range
            self.model_score_flat_frames += int(model_score_range <= 1e-9)

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
class TraceDiagnostics:
    records: int
    changed_records: int
    search_errors: int
    candidate_errors: int
    truncated_candidates: int
    comparable_records: int
    mean_search_minus_baseline_combined_score: float
    mean_search_minus_baseline_tactical_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchDistillDiagnostics:
    dataset_path: str
    model_path: str | None
    checkpoint_path: str | None
    trace_path: str | None
    model_metadata: dict[str, Any]
    overall: dict[str, Any]
    search_changed: dict[str, Any]
    search_unchanged: dict[str, Any]
    search_applied: dict[str, Any]
    search_not_applied: dict[str, Any]
    by_select_context: dict[str, dict[str, Any]]
    metadata_counts: dict[str, Any]
    trace: dict[str, Any] | None
    examples: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagnose_search_distillation(
    *,
    dataset_path: Path,
    model_path: Path | None = None,
    checkpoint_path: Path | None = None,
    trace_path: Path | None = None,
    report_json_path: Path | None = None,
    report_md_path: Path | None = None,
    example_limit: int = 20,
) -> SearchDistillDiagnostics:
    scorer, model_metadata = _load_diagnostic_scorer(
        model_path=model_path,
        checkpoint_path=checkpoint_path,
    )

    overall = AgreementStats()
    changed_stats = AgreementStats()
    unchanged_stats = AgreementStats()
    applied_stats = AgreementStats()
    not_applied_stats = AgreementStats()
    by_context: dict[str, AgreementStats] = defaultdict(AgreementStats)
    metadata_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for frame in _iter_decision_jsonl(dataset_path):
        row = _score_frame(scorer, frame)
        if row is None:
            metadata_counts["skipped_no_legal_options"] += 1
            continue

        metadata = frame.reward_metadata
        collector = str(metadata.get("collector", ""))
        if collector:
            metadata_counts[f"collector:{collector}"] += 1
        applied = bool(metadata.get("phase5_search_applied", False))
        changed = bool(metadata.get("phase5_search_changed", False))
        metadata_counts["search_applied"] += int(applied)
        metadata_counts["search_changed"] += int(changed)
        metadata_counts["search_error"] += int(bool(metadata.get("phase5_search_error")))

        _update_stats(overall, row)
        _update_stats(changed_stats if changed else unchanged_stats, row)
        _update_stats(applied_stats if applied else not_applied_stats, row)
        _update_stats(by_context[f"{frame.select_type}/{frame.context}"], row)

        if changed and len(examples) < example_limit and row["predicted"] != row["search"]:
            examples.append(_diagnostic_example(frame, row))

    trace = None
    if trace_path is not None and trace_path.exists():
        trace = diagnose_search_traces(trace_path).to_dict()

    overall_dict = overall.to_dict()
    changed_dict = changed_stats.to_dict()
    recommendations = _recommendations(overall_dict, changed_dict, trace)

    diagnostics = SearchDistillDiagnostics(
        dataset_path=str(dataset_path.as_posix()),
        model_path=str(model_path.as_posix()) if model_path is not None else None,
        checkpoint_path=(
            str(checkpoint_path.as_posix()) if checkpoint_path is not None else None
        ),
        trace_path=str(trace_path.as_posix()) if trace_path is not None else None,
        model_metadata=model_metadata,
        overall=overall_dict,
        search_changed=changed_dict,
        search_unchanged=unchanged_stats.to_dict(),
        search_applied=applied_stats.to_dict(),
        search_not_applied=not_applied_stats.to_dict(),
        by_select_context={
            key: stats.to_dict()
            for key, stats in sorted(
                by_context.items(),
                key=lambda item: (-item[1].frames, item[0]),
            )
        },
        metadata_counts=dict(sorted(metadata_counts.items())),
        trace=trace,
        examples=examples,
        recommendations=recommendations,
    )

    if report_json_path is not None:
        report_json_path.parent.mkdir(parents=True, exist_ok=True)
        report_json_path.write_text(
            json.dumps(diagnostics.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if report_md_path is not None:
        write_search_distill_diagnostic_markdown(diagnostics, report_md_path)

    return diagnostics


def diagnose_search_traces(path: Path) -> TraceDiagnostics:
    records = changed_records = search_errors = candidate_errors = truncated = 0
    comparable = 0
    combined_margin_sum = 0.0
    tactical_margin_sum = 0.0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            records += 1
            changed_records += int(bool(payload.get("changed")))
            search_errors += int(bool(payload.get("search_error")))
            candidates = list(payload.get("candidates", []) or [])
            candidate_errors += sum(1 for candidate in candidates if candidate.get("error"))
            truncated += sum(1 for candidate in candidates if candidate.get("truncated"))

            baseline = tuple(int(index) for index in payload.get("baseline_indices", []) or [])
            search = tuple(int(index) for index in payload.get("search_indices", []) or [])
            by_indices = {
                tuple(int(index) for index in candidate.get("indices", []) or []): candidate
                for candidate in candidates
            }
            baseline_candidate = by_indices.get(baseline)
            search_candidate = by_indices.get(search)
            if baseline_candidate is None or search_candidate is None:
                continue
            comparable += 1
            combined_margin_sum += _candidate_score(search_candidate, "combined_score") - _candidate_score(
                baseline_candidate,
                "combined_score",
            )
            tactical_margin_sum += _candidate_score(search_candidate, "tactical_score") - _candidate_score(
                baseline_candidate,
                "tactical_score",
            )

    divisor = max(1, comparable)
    return TraceDiagnostics(
        records=records,
        changed_records=changed_records,
        search_errors=search_errors,
        candidate_errors=candidate_errors,
        truncated_candidates=truncated,
        comparable_records=comparable,
        mean_search_minus_baseline_combined_score=(
            combined_margin_sum / divisor if comparable else 0.0
        ),
        mean_search_minus_baseline_tactical_score=(
            tactical_margin_sum / divisor if comparable else 0.0
        ),
    )


def _iter_decision_jsonl(path: Path) -> Iterable[DecisionFrame]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield DecisionFrame.from_dict(json.loads(stripped))


def write_search_distill_diagnostic_markdown(
    diagnostics: SearchDistillDiagnostics,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 5 Search-Distillation Diagnostics",
        "",
        f"- Dataset: `{diagnostics.dataset_path}`",
        f"- Model: `{diagnostics.model_path or 'not provided'}`",
        f"- Checkpoint: `{diagnostics.checkpoint_path or 'not provided'}`",
        f"- Trace: `{diagnostics.trace_path or 'not provided'}`",
        "",
        "## Summary",
        "",
        _stats_line("Overall", diagnostics.overall),
        _stats_line("Search-changed", diagnostics.search_changed),
        _stats_line("Search-unchanged", diagnostics.search_unchanged),
        "",
        "## Changed-Decision Margins",
        "",
        (
            "- Mean model search-minus-baseline score: "
            f"{diagnostics.search_changed['mean_model_search_minus_baseline_score']:.6f}"
        ),
        (
            "- Mean rule search-minus-baseline score: "
            f"{diagnostics.search_changed['mean_rule_search_minus_baseline_score']:.6f}"
        ),
        (
            "- Model prefers search/baseline/tie: "
            f"{diagnostics.search_changed['model_prefers_search']} / "
            f"{diagnostics.search_changed['model_prefers_baseline']} / "
            f"{diagnostics.search_changed['model_ties_search_baseline']}"
        ),
        (
            "- Mean model score range: "
            f"{diagnostics.search_changed['mean_model_score_range']:.6f}"
        ),
        (
            "- Flat model-score frame rate: "
            f"{diagnostics.search_changed['model_score_flat_rate']:.3f}"
        ),
        "",
    ]
    if diagnostics.trace is not None:
        trace = diagnostics.trace
        lines.extend(
            [
                "## Trace Diagnostics",
                "",
                f"- Trace records: {trace['records']}",
                f"- Changed trace records: {trace['changed_records']}",
                f"- Search errors: {trace['search_errors']}",
                f"- Candidate errors: {trace['candidate_errors']}",
                f"- Truncated candidates: {trace['truncated_candidates']}",
                (
                    "- Mean search-minus-baseline combined score: "
                    f"{trace['mean_search_minus_baseline_combined_score']:.6f}"
                ),
                (
                    "- Mean search-minus-baseline tactical score: "
                    f"{trace['mean_search_minus_baseline_tactical_score']:.6f}"
                ),
                "",
            ]
        )

    lines.extend(["## Largest Context Groups", ""])
    for key, stats in list(diagnostics.by_select_context.items())[:12]:
        lines.append(_stats_line(key, stats))

    lines.extend(["", "## Recommendations", ""])
    for recommendation in diagnostics.recommendations:
        lines.append(f"- {recommendation}")

    if diagnostics.examples:
        lines.extend(["", "## Changed-Decision Disagreements", ""])
        for example in diagnostics.examples:
            lines.extend(
                [
                    (
                        f"### {example['select_type']}/{example['context']} "
                        f"game={example.get('game_index')} step={example.get('step_index')}"
                    ),
                    "",
                    f"- Deck: {example.get('deck_label', '')}",
                    f"- Opponent: {example.get('opponent', '')}",
                    f"- Baseline: `{example['baseline']}`",
                    f"- Search: `{example['search']}`",
                    f"- Predicted: `{example['predicted']}`",
                    f"- Model search-minus-baseline score: {example['model_margin']:.6f}",
                    f"- Rule search-minus-baseline score: {example['rule_margin']:.6f}",
                    "",
                ]
            )
            for action in example["top_actions"]:
                lines.append(
                    "- "
                    f"idx={action['index']} model={action['model_score']:.6f} "
                    f"rule={action['rule_score']:.6f} type={action['option_type']} "
                    f"card={action['card_name']}"
                )
            lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _load_diagnostic_scorer(
    *,
    model_path: Path | None,
    checkpoint_path: Path | None,
) -> tuple[Any, dict[str, Any]]:
    if checkpoint_path is not None:
        scorer = TorchCheckpointOptionModel(checkpoint_path)
        return scorer, dict(scorer.metadata)
    if model_path is None:
        raise ValueError("Set either model_path or checkpoint_path for diagnostics.")
    model = LinearOptionModel.load(model_path)
    return model, dict(model.metadata)


def _score_frame(model: Any, frame: DecisionFrame) -> dict[str, Any] | None:
    if not frame.legal_options:
        return None
    scores = model.score_frame(frame)
    target_count = max(1, min(frame.target_count, len(frame.legal_options)))
    ranked = sorted(
        range(len(frame.legal_options)),
        key=lambda index: (scores[index], frame.legal_options[index].rule_score, -index),
        reverse=True,
    )
    predicted = set(ranked[:target_count])
    metadata = frame.reward_metadata
    search = _valid_set(metadata.get("phase5_search_indices", frame.rule_selected_indices), frame)
    baseline = _valid_set(metadata.get("phase5_baseline_indices", frame.rule_selected_indices), frame)
    changed = bool(metadata.get("phase5_search_changed", search != baseline))
    model_margin = _mean_score(scores, search) - _mean_score(scores, baseline)
    rule_scores = [action.rule_score for action in frame.legal_options]
    rule_margin = _mean_score(rule_scores, search) - _mean_score(rule_scores, baseline)
    score_range = max(scores) - min(scores) if scores else 0.0
    return {
        "scores": scores,
        "ranked": ranked,
        "predicted": predicted,
        "search": search,
        "baseline": baseline,
        "changed": changed,
        "model_margin": model_margin,
        "rule_margin": rule_margin,
        "model_score_range": score_range,
    }


def _update_stats(stats: AgreementStats, row: dict[str, Any]) -> None:
    stats.update(
        action_count=len(row["scores"]),
        predicted=row["predicted"],
        search=row["search"],
        baseline=row["baseline"],
        changed=bool(row["changed"]),
        model_margin=float(row["model_margin"]),
        rule_margin=float(row["rule_margin"]),
        model_score_range=float(row["model_score_range"]),
    )


def _valid_set(values: Any, frame: DecisionFrame) -> set[int]:
    output: set[int] = set()
    for value in values or []:
        index = int(value)
        if 0 <= index < len(frame.legal_options):
            output.add(index)
    return output


def _mean_score(scores: Sequence[float], indices: Iterable[int]) -> float:
    selected = [scores[index] for index in indices]
    return sum(selected) / len(selected) if selected else 0.0


def _candidate_score(candidate: dict[str, Any], key: str) -> float:
    try:
        return float(candidate.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _diagnostic_example(frame: DecisionFrame, row: dict[str, Any]) -> dict[str, Any]:
    metadata = frame.reward_metadata
    top_actions = []
    for index in row["ranked"][:8]:
        action = frame.legal_options[index]
        top_actions.append(
            {
                "index": index,
                "model_score": row["scores"][index],
                "rule_score": action.rule_score,
                "option_type": action.option_type,
                "card_name": action.card_name,
                "attack_id": action.attack_id,
                "target_name": action.target_name,
            }
        )
    return {
        "game_index": metadata.get("game_index"),
        "step_index": metadata.get("step_index"),
        "deck_index": metadata.get("deck_index"),
        "deck_label": metadata.get("deck_label"),
        "opponent": metadata.get("opponent"),
        "select_type": frame.select_type,
        "context": frame.context,
        "baseline": sorted(row["baseline"]),
        "search": sorted(row["search"]),
        "predicted": sorted(row["predicted"]),
        "model_margin": row["model_margin"],
        "rule_margin": row["rule_margin"],
        "top_actions": top_actions,
    }


def _recommendations(
    overall: dict[str, Any],
    changed: dict[str, Any],
    trace: dict[str, Any] | None,
) -> list[str]:
    output: list[str] = []
    if changed["frames"] == 0:
        output.append("No changed search decisions were found; generate more search data or lower filters.")
        return output
    if changed["search_hit_rate"] < overall["search_hit_rate"]:
        output.append(
            "Model agreement is weaker on search-changed decisions than overall; rebalance or "
            "oversample changed decisions before more large-scale training."
        )
    if changed["mean_model_search_minus_baseline_score"] < 0:
        output.append(
            "The exported model scores baseline actions above search actions on average for "
            "changed decisions; inspect labels and increase changed-decision loss weight."
        )
    if changed["model_score_flat_rate"] > 0.1:
        output.append(
            "The model assigns effectively identical scores to many changed-decision action "
            "sets; retrain with an action-specific residual head or lower learning rate before "
            "battle evaluation."
        )
    if changed["mean_rule_search_minus_baseline_score"] < 0:
        output.append(
            "Search labels often move away from high rule-score actions; inspect trace examples "
            "to verify the root-search reward is not noisy."
        )
    if trace is not None and trace["mean_search_minus_baseline_combined_score"] <= 0:
        output.append(
            "Trace scores do not show a positive search-over-baseline margin; improve search "
            "scoring before generating more shards."
        )
    if not output:
        output.append(
            "Changed-decision diagnostics look coherent; the next likely bottleneck is model "
            "capacity or dataset scale."
        )
    return output


def _stats_line(label: str, stats: dict[str, Any]) -> str:
    return (
        f"- {label}: frames={stats['frames']}, changed={stats['changed']}, "
        f"search_hit={stats['search_hit_rate']:.3f}, "
        f"search_exact={stats['search_exact_rate']:.3f}, "
        f"baseline_hit={stats['baseline_hit_rate']:.3f}, "
        f"model_margin={stats['mean_model_search_minus_baseline_score']:.6f}"
    )
