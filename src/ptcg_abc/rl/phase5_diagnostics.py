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
    candidate_probes: int
    candidate_errors: int
    truncated_candidates: int
    truncated_candidate_rate: float
    all_candidates_truncated_records: int
    all_candidates_truncated_rate: float
    selected_truncated_records: int
    selected_truncated_rate: float
    changed_selected_truncated_records: int
    changed_selected_truncated_rate: float
    selected_truncated_by_type: dict[str, int]
    selected_truncated_by_matchup: list[dict[str, Any]]
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


SCORE_COMPONENT_FIELDS: tuple[str, ...] = (
    "tactical_score",
    "rule_score",
    "rule_prior",
    "policy_score",
    "policy_prior",
    "neural_action_value",
    "neural_action_value_prior",
    "neural_tactical_score",
    "neural_tactical_prior",
    "prior_score",
    "combined_score",
    "damage_delta",
    "self_damage_delta",
    "rollout_steps",
)


@dataclass
class _OnlineScoreStats:
    count: int = 0
    total: float = 0.0
    low: float | None = None
    high: float | None = None
    positive: int = 0
    negative: int = 0
    zero: int = 0

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.low = value if self.low is None else min(self.low, value)
        self.high = value if self.high is None else max(self.high, value)
        if value > 1e-9:
            self.positive += 1
        elif value < -1e-9:
            self.negative += 1
        else:
            self.zero += 1

    def to_dict(self) -> dict[str, Any]:
        low = self.low if self.low is not None else 0.0
        high = self.high if self.high is not None else 0.0
        return {
            "count": self.count,
            "min": low,
            "max": high,
            "range": high - low if self.count else 0.0,
            "mean": self.total / self.count if self.count else 0.0,
            "positive": self.positive,
            "negative": self.negative,
            "zero": self.zero,
            "positive_rate": self.positive / self.count if self.count else 0.0,
            "negative_rate": self.negative / self.count if self.count else 0.0,
        }


@dataclass
class _ScoreComponentGroup:
    records: int = 0
    changed_records: int = 0
    search_errors: int = 0
    candidate_probes: int = 0
    candidate_errors: int = 0
    selected_records: int = 0
    baseline_records: int = 0
    comparable_records: int = 0
    selected_truncated_records: int = 0
    baseline_truncated_records: int = 0
    all_candidate_stats: dict[str, _OnlineScoreStats] = field(default_factory=dict)
    selected_candidate_stats: dict[str, _OnlineScoreStats] = field(default_factory=dict)
    baseline_candidate_stats: dict[str, _OnlineScoreStats] = field(default_factory=dict)
    margin_stats: dict[str, _OnlineScoreStats] = field(default_factory=dict)
    selected_option_types: Counter[str] = field(default_factory=Counter)
    baseline_option_types: Counter[str] = field(default_factory=Counter)

    def add_record(self, payload: dict[str, Any]) -> None:
        self.records += 1
        self.changed_records += int(bool(payload.get("changed")))
        self.search_errors += int(bool(payload.get("search_error")))
        candidates = list(payload.get("candidates", []) or [])
        self.candidate_probes += len(candidates)
        self.candidate_errors += sum(1 for candidate in candidates if candidate.get("error"))
        for candidate in candidates:
            self._add_candidate_stats(self.all_candidate_stats, candidate)

        selected = _candidate_for_indices(candidates, payload.get("search_indices"))
        baseline = _candidate_for_indices(candidates, payload.get("baseline_indices"))
        if selected is not None:
            self.selected_records += 1
            self.selected_truncated_records += int(bool(selected.get("truncated")))
            self.selected_option_types[str(selected.get("option_type", ""))] += 1
            self._add_candidate_stats(self.selected_candidate_stats, selected)
        if baseline is not None:
            self.baseline_records += 1
            self.baseline_truncated_records += int(bool(baseline.get("truncated")))
            self.baseline_option_types[str(baseline.get("option_type", ""))] += 1
            self._add_candidate_stats(self.baseline_candidate_stats, baseline)
        if selected is None or baseline is None:
            return
        self.comparable_records += 1
        for field_name in SCORE_COMPONENT_FIELDS:
            selected_value = _maybe_float(selected.get(field_name))
            baseline_value = _maybe_float(baseline.get(field_name))
            if selected_value is None or baseline_value is None:
                continue
            self.margin_stats.setdefault(field_name, _OnlineScoreStats()).add(
                selected_value - baseline_value
            )

    def _add_candidate_stats(
        self,
        destination: dict[str, _OnlineScoreStats],
        candidate: dict[str, Any],
    ) -> None:
        for field_name in SCORE_COMPONENT_FIELDS:
            value = _maybe_float(candidate.get(field_name))
            if value is not None:
                destination.setdefault(field_name, _OnlineScoreStats()).add(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "changed_records": self.changed_records,
            "changed_rate": self.changed_records / self.records if self.records else 0.0,
            "search_errors": self.search_errors,
            "candidate_probes": self.candidate_probes,
            "candidate_errors": self.candidate_errors,
            "candidate_error_rate": (
                self.candidate_errors / self.candidate_probes
                if self.candidate_probes
                else 0.0
            ),
            "selected_records": self.selected_records,
            "baseline_records": self.baseline_records,
            "comparable_records": self.comparable_records,
            "selected_truncated_records": self.selected_truncated_records,
            "baseline_truncated_records": self.baseline_truncated_records,
            "selected_option_types": dict(self.selected_option_types.most_common()),
            "baseline_option_types": dict(self.baseline_option_types.most_common()),
            "components": {
                field_name: {
                    "all_candidates": self.all_candidate_stats.get(
                        field_name,
                        _OnlineScoreStats(),
                    ).to_dict(),
                    "selected_candidates": self.selected_candidate_stats.get(
                        field_name,
                        _OnlineScoreStats(),
                    ).to_dict(),
                    "baseline_candidates": self.baseline_candidate_stats.get(
                        field_name,
                        _OnlineScoreStats(),
                    ).to_dict(),
                }
                for field_name in SCORE_COMPONENT_FIELDS
            },
            "selected_minus_baseline": {
                field_name: self.margin_stats.get(
                    field_name,
                    _OnlineScoreStats(),
                ).to_dict()
                for field_name in SCORE_COMPONENT_FIELDS
            },
        }


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
    candidate_probes = 0
    all_candidates_truncated = 0
    selected_truncated = 0
    changed_selected_truncated = 0
    selected_truncated_by_type: Counter[str] = Counter()
    selected_truncated_by_matchup: Counter[tuple[int, str]] = Counter()
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
            candidate_probes += len(candidates)
            candidate_errors += sum(1 for candidate in candidates if candidate.get("error"))
            truncated += sum(1 for candidate in candidates if candidate.get("truncated"))
            if candidates and all(candidate.get("truncated") for candidate in candidates):
                all_candidates_truncated += 1

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
            if search_candidate.get("truncated"):
                selected_truncated += 1
                selected_truncated_by_type[str(search_candidate.get("option_type", ""))] += 1
                selected_truncated_by_matchup[
                    (
                        int(payload.get("deck_index", 0) or 0),
                        str(payload.get("opponent", "")),
                    )
                ] += 1
                if bool(payload.get("changed")):
                    changed_selected_truncated += 1
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
        candidate_probes=candidate_probes,
        candidate_errors=candidate_errors,
        truncated_candidates=truncated,
        truncated_candidate_rate=truncated / candidate_probes if candidate_probes else 0.0,
        all_candidates_truncated_records=all_candidates_truncated,
        all_candidates_truncated_rate=(
            all_candidates_truncated / records if records else 0.0
        ),
        selected_truncated_records=selected_truncated,
        selected_truncated_rate=selected_truncated / divisor if comparable else 0.0,
        changed_selected_truncated_records=changed_selected_truncated,
        changed_selected_truncated_rate=(
            changed_selected_truncated / changed_records if changed_records else 0.0
        ),
        selected_truncated_by_type=dict(selected_truncated_by_type.most_common()),
        selected_truncated_by_matchup=[
            {"deck_index": deck_index, "opponent": opponent, "count": count}
            for (deck_index, opponent), count in selected_truncated_by_matchup.most_common(20)
        ],
        comparable_records=comparable,
        mean_search_minus_baseline_combined_score=(
            combined_margin_sum / divisor if comparable else 0.0
        ),
        mean_search_minus_baseline_tactical_score=(
            tactical_margin_sum / divisor if comparable else 0.0
        ),
    )


def diagnose_search_score_components(path: Path) -> dict[str, Any]:
    overall = _ScoreComponentGroup()
    by_outcome: dict[str, _ScoreComponentGroup] = defaultdict(_ScoreComponentGroup)
    config_counter: Counter[str] = Counter()

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            outcome = str(payload.get("game_outcome") or "unknown")
            overall.add_record(payload)
            by_outcome[outcome].add_record(payload)
            config = payload.get("config")
            if isinstance(config, dict):
                config_counter[_score_config_signature(config)] += 1

    overall_payload = overall.to_dict()
    by_outcome_payload = {
        outcome: group.to_dict()
        for outcome, group in sorted(
            by_outcome.items(),
            key=lambda item: (-item[1].records, item[0]),
        )
    }
    diagnostics = {
        "trace_path": path.as_posix(),
        "records": overall.records,
        "candidate_probes": overall.candidate_probes,
        "comparable_records": overall.comparable_records,
        "outcomes": dict(sorted((key, value.records) for key, value in by_outcome.items())),
        "score_component_fields": list(SCORE_COMPONENT_FIELDS),
        "config_signatures": [
            {"config": json.loads(signature), "records": count}
            for signature, count in config_counter.most_common()
        ],
        "overall": overall_payload,
        "by_outcome": by_outcome_payload,
    }
    diagnostics["weighting_hints"] = _score_component_weighting_hints(diagnostics)
    return diagnostics


def write_search_score_component_markdown(
    diagnostics: dict[str, Any],
    path: Path,
    *,
    trace_path: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trace_label = trace_path.as_posix() if trace_path is not None else diagnostics.get(
        "trace_path",
        "not provided",
    )
    lines = [
        "# Phase 5 Search Score-Component Diagnostics",
        "",
        f"- Trace: `{trace_label}`",
        f"- Records: {diagnostics.get('records', 0)}",
        f"- Candidate probes: {diagnostics.get('candidate_probes', 0)}",
        f"- Comparable records: {diagnostics.get('comparable_records', 0)}",
        "",
        "## Weighting Hints",
        "",
    ]
    hints = list(diagnostics.get("weighting_hints", []) or [])
    if hints:
        lines.extend(f"- {hint}" for hint in hints)
    else:
        lines.append("- No weighting hints were generated.")

    lines.extend(
        [
            "",
            "## Outcomes",
            "",
            (
                "| Outcome | Records | Changed | Comparable | "
                "Selected tactical mean/range | Selected prior mean/range | "
                "Selected combined mean/range | Mean selected-baseline tactical | "
                "Mean selected-baseline prior | Mean selected-baseline combined |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for outcome, group in diagnostics.get("by_outcome", {}).items():
        lines.append(
            "| "
            f"{outcome} | "
            f"{group['records']} | "
            f"{group['changed_records']} | "
            f"{group['comparable_records']} | "
            f"{_mean_range(group, 'tactical_score')} | "
            f"{_mean_range(group, 'prior_score')} | "
            f"{_mean_range(group, 'combined_score')} | "
            f"{_margin_mean(group, 'tactical_score'):.6f} | "
            f"{_margin_mean(group, 'prior_score'):.6f} | "
            f"{_margin_mean(group, 'combined_score'):.6f} |"
        )

    lines.extend(
        [
            "",
            "## Selected Candidate Means",
            "",
            "| Component | Overall | Win | Loss | Draw | Timeout | Error | Unknown |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for field_name in SCORE_COMPONENT_FIELDS:
        lines.append(
            f"| {field_name} | "
            f"{_component_mean(diagnostics['overall'], field_name):.6f} | "
            f"{_outcome_component_mean(diagnostics, 'win', field_name):.6f} | "
            f"{_outcome_component_mean(diagnostics, 'loss', field_name):.6f} | "
            f"{_outcome_component_mean(diagnostics, 'draw', field_name):.6f} | "
            f"{_outcome_component_mean(diagnostics, 'timeout', field_name):.6f} | "
            f"{_outcome_component_mean(diagnostics, 'error', field_name):.6f} | "
            f"{_outcome_component_mean(diagnostics, 'unknown', field_name):.6f} |"
        )

    lines.extend(
        [
            "",
            "## Selected Minus Baseline",
            "",
            "| Component | Overall mean | Win mean | Loss mean | Overall positive rate | Win positive rate | Loss positive rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for field_name in SCORE_COMPONENT_FIELDS:
        lines.append(
            f"| {field_name} | "
            f"{_margin_mean(diagnostics['overall'], field_name):.6f} | "
            f"{_outcome_margin_mean(diagnostics, 'win', field_name):.6f} | "
            f"{_outcome_margin_mean(diagnostics, 'loss', field_name):.6f} | "
            f"{_margin_positive_rate(diagnostics['overall'], field_name):.3f} | "
            f"{_outcome_margin_positive_rate(diagnostics, 'win', field_name):.3f} | "
            f"{_outcome_margin_positive_rate(diagnostics, 'loss', field_name):.3f} |"
        )

    lines.extend(["", "## Config Signatures", ""])
    configs = list(diagnostics.get("config_signatures", []) or [])
    if configs:
        for item in configs:
            config = item["config"]
            lines.append(
                "- "
                f"records={item['records']} "
                f"rule_prior_weight={config.get('rule_prior_weight')} "
                f"policy_prior_weight={config.get('policy_prior_weight')} "
                f"neural_action_value_weight={config.get('neural_action_value_weight')} "
                f"neural_tactical_weight={config.get('neural_tactical_weight')}"
            )
    else:
        lines.append("- None")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


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
                f"- Candidate probes: {trace['candidate_probes']}",
                f"- Candidate errors: {trace['candidate_errors']}",
                f"- Truncated candidates: {trace['truncated_candidates']}",
                f"- Truncated candidate rate: {trace['truncated_candidate_rate']:.3f}",
                (
                    "- All-candidates-truncated records: "
                    f"{trace['all_candidates_truncated_records']}"
                ),
                f"- Selected-truncated records: {trace['selected_truncated_records']}",
                (
                    "- Changed selected-truncated records: "
                    f"{trace['changed_selected_truncated_records']}"
                ),
                (
                    "- Changed selected-truncated rate: "
                    f"{trace['changed_selected_truncated_rate']:.3f}"
                ),
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


def write_trace_diagnostic_markdown(
    diagnostics: TraceDiagnostics,
    path: Path,
    *,
    trace_path: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = diagnostics.to_dict()
    lines = [
        "# Phase 5 Search Trace Diagnostics",
        "",
        f"- Trace: `{trace_path.as_posix() if trace_path is not None else 'not provided'}`",
        "",
        "## Summary",
        "",
        f"- Records: {payload['records']}",
        f"- Changed records: {payload['changed_records']}",
        f"- Search errors: {payload['search_errors']}",
        f"- Candidate probes: {payload['candidate_probes']}",
        f"- Candidate errors: {payload['candidate_errors']}",
        f"- Truncated candidates: {payload['truncated_candidates']}",
        f"- Truncated candidate rate: {payload['truncated_candidate_rate']:.3f}",
        (
            "- All-candidates-truncated records: "
            f"{payload['all_candidates_truncated_records']}"
        ),
        (
            "- All-candidates-truncated rate: "
            f"{payload['all_candidates_truncated_rate']:.3f}"
        ),
        f"- Selected-truncated records: {payload['selected_truncated_records']}",
        f"- Selected-truncated rate: {payload['selected_truncated_rate']:.3f}",
        (
            "- Changed selected-truncated records: "
            f"{payload['changed_selected_truncated_records']}"
        ),
        (
            "- Changed selected-truncated rate: "
            f"{payload['changed_selected_truncated_rate']:.3f}"
        ),
        "",
        "## Selected-Truncated By Type",
        "",
    ]
    by_type = payload["selected_truncated_by_type"]
    if by_type:
        for option_type, count in by_type.items():
            lines.append(f"- {option_type or 'UNKNOWN'}: {count}")
    else:
        lines.append("- None")

    lines.extend(["", "## Selected-Truncated By Matchup", ""])
    by_matchup = payload["selected_truncated_by_matchup"]
    if by_matchup:
        lines.extend(
            [
                "| Deck | Opponent | Count |",
                "| ---: | --- | ---: |",
            ]
        )
        for row in by_matchup:
            lines.append(f"| {row['deck_index']} | {row['opponent']} | {row['count']} |")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Margins",
            "",
            (
                "- Mean search-minus-baseline combined score: "
                f"{payload['mean_search_minus_baseline_combined_score']:.6f}"
            ),
            (
                "- Mean search-minus-baseline tactical score: "
                f"{payload['mean_search_minus_baseline_tactical_score']:.6f}"
            ),
        ]
    )
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


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _candidate_for_indices(
    candidates: Sequence[dict[str, Any]],
    indices: Any,
) -> dict[str, Any] | None:
    wanted = tuple(int(index) for index in (indices or []))
    for candidate in candidates:
        candidate_indices = tuple(int(index) for index in candidate.get("indices", []) or [])
        if candidate_indices == wanted:
            return candidate
    return None


def _score_config_signature(config: dict[str, Any]) -> str:
    keys = [
        "rule_prior_weight",
        "policy_prior_weight",
        "neural_action_value_weight",
        "neural_tactical_weight",
        "damage_weight",
        "self_damage_weight",
        "prize_weight",
        "opponent_prize_weight",
        "setup_weight",
        "hand_weight",
        "terminal_win_score",
        "terminal_loss_score",
        "truncated_penalty",
        "top_k",
        "max_rollout_steps",
    ]
    return json.dumps(
        {key: config.get(key) for key in keys if key in config},
        sort_keys=True,
    )


def _score_component_weighting_hints(diagnostics: dict[str, Any]) -> list[str]:
    overall = diagnostics.get("overall", {})
    tactical = _component_stats(overall, "tactical_score")
    prior = _component_stats(overall, "prior_score")
    policy_prior = _component_stats(overall, "policy_prior")
    q_prior = _component_stats(overall, "neural_action_value_prior")
    neural_tactical_prior = _component_stats(overall, "neural_tactical_prior")
    combined_margin = _margin_stats(overall, "combined_score")
    tactical_margin = _margin_stats(overall, "tactical_score")
    prior_margin = _margin_stats(overall, "prior_score")

    hints: list[str] = []
    tactical_range = float(tactical.get("range", 0.0))
    prior_range = float(prior.get("range", 0.0))
    if tactical_range > 0 and prior_range > 0:
        ratio = tactical_range / prior_range
        hints.append(
            "Selected tactical-score range is "
            f"{ratio:.2f}x selected prior-score range "
            f"({tactical_range:.6f} vs {prior_range:.6f})."
        )
        if ratio > 5.0:
            hints.append(
                "Tactical score is much wider than the weighted neural/rule prior; "
                "increase prior weights or normalize tactical before combining if "
                "wins depend on lower-tactical neural choices."
            )
        elif ratio < 0.2:
            hints.append(
                "Weighted priors are much wider than tactical score; reduce prior "
                "weights if the agent ignores immediate board gains."
            )
    elif tactical_range > 0:
        hints.append(
            "Selected prior-score range is zero while tactical varies; current "
            "prior weights are not changing action ranking in these traces."
        )

    prior_sources = {
        "policy_prior": policy_prior,
        "neural_action_value_prior": q_prior,
        "neural_tactical_prior": neural_tactical_prior,
    }
    for name, stats in prior_sources.items():
        if int(stats.get("count", 0)) and float(stats.get("range", 0.0)) == 0.0:
            hints.append(f"{name} is flat across selected candidates in this trace.")

    hints.append(
        "Mean selected-minus-baseline margins: "
        f"combined={float(combined_margin.get('mean', 0.0)):.6f}, "
        f"tactical={float(tactical_margin.get('mean', 0.0)):.6f}, "
        f"prior={float(prior_margin.get('mean', 0.0)):.6f}."
    )
    if float(combined_margin.get("mean", 0.0)) < 0:
        hints.append(
            "Selected candidates score below the baseline on average; inspect "
            "failed traces before raising exploration or PPO pressure."
        )

    by_outcome = diagnostics.get("by_outcome", {})
    win_combined = _outcome_margin_mean(diagnostics, "win", "combined_score")
    loss_combined = _outcome_margin_mean(diagnostics, "loss", "combined_score")
    if by_outcome.get("win") and by_outcome.get("loss"):
        hints.append(
            "Win/loss mean selected-minus-baseline combined margins are "
            f"{win_combined:.6f} / {loss_combined:.6f}."
        )
        if loss_combined > win_combined:
            hints.append(
                "Losing sequences have higher selected-over-baseline combined "
                "margins than winning sequences; the scoring function may be "
                "confident in the wrong short-horizon signals."
            )

    return hints


def _component_stats(group: dict[str, Any], field_name: str) -> dict[str, Any]:
    return (
        group.get("components", {})
        .get(field_name, {})
        .get("selected_candidates", {})
    )


def _margin_stats(group: dict[str, Any], field_name: str) -> dict[str, Any]:
    return group.get("selected_minus_baseline", {}).get(field_name, {})


def _component_mean(group: dict[str, Any], field_name: str) -> float:
    return float(_component_stats(group, field_name).get("mean", 0.0))


def _margin_mean(group: dict[str, Any], field_name: str) -> float:
    return float(_margin_stats(group, field_name).get("mean", 0.0))


def _margin_positive_rate(group: dict[str, Any], field_name: str) -> float:
    return float(_margin_stats(group, field_name).get("positive_rate", 0.0))


def _outcome_group(
    diagnostics: dict[str, Any],
    outcome: str,
) -> dict[str, Any]:
    return diagnostics.get("by_outcome", {}).get(outcome, {})


def _outcome_component_mean(
    diagnostics: dict[str, Any],
    outcome: str,
    field_name: str,
) -> float:
    return _component_mean(_outcome_group(diagnostics, outcome), field_name)


def _outcome_margin_mean(
    diagnostics: dict[str, Any],
    outcome: str,
    field_name: str,
) -> float:
    return _margin_mean(_outcome_group(diagnostics, outcome), field_name)


def _outcome_margin_positive_rate(
    diagnostics: dict[str, Any],
    outcome: str,
    field_name: str,
) -> float:
    return _margin_positive_rate(_outcome_group(diagnostics, outcome), field_name)


def _mean_range(group: dict[str, Any], field_name: str) -> str:
    stats = _component_stats(group, field_name)
    return f"{float(stats.get('mean', 0.0)):.6f}/{float(stats.get('range', 0.0)):.6f}"


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
