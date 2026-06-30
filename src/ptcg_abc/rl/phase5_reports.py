from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def compare_benchmark_reports(
    *,
    baseline_path: Path,
    candidate_path: Path,
    output_json: Path | None = None,
    output_md: Path | None = None,
) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    payload = {
        "baseline_path": baseline_path.as_posix(),
        "candidate_path": candidate_path.as_posix(),
        "baseline": _report_totals(baseline),
        "candidate": _report_totals(candidate),
        "overall_delta": {},
        "deck_deltas": [],
        "matchup_deltas": [],
    }
    payload["overall_delta"] = _delta(payload["baseline"], payload["candidate"])
    baseline_decks = _aggregate_rows(baseline.get("rows", []), key_fields=("deck_index",))
    candidate_decks = _aggregate_rows(candidate.get("rows", []), key_fields=("deck_index",))
    payload["deck_deltas"] = _joined_deltas(baseline_decks, candidate_decks)
    baseline_matchups = _aggregate_rows(
        baseline.get("rows", []),
        key_fields=("deck_index", "opponent"),
    )
    candidate_matchups = _aggregate_rows(
        candidate.get("rows", []),
        key_fields=("deck_index", "opponent"),
    )
    payload["matchup_deltas"] = _joined_deltas(baseline_matchups, candidate_matchups)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(_comparison_markdown(payload), encoding="utf-8")
    return payload


def _report_totals(report: dict[str, Any]) -> dict[str, Any]:
    return _sum_rows(report.get("rows", []))


def _aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = "|".join(str(row.get(field, "")) for field in key_fields)
        if key not in output:
            output[key] = {
                "key": key,
                **{field: row.get(field) for field in key_fields},
                "deck_label": row.get("deck_label"),
                "archetype": row.get("archetype"),
                "opponent": row.get("opponent"),
                "games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "timeouts": 0,
                "errors": 0,
            }
        target = output[key]
        for field in ("games", "wins", "losses", "draws", "timeouts", "errors"):
            target[field] += int(row.get(field, 0) or 0)
    for row in output.values():
        row["win_rate"] = row["wins"] / row["games"] if row["games"] else 0.0
    return output


def _joined_deltas(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for key in sorted(set(baseline) | set(candidate)):
        base = baseline.get(key, _empty_row(key))
        cand = candidate.get(key, _empty_row(key))
        output.append(
            {
                "key": key,
                "baseline": base,
                "candidate": cand,
                "delta": _delta(base, cand),
            }
        )
    return output


def _sum_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = {
        "games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "timeouts": 0,
        "errors": 0,
    }
    for row in rows:
        for field in total:
            total[field] += int(row.get(field, 0) or 0)
    total["win_rate"] = total["wins"] / total["games"] if total["games"] else 0.0
    return total


def _delta(base: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    return {
        "games": int(cand.get("games", 0) or 0) - int(base.get("games", 0) or 0),
        "wins": int(cand.get("wins", 0) or 0) - int(base.get("wins", 0) or 0),
        "losses": int(cand.get("losses", 0) or 0) - int(base.get("losses", 0) or 0),
        "draws": int(cand.get("draws", 0) or 0) - int(base.get("draws", 0) or 0),
        "timeouts": int(cand.get("timeouts", 0) or 0)
        - int(base.get("timeouts", 0) or 0),
        "errors": int(cand.get("errors", 0) or 0) - int(base.get("errors", 0) or 0),
        "win_rate": float(cand.get("win_rate", 0.0) or 0.0)
        - float(base.get("win_rate", 0.0) or 0.0),
    }


def _empty_row(key: str) -> dict[str, Any]:
    return {
        "key": key,
        "games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "timeouts": 0,
        "errors": 0,
        "win_rate": 0.0,
    }


def _comparison_markdown(payload: dict[str, Any]) -> str:
    baseline = payload["baseline"]
    candidate = payload["candidate"]
    delta = payload["overall_delta"]
    lines = [
        "# Phase 5 Benchmark Comparison",
        "",
        f"Baseline: `{payload['baseline_path']}`",
        f"Candidate: `{payload['candidate_path']}`",
        "",
        "## Overall",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "| --- | ---: | ---: | ---: |",
        f"| Games | {baseline['games']} | {candidate['games']} | {delta['games']} |",
        f"| Wins | {baseline['wins']} | {candidate['wins']} | {delta['wins']} |",
        f"| Losses | {baseline['losses']} | {candidate['losses']} | {delta['losses']} |",
        f"| Draws | {baseline['draws']} | {candidate['draws']} | {delta['draws']} |",
        f"| Timeouts | {baseline['timeouts']} | {candidate['timeouts']} | {delta['timeouts']} |",
        f"| Errors | {baseline['errors']} | {candidate['errors']} | {delta['errors']} |",
        f"| Win rate | {baseline['win_rate']:.3f} | {candidate['win_rate']:.3f} | {delta['win_rate']:+.3f} |",
        "",
        "## Deck Deltas",
        "",
        "| Key | Wins | Win rate | Errors | Timeouts |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["deck_deltas"]:
        row_delta = row["delta"]
        label = row["candidate"].get("archetype") or row["baseline"].get("archetype") or row["key"]
        lines.append(
            f"| {label} | {row_delta['wins']:+d} | "
            f"{row_delta['win_rate']:+.3f} | {row_delta['errors']:+d} | "
            f"{row_delta['timeouts']:+d} |"
        )
    lines.append("")
    return "\n".join(lines)
