"""Build a Phase 5 Alpha full-agent evaluation dashboard.

The script intentionally depends only on the Python standard library so it can
run in the local Codex sandbox and on ERAWAN without extra packages.
"""

from __future__ import annotations

import argparse
import csv
import glob
import html
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ITERATION_RE = re.compile(r"iter0*(\d+)", re.IGNORECASE)
REQUIRED_SAMPLE_MARKER = "Required benchmark sample"


@dataclass(frozen=True)
class Aggregate:
    games: int
    wins: int
    losses: int
    draws: int
    timeouts: int
    errors: int

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0


def parse_iteration(path: Path) -> int:
    match = ITERATION_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse iteration from {path}")
    return int(match.group(1))


def expand_inputs(inputs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        matches = [Path(p) for p in glob.glob(item)]
        if matches:
            paths.extend(matches)
        else:
            paths.append(Path(item))
    unique = {path.resolve(): path for path in paths}
    return sorted(unique.values(), key=lambda p: (parse_iteration(p), p.name))


def aggregate_rows(rows: Iterable[dict]) -> Aggregate:
    rows = list(rows)
    return Aggregate(
        games=sum(int(row.get("games", 0)) for row in rows),
        wins=sum(int(row.get("wins", 0)) for row in rows),
        losses=sum(int(row.get("losses", 0)) for row in rows),
        draws=sum(int(row.get("draws", 0)) for row in rows),
        timeouts=sum(int(row.get("timeouts", 0)) for row in rows),
        errors=sum(int(row.get("errors", 0)) for row in rows),
    )


def deck_name(row: dict) -> str:
    return str(row.get("archetype") or row.get("deck_label", "Unknown").split(" / ")[0])


def load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    rows = list(data.get("rows", []))
    if not rows:
        raise ValueError(f"No matchup rows in {path}")

    iteration = parse_iteration(path)
    overall = aggregate_rows(rows)
    required_rows = [
        row
        for row in rows
        if REQUIRED_SAMPLE_MARKER in str(row.get("opponent_deck_label", ""))
    ]
    required = aggregate_rows(required_rows)

    deck_groups: dict[int, list[dict]] = defaultdict(list)
    required_deck_groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        deck_groups[int(row["deck_index"])].append(row)
    for row in required_rows:
        required_deck_groups[int(row["deck_index"])].append(row)

    decks = []
    for deck_index in sorted(deck_groups):
        group = deck_groups[deck_index]
        sample_group = required_deck_groups.get(deck_index, [])
        decks.append(
            {
                "iteration": iteration,
                "deck_index": deck_index,
                "deck_name": deck_name(group[0]),
                "overall": aggregate_rows(group),
                "required": aggregate_rows(sample_group),
            }
        )

    return {
        "iteration": iteration,
        "path": path,
        "overall": overall,
        "required": required,
        "decks": decks,
        "search": data.get("search_telemetry", {}),
    }


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def fmt_int(value: float | int | None) -> str:
    if value is None:
        return ""
    return f"{int(value):,}"


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def write_summary_csv(path: Path, reports: list[dict], best: dict) -> None:
    fieldnames = [
        "iteration",
        "games",
        "wins",
        "win_rate",
        "delta_wins_vs_best",
        "draws",
        "timeouts",
        "errors",
        "required_sample_games",
        "required_sample_wins",
        "required_sample_win_rate",
        "required_sample_delta_wins_vs_best",
        "searched_decisions",
        "search_changed_decisions",
        "search_change_rate",
        "search_errors",
        "candidate_errors",
        "truncated_candidates",
        "avg_search_seconds",
        "max_search_seconds",
        "source_json",
    ]
    best_required_wins = max(report["required"].wins for report in reports)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            search = report["search"]
            writer.writerow(
                {
                    "iteration": report["iteration"],
                    "games": report["overall"].games,
                    "wins": report["overall"].wins,
                    "win_rate": f"{report['overall'].win_rate:.6f}",
                    "delta_wins_vs_best": report["overall"].wins - best["overall"].wins,
                    "draws": report["overall"].draws,
                    "timeouts": report["overall"].timeouts,
                    "errors": report["overall"].errors,
                    "required_sample_games": report["required"].games,
                    "required_sample_wins": report["required"].wins,
                    "required_sample_win_rate": f"{report['required'].win_rate:.6f}",
                    "required_sample_delta_wins_vs_best": (
                        report["required"].wins - best_required_wins
                    ),
                    "searched_decisions": search.get("searched_decisions", ""),
                    "search_changed_decisions": search.get("changed_decisions", ""),
                    "search_change_rate": search.get("change_rate", ""),
                    "search_errors": search.get("search_errors", ""),
                    "candidate_errors": search.get("candidate_errors", ""),
                    "truncated_candidates": search.get("truncated_candidates", ""),
                    "avg_search_seconds": search.get("avg_search_seconds", ""),
                    "max_search_seconds": search.get("max_search_seconds", ""),
                    "source_json": report["path"].name,
                }
            )


def write_per_deck_csv(path: Path, reports: list[dict]) -> None:
    fieldnames = [
        "iteration",
        "deck_index",
        "deck_name",
        "games",
        "wins",
        "win_rate",
        "draws",
        "timeouts",
        "errors",
        "required_sample_games",
        "required_sample_wins",
        "required_sample_win_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            for deck in report["decks"]:
                overall = deck["overall"]
                required = deck["required"]
                writer.writerow(
                    {
                        "iteration": deck["iteration"],
                        "deck_index": deck["deck_index"],
                        "deck_name": deck["deck_name"],
                        "games": overall.games,
                        "wins": overall.wins,
                        "win_rate": f"{overall.win_rate:.6f}",
                        "draws": overall.draws,
                        "timeouts": overall.timeouts,
                        "errors": overall.errors,
                        "required_sample_games": required.games,
                        "required_sample_wins": required.wins,
                        "required_sample_win_rate": f"{required.win_rate:.6f}",
                    }
                )


def write_matchup_csv(path: Path, reports: list[dict]) -> None:
    fieldnames = [
        "iteration",
        "deck_index",
        "deck_name",
        "opponent",
        "games",
        "wins",
        "losses",
        "draws",
        "timeouts",
        "errors",
        "win_rate",
        "required_sample_opponent",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            with report["path"].open("r", encoding="utf-8") as source:
                rows = json.load(source)["rows"]
            for row in rows:
                games = int(row.get("games", 0))
                wins = int(row.get("wins", 0))
                writer.writerow(
                    {
                        "iteration": report["iteration"],
                        "deck_index": row.get("deck_index"),
                        "deck_name": deck_name(row),
                        "opponent": row.get("opponent", ""),
                        "games": games,
                        "wins": wins,
                        "losses": row.get("losses", 0),
                        "draws": row.get("draws", 0),
                        "timeouts": row.get("timeouts", 0),
                        "errors": row.get("errors", 0),
                        "win_rate": f"{safe_rate(wins, games):.6f}",
                        "required_sample_opponent": REQUIRED_SAMPLE_MARKER
                        in str(row.get("opponent_deck_label", "")),
                    }
                )


def chart_xy(
    series: list[tuple[str, list[tuple[int, float]], str]],
    *,
    y_label: str,
    y_min: float | None = None,
    y_max: float | None = None,
    width: int = 900,
    height: int = 300,
    percent: bool = True,
) -> str:
    all_points = [point for _, points, _ in series for point in points]
    xs = [x for x, _ in all_points]
    ys = [y for _, y in all_points]
    if not all_points:
        return ""
    min_x, max_x = min(xs), max(xs)
    min_y = min(ys) if y_min is None else y_min
    max_y = max(ys) if y_max is None else y_max
    if math.isclose(min_y, max_y):
        min_y -= 0.01
        max_y += 0.01
    y_pad = (max_y - min_y) * 0.08
    min_y = max(0.0, min_y - y_pad) if percent else min_y - y_pad
    max_y = max_y + y_pad

    left, right, top, bottom = 64, 24, 24, 46
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(x: int) -> float:
        if max_x == min_x:
            return left + plot_w / 2
        return left + (x - min_x) / (max_x - min_x) * plot_w

    def sy(y: float) -> float:
        return top + (max_y - y) / (max_y - min_y) * plot_h

    y_ticks = [min_y + (max_y - min_y) * i / 4 for i in range(5)]
    x_ticks = sorted(set(xs))
    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">',
        f'<text x="14" y="18" class="axis-label">{html.escape(y_label)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>',
    ]
    for tick in y_ticks:
        y = sy(tick)
        label = pct(tick) if percent else f"{tick:.0f}"
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" class="grid"/>'
        )
        parts.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end">{label}</text>')
    for tick in x_ticks:
        x = sx(tick)
        parts.append(f'<text x="{x:.2f}" y="{top + plot_h + 26}" text-anchor="middle">{tick}</text>')

    legend_x = left
    for name, points, color in series:
        coords = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
        parts.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for x, y in points:
            parts.append(
                f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4" fill="{color}">'
                f"<title>Iter {x}: {pct(y) if percent else f'{y:.0f}'}</title></circle>"
            )
        parts.append(f'<rect x="{legend_x}" y="{height - 16}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 18}" y="{height - 6}">{html.escape(name)}</text>')
        legend_x += 180
    parts.append("</svg>")
    return "\n".join(parts)


def chart_bars(
    reports: list[dict],
    *,
    best_wins: int,
    width: int = 900,
    height: int = 280,
) -> str:
    values = [(report["iteration"], report["overall"].wins) for report in reports]
    if not values:
        return ""
    left, right, top, bottom = 64, 24, 22, 46
    plot_w = width - left - right
    plot_h = height - top - bottom
    min_v = min(v for _, v in values) - 20
    max_v = max(v for _, v in values) + 20
    span = max_v - min_v
    bar_gap = 8
    bar_w = (plot_w - bar_gap * (len(values) - 1)) / len(values)

    def sy(value: float) -> float:
        return top + (max_v - value) / span * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">',
        '<text x="14" y="18" class="axis-label">Wins</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>',
    ]
    for i in range(5):
        tick = min_v + span * i / 4
        y = sy(tick)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" class="grid"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end">{tick:.0f}</text>')
    best_y = sy(best_wins)
    parts.append(
        f'<line x1="{left}" y1="{best_y:.2f}" x2="{left + plot_w}" y2="{best_y:.2f}" '
        'class="best-line"/>'
    )
    parts.append(f'<text x="{left + plot_w - 4}" y="{best_y - 6:.2f}" text-anchor="end">best {best_wins}</text>')
    for i, (iteration, wins) in enumerate(values):
        x = left + i * (bar_w + bar_gap)
        y = sy(wins)
        h = top + plot_h - y
        fill = "#2563eb" if wins == best_wins else "#64748b"
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" '
            f'rx="3" fill="{fill}"><title>Iter {iteration}: {wins} wins</title></rect>'
        )
        parts.append(f'<text x="{x + bar_w / 2:.2f}" y="{top + plot_h + 26}" text-anchor="middle">{iteration}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def heat_color(rate: float) -> str:
    # 0.0 red, 0.5 amber, 0.8+ green.
    clamped = max(0.0, min(0.8, rate))
    hue = 5 + (clamped / 0.8) * 130
    return f"hsl({hue:.1f} 72% 78%)"


def heatmap_table(reports: list[dict], *, required: bool = False) -> str:
    iterations = [report["iteration"] for report in reports]
    deck_map: dict[int, dict[int, dict]] = defaultdict(dict)
    deck_names: dict[int, str] = {}
    for report in reports:
        for deck in report["decks"]:
            deck_map[deck["deck_index"]][report["iteration"]] = deck
            deck_names[deck["deck_index"]] = deck["deck_name"]

    headers = "".join(f"<th>Iter {iteration}</th>" for iteration in iterations)
    rows = [
        "<table class=\"heatmap\">",
        f"<thead><tr><th>Deck</th><th>Archetype</th>{headers}</tr></thead>",
        "<tbody>",
    ]
    for deck_index in sorted(deck_map):
        cells = []
        for iteration in iterations:
            deck = deck_map[deck_index].get(iteration)
            if not deck:
                cells.append("<td></td>")
                continue
            agg = deck["required"] if required else deck["overall"]
            title = f"{agg.wins}/{agg.games}"
            cells.append(
                f'<td style="background:{heat_color(agg.win_rate)}">'
                f'<span title="{html.escape(title)}">{pct(agg.win_rate)}</span></td>'
            )
        rows.append(
            f"<tr><th>{deck_index}</th><td>{html.escape(deck_names[deck_index])}</td>{''.join(cells)}</tr>"
        )
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)


def table(headers: list[str], rows: list[list[str]], *, class_name: str = "") -> str:
    class_attr = f' class="{class_name}"' if class_name else ""
    out = [f"<table{class_attr}>", "<thead><tr>"]
    out.extend(f"<th>{html.escape(header)}</th>" for header in headers)
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        out.extend(f"<td>{cell}</td>" for cell in row)
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def write_dashboard(path: Path, reports: list[dict], summary_csv: Path, per_deck_csv: Path, matchup_csv: Path) -> None:
    best = max(reports, key=lambda report: (report["overall"].wins, report["iteration"]))
    best_required = max(reports, key=lambda report: (report["required"].wins, report["iteration"]))
    latest = max(reports, key=lambda report: report["iteration"])
    iterations = [report["iteration"] for report in reports]
    missing_iterations = sorted(set(range(min(iterations), max(iterations) + 1)) - set(iterations))

    overall_series = [
        (report["iteration"], report["overall"].win_rate)
        for report in reports
    ]
    required_series = [
        (report["iteration"], report["required"].win_rate)
        for report in reports
    ]

    overall_rows = []
    for report in reports:
        overall = report["overall"]
        required = report["required"]
        overall_rows.append(
            [
                str(report["iteration"]),
                fmt_int(overall.wins),
                fmt_int(overall.games),
                pct(overall.win_rate),
                f"{overall.wins - best['overall'].wins:+d}",
                fmt_int(required.wins),
                fmt_int(required.games),
                pct(required.win_rate),
                f"{required.wins - best_required['required'].wins:+d}",
                fmt_int(overall.draws),
                fmt_int(overall.timeouts),
                fmt_int(overall.errors),
            ]
        )

    latest_decks = sorted(latest["decks"], key=lambda deck: deck["overall"].win_rate, reverse=True)
    latest_deck_rows = [
        [
            str(deck["deck_index"]),
            html.escape(deck["deck_name"]),
            fmt_int(deck["overall"].wins),
            fmt_int(deck["overall"].games),
            pct(deck["overall"].win_rate),
            fmt_int(deck["required"].wins),
            fmt_int(deck["required"].games),
            pct(deck["required"].win_rate),
        ]
        for deck in latest_decks
    ]

    deck_delta_rows = []
    deck_indices = sorted({deck["deck_index"] for report in reports for deck in report["decks"]})
    for deck_index in deck_indices:
        deck_versions = [
            deck
            for report in reports
            for deck in report["decks"]
            if deck["deck_index"] == deck_index
        ]
        best_deck = max(deck_versions, key=lambda deck: (deck["overall"].wins, deck["iteration"]))
        latest_deck = next(deck for deck in latest["decks"] if deck["deck_index"] == deck_index)
        deck_delta_rows.append(
            [
                str(deck_index),
                html.escape(latest_deck["deck_name"]),
                str(best_deck["iteration"]),
                fmt_int(best_deck["overall"].wins),
                pct(best_deck["overall"].win_rate),
                fmt_int(latest_deck["overall"].wins),
                pct(latest_deck["overall"].win_rate),
                f"{latest_deck['overall'].wins - best_deck['overall'].wins:+d}",
            ]
        )

    search_rows = []
    for report in reports:
        search = report["search"]
        search_rows.append(
            [
                str(report["iteration"]),
                fmt_int(search.get("searched_decisions")),
                fmt_int(search.get("changed_decisions")),
                pct(float(search.get("change_rate", 0.0))),
                fmt_int(search.get("search_errors")),
                fmt_int(search.get("candidate_errors")),
                fmt_int(search.get("truncated_candidates")),
                f"{float(search.get('avg_search_seconds', 0.0)):.4f}",
                f"{float(search.get('max_search_seconds', 0.0)):.4f}",
            ]
        )

    source_rows = [
        [str(report["iteration"]), html.escape(report["path"].name)]
        for report in reports
    ]

    css = """
    :root { color-scheme: light; --ink:#172033; --muted:#64748b; --line:#d9e2ef; --bg:#f6f8fb; --panel:#ffffff; }
    * { box-sizing: border-box; }
    body { margin: 0; font: 14px/1.45 "Segoe UI", Arial, sans-serif; color: var(--ink); background: var(--bg); }
    header { padding: 28px 32px 18px; background: #111827; color: white; }
    header p { margin: 6px 0 0; color: #cbd5e1; }
    main { padding: 24px 32px 40px; max-width: 1400px; margin: 0 auto; }
    h1 { margin: 0; font-size: 28px; font-weight: 700; }
    h2 { margin: 26px 0 12px; font-size: 20px; }
    h3 { margin: 20px 0 8px; font-size: 16px; }
    a { color: #2563eb; }
    .cards { display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 12px; margin: 18px 0 10px; }
    .card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; }
    .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
    .value { font-size: 24px; font-weight: 700; margin-top: 4px; }
    .note { color: var(--muted); margin-top: 4px; }
    .grid2 { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; overflow: auto; }
    table { width: 100%; border-collapse: collapse; background: var(--panel); }
    th, td { border-bottom: 1px solid var(--line); padding: 7px 9px; text-align: right; white-space: nowrap; }
    th:first-child, td:first-child, td:nth-child(2) { text-align: left; }
    thead th { position: sticky; top: 0; background: #eef3f8; z-index: 1; }
    .heatmap th, .heatmap td { text-align: center; }
    .heatmap td:nth-child(2), .heatmap th:nth-child(2) { text-align: left; }
    .chart { width: 100%; min-height: 240px; }
    .axis { stroke: #475569; stroke-width: 1.2; }
    .grid { stroke: #e2e8f0; stroke-width: 1; }
    .best-line { stroke: #dc2626; stroke-width: 2; stroke-dasharray: 6 4; }
    svg text { fill: #334155; font-size: 12px; }
    .axis-label { fill: #64748b; font-weight: 600; }
    .callout { padding: 12px 14px; background: #fff7ed; border: 1px solid #fed7aa; border-radius: 8px; }
    .files { font-family: Consolas, "Courier New", monospace; font-size: 12px; }
    @media (max-width: 900px) { .cards, .grid2 { grid-template-columns: 1fr; } main { padding: 16px; } header { padding: 20px 16px; } }
    """

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phase 5 Alpha Evaluation Dashboard</title>
  <style>{css}</style>
</head>
<body>
<header>
  <h1>Phase 5 Alpha Evaluation Dashboard</h1>
  <p>Full-agent specialists vs 13-deck rule-agent league, 30 games per matchup. Generated from iterations {", ".join(str(i) for i in iterations)}.</p>
</header>
<main>
  <section class="cards">
    <div class="card"><div class="label">Best Overall</div><div class="value">Iter {best["iteration"]}</div><div class="note">{best["overall"].wins:,}/{best["overall"].games:,} wins, {pct(best["overall"].win_rate)}</div></div>
    <div class="card"><div class="label">Latest</div><div class="value">Iter {latest["iteration"]}</div><div class="note">{latest["overall"].wins:,}/{latest["overall"].games:,} wins, {pct(latest["overall"].win_rate)} ({latest["overall"].wins - best["overall"].wins:+d} vs best)</div></div>
    <div class="card"><div class="label">Best Required Sample</div><div class="value">Iter {best_required["iteration"]}</div><div class="note">{best_required["required"].wins:,}/{best_required["required"].games:,} wins, {pct(best_required["required"].win_rate)}</div></div>
    <div class="card"><div class="label">Training Status</div><div class="value">Stopped at 10</div><div class="note">Finish eval analysis; do not queue iter11 without a plan change.</div></div>
  </section>

  <section class="callout">
    <strong>Conclusion:</strong> iteration {best["iteration"]} remains the promotion/package candidate. Iterations 8, 9, and 10 did not beat its {best["overall"].wins:,}/{best["overall"].games:,} overall gate.
    Missing eval iteration(s): {", ".join(str(i) for i in missing_iterations) if missing_iterations else "none"}.
  </section>

  <section class="grid2">
    <div class="panel">
      <h2>Win-Rate Trend</h2>
      {chart_xy([
        ("Overall", overall_series, "#2563eb"),
        ("Required sample", required_series, "#ea580c"),
      ], y_label="Win rate", y_min=0.38, y_max=0.56)}
    </div>
    <div class="panel">
      <h2>Total Wins</h2>
      {chart_bars(reports, best_wins=best["overall"].wins)}
    </div>
  </section>

  <section class="panel">
    <h2>Overall Iterations</h2>
    {table([
        "Iter", "Wins", "Games", "Win rate", "Delta vs best",
        "Req wins", "Req games", "Req rate", "Req delta", "Draws", "Timeouts", "Errors"
    ], overall_rows)}
  </section>

  <section class="panel">
    <h2>Per-Deck Overall Heatmap</h2>
    {heatmap_table(reports)}
  </section>

  <section class="panel">
    <h2>Per-Deck Required-Sample Heatmap</h2>
    {heatmap_table(reports, required=True)}
  </section>

  <section class="grid2">
    <div class="panel">
      <h2>Latest Iteration Deck Ranking</h2>
      {table(["Deck", "Archetype", "Wins", "Games", "Rate", "Req wins", "Req games", "Req rate"], latest_deck_rows)}
    </div>
    <div class="panel">
      <h2>Latest vs Per-Deck Best</h2>
      {table(["Deck", "Archetype", "Best iter", "Best wins", "Best rate", "Latest wins", "Latest rate", "Delta"], deck_delta_rows)}
    </div>
  </section>

  <section class="panel">
    <h2>Search Telemetry</h2>
    {table(["Iter", "Searched", "Changed", "Change rate", "Search errors", "Candidate errors", "Truncated", "Avg sec", "Max sec"], search_rows)}
  </section>

  <section class="panel">
    <h2>Artifacts</h2>
    <p>Derived CSVs: <span class="files">{html.escape(summary_csv.name)}</span>, <span class="files">{html.escape(per_deck_csv.name)}</span>, <span class="files">{html.escape(matchup_csv.name)}</span>.</p>
    <p>Source eval JSONs were read from local files and are listed below; raw uploaded reports are intentionally not copied into the repo dashboard bundle.</p>
    {table(["Iter", "Source JSON"], source_rows, class_name="files")}
  </section>
</main>
</body>
</html>
"""
    path.write_text(html_doc, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Input JSON path or glob. Repeatable.")
    parser.add_argument("--output-dir", default="reports", help="Directory for dashboard artifacts.")
    parser.add_argument("--prefix", default="phase5_alpha_eval_dashboard", help="Output file prefix.")
    args = parser.parse_args()

    input_paths = expand_inputs(args.input)
    reports = [load_report(path) for path in input_paths]
    # If duplicate iterations are supplied, keep the last path in sorted order.
    by_iteration = {report["iteration"]: report for report in reports}
    reports = [by_iteration[iteration] for iteration in sorted(by_iteration)]
    if not reports:
        raise SystemExit("No reports loaded")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{args.prefix}.html"
    summary_csv = output_dir / f"{args.prefix}_summary.csv"
    per_deck_csv = output_dir / f"{args.prefix}_per_deck.csv"
    matchup_csv = output_dir / f"{args.prefix}_matchups.csv"

    best = max(reports, key=lambda report: (report["overall"].wins, report["iteration"]))
    write_summary_csv(summary_csv, reports, best)
    write_per_deck_csv(per_deck_csv, reports)
    write_matchup_csv(matchup_csv, reports)
    write_dashboard(html_path, reports, summary_csv, per_deck_csv, matchup_csv)

    print(f"Wrote {html_path}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {per_deck_csv}")
    print(f"Wrote {matchup_csv}")
    print(
        "Best overall: "
        f"iter {best['iteration']} {best['overall'].wins}/{best['overall'].games} "
        f"({best['overall'].win_rate:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
