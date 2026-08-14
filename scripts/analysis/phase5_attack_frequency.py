"""Count chosen attack commands by attack ID and engine-provided attack name."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ptcg_abc.agent.rule_based import _get
from ptcg_abc.simulator import load_engine_metadata


def _attack_name(attack: Any | None, attack_id: int) -> str:
    if attack is None:
        return f"Unknown attack {attack_id}"
    for field in ("name", "attackName"):
        value = _get(attack, field)
        if value:
            return str(value)
    return f"Attack {attack_id}"


def summarize(path: Path, sample_dir: Path) -> dict[str, Any]:
    _, attack_data = load_engine_metadata(sample_dir)
    attacks = {
        int(_get(attack, "attackId")): attack
        for attack in attack_data
        if _get(attack, "attackId") is not None
    }
    counts: Counter[int] = Counter()
    games: set[int] = set()
    decision_count = 0
    attack_decisions = 0
    opponent_counts: dict[str, Counter[int]] = defaultdict(Counter)

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON on line {line_number}: {exc}") from exc
            decision_count += 1
            decision = row.get("decision", {})
            metadata = decision.get("reward_metadata", {})
            game_index = metadata.get("game_index")
            if game_index is not None:
                games.add(int(game_index))
            opponent = str(metadata.get("opponent_agent_key") or "unknown")
            chosen = {int(index) for index in row.get("chosen_indices", [])}
            selected_attacks: list[int] = []
            for action in decision.get("legal_options", []):
                if int(action.get("index", -1)) not in chosen:
                    continue
                if str(action.get("option_type", "")) != "ATTACK":
                    continue
                attack_id = action.get("attack_id")
                if attack_id is not None:
                    selected_attacks.append(int(attack_id))
            if selected_attacks:
                attack_decisions += 1
            for attack_id in selected_attacks:
                counts[attack_id] += 1
                opponent_counts[opponent][attack_id] += 1

    total_attacks = sum(counts.values())
    rows = [
        {
            "attack_id": attack_id,
            "attack_name": _attack_name(attacks.get(attack_id), attack_id),
            "count": count,
            "share_of_attacks": count / total_attacks if total_attacks else 0.0,
            "per_game": count / len(games) if games else 0.0,
            "opponent_counts": {
                opponent: values[attack_id]
                for opponent, values in sorted(opponent_counts.items())
                if values[attack_id]
            },
        }
        for attack_id, count in counts.most_common()
    ]
    phantom_dive_count = sum(
        row["count"] for row in rows if row["attack_name"].casefold() == "phantom dive"
    )
    return {
        "trajectory_path": path.as_posix(),
        "games": len(games),
        "decisions": decision_count,
        "attack_decisions": attack_decisions,
        "total_attack_commands": total_attacks,
        "attack_rate_per_game": total_attacks / len(games) if games else 0.0,
        "phantom_dive_count": phantom_dive_count,
        "attacks": rows,
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 5 Attack Frequency",
        "",
        f"Games: {summary['games']}",
        f"Recorded decisions: {summary['decisions']}",
        f"Attack commands: {summary['total_attack_commands']}",
        f"Phantom Dive count: {summary['phantom_dive_count']}",
        "",
        "| Attack | ID | Count | Share | Per game |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["attacks"]:
        lines.append(
            f"| {row['attack_name']} | {row['attack_id']} | {row['count']} | "
            f"{row['share_of_attacks']:.2%} | {row['per_game']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.trajectory, args.sample_dir)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.report_md.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
