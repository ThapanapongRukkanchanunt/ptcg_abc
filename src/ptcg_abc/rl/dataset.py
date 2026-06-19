from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ptcg_abc.rl.records import DecisionFrame, TrajectoryStep


def write_decision_jsonl(frames: Iterable[DecisionFrame], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for frame in frames:
            handle.write(json.dumps(frame.to_dict(), sort_keys=True) + "\n")
            count += 1
    return count


def append_decision_jsonl(frame: DecisionFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(frame.to_dict(), sort_keys=True) + "\n")


def read_decision_jsonl(path: Path) -> list[DecisionFrame]:
    frames = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                frames.append(DecisionFrame.from_dict(json.loads(stripped)))
    return frames


def append_trajectory_jsonl(step: TrajectoryStep, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(step.to_dict(), sort_keys=True) + "\n")


def read_trajectory_jsonl(path: Path) -> list[TrajectoryStep]:
    steps = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                steps.append(TrajectoryStep.from_dict(json.loads(stripped)))
    return steps
