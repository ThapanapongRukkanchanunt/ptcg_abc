from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from ptcg_abc.agent import RuleBasedAgent


@dataclass(frozen=True)
class BattleSmokeResult:
    started: bool
    steps: int
    finished: bool
    result: int | None
    start_error_player: int | None = None
    start_error_type: int | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _with_sample_submission_on_path(sample_dir: Path) -> list[str]:
    previous_path = list(sys.path)
    sys.path.insert(0, str(sample_dir.resolve()))
    return previous_path


def run_battle_smoke(
    deck0: list[int],
    deck1: list[int],
    *,
    sample_dir: Path,
    max_steps: int = 50,
) -> BattleSmokeResult:
    if len(deck0) != 60 or len(deck1) != 60:
        raise ValueError("Both decks must contain 60 card IDs.")
    if not (sample_dir / "cg" / "game.py").exists():
        raise FileNotFoundError(f"Kaggle sample submission not found at {sample_dir}.")

    previous_path = _with_sample_submission_on_path(sample_dir)
    try:
        from cg.api import to_observation_class
        from cg.game import battle_finish, battle_select, battle_start
    finally:
        sys.path = previous_path

    agents = [RuleBasedAgent(deck0), RuleBasedAgent(deck1)]
    obs_dict = None
    steps = 0
    try:
        obs_dict, start_data = battle_start(deck0, deck1)
        if obs_dict is None:
            return BattleSmokeResult(
                started=False,
                steps=0,
                finished=False,
                result=None,
                start_error_player=int(start_data.errorPlayer),
                start_error_type=int(start_data.errorType),
            )

        for steps in range(max_steps):
            obs = to_observation_class(obs_dict)
            current = getattr(obs, "current", None)
            result = getattr(current, "result", None)
            if result is not None and int(result) != -1:
                return BattleSmokeResult(
                    started=True,
                    steps=steps,
                    finished=True,
                    result=int(result),
                    start_error_player=int(start_data.errorPlayer),
                    start_error_type=int(start_data.errorType),
                )

            your_index = int(getattr(current, "yourIndex", 0) or 0)
            choice = agents[your_index].act(obs)
            obs_dict = battle_select(choice)

        return BattleSmokeResult(
            started=True,
            steps=max_steps,
            finished=False,
            result=None,
            start_error_player=int(start_data.errorPlayer),
            start_error_type=int(start_data.errorType),
        )
    except Exception as exc:
        return BattleSmokeResult(
            started=obs_dict is not None,
            steps=steps,
            finished=False,
            result=None,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if obs_dict is not None:
            battle_finish()
