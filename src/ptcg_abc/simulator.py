from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

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


@dataclass(frozen=True)
class BattleResult:
    started: bool
    steps: int
    finished: bool
    result: int | None
    winner: int | None
    leader: int | None
    prize_counts: tuple[int, int] | None
    start_error_player: int | None = None
    start_error_type: int | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _with_sample_submission_on_path(sample_dir: Path) -> list[str]:
    previous_path = list(sys.path)
    sys.path.insert(0, str(sample_dir.resolve()))
    return previous_path


def load_engine_metadata(sample_dir: Path) -> tuple[list[Any], list[Any]]:
    if not (sample_dir / "cg" / "api.py").exists():
        raise FileNotFoundError(f"Kaggle sample submission not found at {sample_dir}.")
    previous_path = _with_sample_submission_on_path(sample_dir)
    try:
        from cg.api import all_attack, all_card_data
    finally:
        sys.path = previous_path
    return all_card_data(), all_attack()


def _prize_counts(obs: Any) -> tuple[int, int] | None:
    current = getattr(obs, "current", None)
    players = list(getattr(current, "players", []) or [])
    if len(players) < 2:
        return None
    return (len(getattr(players[0], "prize", []) or []), len(getattr(players[1], "prize", []) or []))


def _leader_from_prizes(prize_counts: tuple[int, int] | None) -> int | None:
    if prize_counts is None or prize_counts[0] == prize_counts[1]:
        return None
    return 0 if prize_counts[0] < prize_counts[1] else 1


def run_battle(
    deck0: list[int],
    deck1: list[int],
    *,
    sample_dir: Path,
    agent0: Any | None = None,
    agent1: Any | None = None,
    card_data: Sequence[Any] | None = None,
    attack_data: Sequence[Any] | None = None,
    max_steps: int = 600,
) -> BattleResult:
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

    if card_data is None or attack_data is None:
        card_data, attack_data = load_engine_metadata(sample_dir)
    agent0 = agent0 or RuleBasedAgent(deck0, card_data=card_data, attack_data=attack_data)
    agent1 = agent1 or RuleBasedAgent(deck1, card_data=card_data, attack_data=attack_data)
    agents = [agent0, agent1]

    obs_dict = None
    obs = None
    steps = 0
    try:
        obs_dict, start_data = battle_start(deck0, deck1)
        if obs_dict is None:
            return BattleResult(
                started=False,
                steps=0,
                finished=False,
                result=None,
                winner=None,
                leader=None,
                prize_counts=None,
                start_error_player=int(start_data.errorPlayer),
                start_error_type=int(start_data.errorType),
            )

        for steps in range(max_steps):
            obs = to_observation_class(obs_dict)
            current = getattr(obs, "current", None)
            result = getattr(current, "result", None)
            if result is not None and int(result) != -1:
                winner = int(result) if int(result) in {0, 1} else None
                prizes = _prize_counts(obs)
                return BattleResult(
                    started=True,
                    steps=steps,
                    finished=True,
                    result=int(result),
                    winner=winner,
                    leader=_leader_from_prizes(prizes),
                    prize_counts=prizes,
                    start_error_player=int(start_data.errorPlayer),
                    start_error_type=int(start_data.errorType),
                )

            your_index = int(getattr(current, "yourIndex", 0) or 0)
            acting_agent = agents[your_index]
            choice = acting_agent.act(obs)
            obs_dict = battle_select(choice)
            observe_after_action = getattr(acting_agent, "observe_after_action", None)
            if obs_dict is not None and callable(observe_after_action):
                observe_after_action(
                    to_observation_class(obs_dict),
                    actor_index=your_index,
                )

        if obs_dict is not None:
            obs = to_observation_class(obs_dict)
        prizes = _prize_counts(obs)
        return BattleResult(
            started=True,
            steps=max_steps,
            finished=False,
            result=None,
            winner=None,
            leader=_leader_from_prizes(prizes),
            prize_counts=prizes,
            start_error_player=int(start_data.errorPlayer),
            start_error_type=int(start_data.errorType),
        )
    except Exception as exc:
        prizes = _prize_counts(obs) if obs is not None else None
        return BattleResult(
            started=obs_dict is not None,
            steps=steps,
            finished=False,
            result=None,
            winner=None,
            leader=_leader_from_prizes(prizes),
            prize_counts=prizes,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if obs_dict is not None:
            battle_finish()


def run_battle_smoke(
    deck0: list[int],
    deck1: list[int],
    *,
    sample_dir: Path,
    max_steps: int = 50,
) -> BattleSmokeResult:
    card_data, attack_data = load_engine_metadata(sample_dir)
    result = run_battle(
        deck0,
        deck1,
        sample_dir=sample_dir,
        card_data=card_data,
        attack_data=attack_data,
        max_steps=max_steps,
    )
    return BattleSmokeResult(
        started=result.started,
        steps=result.steps,
        finished=result.finished,
        result=result.result,
        start_error_player=result.start_error_player,
        start_error_type=result.start_error_type,
        error=result.error,
    )
