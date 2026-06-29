from __future__ import annotations

import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path

from ptcg_abc.corpus import write_deck_csv


MAIN_PY = '''from __future__ import annotations

import os

from cg.api import all_attack, all_card_data, to_observation_class
from ptcg_abc.agent import RuleBasedAgent


_AGENT = None


def read_deck_csv() -> list[int]:
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle.readlines() if line.strip()]
    return [int(value) for value in lines[:60]]


def agent(obs_dict: dict) -> list[int]:
    global _AGENT
    obs = to_observation_class(obs_dict)
    if _AGENT is None:
        _AGENT = RuleBasedAgent(
            read_deck_csv(),
            card_data=all_card_data(),
            attack_data=all_attack(),
        )
    return _AGENT.act(obs)
'''

HYBRID_RL_MAIN_PY = '''from __future__ import annotations

import os

from cg.api import all_attack, all_card_data, to_observation_class
from ptcg_abc.agent import HybridRlAgent


_AGENT = None


def read_deck_csv() -> list[int]:
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle.readlines() if line.strip()]
    return [int(value) for value in lines[:60]]


def model_path() -> str | None:
    file_path = "model.json"
    if os.path.exists(file_path):
        return file_path
    kaggle_path = "/kaggle_simulations/agent/" + file_path
    if os.path.exists(kaggle_path):
        return kaggle_path
    return None


def agent(obs_dict: dict) -> list[int]:
    global _AGENT
    obs = to_observation_class(obs_dict)
    if _AGENT is None:
        _AGENT = HybridRlAgent(
            read_deck_csv(),
            card_data=all_card_data(),
            attack_data=all_attack(),
            model_path=model_path(),
        )
    return _AGENT.act(obs)
'''

PHASE5_SEARCH_MAIN_PY = '''from __future__ import annotations

import os
from collections import Counter
from typing import Any

from cg.api import all_attack, all_card_data, to_observation_class
from ptcg_abc.agent import Phase5SearchPolicyAgent
from ptcg_abc.evaluation import phase5_league_prepared_decks


_AGENT = None
_OPPONENT_PRIORS = None
_HERE = os.path.dirname(os.path.abspath(__file__))


def _agent_root() -> str:
    if os.path.exists(os.path.join(_HERE, "cg")):
        return _HERE
    if os.path.exists("cg"):
        return os.getcwd()
    return "/kaggle_simulations/agent"


def _local_path(name: str) -> str:
    path = name
    if os.path.exists(path):
        return path
    return os.path.join(_agent_root(), name)


def read_deck_csv() -> list[int]:
    with open(_local_path("deck.csv"), "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle.readlines() if line.strip()]
    return [int(value) for value in lines[:60]]


def model_path() -> str:
    return _local_path("model.pt")


def opponent_priors() -> list[tuple[str, list[int]]]:
    global _OPPONENT_PRIORS
    if _OPPONENT_PRIORS is None:
        _OPPONENT_PRIORS = [
            (deck.label, list(deck.card_ids))
            for deck in phase5_league_prepared_decks()
        ]
    return _OPPONENT_PRIORS


def _card_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, dict):
        output: list[int] = []
        for key in ("id", "cardID", "cardId", "card_ids", "cards"):
            if key in value:
                output.extend(_card_ids(value[key]))
        return output
    if isinstance(value, (list, tuple)):
        output = []
        for item in value:
            output.extend(_card_ids(item))
        return output
    for name in ("id", "cardID", "cardId"):
        if hasattr(value, name):
            try:
                return [int(getattr(value, name))]
            except (TypeError, ValueError):
                return []
    for name in ("cards", "card_ids"):
        if hasattr(value, name):
            return _card_ids(getattr(value, name))
    return []


def _visible_opponent_ids(obs: Any) -> list[int]:
    current = getattr(obs, "current", None)
    players = list(getattr(current, "players", []) or [])
    your_index = int(getattr(current, "yourIndex", 0) or 0)
    opponent_index = 1 - your_index
    opponent = players[opponent_index] if 0 <= opponent_index < len(players) else None
    ids: list[int] = []
    for zone_name in ("active", "bench", "discard", "lostZone"):
        ids.extend(_card_ids(getattr(opponent, zone_name, None)))
    return ids


def choose_opponent_deck(obs: Any) -> list[int]:
    priors = opponent_priors()
    visible = _visible_opponent_ids(obs)
    if not visible:
        for label, deck in priors:
            if label.startswith("Crustle / Required benchmark sample"):
                return list(deck)
        return list(priors[0][1])
    visible_counts = Counter(visible)
    best_score = None
    best_deck = priors[0][1]
    for _, deck in priors:
        deck_counts = Counter(deck)
        overlap = sum(min(count, deck_counts.get(card_id, 0)) for card_id, count in visible_counts.items())
        coverage = sum(1 for card_id in visible_counts if card_id in deck_counts)
        score = (overlap, coverage)
        if best_score is None or score > best_score:
            best_score = score
            best_deck = deck
    return list(best_deck)


def agent(obs_dict: dict) -> list[int]:
    global _AGENT
    obs = to_observation_class(obs_dict)
    if _AGENT is None:
        _AGENT = Phase5SearchPolicyAgent(
            read_deck_csv(),
            opponent_deck_ids=choose_opponent_deck(obs),
            sample_dir=_agent_root(),
            card_data=all_card_data(),
            attack_data=all_attack(),
            checkpoint_path=model_path(),
        )
    _AGENT.opponent_deck_ids = choose_opponent_deck(obs)
    return _AGENT.act(obs)
'''


@dataclass(frozen=True)
class SubmissionBuildResult:
    output_dir: Path
    tar_path: Path
    main_path: Path
    deck_path: Path


def _ignore_generated(dir_name: str, names: list[str]) -> set[str]:
    return {name for name in names if name == "__pycache__" or name.endswith(".pyc")}


def _copy_agent_package(src_root: Path, output_dir: Path) -> None:
    package_dir = output_dir / "ptcg_abc"
    agent_dir = package_dir / "agent"
    rl_dir = package_dir / "rl"
    agent_dir.mkdir(parents=True, exist_ok=True)
    rl_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_root / "ptcg_abc" / "__init__.py", package_dir / "__init__.py")
    shutil.copy2(src_root / "ptcg_abc" / "agent" / "__init__.py", agent_dir / "__init__.py")
    shutil.copy2(src_root / "ptcg_abc" / "agent" / "hybrid_rl.py", agent_dir / "hybrid_rl.py")
    shutil.copy2(src_root / "ptcg_abc" / "agent" / "rule_based.py", agent_dir / "rule_based.py")
    shutil.copy2(src_root / "ptcg_abc" / "agent" / "random_agent.py", agent_dir / "random_agent.py")
    for name in [
        "__init__.py",
        "dataset.py",
        "featurizer.py",
        "guidance.py",
        "model.py",
        "records.py",
        "rewards.py",
    ]:
        shutil.copy2(src_root / "ptcg_abc" / "rl" / name, rl_dir / name)


def _copy_full_package(src_root: Path, output_dir: Path) -> None:
    shutil.copytree(
        src_root / "ptcg_abc",
        output_dir / "ptcg_abc",
        dirs_exist_ok=True,
        ignore=_ignore_generated,
    )


def build_submission_bundle(
    *,
    deck_ids: list[int],
    sample_dir: Path,
    output_dir: Path,
    tar_path: Path | None = None,
    src_root: Path = Path("src"),
) -> SubmissionBuildResult:
    if not (sample_dir / "cg").exists():
        raise FileNotFoundError(f"Kaggle sample cg package not found at {sample_dir}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    main_path = output_dir / "main.py"
    deck_path = output_dir / "deck.csv"
    main_path.write_text(MAIN_PY, encoding="utf-8")
    write_deck_csv(deck_ids, deck_path)

    shutil.copytree(
        sample_dir / "cg",
        output_dir / "cg",
        dirs_exist_ok=True,
        ignore=_ignore_generated,
    )
    _copy_agent_package(src_root, output_dir)

    tar_path = tar_path or (output_dir / "submission.tar.gz")
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w:gz") as tar:
        for name in ["main.py", "deck.csv", "cg", "ptcg_abc"]:
            tar.add(output_dir / name, arcname=name)

    return SubmissionBuildResult(
        output_dir=output_dir,
        tar_path=tar_path,
        main_path=main_path,
        deck_path=deck_path,
    )


def build_hybrid_rl_submission_bundle(
    *,
    deck_ids: list[int],
    sample_dir: Path,
    output_dir: Path,
    model_path: Path | None = None,
    tar_path: Path | None = None,
    src_root: Path = Path("src"),
) -> SubmissionBuildResult:
    if not (sample_dir / "cg").exists():
        raise FileNotFoundError(f"Kaggle sample cg package not found at {sample_dir}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    main_path = output_dir / "main.py"
    deck_path = output_dir / "deck.csv"
    main_path.write_text(HYBRID_RL_MAIN_PY, encoding="utf-8")
    write_deck_csv(deck_ids, deck_path)
    if model_path is not None and model_path.exists():
        shutil.copy2(model_path, output_dir / "model.json")

    shutil.copytree(
        sample_dir / "cg",
        output_dir / "cg",
        dirs_exist_ok=True,
        ignore=_ignore_generated,
    )
    _copy_agent_package(src_root, output_dir)

    tar_path = tar_path or (output_dir / "submission.tar.gz")
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w:gz") as tar:
        names = ["main.py", "deck.csv", "cg", "ptcg_abc"]
        if (output_dir / "model.json").exists():
            names.append("model.json")
        for name in names:
            tar.add(output_dir / name, arcname=name)

    return SubmissionBuildResult(
        output_dir=output_dir,
        tar_path=tar_path,
        main_path=main_path,
        deck_path=deck_path,
    )


def build_phase5_search_submission_bundle(
    *,
    deck_ids: list[int],
    sample_dir: Path,
    output_dir: Path,
    model_path: Path,
    tar_path: Path | None = None,
    src_root: Path = Path("src"),
) -> SubmissionBuildResult:
    if not (sample_dir / "cg").exists():
        raise FileNotFoundError(f"Kaggle sample cg package not found at {sample_dir}.")
    if not model_path.exists():
        raise FileNotFoundError(f"Phase 5 checkpoint not found at {model_path}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    main_path = output_dir / "main.py"
    deck_path = output_dir / "deck.csv"
    main_path.write_text(PHASE5_SEARCH_MAIN_PY, encoding="utf-8")
    write_deck_csv(deck_ids, deck_path)
    shutil.copy2(model_path, output_dir / "model.pt")

    shutil.copytree(
        sample_dir / "cg",
        output_dir / "cg",
        dirs_exist_ok=True,
        ignore=_ignore_generated,
    )
    _copy_full_package(src_root, output_dir)

    tar_path = tar_path or (output_dir / "submission.tar.gz")
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w:gz") as tar:
        for name in ["main.py", "deck.csv", "model.pt", "cg", "ptcg_abc"]:
            tar.add(output_dir / name, arcname=name)

    return SubmissionBuildResult(
        output_dir=output_dir,
        tar_path=tar_path,
        main_path=main_path,
        deck_path=deck_path,
    )
