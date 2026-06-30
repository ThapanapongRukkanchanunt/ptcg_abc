from __future__ import annotations

import shutil
import tarfile
import zipfile
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

from cg.api import all_attack, all_card_data, to_observation_class
from ptcg_abc.agent import Phase5SearchPolicyAgent
from ptcg_abc.rl.phase5_belief import infer_opponent_deck, phase5_league_opponent_priors


_AGENT = None
_OPPONENT_PRIORS = None
_HERE = os.path.dirname(os.path.abspath(globals().get("__file__", "")))


def _agent_root() -> str:
    candidates = [
        _HERE,
        os.getcwd(),
        "/kaggle_simulations/agent",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(os.path.join(candidate, "cg")):
            return candidate
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


def opponent_priors():
    global _OPPONENT_PRIORS
    if _OPPONENT_PRIORS is None:
        _OPPONENT_PRIORS = phase5_league_opponent_priors()
    return _OPPONENT_PRIORS


def choose_opponent_deck(obs: Any) -> list[int]:
    return list(infer_opponent_deck(obs, opponent_priors()).card_ids)


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
    zip_path: Path | None = None


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


def _write_direct_zip(output_dir: Path, zip_path: Path, names: list[str]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            path = output_dir / name
            if path.is_dir():
                for child in path.rglob("*"):
                    if child.is_file():
                        archive.write(child, child.relative_to(output_dir).as_posix())
            else:
                archive.write(path, name)


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
    zip_path: Path | None = None,
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
    names = ["main.py", "deck.csv", "model.pt", "cg", "ptcg_abc"]
    with tarfile.open(tar_path, "w:gz") as tar:
        for name in names:
            tar.add(output_dir / name, arcname=name)
    if zip_path is not None:
        _write_direct_zip(output_dir, zip_path, names)

    return SubmissionBuildResult(
        output_dir=output_dir,
        tar_path=tar_path,
        main_path=main_path,
        deck_path=deck_path,
        zip_path=zip_path,
    )
