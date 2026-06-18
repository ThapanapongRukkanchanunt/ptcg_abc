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
    agent_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_root / "ptcg_abc" / "__init__.py", package_dir / "__init__.py")
    shutil.copy2(src_root / "ptcg_abc" / "agent" / "__init__.py", agent_dir / "__init__.py")
    shutil.copy2(src_root / "ptcg_abc" / "agent" / "rule_based.py", agent_dir / "rule_based.py")
    shutil.copy2(src_root / "ptcg_abc" / "agent" / "random_agent.py", agent_dir / "random_agent.py")


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
