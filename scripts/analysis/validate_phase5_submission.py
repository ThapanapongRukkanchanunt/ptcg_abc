"""Validate a built Phase 5 Kaggle submission directory without playing games."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def validate_submission(path: Path) -> dict[str, Any]:
    required = ("main.py", "deck.csv", "model.pt", "cg", "ptcg_abc")
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise ValueError(f"{path}: missing required entries: {', '.join(missing)}")

    deck_ids = [
        int(line.strip())
        for line in (path / "deck.csv").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(deck_ids) != 60:
        raise ValueError(f"{path}: expected 60 deck cards, found {len(deck_ids)}")

    previous_cwd = Path.cwd()
    previous_path = list(sys.path)
    namespace: dict[str, Any] = {}
    try:
        os.chdir(path)
        sys.path.insert(0, str(Path.cwd()))
        exec(compile(Path("main.py").read_text(encoding="utf-8"), "main.py", "exec"), namespace)
        if not callable(namespace.get("agent")):
            raise ValueError(f"{path}: main.py does not expose callable agent")
        if len(namespace["read_deck_csv"]()) != 60:
            raise ValueError(f"{path}: packaged read_deck_csv did not return 60 cards")
        import torch

        checkpoint = torch.load("model.pt", map_location="cpu", weights_only=False)
    finally:
        sys.path[:] = previous_path
        os.chdir(previous_cwd)

    return {
        "path": path.as_posix(),
        "deck_cards": len(deck_ids),
        "agent_callable": True,
        "checkpoint_type": type(checkpoint).__name__,
        "checkpoint_keys": len(checkpoint) if hasattr(checkpoint, "__len__") else None,
        "executed_without_file_global": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([validate_submission(path) for path in args.submission_dir], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
