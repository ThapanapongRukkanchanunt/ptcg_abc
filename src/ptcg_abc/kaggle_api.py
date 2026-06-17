from __future__ import annotations

import json
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ptcg_abc.config import COMPETITION_SLUG
from ptcg_abc.http_client import basic_auth_header, request_bytes, request_json


KAGGLE_API_BASE = "https://www.kaggle.com/api/v1"


@dataclass(frozen=True)
class KaggleCredentials:
    username: str
    key: str

    @property
    def auth_header(self) -> str:
        return basic_auth_header(self.username, self.key)


class KaggleCredentialsError(RuntimeError):
    pass


def read_kaggle_credentials() -> KaggleCredentials:
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        return KaggleCredentials(username=username, key=key)

    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    config_path = config_dir / "kaggle.json"
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))
        username = data.get("username")
        key = data.get("key")
        if username and key:
            return KaggleCredentials(username=username, key=key)

    raise KaggleCredentialsError(
        "Kaggle credentials were not found. Set KAGGLE_USERNAME and KAGGLE_KEY, "
        "or place kaggle.json in %USERPROFILE%\\.kaggle."
    )


def list_competition_files(
    *,
    competition: str = COMPETITION_SLUG,
    credentials: KaggleCredentials | None = None,
) -> object:
    credentials = credentials or read_kaggle_credentials()
    url = f"{KAGGLE_API_BASE}/competitions/data/list/{competition}"
    return request_json(url, auth_header=credentials.auth_header)


def download_competition_archive(
    destination: Path,
    *,
    competition: str = COMPETITION_SLUG,
    credentials: KaggleCredentials | None = None,
) -> Path:
    credentials = credentials or read_kaggle_credentials()
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / f"{competition}.zip"
    url = f"{KAGGLE_API_BASE}/competitions/data/download-all/{competition}"
    archive_path.write_bytes(request_bytes(url, auth_header=credentials.auth_header, timeout=300))
    return archive_path


def extract_archive(archive_path: Path, destination: Path, *, refresh: bool = False) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    if refresh:
        for path in sorted(destination.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(destination)
    return [path for path in destination.rglob("*") if path.is_file()]


def setup_kaggle_data(
    raw_dir: Path,
    input_dir: Path,
    *,
    competition: str = COMPETITION_SLUG,
    archive_path: Path | None = None,
    refresh: bool = False,
) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    if archive_path is None:
        credentials = read_kaggle_credentials()
        file_list = list_competition_files(competition=competition, credentials=credentials)
        (raw_dir / "files.json").write_text(json.dumps(file_list, indent=2), encoding="utf-8")
    else:
        file_list = {"source": "local_archive", "path": str(archive_path)}
        (raw_dir / "files.json").write_text(json.dumps(file_list, indent=2), encoding="utf-8")

    destination_archive = raw_dir / f"{competition}.zip"
    if archive_path is not None:
        if archive_path.resolve() != destination_archive.resolve():
            shutil.copy2(archive_path, destination_archive)
        archive_path = destination_archive
    elif refresh or not destination_archive.exists():
        archive_path = download_competition_archive(
            raw_dir, competition=competition, credentials=credentials
        )
    else:
        archive_path = destination_archive

    extracted = extract_archive(archive_path, input_dir, refresh=refresh)
    return {
        "competition": competition,
        "archive_path": str(archive_path),
        "input_dir": str(input_dir),
        "file_count": len(extracted),
        "files": [str(path) for path in extracted],
    }
