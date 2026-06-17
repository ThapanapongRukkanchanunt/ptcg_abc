from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ptcg_abc.normalize import clean_text, normalize_card_name


NAME_COLUMNS = {
    "name",
    "card",
    "card_name",
    "cardname",
    "card name",
    "english_name",
    "name_en",
    "en_name",
    "display_name",
}
LEGAL_COLUMNS = {
    "legal",
    "is_legal",
    "is legal",
    "standard_legal",
    "standard legal",
    "allowed",
    "valid",
}
SUPPORTED_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl", ".txt"}


@dataclass
class LegalCardCandidate:
    path: Path
    names: set[str]
    source_column: str
    legal_column: str | None
    score: float
    sample: list[str]

    @property
    def count(self) -> int:
        return len(self.names)


def _path_score(path: Path) -> int:
    text = path.as_posix().casefold()
    score = 0
    if "legal" in text:
        score += 6
    if "card" in text:
        score += 4
    if "regulation" in text:
        score += 2
    if "submission" in text:
        score -= 5
    if "sample" in text:
        score -= 3
    if "deck" in text:
        score -= 2
    return score


def _is_truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = clean_text(str(value)).casefold()
    return text in {"1", "true", "yes", "y", "legal", "allowed", "valid", "standard"}


def _score_candidate(path: Path, count: int, has_legal_column: bool) -> float:
    score = float(_path_score(path))
    if has_legal_column:
        score += 4
    if 50 <= count <= 2500:
        score += 3
    elif count > 2500:
        score -= 1
    score += min(count, 2000) / 1000
    return score


def _candidate_from_rows(path: Path, rows: list[dict[str, object]]) -> list[LegalCardCandidate]:
    if not rows:
        return []

    columns = {column for row in rows for column in row.keys()}
    normalized_columns = {clean_text(column).casefold(): column for column in columns}
    name_columns = [
        original for normalized, original in normalized_columns.items() if normalized in NAME_COLUMNS
    ]
    legal_columns = [
        original for normalized, original in normalized_columns.items() if normalized in LEGAL_COLUMNS
    ]
    if not name_columns:
        return []

    candidates: list[LegalCardCandidate] = []
    for name_column in name_columns:
        legal_column = legal_columns[0] if legal_columns else None
        names: set[str] = set()
        for row in rows:
            if legal_column and not _is_truthy(row.get(legal_column)):
                continue
            raw_name = row.get(name_column)
            if raw_name is None:
                continue
            name = clean_text(str(raw_name))
            if name:
                names.add(name)

        if names:
            candidates.append(
                LegalCardCandidate(
                    path=path,
                    names=names,
                    source_column=name_column,
                    legal_column=legal_column,
                    score=_score_candidate(path, len(names), legal_column is not None),
                    sample=sorted(names)[:8],
                )
            )
    return candidates


def _read_delimited(path: Path) -> list[LegalCardCandidate]:
    delimiter = "\t" if path.suffix.casefold() == ".tsv" else ","
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = [dict(row) for row in reader]
    return _candidate_from_rows(path, rows)


def _flatten_json_records(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        if all(isinstance(item, dict) for item in value):
            return [dict(item) for item in value]
        return []
    if isinstance(value, dict):
        records: list[dict[str, object]] = []
        for child in value.values():
            records.extend(_flatten_json_records(child))
        return records
    return []


def _read_json(path: Path) -> list[LegalCardCandidate]:
    if path.suffix.casefold() == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
        return _candidate_from_rows(path, rows)

    value = json.loads(path.read_text(encoding="utf-8-sig"))
    return _candidate_from_rows(path, _flatten_json_records(value))


def _read_txt(path: Path) -> list[LegalCardCandidate]:
    names = {clean_text(line) for line in path.read_text(encoding="utf-8-sig").splitlines()}
    names = {name for name in names if name and not name.startswith("#")}
    if len(names) < 10:
        return []
    return [
        LegalCardCandidate(
            path=path,
            names=names,
            source_column="line",
            legal_column=None,
            score=_score_candidate(path, len(names), False),
            sample=sorted(names)[:8],
        )
    ]


def iter_supported_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.suffix.casefold() in SUPPORTED_SUFFIXES:
            yield root
        return
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES:
            yield path


def discover_legal_card_candidates(root: Path) -> list[LegalCardCandidate]:
    candidates: list[LegalCardCandidate] = []
    for path in iter_supported_files(root):
        try:
            suffix = path.suffix.casefold()
            if suffix in {".csv", ".tsv"}:
                candidates.extend(_read_delimited(path))
            elif suffix in {".json", ".jsonl"}:
                candidates.extend(_read_json(path))
            elif suffix == ".txt":
                candidates.extend(_read_txt(path))
        except (csv.Error, json.JSONDecodeError, UnicodeDecodeError):
            continue
    candidates.sort(key=lambda candidate: (candidate.score, candidate.count), reverse=True)
    return candidates


def choose_legal_card_candidate(root: Path, *, source: Path | None = None) -> LegalCardCandidate:
    candidates = discover_legal_card_candidates(source or root)
    if not candidates:
        raise RuntimeError(f"No legal card list candidate found under {source or root}.")
    best = candidates[0]
    if source is None and len(candidates) > 1:
        second = candidates[1]
        if best.score - second.score < 1 and best.count != second.count:
            raise RuntimeError(
                "Multiple plausible legal card lists were found. Re-run with --legal-source. "
                f"Top candidates: {best.path} ({best.count}), {second.path} ({second.count})."
            )
    return best


def write_legal_cards(names: set[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sorted(names)) + "\n", encoding="utf-8")


def load_legal_cards(path: Path) -> dict[str, str]:
    cards: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        name = clean_text(line)
        if name and not name.startswith("#"):
            cards[normalize_card_name(name)] = name
    return cards


def write_candidate_report(candidates: list[LegalCardCandidate], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Legal Card Candidate Sources",
        "",
        "| Rank | Score | Count | File | Name column | Legal column | Sample |",
        "| ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for index, candidate in enumerate(candidates, start=1):
        sample = ", ".join(candidate.sample[:5])
        lines.append(
            f"| {index} | {candidate.score:.2f} | {candidate.count} | "
            f"`{candidate.path.as_posix()}` | `{candidate.source_column}` | "
            f"`{candidate.legal_column or ''}` | {sample} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
