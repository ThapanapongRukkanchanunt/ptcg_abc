from __future__ import annotations

import contextlib
import importlib
import importlib.util
import json
import sys
import types
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


PUBLIC_AGENT_SOURCES: tuple[dict[str, str], ...] = (
    {
        "key": "roman_v7_crustle_lucario",
        "label": "RomanRozen Crustle+Lucario V7 LB960+",
        "source_ref": "romanrozen/strong-start-crustle-lucario-agent-v7-lb-960",
        "url": "https://www.kaggle.com/code/romanrozen/strong-start-crustle-lucario-agent-v7-lb-960",
    },
    {
        "key": "kokinn_lucario_search_915",
        "label": "Kokinnwakashuu public 915+ Lucario search",
        "source_ref": "kokinnwakashuu/ptcg-public-915-lucario-search-baseline",
        "url": "https://www.kaggle.com/code/kokinnwakashuu/ptcg-public-915-lucario-search-baseline",
    },
    {
        "key": "penguin_public_scores_915",
        "label": "Penguin public scores 915+",
        "source_ref": "penguin069/public-scores-915",
        "url": "https://www.kaggle.com/code/penguin069/public-scores-915",
    },
    {
        "key": "alyce_lucario_v2_bot",
        "label": "Alyce Lucario Deck V2 bot",
        "source_ref": "alycemiki/lucario-deck-v2-play-like-a-bot",
        "url": "https://www.kaggle.com/code/alycemiki/lucario-deck-v2-play-like-a-bot",
    },
    {
        "key": "yaroslav_lucario_v2_crustle_aware",
        "label": "Yaroslav Mega Lucario V2 crustle-aware",
        "source_ref": "yaroslavkholmirzayev/mega-lucario-v2-crustle-aware-best-submit",
        "url": "https://www.kaggle.com/code/yaroslavkholmirzayev/mega-lucario-v2-crustle-aware-best-submit",
    },
    {
        "key": "kacchan_lucario_anti_wall",
        "label": "Kacchan Crustle-aware Mega Lucario anti-wall",
        "source_ref": "kacchanwriting/crustle-aware-mega-lucario-ex-anti-wall",
        "url": "https://www.kaggle.com/code/kacchanwriting/crustle-aware-mega-lucario-ex-anti-wall",
    },
    {
        "key": "biohack_day2_new",
        "label": "Biohack44 Beating the Day-2 new",
        "source_ref": "biohack44/beating-the-day-2-new",
        "url": "https://www.kaggle.com/code/biohack44/beating-the-day-2-new",
    },
    {
        "key": "makthanithin_1084_5_baseline",
        "label": "Makthanithin 1084.5 baseline",
        "source_ref": "makthanithin/pokemon-tcg-ai-battle-1084-5-baseline",
        "url": "https://www.kaggle.com/code/makthanithin/pokemon-tcg-ai-battle-1084-5-baseline",
    },
    {
        "key": "seokjeongeum_strong_start_psychic_v8",
        "label": "Seokjeongeum Strong Start Psychic Anti-Meta V8",
        "source_ref": "seokjeongeum/strong-start-psychic-anti-meta-v8-lb-1100",
        "url": "https://www.kaggle.com/code/seokjeongeum/strong-start-psychic-anti-meta-v8-lb-1100",
    },
    {
        "key": "seokjeongeum_strong_start_baseline_v10",
        "label": "Seokjeongeum Strong Start Baseline Agent V10",
        "source_ref": "seokjeongeum/strong-start-baseline-agent-v10-lb-950",
        "url": "https://www.kaggle.com/code/seokjeongeum/strong-start-baseline-agent-v10-lb-950",
    },
    {
        "key": "pixiux_mega_lucario_ex_v63",
        "label": "Pixiux Mega Lucario ex v63",
        "source_ref": "pixiux/ptcg-mega-lucario-ex-v63",
        "url": "https://www.kaggle.com/code/pixiux/ptcg-mega-lucario-ex-v63",
    },
    {
        "key": "seokjeongeum_crustle_tusk_lo_1208",
        "label": "Seokjeongeum Crustle/Tusk LibraryOut 1208",
        "source_ref": "seokjeongeum/max-elo-1208-libraryout-w-crustle-great-tusk",
        "url": "https://www.kaggle.com/code/seokjeongeum/max-elo-1208-libraryout-w-crustle-great-tusk",
    },
    {
        "key": "makthanithin_mega_lucario_v62",
        "label": "Makthanithin Mega Lucario ex v62",
        "source_ref": "makthanithin/ptcg-mega-lucario-ex-v62",
        "url": "https://www.kaggle.com/code/makthanithin/ptcg-mega-lucario-ex-v62",
    },
    {
        "key": "pilkwang_lucario_alakazam",
        "label": "Pilkwang Lucario & Alakazam",
        "source_ref": "pilkwang/pokemon-tcg-lucario-alakazam",
        "url": "https://www.kaggle.com/code/pilkwang/pokemon-tcg-lucario-alakazam",
    },
    {
        "key": "rauffauzanrambe_ptcg_advanced",
        "label": "Rauffauzanrambe PTCG Advanced",
        "source_ref": "rauffauzanrambe/pokemon-ai-battle-best-ptcg-advanced",
        "url": "https://www.kaggle.com/code/rauffauzanrambe/pokemon-ai-battle-best-ptcg-advanced",
    },
    {
        "key": "rv1922_ai_battle_challenge",
        "label": "RV1922 AI Battle Challenge",
        "source_ref": "rv1922/ai-battle-challenge",
        "url": "https://www.kaggle.com/code/rv1922/ai-battle-challenge",
    },
    {
        "key": "plamen06_pokemon_steel",
        "label": "Plamen06 Pokemon Steel",
        "source_ref": "plamen06/pokemon-steel",
        "url": "https://www.kaggle.com/code/plamen06/pokemon-steel",
    },
    {
        "key": "seokjeongeum_pure_dragapult_ex",
        "label": "Seokjeongeum Pure Dragapult ex",
        "source_ref": "seokjeongeum/pure-dragapult-ex-deck",
        "url": "https://www.kaggle.com/code/seokjeongeum/pure-dragapult-ex-deck",
    },
    {
        "key": "seokjeongeum_mega_reinforcement",
        "label": "Seokjeongeum Mega Reinforcement",
        "source_ref": "seokjeongeum/mega-pokemon-reinforcement-ai-battle",
        "url": "https://www.kaggle.com/code/seokjeongeum/mega-pokemon-reinforcement-ai-battle",
    },
    {
        "key": "aman5153684_crustle_aware_fighting",
        "label": "Aman5153684 Crustle-aware Fighting",
        "source_ref": "aman5153684/a-crustle-aware-fighting-agent",
        "url": "https://www.kaggle.com/code/aman5153684/a-crustle-aware-fighting-agent",
    },
    {
        "key": "sample_lucario",
        "label": "Official sample Mega Lucario ex",
        "source_ref": "kiyotah/a-sample-rule-based-agent-mega-lucario-ex-deck",
        "url": "https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-mega-lucario-ex-deck",
    },
    {
        "key": "sample_dragapult",
        "label": "Official sample Dragapult ex",
        "source_ref": "kiyotah/a-sample-rule-based-agent-dragapult-ex-deck",
        "url": "https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-dragapult-ex-deck",
    },
    {
        "key": "sample_iono",
        "label": "Official sample Iono",
        "source_ref": "kiyotah/a-sample-rule-based-agent-iono-s-deck",
        "url": "https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-iono-s-deck",
    },
    {
        "key": "sample_abomasnow",
        "label": "Official sample Mega Abomasnow ex",
        "source_ref": "kiyotah/a-sample-rule-based-agent-mega-abomasnow-ex-deck",
        "url": "https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-mega-abomasnow-ex-deck",
    },
)

DECK_VARIABLE_NAMES = (
    "deck_ids",
    "DECK_IDS",
    "deck",
    "DECK",
    "MY_DECK",
    "POKEMON_TCG_DECK",
    "SAMPLE_DECK",
    "SAMPLE_DRAGAPULT_DECK",
    "SAMPLE_LUCARIO_DECK",
    "SAMPLE_ABOMASNOW_DECK",
    "SAMPLE_IONO_DECK",
)


@dataclass(frozen=True)
class PublicAgentSource:
    key: str
    label: str
    source_ref: str
    url: str

    @property
    def is_sample(self) -> bool:
        return self.key.startswith("sample_")

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "key": self.key,
            "label": self.label,
            "source_ref": self.source_ref,
            "url": self.url,
            "is_sample": self.is_sample,
        }


@dataclass(frozen=True)
class PublicAgentStatus:
    source: PublicAgentSource
    status: str
    path: str | None = None
    deck_ids: list[int] | None = None
    error: str | None = None
    built_in: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.source.to_dict(),
            "status": self.status,
            "path": self.path,
            "deck_ids": list(self.deck_ids or []),
            "error": self.error,
            "built_in": self.built_in,
        }


class PublicAgentAdapter:
    def __init__(
        self,
        *,
        source: PublicAgentSource,
        deck_ids: Sequence[int],
        agent_factory: Callable[[], Any],
    ) -> None:
        self.source = source
        self.deck_ids = list(deck_ids)
        self._agent = agent_factory()

    def act(self, observation: Any) -> list[int]:
        agent = self._agent
        if hasattr(agent, "act"):
            return list(agent.act(observation))
        if callable(agent):
            payload = asdict(observation) if is_dataclass(observation) else observation
            return list(agent(payload))
        raise TypeError(f"Loaded public agent {self.source.key} has no act() or callable.")


@dataclass(frozen=True)
class LoadedPublicAgent:
    source: PublicAgentSource
    path: Path | None
    deck_ids: list[int]
    make_agent: Callable[[], PublicAgentAdapter]
    built_in: bool = False

    @property
    def key(self) -> str:
        return self.source.key

    @property
    def label(self) -> str:
        return self.source.label

    def to_status(self) -> PublicAgentStatus:
        return PublicAgentStatus(
            source=self.source,
            status="available",
            path=self.path.as_posix() if self.path is not None else None,
            deck_ids=list(self.deck_ids),
            built_in=self.built_in,
        )


@contextlib.contextmanager
def _temporary_sys_path(paths: Iterable[Path]):
    previous = list(sys.path)
    for path in reversed([p for p in paths if p is not None]):
        sys.path.insert(0, str(path.resolve()))
    try:
        yield
    finally:
        sys.path = previous


def public_agent_sources(
    *,
    roster_notebook: Path | None = None,
    include_public: bool = True,
    include_samples: bool = True,
) -> list[PublicAgentSource]:
    records = _load_sources_from_notebook(roster_notebook) if roster_notebook else PUBLIC_AGENT_SOURCES
    sources = [PublicAgentSource(**record) for record in records]
    return [
        source
        for source in sources
        if (include_public and not source.is_sample) or (include_samples and source.is_sample)
    ]


def discover_public_agents(
    *,
    roots: Sequence[Path],
    sample_dir: Path,
    roster_notebook: Path | None = None,
    include_public: bool = True,
    include_samples: bool = True,
    include_builtin_samples: bool = True,
) -> tuple[list[LoadedPublicAgent], list[PublicAgentStatus]]:
    loaded: list[LoadedPublicAgent] = []
    statuses: list[PublicAgentStatus] = []
    for source in public_agent_sources(
        roster_notebook=roster_notebook,
        include_public=include_public,
        include_samples=include_samples,
    ):
        try:
            agent = load_public_agent(
                source,
                roots=roots,
                sample_dir=sample_dir,
                include_builtin_samples=include_builtin_samples,
            )
        except Exception as exc:
            statuses.append(
                PublicAgentStatus(
                    source=source,
                    status="unavailable",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        loaded.append(agent)
        statuses.append(agent.to_status())
    return loaded, statuses


def load_public_agent(
    source: PublicAgentSource,
    *,
    roots: Sequence[Path],
    sample_dir: Path,
    include_builtin_samples: bool = True,
) -> LoadedPublicAgent:
    if include_builtin_samples and source.key == "sample_dragapult":
        try:
            return _load_builtin_sample_dragapult(source)
        except Exception:
            pass
    if include_builtin_samples and source.key == "sample_lucario":
        try:
            return _load_builtin_sample_lucario(source, sample_dir=sample_dir)
        except Exception:
            pass
    candidates = _candidate_paths(source, roots)
    errors: list[str] = []
    for path in candidates:
        try:
            return _load_external_agent(source, path, sample_dir=sample_dir)
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    if not candidates:
        raise FileNotFoundError(
            f"No local agent file found for {source.key}; searched roots: "
            + ", ".join(root.as_posix() for root in roots)
        )
    raise ValueError("; ".join(errors))


def _load_sources_from_notebook(path: Path) -> tuple[dict[str, str], ...]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        if "AGENT_SOURCES" not in source:
            continue
        namespace: dict[str, Any] = {}
        exec(compile(source, str(path), "exec"), namespace)
        records = namespace.get("AGENT_SOURCES")
        if isinstance(records, list):
            return tuple(dict(record) for record in records)
    raise ValueError(f"No AGENT_SOURCES cell found in {path}.")


def _load_builtin_sample_dragapult(source: PublicAgentSource) -> LoadedPublicAgent:
    from ptcg_abc.agent.sample_dragapult_agent import SAMPLE_DRAGAPULT_DECK, SampleDragapultAgent

    deck_ids = _validated_deck(SAMPLE_DRAGAPULT_DECK)

    def factory() -> PublicAgentAdapter:
        return PublicAgentAdapter(
            source=source,
            deck_ids=deck_ids,
            agent_factory=SampleDragapultAgent,
        )

    return LoadedPublicAgent(
        source=source,
        path=None,
        deck_ids=deck_ids,
        make_agent=factory,
        built_in=True,
    )


def _load_builtin_sample_lucario(
    source: PublicAgentSource,
    *,
    sample_dir: Path,
) -> LoadedPublicAgent:
    from ptcg_abc.agent.rule_based import RuleBasedAgent
    from ptcg_abc.evaluation import REQUIRED_PHASE3_SAMPLE_DECKS
    from ptcg_abc.simulator import load_engine_metadata

    deck_ids = _validated_deck(_sample_deck_by_name(REQUIRED_PHASE3_SAMPLE_DECKS, "lucario"))
    card_data, attack_data = load_engine_metadata(sample_dir)

    def factory() -> PublicAgentAdapter:
        return PublicAgentAdapter(
            source=source,
            deck_ids=deck_ids,
            agent_factory=lambda: RuleBasedAgent(
                deck_ids,
                card_data=card_data,
                attack_data=attack_data,
            ),
        )

    return LoadedPublicAgent(
        source=source,
        path=None,
        deck_ids=deck_ids,
        make_agent=factory,
        built_in=True,
    )


def _sample_deck_by_name(
    records: Sequence[tuple[str, Sequence[int], str]],
    needle: str,
) -> list[int]:
    clean = needle.casefold()
    for name, card_ids, _ in records:
        if clean in name.casefold():
            return list(card_ids)
    raise ValueError(f"No required sample deck matching {needle!r}.")


def _load_external_agent(
    source: PublicAgentSource,
    path: Path,
    *,
    sample_dir: Path,
) -> LoadedPublicAgent:
    module_name = f"_ptcg_public_agent_{source.key}_{abs(hash(path.resolve()))}"

    def load_module() -> types.ModuleType:
        with _temporary_sys_path([sample_dir, path.parent]):
            if path.suffix.lower() == ".ipynb":
                return _module_from_notebook(path, module_name)
            return _module_from_py(path, module_name)

    module = load_module()
    deck_ids = _extract_deck_ids(module)
    agent_factory = _extract_agent_factory(module)

    def factory() -> PublicAgentAdapter:
        fresh_module = load_module()
        fresh_factory = _extract_agent_factory(fresh_module)
        return PublicAgentAdapter(
            source=source,
            deck_ids=deck_ids,
            agent_factory=fresh_factory,
        )

    # Validate the originally loaded module has both pieces before returning.
    agent_factory()
    return LoadedPublicAgent(
        source=source,
        path=path,
        deck_ids=deck_ids,
        make_agent=factory,
    )


def _module_from_py(path: Path, module_name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _module_from_notebook(path: Path, module_name: str) -> types.ModuleType:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    sys.modules[module_name] = module
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def _candidate_paths(source: PublicAgentSource, roots: Sequence[Path]) -> list[Path]:
    slugs = _source_slugs(source)
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for slug in slugs:
            candidates.extend(
                [
                    root / source.key / "submission.py",
                    root / source.key / "main.py",
                    root / source.key / "agent.py",
                    root / f"{source.key}.py",
                    root / f"{source.key}.ipynb",
                    root / slug / "submission.py",
                    root / slug / "main.py",
                    root / slug / "agent.py",
                    root / f"{slug}.py",
                    root / f"{slug}.ipynb",
                ]
            )
        for path in root.rglob("*.py"):
            if _path_matches_source(path, source, slugs):
                candidates.append(path)
        for path in root.rglob("*.ipynb"):
            if _path_matches_source(path, source, slugs):
                candidates.append(path)
    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not candidate.exists() or not candidate.is_file():
            continue
        seen.add(resolved)
        ordered.append(candidate)
    return ordered


def _source_slugs(source: PublicAgentSource) -> list[str]:
    parts = [source.key]
    if "/" in source.source_ref:
        owner, kernel = source.source_ref.split("/", 1)
        parts.extend([kernel, f"{owner}-{kernel}", source.source_ref.replace("/", "_")])
    return list(dict.fromkeys(parts))


def _path_matches_source(path: Path, source: PublicAgentSource, slugs: Sequence[str]) -> bool:
    haystack = path.as_posix().casefold().replace("-", "_")
    return any(slug.casefold().replace("-", "_") in haystack for slug in slugs)


def _extract_deck_ids(module: types.ModuleType) -> list[int]:
    for name in DECK_VARIABLE_NAMES:
        if hasattr(module, name):
            try:
                return _validated_deck(getattr(module, name))
            except ValueError:
                continue
    for value in module.__dict__.values():
        deck = getattr(value, "deck_ids", None)
        if deck is not None:
            try:
                return _validated_deck(deck)
            except ValueError:
                continue
    raise ValueError("No 60-card deck list found in public agent module.")


def _extract_agent_factory(module: types.ModuleType) -> Callable[[], Any]:
    agent_func = getattr(module, "agent", None)
    if callable(agent_func):
        return lambda: getattr(module, "agent")
    for value in module.__dict__.values():
        if isinstance(value, type) and hasattr(value, "act"):
            try:
                instance = value()
            except Exception:
                continue
            if hasattr(instance, "act"):
                return value
    raise ValueError("No callable agent() or zero-argument class with act() found.")


def _validated_deck(value: Any) -> list[int]:
    if callable(value):
        value = value()
    try:
        deck = [int(card_id) for card_id in list(value)]
    except Exception as exc:
        raise ValueError(f"Deck value is not a sequence of ints: {exc}") from exc
    if len(deck) != 60:
        raise ValueError(f"Deck has {len(deck)} cards, expected 60.")
    return deck
