from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ptcg_abc.agent import RuleBasedAgent
from ptcg_abc.agent.rule_based import _select_context_name, _select_type_name
from ptcg_abc.evaluation import phase3_tournament_559_prepared_decks, required_phase3_prepared_decks
from ptcg_abc.rl.board_image import (
    SnapshotImage,
    SnapshotManifest,
    render_tabletop_snapshot,
    safe_filename,
    write_manifest,
)
from ptcg_abc.rl.card_art import CardArtCache
from ptcg_abc.rl.featurizer import card_lookup
from ptcg_abc.simulator import load_engine_metadata, run_battle


@dataclass
class SnapshotRuleAgent:
    deck_ids: Sequence[int]
    label: str
    output_dir: Path
    card_data: Sequence[Any]
    attack_data: Sequence[Any]
    card_art: CardArtCache | None = None
    record: bool = True
    image_limit: int = 0
    turns_per_player: int = 0

    def __post_init__(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.agent = RuleBasedAgent(
                self.deck_ids,
                card_data=self.card_data,
                attack_data=self.attack_data,
            )
        self.card_by_id = card_lookup(self.card_data)
        self.images: list[SnapshotImage] = []

    def act(self, observation: Any) -> list[int]:
        selected = self.agent.act(observation)
        if self._should_record(observation):
            self._write_snapshot(observation, selected)
        return selected

    def _should_record(self, observation: Any) -> bool:
        if not self.record:
            return False
        if self.image_limit > 0 and len(self.images) >= self.image_limit:
            return False
        if self.turns_per_player <= 0:
            return True
        current = getattr(observation, "current", None)
        turn = int(getattr(current, "turn", 0) or 0)
        if turn <= 0:
            return False
        player_turn = ((turn - 1) // 2) + 1
        return player_turn <= self.turns_per_player

    def _write_snapshot(self, observation: Any, selected: list[int]) -> None:
        current = getattr(observation, "current", None)
        select = getattr(observation, "select", None)
        player_index = int(getattr(current, "yourIndex", 0) or 0)
        select_type = _select_name(select, "type")
        context = _select_name(select, "context")
        step = len(self.images) + 1
        filename = safe_filename(
            f"{step:04d}_p{player_index}_{self.label}_{select_type}_{context}.png"
        )
        path = self.output_dir / filename
        title = f"{self.label} | step {step:04d} | player {player_index}"
        render_tabletop_snapshot(
            observation,
            card_by_id=self.card_by_id,
            card_art=self.card_art,
            selected_indices=selected,
            title=title,
            output_path=path,
        )
        self.images.append(
            SnapshotImage(
                path=str(path.as_posix()),
                step=step,
                player_index=player_index,
                label=self.label,
                select_type=select_type,
                context=context,
                selected_indices=list(selected),
            )
        )


def run_rule_vs_benchmark_snapshots(
    *,
    sample_dir: Path,
    output_dir: Path,
    our_deck_index: int = 9,
    benchmark_index: int = 4,
    max_steps: int = 120,
    record_player: str = "both",
    image_limit: int = 0,
    turns_per_player: int = 0,
    card_art_pdf: Path | None = None,
    card_art_dir: Path | None = None,
) -> SnapshotManifest:
    card_data, attack_data = load_engine_metadata(sample_dir)
    our_decks = {deck.index: deck for deck in phase3_tournament_559_prepared_decks()}
    benchmark_decks = {
        deck.index: deck for deck in required_phase3_prepared_decks(start_index=1)
    }
    if our_deck_index not in our_decks:
        raise ValueError(f"Unknown prepared deck index: {our_deck_index}.")
    if benchmark_index not in benchmark_decks:
        raise ValueError(f"Unknown benchmark deck index: {benchmark_index}.")
    our_deck = our_decks[our_deck_index]
    benchmark_deck = benchmark_decks[benchmark_index]

    output_dir.mkdir(parents=True, exist_ok=True)
    card_art = (
        CardArtCache(pdf_path=card_art_pdf, cache_dir=card_art_dir)
        if card_art_pdf is not None and card_art_dir is not None
        else None
    )
    our_agent = SnapshotRuleAgent(
        our_deck.card_ids,
        label=f"our-{our_deck.index}-{our_deck.archetype}",
        output_dir=output_dir,
        card_data=card_data,
        attack_data=attack_data,
        card_art=card_art,
        record=record_player in {"both", "ours"},
        image_limit=image_limit,
        turns_per_player=turns_per_player,
    )
    benchmark_agent = SnapshotRuleAgent(
        benchmark_deck.card_ids,
        label=f"benchmark-{benchmark_deck.index}-{benchmark_deck.archetype}",
        output_dir=output_dir,
        card_data=card_data,
        attack_data=attack_data,
        card_art=card_art,
        record=record_player in {"both", "benchmark"},
        image_limit=image_limit,
        turns_per_player=turns_per_player,
    )
    result = run_battle(
        our_deck.card_ids,
        benchmark_deck.card_ids,
        sample_dir=sample_dir,
        agent0=our_agent,
        agent1=benchmark_agent,
        card_data=card_data,
        attack_data=attack_data,
        max_steps=max_steps,
    )
    images = our_agent.images + benchmark_agent.images
    images.sort(key=lambda image: image.path)
    manifest = SnapshotManifest(
        output_dir=str(output_dir.as_posix()),
        images=images,
        battle_result=result.to_dict(),
        our_deck=our_deck.label,
        benchmark_deck=benchmark_deck.label,
        max_steps=max_steps,
    )
    write_manifest(manifest, output_dir / "manifest.json")
    return manifest


def _select_name(select: Any, name: str) -> str:
    if name == "type":
        return _select_type_name(select)
    return _select_context_name(select)
