from __future__ import annotations

import contextlib
import importlib
import io
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ptcg_abc.agent import RandomAgent, RuleBasedAgent
from ptcg_abc.card_db import CardIdLookup
from ptcg_abc.corpus import deck_label, deck_to_card_ids
from ptcg_abc.models import Decklist
from ptcg_abc.simulator import BattleResult, load_engine_metadata, run_battle


@dataclass
class PreparedDeck:
    index: int
    deck: Decklist
    card_ids: list[int]

    @property
    def archetype(self) -> str:
        return self.deck.archetype.name

    @property
    def label(self) -> str:
        return deck_label(self.deck)


@dataclass
class ArchetypeScore:
    archetype: str
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    timeouts: int = 0
    prize_leads: int = 0
    errors: int = 0
    points: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["win_rate"] = self.win_rate
        return data


@dataclass
class MatchupRecord:
    archetype_a: str
    archetype_b: str
    games: int
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0
    timeouts: int = 0
    errors: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RandomEvalSummary:
    games: int
    wins: int
    losses: int
    draws: int
    timeouts: int
    errors: int
    win_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Phase3CloseoutResult:
    selected_archetype: str
    selected_deck_index: int
    selected_deck_label: str
    submission_tar: str
    random_eval: RandomEvalSummary
    archetype_scores: list[ArchetypeScore]
    matchups: list[MatchupRecord]
    games_per_matchup: int
    random_games: int
    max_steps: int
    seed: int

    def to_dict(self) -> dict:
        return {
            "selected_archetype": self.selected_archetype,
            "selected_deck_index": self.selected_deck_index,
            "selected_deck_label": self.selected_deck_label,
            "submission_tar": self.submission_tar,
            "random_eval": self.random_eval.to_dict(),
            "archetype_scores": [score.to_dict() for score in self.archetype_scores],
            "matchups": [matchup.to_dict() for matchup in self.matchups],
            "games_per_matchup": self.games_per_matchup,
            "random_games": self.random_games,
            "max_steps": self.max_steps,
            "seed": self.seed,
        }


@dataclass
class SampleDragapultBenchmarkRow:
    deck_index: int
    deck_label: str
    archetype: str
    games: int
    wins: int = 0
    losses: int = 0
    draws: int = 0
    timeouts: int = 0
    errors: int = 0
    win_rate: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SampleDragapultBenchmarkResult:
    sample_deck_label: str
    games_per_deck: int
    max_steps: int
    rows: list[SampleDragapultBenchmarkRow]

    def to_dict(self) -> dict:
        return {
            "sample_deck_label": self.sample_deck_label,
            "games_per_deck": self.games_per_deck,
            "max_steps": self.max_steps,
            "rows": [row.to_dict() for row in self.rows],
        }


def prepare_decks(decks: list[Decklist], lookup: CardIdLookup) -> list[PreparedDeck]:
    return [
        PreparedDeck(index=index, deck=deck, card_ids=deck_to_card_ids(deck, lookup))
        for index, deck in enumerate(decks, start=1)
    ]


def group_by_archetype(decks: list[PreparedDeck]) -> dict[str, list[PreparedDeck]]:
    grouped: dict[str, list[PreparedDeck]] = defaultdict(list)
    for deck in decks:
        grouped[deck.archetype].append(deck)
    return dict(sorted(grouped.items(), key=lambda item: item[0].casefold()))


def representative_decks(decks: list[PreparedDeck]) -> list[PreparedDeck]:
    grouped = group_by_archetype(decks)
    return [items[0] for _, items in grouped.items()]


def _record_game(
    scores: dict[str, ArchetypeScore],
    archetype0: str,
    archetype1: str,
    result: BattleResult,
) -> tuple[int | None, bool, bool]:
    scores[archetype0].games += 1
    scores[archetype1].games += 1
    if result.error:
        scores[archetype0].errors += 1
        scores[archetype1].errors += 1
        scores[archetype0].draws += 1
        scores[archetype1].draws += 1
        return None, False, True
    if result.winner == 0:
        scores[archetype0].wins += 1
        scores[archetype1].losses += 1
        scores[archetype0].points += 3
        return 0, False, False
    if result.winner == 1:
        scores[archetype1].wins += 1
        scores[archetype0].losses += 1
        scores[archetype1].points += 3
        return 1, False, False

    timeout = not result.finished
    if timeout:
        scores[archetype0].timeouts += 1
        scores[archetype1].timeouts += 1
    if result.leader == 0:
        scores[archetype0].prize_leads += 1
        scores[archetype0].points += 1
    elif result.leader == 1:
        scores[archetype1].prize_leads += 1
        scores[archetype1].points += 1
    else:
        scores[archetype0].draws += 1
        scores[archetype1].draws += 1
    return result.leader, timeout, False


def run_archetype_sweep(
    prepared_decks: list[PreparedDeck],
    *,
    sample_dir: Path,
    games_per_matchup: int = 10,
    max_steps: int = 600,
) -> tuple[list[ArchetypeScore], list[MatchupRecord]]:
    card_data, attack_data = load_engine_metadata(sample_dir)
    reps = representative_decks(prepared_decks)
    scores = {deck.archetype: ArchetypeScore(archetype=deck.archetype) for deck in reps}
    matchups: list[MatchupRecord] = []

    for left_index, deck_a in enumerate(reps):
        for deck_b in reps[left_index + 1 :]:
            matchup = MatchupRecord(
                archetype_a=deck_a.archetype,
                archetype_b=deck_b.archetype,
                games=games_per_matchup,
            )
            for game_index in range(games_per_matchup):
                swap = game_index % 2 == 1
                deck0 = deck_b if swap else deck_a
                deck1 = deck_a if swap else deck_b
                result = run_battle(
                    deck0.card_ids,
                    deck1.card_ids,
                    sample_dir=sample_dir,
                    card_data=card_data,
                    attack_data=attack_data,
                    max_steps=max_steps,
                )
                winner, timeout, errored = _record_game(
                    scores,
                    deck0.archetype,
                    deck1.archetype,
                    result,
                )
                if errored:
                    matchup.errors += 1
                elif winner is None:
                    matchup.draws += 1
                elif (winner == 0 and deck0 is deck_a) or (winner == 1 and deck1 is deck_a):
                    matchup.wins_a += 1
                else:
                    matchup.wins_b += 1
                if timeout:
                    matchup.timeouts += 1
            matchups.append(matchup)

    ordered_scores = sorted(
        scores.values(),
        key=lambda score: (score.points, score.wins, score.win_rate, -score.losses, score.archetype),
        reverse=True,
    )
    return ordered_scores, matchups


def run_random_evaluation(
    deck: PreparedDeck,
    *,
    sample_dir: Path,
    games: int = 20,
    max_steps: int = 600,
    seed: int = 20260617,
) -> RandomEvalSummary:
    card_data, attack_data = load_engine_metadata(sample_dir)
    wins = losses = draws = timeouts = errors = 0
    for game_index in range(games):
        rule_is_player0 = game_index % 2 == 0
        rule_agent = RuleBasedAgent(deck.card_ids, card_data=card_data, attack_data=attack_data)
        random_agent = RandomAgent(deck.card_ids, seed=seed + game_index)
        result = run_battle(
            deck.card_ids,
            deck.card_ids,
            sample_dir=sample_dir,
            agent0=rule_agent if rule_is_player0 else random_agent,
            agent1=random_agent if rule_is_player0 else rule_agent,
            card_data=card_data,
            attack_data=attack_data,
            max_steps=max_steps,
        )
        if result.error:
            errors += 1
            draws += 1
        elif result.winner is None:
            timeouts += int(not result.finished)
            if result.leader is None:
                draws += 1
            elif (result.leader == 0 and rule_is_player0) or (result.leader == 1 and not rule_is_player0):
                wins += 1
            else:
                losses += 1
        elif (result.winner == 0 and rule_is_player0) or (result.winner == 1 and not rule_is_player0):
            wins += 1
        else:
            losses += 1
    return RandomEvalSummary(
        games=games,
        wins=wins,
        losses=losses,
        draws=draws,
        timeouts=timeouts,
        errors=errors,
        win_rate=wins / games if games else 0.0,
    )


def choose_deck_from_best_archetype(
    prepared_decks: list[PreparedDeck],
    scores: list[ArchetypeScore],
    *,
    seed: int = 20260617,
) -> PreparedDeck:
    if not scores:
        raise ValueError("No archetype scores available.")
    best_archetype = scores[0].archetype
    candidates = [deck for deck in prepared_decks if deck.archetype == best_archetype]
    if not candidates:
        raise ValueError(f"No decks available for selected archetype {best_archetype}.")
    return random.Random(seed).choice(candidates)


def write_closeout_reports(result: Phase3CloseoutResult, *, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    lines = [
        "# Phase 3 Closeout Report",
        "",
        f"Selected archetype: `{result.selected_archetype}`",
        f"Selected deck: `{result.selected_deck_label}`",
        f"Selected deck index: `{result.selected_deck_index}`",
        f"Submission bundle: `{result.submission_tar}`",
        "",
        "## Random-Agent Evaluation",
        "",
        f"- Games: {result.random_eval.games}",
        f"- Wins: {result.random_eval.wins}",
        f"- Losses: {result.random_eval.losses}",
        f"- Draws: {result.random_eval.draws}",
        f"- Timeouts: {result.random_eval.timeouts}",
        f"- Errors: {result.random_eval.errors}",
        f"- Win rate: {result.random_eval.win_rate:.3f}",
        "",
        "## Archetype Ranking",
        "",
        "| Rank | Archetype | Points | Wins | Losses | Draws | Prize leads | Timeouts | Errors | Win rate |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, score in enumerate(result.archetype_scores, start=1):
        lines.append(
            f"| {rank} | {score.archetype} | {score.points} | {score.wins} | "
            f"{score.losses} | {score.draws} | {score.prize_leads} | {score.timeouts} | "
            f"{score.errors} | {score.win_rate:.3f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_sample_dragapult_module(sample_dir: Path):
    previous_path = list(sys.path)
    sys.path.insert(0, str(sample_dir.resolve()))
    try:
        return importlib.import_module("ptcg_abc.agent.sample_dragapult_agent")
    finally:
        sys.path = previous_path


def _quiet_rule_agent(deck_ids: list[int], card_data: list[Any], attack_data: list[Any]) -> RuleBasedAgent:
    with contextlib.redirect_stdout(io.StringIO()):
        return RuleBasedAgent(deck_ids, card_data=card_data, attack_data=attack_data)


def run_sample_dragapult_benchmark(
    prepared_decks: list[PreparedDeck],
    *,
    sample_dir: Path,
    games_per_deck: int = 10,
    max_steps: int = 600,
) -> SampleDragapultBenchmarkResult:
    card_data, attack_data = load_engine_metadata(sample_dir)
    sample_module = _load_sample_dragapult_module(sample_dir)
    sample_deck = list(sample_module.SAMPLE_DRAGAPULT_DECK)
    rows: list[SampleDragapultBenchmarkRow] = []

    for prepared in prepared_decks:
        row = SampleDragapultBenchmarkRow(
            deck_index=prepared.index,
            deck_label=prepared.label,
            archetype=prepared.archetype,
            games=games_per_deck,
        )
        for game_index in range(games_per_deck):
            our_is_player0 = game_index % 2 == 0
            sample_agent = sample_module.SampleDragapultAgent()
            rule_agent = _quiet_rule_agent(prepared.card_ids, card_data, attack_data)
            result = run_battle(
                prepared.card_ids if our_is_player0 else sample_deck,
                sample_deck if our_is_player0 else prepared.card_ids,
                sample_dir=sample_dir,
                agent0=rule_agent if our_is_player0 else sample_agent,
                agent1=sample_agent if our_is_player0 else rule_agent,
                card_data=card_data,
                attack_data=attack_data,
                max_steps=max_steps,
            )
            if result.error:
                row.errors += 1
                row.draws += 1
            else:
                if result.winner is None:
                    if not result.finished:
                        row.timeouts += 1
                    effective_winner = result.leader
                else:
                    effective_winner = result.winner

                if effective_winner is None:
                    row.draws += 1
                elif (effective_winner == 0 and our_is_player0) or (
                    effective_winner == 1 and not our_is_player0
                ):
                    row.wins += 1
                else:
                    row.losses += 1
        row.win_rate = row.wins / row.games if row.games else 0.0
        rows.append(row)

    return SampleDragapultBenchmarkResult(
        sample_deck_label="Kiyotah sample Dragapult ex deck",
        games_per_deck=games_per_deck,
        max_steps=max_steps,
        rows=rows,
    )


def write_sample_dragapult_benchmark_report(
    result: SampleDragapultBenchmarkResult,
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    lines = [
        "# Sample Dragapult Benchmark",
        "",
        f"Opponent: `{result.sample_deck_label}`",
        f"Games per deck: {result.games_per_deck}",
        f"Max selections per game: {result.max_steps}",
        "",
        "| Deck | Archetype | Wins | Losses | Draws | Timeouts | Errors | Win rate |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result.rows:
        lines.append(
            f"| {row.deck_index} | {row.archetype} | {row.wins} | {row.losses} | "
            f"{row.draws} | {row.timeouts} | {row.errors} | {row.win_rate:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Deck Labels",
            "",
            "| Deck | Label |",
            "| ---: | --- |",
        ]
    )
    for row in result.rows:
        lines.append(f"| {row.deck_index} | {row.deck_label} |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
