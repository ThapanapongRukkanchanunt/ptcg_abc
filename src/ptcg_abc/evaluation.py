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
from ptcg_abc.models import Archetype, CardLine, Decklist, TournamentResult, Variant
from ptcg_abc.simulator import BattleResult, load_engine_metadata, run_battle


REQUIRED_PHASE3_BENCHMARK_DECKS = (
    "Crustle",
    "Mega Lucario",
    "Mega Abomasnow",
    "Iono",
)

REQUIRED_PHASE3_SAMPLE_DECKS = (
    (
        "Crustle",
        [
            *([344] * 4),
            *([345] * 3),
            *([756] * 3),
            117,
            *([1227] * 4),
            *([1182] * 4),
            *([1219] * 4),
            *([1194] * 2),
            *([1225] * 2),
            1197,
            1187,
            *([1122] * 4),
            *([1147] * 4),
            *([1121] * 3),
            *([1086] * 2),
            1123,
            1159,
            1257,
            1242,
            *([11] * 4),
            *([18] * 4),
            *([14] * 4),
            *([20] * 2),
        ],
        "https://limitlesstcg.com/decks/list/26474",
    ),
    (
        "Mega Lucario ex",
        [
            *([673] * 2),
            *([674] * 2),
            *([675] * 2),
            *([676] * 3),
            *([677] * 3),
            *([678] * 4),
            *([1102] * 4),
            *([1123] * 2),
            *([1141] * 4),
            *([1142] * 4),
            *([1152] * 4),
            1159,
            *([1182] * 2),
            *([1192] * 4),
            *([1227] * 4),
            *([1252] * 2),
            *([6] * 13),
        ],
        "https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-mega-lucario-ex-deck",
    ),
    (
        "Mega Abomasnow ex",
        [
            *([721] * 2),
            *([722] * 4),
            *([723] * 4),
            *([1121] * 4),
            1126,
            *([1192] * 4),
            *([1227] * 4),
            *([1262] * 3),
            *([3] * 34),
        ],
        "https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-mega-abomasnow-ex-deck",
    ),
    (
        "Iono's Bellibolt ex",
        [
            *([265] * 3),
            *([268] * 3),
            *([269] * 3),
            *([270] * 3),
            *([271] * 3),
            *([1086] * 3),
            *([1097] * 2),
            1110,
            1118,
            *([1121] * 3),
            *([1152] * 2),
            *([1227] * 4),
            *([1233] * 4),
            *([1254] * 3),
            *([4] * 22),
        ],
        "https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-iono-s-deck",
    ),
)

TOURNAMENT_559_SOURCE_URL = "https://limitlesstcg.com/tournaments/559/decklists"
TOURNAMENT_559_REQUESTED_RANKS = (1, 2, 3, 4, 9, 10, 11, 18, 22)
TOURNAMENT_559_SUBSTITUTIONS = (
    {
        "rank": 2,
        "original_card": "Pokemon Center Lady",
        "replacement_card": "Cook",
        "reason": "Pokemon Center Lady is not present in the Kaggle card table.",
    },
)

TOURNAMENT_559_DECKS = (
    (
        1,
        "Alakazam Dudunsparce",
        "Cerys Jones",
        [
            109, 109, 109, 109, 742, 742, 742, 742, 245, 245, 245, 65, 65, 65, 66,
            66, 66, 140, 222, 521, 142, 858, 1231, 1231, 1231, 1231, 1225, 1225,
            1225, 1182, 1182, 1184, 1086, 1086, 1086, 1086, 1152, 1152, 1152,
            1152, 1079, 1079, 1079, 1081, 1081, 1129, 1097, 1161, 1161, 1156,
            1266, 1266, 1266, 1266, 19, 19, 19, 19, 5, 13,
        ],
    ),
    (
        2,
        "Crustle",
        "Rahul Reddy",
        [
            756, 756, 756, 756, 344, 344, 344, 345, 345, 345, 117, 858, 1227,
            1227, 1227, 1227, 1182, 1182, 1182, 1182, 1219, 1219, 1219, 1219,
            1225, 1225, 1186, 1197, 1212, 1190, 1120, 1120, 1120, 1120, 1147,
            1147, 1147, 1147, 1122, 1122, 1122, 1086, 1086, 1121, 1123, 1159,
            1257, 14, 14, 14, 14, 18, 18, 18, 18, 11, 11, 11, 11, 6,
        ],
    ),
    (
        3,
        "Dragapult Dusknoir",
        "Justin Newdorf",
        [
            119, 119, 119, 119, 120, 120, 120, 120, 121, 121, 121, 131, 131, 132,
            132, 133, 140, 112, 235, 1071, 791, 1227, 1227, 1227, 1227, 1182,
            1182, 1182, 1198, 1198, 1198, 1231, 1086, 1086, 1086, 1086, 1152,
            1152, 1152, 1152, 1121, 1121, 1121, 1121, 1120, 1120, 1120, 1097,
            1097, 1080, 1256, 1256, 2, 2, 2, 2, 5, 5, 5, 7,
        ],
    ),
    (
        4,
        "Dragapult",
        "Kira Melville",
        [
            119, 119, 119, 119, 120, 120, 120, 120, 121, 121, 121, 112, 112, 235,
            235, 1071, 140, 791, 1227, 1227, 1227, 1227, 1182, 1182, 1182, 1198,
            1198, 1198, 1213, 1120, 1120, 1120, 1120, 1086, 1086, 1086, 1086,
            1152, 1152, 1152, 1152, 1121, 1121, 1121, 1121, 1097, 1097, 1080,
            1256, 1256, 1260, 2, 2, 2, 2, 5, 5, 5, 7, 7,
        ],
    ),
    (
        9,
        "Dragapult Dudunsparce",
        "Alex Engeriser",
        [
            119, 119, 119, 119, 120, 120, 120, 120, 121, 121, 121, 65, 65, 66,
            66, 306, 112, 112, 140, 1071, 235, 791, 1227, 1227, 1227, 1227,
            1182, 1182, 1182, 1198, 1198, 1198, 1213, 1152, 1152, 1152, 1152,
            1086, 1086, 1086, 1086, 1121, 1121, 1121, 1121, 1097, 1097, 1080,
            1260, 1260, 2, 2, 2, 2, 7, 7, 7, 5, 5, 5,
        ],
    ),
    (
        10,
        "Hydrapple",
        "Cody Hope",
        [
            96, 96, 96, 96, 708, 708, 709, 709, 710, 710, 42, 42, 93, 93, 150,
            150, 1071, 1071, 655, 920, 140, 1227, 1227, 1227, 1227, 1182, 1182,
            1231, 1231, 1201, 1184, 1188, 1121, 1121, 1121, 1121, 1094, 1094,
            1094, 1094, 1080, 1097, 1261, 1261, 1261, 1261, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1,
        ],
    ),
    (
        11,
        "Raging Bolt Ogerpon",
        "Larry Russano",
        [
            756, 756, 756, 96, 96, 96, 1071, 1071, 1071, 184, 184, 63, 63, 209,
            978, 272, 108, 140, 1198, 1198, 1198, 1198, 1182, 1182, 1205, 1205,
            1188, 1227, 1121, 1121, 1121, 1121, 1116, 1116, 1116, 1116, 1097,
            1097, 1098, 1098, 1080, 1250, 1250, 1250, 1250, 1, 1, 1, 1, 1, 1, 1,
            1, 5, 5, 4, 4, 6, 6, 3,
        ],
    ),
    (
        18,
        "Dragapult Blaziken",
        "Ivan Lukens",
        [
            119, 119, 119, 119, 120, 120, 120, 120, 121, 121, 324, 324, 325,
            326, 326, 112, 112, 235, 272, 140, 1071, 1227, 1227, 1227, 1227,
            1182, 1182, 1182, 1231, 1231, 1198, 1198, 1086, 1086, 1086, 1086,
            1121, 1121, 1121, 1121, 1152, 1152, 1152, 1152, 1079, 1079, 1079,
            1097, 1097, 1080, 1250, 1256, 2, 2, 2, 5, 5, 5, 7, 7,
        ],
    ),
    (
        22,
        "Ogerpon Box",
        "Matthew Hubbard",
        [
            756, 756, 756, 756, 1071, 1071, 1071, 96, 96, 96, 272, 272, 184, 184,
            108, 140, 209, 63, 979, 1198, 1198, 1198, 1198, 1182, 1182, 1188,
            1227, 1205, 1223, 1116, 1116, 1116, 1116, 1121, 1121, 1121, 1121,
            1098, 1098, 1097, 1097, 1159, 1250, 1250, 1250, 1, 1, 1, 1, 1, 1, 1,
            1, 5, 5, 4, 4, 6, 6, 3,
        ],
    ),
)


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
class SampleDragapultDebugGame:
    deck_index: int
    deck_label: str
    archetype: str
    game_index: int
    outcome: str
    our_player_index: int
    steps: int
    prize_counts: tuple[int, int] | None
    error: str | None
    trace: list[dict[str, Any]]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SampleDragapultBenchmarkResult:
    sample_deck_label: str
    games_per_deck: int
    max_steps: int
    rows: list[SampleDragapultBenchmarkRow]
    debug_games: list[SampleDragapultDebugGame] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sample_deck_label": self.sample_deck_label,
            "games_per_deck": self.games_per_deck,
            "max_steps": self.max_steps,
            "rows": [row.to_dict() for row in self.rows],
            "debug_games": [game.to_dict() for game in self.debug_games],
        }


@dataclass
class Phase3RequiredBenchmarkRow:
    deck_index: int
    deck_label: str
    archetype: str
    tournament_rank: int
    opponent: str
    opponent_deck_label: str
    games: int
    wins: int = 0
    losses: int = 0
    draws: int = 0
    timeouts: int = 0
    errors: int = 0
    win_rate: float = 0.0
    search_telemetry: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        if not self.search_telemetry:
            payload.pop("search_telemetry", None)
        return payload


@dataclass
class Phase3RequiredDebugGame:
    deck_index: int
    deck_label: str
    archetype: str
    tournament_rank: int
    opponent: str
    game_index: int
    outcome: str
    our_player_index: int
    steps: int
    prize_counts: tuple[int, int] | None
    error: str | None
    trace: list[dict[str, Any]]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Phase3RequiredBenchmarkResult:
    our_deck_source_url: str
    requested_ranks: tuple[int, ...]
    substitutions: tuple[dict[str, Any], ...]
    games_per_matchup: int
    max_steps: int
    rows: list[Phase3RequiredBenchmarkRow]
    debug_games: list[Phase3RequiredDebugGame] = field(default_factory=list)
    search_telemetry: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = {
            "our_deck_source_url": self.our_deck_source_url,
            "requested_ranks": list(self.requested_ranks),
            "substitutions": list(self.substitutions),
            "games_per_matchup": self.games_per_matchup,
            "max_steps": self.max_steps,
            "rows": [row.to_dict() for row in self.rows],
            "debug_games": [game.to_dict() for game in self.debug_games],
        }
        if self.search_telemetry:
            payload["search_telemetry"] = dict(self.search_telemetry)
        return payload


def phase3_benchmark_deck_coverage(rows: list[SampleDragapultBenchmarkRow]) -> list[dict[str, Any]]:
    coverage = []
    haystacks = [
        f"{row.archetype} {row.deck_label}".casefold()
        for row in rows
    ]
    for required in REQUIRED_PHASE3_BENCHMARK_DECKS:
        needle = required.casefold()
        matches = [
            row.deck_index
            for row, haystack in zip(rows, haystacks, strict=False)
            if needle in haystack
        ]
        coverage.append(
            {
                "required_deck": required,
                "status": "covered" if matches else "missing",
                "deck_indices": matches,
            }
        )
    return coverage


def required_phase3_prepared_decks(start_index: int) -> list[PreparedDeck]:
    prepared = []
    for offset, (name, card_ids, source_url) in enumerate(REQUIRED_PHASE3_SAMPLE_DECKS):
        if len(card_ids) != 60:
            raise ValueError(f"Required benchmark deck {name} has {len(card_ids)} cards, expected 60.")
        deck = Decklist(
            archetype=Archetype(
                rank=1000 + offset,
                name=name,
                deck_id=f"required-{offset + 1}",
                points=None,
                share=None,
                source_url=source_url,
            ),
            variant=Variant(name="Required benchmark sample", value=None, source_url=source_url),
            result=TournamentResult(
                event_name="Phase 3 required benchmark",
                event_date="2026-06-18",
                placement="Sample",
                placement_rank=1000 + offset,
                player="Required sample",
                decklist_url=source_url,
                source_url=source_url,
                page_order=offset,
            ),
            title=name,
            cards=[CardLine(count=1, name=str(card_id), section="Card IDs") for card_id in card_ids],
            total_cards=60,
            fingerprint=f"required-phase3-{offset + 1}",
            source_url=source_url,
        )
        prepared.append(PreparedDeck(index=start_index + offset, deck=deck, card_ids=list(card_ids)))
    return prepared


def phase3_tournament_559_prepared_decks(start_index: int = 1) -> list[PreparedDeck]:
    prepared = []
    for offset, (rank, archetype, player, card_ids) in enumerate(TOURNAMENT_559_DECKS):
        if len(card_ids) != 60:
            raise ValueError(
                f"Tournament 559 rank {rank} deck has {len(card_ids)} cards, expected 60."
            )
        deck = Decklist(
            archetype=Archetype(
                rank=rank,
                name=archetype,
                deck_id=f"tournament-559-rank-{rank}",
                points=None,
                share=None,
                source_url=TOURNAMENT_559_SOURCE_URL,
            ),
            variant=Variant(
                name=f"Rank {rank}",
                value=None,
                source_url=TOURNAMENT_559_SOURCE_URL,
            ),
            result=TournamentResult(
                event_name="Regional Indianapolis, IN",
                event_date="2026-05-30",
                placement=f"Rank {rank}",
                placement_rank=rank,
                player=player,
                decklist_url=TOURNAMENT_559_SOURCE_URL,
                source_url=TOURNAMENT_559_SOURCE_URL,
                page_order=offset,
            ),
            title=archetype,
            cards=[CardLine(count=1, name=str(card_id), section="Card IDs") for card_id in card_ids],
            total_cards=60,
            fingerprint=f"tournament-559-rank-{rank}",
            source_url=TOURNAMENT_559_SOURCE_URL,
        )
        prepared.append(PreparedDeck(index=start_index + offset, deck=deck, card_ids=list(card_ids)))
    return prepared


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


def _quiet_rule_agent(
    deck_ids: list[int],
    card_data: list[Any],
    attack_data: list[Any],
    *,
    trace: list[Any] | None = None,
    trace_limit: int = 0,
) -> RuleBasedAgent:
    with contextlib.redirect_stdout(io.StringIO()):
        return RuleBasedAgent(
            deck_ids,
            card_data=card_data,
            attack_data=attack_data,
            trace=trace,
            trace_limit=trace_limit,
        )


def run_sample_dragapult_benchmark(
    prepared_decks: list[PreparedDeck],
    *,
    sample_dir: Path,
    games_per_deck: int = 10,
    max_steps: int = 600,
    debug_limit_per_deck: int = 2,
    trace_limit: int = 80,
) -> SampleDragapultBenchmarkResult:
    card_data, attack_data = load_engine_metadata(sample_dir)
    sample_module = _load_sample_dragapult_module(sample_dir)
    sample_deck = list(sample_module.SAMPLE_DRAGAPULT_DECK)
    rows: list[SampleDragapultBenchmarkRow] = []
    debug_games: list[SampleDragapultDebugGame] = []

    for prepared in prepared_decks:
        row = SampleDragapultBenchmarkRow(
            deck_index=prepared.index,
            deck_label=prepared.label,
            archetype=prepared.archetype,
            games=games_per_deck,
        )
        kept_debug_games = 0
        for game_index in range(games_per_deck):
            our_is_player0 = game_index % 2 == 0
            trace = []
            sample_agent = sample_module.SampleDragapultAgent()
            rule_agent = _quiet_rule_agent(
                prepared.card_ids,
                card_data,
                attack_data,
                trace=trace,
                trace_limit=trace_limit,
            )
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
            timeout = False
            outcome = "draw"
            if result.error:
                row.errors += 1
                row.draws += 1
                outcome = "error"
            else:
                if result.winner is None:
                    if not result.finished:
                        row.timeouts += 1
                        timeout = True
                    effective_winner = result.leader
                else:
                    effective_winner = result.winner

                if effective_winner is None:
                    row.draws += 1
                    outcome = "timeout_draw" if timeout else "draw"
                elif (effective_winner == 0 and our_is_player0) or (
                    effective_winner == 1 and not our_is_player0
                ):
                    row.wins += 1
                    outcome = "win"
                else:
                    row.losses += 1
                    outcome = "timeout_loss" if timeout else "loss"

            if (
                debug_limit_per_deck > 0
                and kept_debug_games < debug_limit_per_deck
                and outcome in {"loss", "timeout_loss", "timeout_draw", "error"}
            ):
                debug_games.append(
                    SampleDragapultDebugGame(
                        deck_index=prepared.index,
                        deck_label=prepared.label,
                        archetype=prepared.archetype,
                        game_index=game_index + 1,
                        outcome=outcome,
                        our_player_index=0 if our_is_player0 else 1,
                        steps=result.steps,
                        prize_counts=result.prize_counts,
                        error=result.error,
                        trace=[entry.to_dict() for entry in trace],
                    )
                )
                kept_debug_games += 1
        row.win_rate = row.wins / row.games if row.games else 0.0
        rows.append(row)

    return SampleDragapultBenchmarkResult(
        sample_deck_label="Kiyotah sample Dragapult ex deck",
        games_per_deck=games_per_deck,
        max_steps=max_steps,
        rows=rows,
        debug_games=debug_games,
    )


def _record_phase3_required_outcome(
    row: Phase3RequiredBenchmarkRow,
    result: BattleResult,
    *,
    our_is_player0: bool,
) -> tuple[str, bool]:
    timeout = False
    if result.error:
        row.errors += 1
        row.draws += 1
        return "error", False

    if result.winner is None:
        if not result.finished:
            row.timeouts += 1
            timeout = True
        effective_winner = result.leader
    else:
        effective_winner = result.winner

    if effective_winner is None:
        row.draws += 1
        return ("timeout_draw" if timeout else "draw"), timeout

    if (effective_winner == 0 and our_is_player0) or (
        effective_winner == 1 and not our_is_player0
    ):
        row.wins += 1
        return ("timeout_win" if timeout else "win"), timeout

    row.losses += 1
    return ("timeout_loss" if timeout else "loss"), timeout


def run_phase3_required_benchmark(
    our_decks: list[PreparedDeck],
    benchmark_decks: list[PreparedDeck],
    *,
    sample_dir: Path,
    games_per_matchup: int = 10,
    max_steps: int = 600,
    debug_limit_per_matchup: int = 1,
    trace_limit: int = 60,
) -> Phase3RequiredBenchmarkResult:
    card_data, attack_data = load_engine_metadata(sample_dir)
    rows: list[Phase3RequiredBenchmarkRow] = []
    debug_games: list[Phase3RequiredDebugGame] = []

    for our_deck in our_decks:
        for benchmark_deck in benchmark_decks:
            row = Phase3RequiredBenchmarkRow(
                deck_index=our_deck.index,
                deck_label=our_deck.label,
                archetype=our_deck.archetype,
                tournament_rank=our_deck.deck.result.placement_rank,
                opponent=benchmark_deck.archetype,
                opponent_deck_label=benchmark_deck.label,
                games=games_per_matchup,
            )
            kept_debug_games = 0
            for game_index in range(games_per_matchup):
                our_is_player0 = game_index % 2 == 0
                trace = []
                our_agent = _quiet_rule_agent(
                    our_deck.card_ids,
                    card_data,
                    attack_data,
                    trace=trace,
                    trace_limit=trace_limit,
                )
                benchmark_agent = _quiet_rule_agent(
                    benchmark_deck.card_ids,
                    card_data,
                    attack_data,
                )
                result = run_battle(
                    our_deck.card_ids if our_is_player0 else benchmark_deck.card_ids,
                    benchmark_deck.card_ids if our_is_player0 else our_deck.card_ids,
                    sample_dir=sample_dir,
                    agent0=our_agent if our_is_player0 else benchmark_agent,
                    agent1=benchmark_agent if our_is_player0 else our_agent,
                    card_data=card_data,
                    attack_data=attack_data,
                    max_steps=max_steps,
                )
                outcome, _ = _record_phase3_required_outcome(
                    row,
                    result,
                    our_is_player0=our_is_player0,
                )
                if (
                    debug_limit_per_matchup > 0
                    and kept_debug_games < debug_limit_per_matchup
                    and outcome in {"loss", "timeout_loss", "timeout_draw", "error"}
                ):
                    debug_games.append(
                        Phase3RequiredDebugGame(
                            deck_index=our_deck.index,
                            deck_label=our_deck.label,
                            archetype=our_deck.archetype,
                            tournament_rank=our_deck.deck.result.placement_rank,
                            opponent=benchmark_deck.archetype,
                            game_index=game_index + 1,
                            outcome=outcome,
                            our_player_index=0 if our_is_player0 else 1,
                            steps=result.steps,
                            prize_counts=result.prize_counts,
                            error=result.error,
                            trace=[entry.to_dict() for entry in trace],
                        )
                    )
                    kept_debug_games += 1
            row.win_rate = row.wins / row.games if row.games else 0.0
            rows.append(row)

    return Phase3RequiredBenchmarkResult(
        our_deck_source_url=TOURNAMENT_559_SOURCE_URL,
        requested_ranks=TOURNAMENT_559_REQUESTED_RANKS,
        substitutions=TOURNAMENT_559_SUBSTITUTIONS,
        games_per_matchup=games_per_matchup,
        max_steps=max_steps,
        rows=rows,
        debug_games=debug_games,
    )


def _phase3_required_totals(rows: list[Phase3RequiredBenchmarkRow]) -> dict[str, Any]:
    games = sum(row.games for row in rows)
    wins = sum(row.wins for row in rows)
    losses = sum(row.losses for row in rows)
    draws = sum(row.draws for row in rows)
    timeouts = sum(row.timeouts for row in rows)
    errors = sum(row.errors for row in rows)
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "timeouts": timeouts,
        "errors": errors,
        "win_rate": wins / games if games else 0.0,
    }


def _phase3_required_deck_totals(rows: list[Phase3RequiredBenchmarkRow]) -> list[dict[str, Any]]:
    totals: dict[int, dict[str, Any]] = {}
    for row in rows:
        data = totals.setdefault(
            row.deck_index,
            {
                "deck_index": row.deck_index,
                "tournament_rank": row.tournament_rank,
                "archetype": row.archetype,
                "deck_label": row.deck_label,
                "games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "timeouts": 0,
                "errors": 0,
            },
        )
        data["games"] += row.games
        data["wins"] += row.wins
        data["losses"] += row.losses
        data["draws"] += row.draws
        data["timeouts"] += row.timeouts
        data["errors"] += row.errors
    ordered = sorted(totals.values(), key=lambda item: item["deck_index"])
    for data in ordered:
        data["win_rate"] = data["wins"] / data["games"] if data["games"] else 0.0
    return ordered


def write_phase3_required_benchmark_report(
    result: Phase3RequiredBenchmarkResult,
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    totals = _phase3_required_totals(result.rows)
    deck_totals = _phase3_required_deck_totals(result.rows)
    requested_ranks = ", ".join(str(rank) for rank in result.requested_ranks)
    lines = [
        "# Phase 3 Required Benchmark",
        "",
        f"Our deck source: {result.our_deck_source_url}",
        f"Our requested ranks: {requested_ranks}",
        "Benchmark decks: Crustle, Mega Lucario ex, Mega Abomasnow ex, Iono's Bellibolt ex",
        f"Games per matchup: {result.games_per_matchup}",
        f"Max selections per game: {result.max_steps}",
        "",
        "## Overall",
        "",
        f"- Games: {totals['games']}",
        f"- Wins: {totals['wins']}",
        f"- Losses: {totals['losses']}",
        f"- Draws: {totals['draws']}",
        f"- Timeouts: {totals['timeouts']}",
        f"- Errors: {totals['errors']}",
        f"- Win rate: {totals['win_rate']:.3f}",
        "",
    ]
    if result.substitutions:
        lines.extend(
            [
                "## Temporary Substitutions",
                "",
                "| Rank | Original | Replacement | Reason |",
                "| ---: | --- | --- | --- |",
            ]
        )
        for substitution in result.substitutions:
            lines.append(
                f"| {substitution['rank']} | {substitution['original_card']} | "
                f"{substitution['replacement_card']} | {substitution['reason']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Per-Deck Totals",
            "",
            "| Deck | Tournament rank | Archetype | Wins | Losses | Draws | Timeouts | Errors | Win rate |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in deck_totals:
        lines.append(
            f"| {row['deck_index']} | {row['tournament_rank']} | {row['archetype']} | "
            f"{row['wins']} | {row['losses']} | {row['draws']} | {row['timeouts']} | "
            f"{row['errors']} | {row['win_rate']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Matchups",
            "",
            "| Deck | Tournament rank | Archetype | Opponent | Wins | Losses | Draws | Timeouts | Errors | Win rate |",
            "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result.rows:
        lines.append(
            f"| {row.deck_index} | {row.tournament_rank} | {row.archetype} | {row.opponent} | "
            f"{row.wins} | {row.losses} | {row.draws} | {row.timeouts} | "
            f"{row.errors} | {row.win_rate:.3f} |"
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
    seen_labels = set()
    for row in result.rows:
        if row.deck_index in seen_labels:
            continue
        seen_labels.add(row.deck_index)
        lines.append(f"| {row.deck_index} | {row.deck_label} |")

    if result.debug_games:
        lines.extend(
            [
                "",
                "## Debug Samples",
                "",
                "These compact traces are captured from the first loss, timeout, or error in each matchup.",
                "",
            ]
        )
        for game in result.debug_games[:20]:
            lines.extend(
                [
                    f"### Deck {game.deck_index} vs {game.opponent}, game {game.game_index}: {game.outcome}",
                    "",
                    f"- Our player index: {game.our_player_index}",
                    f"- Steps: {game.steps}",
                    f"- Prize counts: `{game.prize_counts}`",
                ]
            )
            if game.error:
                lines.append(f"- Error: `{game.error}`")
            lines.append("")

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
            "## Phase 3 Coverage Targets",
            "",
            "These named decks are required for the final Phase 3 benchmark target.",
            "",
            "| Required deck | Status | Corpus deck indices |",
            "| --- | --- | --- |",
        ]
    )
    for row in phase3_benchmark_deck_coverage(result.rows):
        indices = ", ".join(str(index) for index in row["deck_indices"]) or "-"
        lines.append(f"| {row['required_deck']} | {row['status']} | {indices} |")
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

    if result.debug_games:
        lines.extend(
            [
                "",
                "## Debug Samples",
                "",
                "These are compact traces from early losses, timeouts, or errors. They show the rule "
                "agent's selected option and the first prize-map route it was trying to serve.",
                "",
            ]
        )
        for game in result.debug_games:
            lines.extend(
                [
                    f"### Deck {game.deck_index}, game {game.game_index}: {game.outcome}",
                    "",
                    f"- Label: `{game.deck_label}`",
                    f"- Our player index: {game.our_player_index}",
                    f"- Steps: {game.steps}",
                    f"- Prize counts: `{game.prize_counts}`",
                ]
            )
            if game.error:
                lines.append(f"- Error: `{game.error}`")
            lines.extend(
                [
                    "",
                    "| Turn | Context | Selected | Prize map | Key attackers |",
                    "| ---: | --- | --- | --- | --- |",
                ]
            )
            interesting = [
                entry
                for entry in game.trace
                if entry["prize_map"]["steps"]
                or entry["context"] in {"MAIN", "ATTACK", "TO_ACTIVE", "SWITCH", "TO_HAND"}
            ][:10]
            for entry in interesting:
                selected = "; ".join(
                    f"{option['type']}#{option['index']} {option.get('card', {}).get('name', '')}"
                    f"{' atk=' + str(option['attack_id']) if 'attack_id' in option else ''}"
                    for option in entry["selected_options"]
                )
                steps = entry["prize_map"]["steps"]
                if steps:
                    first = steps[0]
                    route = (
                        f"{first['attacker']['name']} -> atk {first['attack_id']} -> "
                        f"{first['target']['area']} {first['target']['name']} "
                        f"({first['prizes_taken']} prizes, setup {first['setup_cost']})"
                    )
                else:
                    route = "none"
                attackers = ", ".join(attacker["name"] for attacker in entry["key_attackers"][:4])
                lines.append(
                    f"| {entry['turn']} | {entry['context']} | {selected or 'none'} | "
                    f"{route} | {attackers or 'none'} |"
                )
            lines.append("")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sample_totals(result: SampleDragapultBenchmarkResult) -> dict[str, Any]:
    games = sum(row.games for row in result.rows)
    wins = sum(row.wins for row in result.rows)
    losses = sum(row.losses for row in result.rows)
    draws = sum(row.draws for row in result.rows)
    timeouts = sum(row.timeouts for row in result.rows)
    errors = sum(row.errors for row in result.rows)
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "timeouts": timeouts,
        "errors": errors,
        "win_rate": wins / games if games else 0.0,
    }


def sample_dragapult_benchmark_from_dict(data: dict[str, Any]) -> SampleDragapultBenchmarkResult:
    return SampleDragapultBenchmarkResult(
        sample_deck_label=str(data["sample_deck_label"]),
        games_per_deck=int(data["games_per_deck"]),
        max_steps=int(data["max_steps"]),
        rows=[SampleDragapultBenchmarkRow(**row) for row in data["rows"]],
        debug_games=[
            SampleDragapultDebugGame(**game) for game in data.get("debug_games", [])
        ],
    )


def write_sample_dragapult_comparison_report(
    baseline: SampleDragapultBenchmarkResult,
    current: SampleDragapultBenchmarkResult,
    *,
    markdown_path: Path,
) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_by_index = {row.deck_index: row for row in baseline.rows}
    base_totals = _sample_totals(baseline)
    current_totals = _sample_totals(current)

    lines = [
        "# Sample Dragapult Benchmark Comparison",
        "",
        "| Metric | Previous | Current | Delta |",
        "| --- | ---: | ---: | ---: |",
        f"| Wins | {base_totals['wins']} | {current_totals['wins']} | "
        f"{current_totals['wins'] - base_totals['wins']:+d} |",
        f"| Losses | {base_totals['losses']} | {current_totals['losses']} | "
        f"{current_totals['losses'] - base_totals['losses']:+d} |",
        f"| Draws | {base_totals['draws']} | {current_totals['draws']} | "
        f"{current_totals['draws'] - base_totals['draws']:+d} |",
        f"| Timeouts | {base_totals['timeouts']} | {current_totals['timeouts']} | "
        f"{current_totals['timeouts'] - base_totals['timeouts']:+d} |",
        f"| Errors | {base_totals['errors']} | {current_totals['errors']} | "
        f"{current_totals['errors'] - base_totals['errors']:+d} |",
        f"| Win rate | {base_totals['win_rate']:.3f} | {current_totals['win_rate']:.3f} | "
        f"{current_totals['win_rate'] - base_totals['win_rate']:+.3f} |",
        "",
        "## Deck-Level Changes",
        "",
        "| Deck | Archetype | Previous | Current | Win delta | Timeout delta |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in current.rows:
        old = baseline_by_index.get(row.deck_index)
        if old is None:
            continue
        lines.append(
            f"| {row.deck_index} | {row.archetype} | {old.wins}-{old.losses}-{old.draws} | "
            f"{row.wins}-{row.losses}-{row.draws} | {row.wins - old.wins:+d} | "
            f"{row.timeouts - old.timeouts:+d} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
