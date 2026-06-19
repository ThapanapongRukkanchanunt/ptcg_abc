from __future__ import annotations

import contextlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from ptcg_abc.agent import HybridRlAgent, RuleBasedAgent
from ptcg_abc.evaluation import (
    Phase3RequiredBenchmarkResult,
    Phase3RequiredBenchmarkRow,
    PreparedDeck,
    TOURNAMENT_559_REQUESTED_RANKS,
    TOURNAMENT_559_SOURCE_URL,
    TOURNAMENT_559_SUBSTITUTIONS,
    phase3_tournament_559_prepared_decks,
    required_phase3_prepared_decks,
)
from ptcg_abc.rl.dataset import append_decision_jsonl, append_trajectory_jsonl, read_decision_jsonl
from ptcg_abc.rl.featurizer import attack_lookup, card_lookup, make_decision_frame
from ptcg_abc.rl.guidance import default_guidance_rules
from ptcg_abc.rl.model import (
    LinearOptionModel,
    TrainingSummary,
    train_behavior_cloning_model,
    train_reward_weighted_model,
)
from ptcg_abc.rl.records import DecisionFrame, TrajectoryStep
from ptcg_abc.rl.rewards import reward_from_result_metadata
from ptcg_abc.rl.torch_backend import train_torch_bc_model
from ptcg_abc.simulator import BattleResult, load_engine_metadata, run_battle


@dataclass(frozen=True)
class CollectionSummary:
    games_requested: int
    games_started: int
    decisions: int
    output_path: str
    errors: int
    timeouts: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RolloutSummary:
    agent: str
    games_requested: int
    games_started: int
    steps: int
    output_path: str
    wins: int
    losses: int
    draws: int
    errors: int
    timeouts: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_bc_demonstrations(
    *,
    sample_dir: Path,
    output_path: Path,
    games: int = 36,
    max_steps: int = 120,
) -> CollectionSummary:
    card_data, attack_data = load_engine_metadata(sample_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    our_decks = phase3_tournament_559_prepared_decks()
    benchmark_decks = required_phase3_prepared_decks(start_index=1)
    matchups = [(our, benchmark) for our in our_decks for benchmark in benchmark_decks]
    decisions = games_started = errors = timeouts = 0

    for game_index in range(games):
        our_deck, benchmark_deck = matchups[game_index % len(matchups)]
        our_is_player0 = game_index % 2 == 0
        recorder = RecordingRuleAgent(
            our_deck.card_ids,
            card_data=card_data,
            attack_data=attack_data,
            reward_metadata={
                "game_index": game_index + 1,
                "deck_index": our_deck.index,
                "deck_label": our_deck.label,
                "opponent": benchmark_deck.archetype,
                "collector": "rule_bc",
            },
        )
        benchmark_agent = _quiet_rule_agent(benchmark_deck.card_ids, card_data, attack_data)
        result = run_battle(
            our_deck.card_ids if our_is_player0 else benchmark_deck.card_ids,
            benchmark_deck.card_ids if our_is_player0 else our_deck.card_ids,
            sample_dir=sample_dir,
            agent0=recorder if our_is_player0 else benchmark_agent,
            agent1=benchmark_agent if our_is_player0 else recorder,
            card_data=card_data,
            attack_data=attack_data,
            max_steps=max_steps,
        )
        games_started += int(result.started)
        errors += int(result.error is not None)
        timeouts += int(result.started and not result.finished and result.error is None)
        final_metadata = _result_metadata(result, 0 if our_is_player0 else 1)
        for frame in recorder.frames:
            append_decision_jsonl(_with_metadata(frame, final_metadata), output_path)
            decisions += 1

    return CollectionSummary(
        games_requested=games,
        games_started=games_started,
        decisions=decisions,
        output_path=str(output_path.as_posix()),
        errors=errors,
        timeouts=timeouts,
    )


def train_bc_from_jsonl(
    *,
    dataset_path: Path,
    model_path: Path,
    report_path: Path | None = None,
    epochs: int = 1,
) -> TrainingSummary:
    frames = read_decision_jsonl(dataset_path)
    model, summary = train_behavior_cloning_model(frames, epochs=epochs)
    model.save(model_path)
    summary = TrainingSummary(
        frames=summary.frames,
        actions=summary.actions,
        positives=summary.positives,
        negatives=summary.negatives,
        epochs=summary.epochs,
        model_path=str(model_path.as_posix()),
        accuracy=summary.accuracy,
    )
    if report_path is not None:
        _write_json_report(summary.to_dict(), report_path)
    return summary


def train_torch_bc_from_jsonl(
    *,
    dataset_path: Path,
    checkpoint_path: Path,
    export_model_path: Path,
    report_path: Path | None = None,
    epochs: int = 1,
    learning_rate: float = 0.02,
) -> Any:
    frames = read_decision_jsonl(dataset_path)
    summary = train_torch_bc_model(
        frames,
        checkpoint_path=checkpoint_path,
        export_model_path=export_model_path,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    if report_path is not None:
        _write_json_report(summary.to_dict(), report_path)
    return summary


def rollout_games(
    *,
    sample_dir: Path,
    output_path: Path,
    agent_kind: str = "hybrid",
    model_path: Path | None = None,
    games: int = 36,
    max_steps: int = 120,
    guidance_rules: Sequence[str] | None = None,
) -> RolloutSummary:
    card_data, attack_data = load_engine_metadata(sample_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    our_decks = phase3_tournament_559_prepared_decks()
    benchmark_decks = required_phase3_prepared_decks(start_index=1)
    matchups = [(our, benchmark) for our in our_decks for benchmark in benchmark_decks]
    games_started = wins = losses = draws = errors = timeouts = steps_written = 0

    for game_index in range(games):
        our_deck, benchmark_deck = matchups[game_index % len(matchups)]
        our_is_player0 = game_index % 2 == 0
        player_index = 0 if our_is_player0 else 1
        recorder = RecordingPolicyAgent(
            _make_agent(
                agent_kind,
                our_deck.card_ids,
                card_data,
                attack_data,
                model_path=model_path,
                guidance_rules=guidance_rules,
            ),
            our_deck.card_ids,
            card_data=card_data,
            attack_data=attack_data,
            reward_metadata={
                "game_index": game_index + 1,
                "deck_index": our_deck.index,
                "deck_label": our_deck.label,
                "opponent": benchmark_deck.archetype,
                "collector": "rollout",
                "agent": agent_kind,
            },
        )
        benchmark_agent = _quiet_rule_agent(benchmark_deck.card_ids, card_data, attack_data)
        result = run_battle(
            our_deck.card_ids if our_is_player0 else benchmark_deck.card_ids,
            benchmark_deck.card_ids if our_is_player0 else our_deck.card_ids,
            sample_dir=sample_dir,
            agent0=recorder if our_is_player0 else benchmark_agent,
            agent1=benchmark_agent if our_is_player0 else recorder,
            card_data=card_data,
            attack_data=attack_data,
            max_steps=max_steps,
        )
        games_started += int(result.started)
        outcome = _agent_outcome(result, player_index)
        wins += int(outcome == "win")
        losses += int(outcome == "loss")
        draws += int(outcome == "draw")
        errors += int(result.error is not None)
        timeouts += int(result.started and not result.finished and result.error is None)
        final_metadata = _result_metadata(result, player_index)
        reward = reward_from_result_metadata(final_metadata)
        for frame, chosen_indices in recorder.frames:
            frame = _with_metadata(frame, final_metadata)
            append_trajectory_jsonl(
                TrajectoryStep(
                    decision=frame,
                    chosen_indices=chosen_indices,
                    reward=reward,
                    terminal=result.finished,
                    truncated=not result.finished and result.error is None,
                ),
                output_path,
            )
            steps_written += 1

    return RolloutSummary(
        agent=agent_kind,
        games_requested=games,
        games_started=games_started,
        steps=steps_written,
        output_path=str(output_path.as_posix()),
        wins=wins,
        losses=losses,
        draws=draws,
        errors=errors,
        timeouts=timeouts,
    )


def train_ppo_from_rollouts(
    *,
    rollout_path: Path,
    model_path: Path,
    output_path: Path,
    report_path: Path | None = None,
    epochs: int = 1,
) -> TrainingSummary:
    from ptcg_abc.rl.dataset import read_trajectory_jsonl

    steps = read_trajectory_jsonl(rollout_path)
    base_model = LinearOptionModel.load(model_path) if model_path.exists() else None
    model, summary = train_reward_weighted_model(steps, base_model=base_model, epochs=epochs)
    model.save(output_path)
    summary = TrainingSummary(
        frames=summary.frames,
        actions=summary.actions,
        positives=summary.positives,
        negatives=summary.negatives,
        epochs=summary.epochs,
        model_path=str(output_path.as_posix()),
        accuracy=summary.accuracy,
    )
    if report_path is not None:
        _write_json_report(summary.to_dict(), report_path)
    return summary


def run_phase4_required_benchmark(
    *,
    sample_dir: Path,
    agent_kind: str = "hybrid",
    model_path: Path | None = None,
    games_per_matchup: int = 1,
    max_steps: int = 120,
    guidance_rules: Sequence[str] | None = None,
) -> Phase3RequiredBenchmarkResult:
    card_data, attack_data = load_engine_metadata(sample_dir)
    our_decks = phase3_tournament_559_prepared_decks()
    benchmark_decks = required_phase3_prepared_decks(start_index=1)
    rows: list[Phase3RequiredBenchmarkRow] = []

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
            for game_index in range(games_per_matchup):
                our_is_player0 = game_index % 2 == 0
                our_agent = _make_agent(
                    agent_kind,
                    our_deck.card_ids,
                    card_data,
                    attack_data,
                    model_path=model_path,
                    guidance_rules=guidance_rules,
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
                _record_row_outcome(row, result, our_is_player0=our_is_player0)
            row.win_rate = row.wins / row.games if row.games else 0.0
            rows.append(row)

    return Phase3RequiredBenchmarkResult(
        our_deck_source_url=TOURNAMENT_559_SOURCE_URL,
        requested_ranks=TOURNAMENT_559_REQUESTED_RANKS,
        substitutions=TOURNAMENT_559_SUBSTITUTIONS,
        games_per_matchup=games_per_matchup,
        max_steps=max_steps,
        rows=rows,
        debug_games=[],
    )


def write_phase4_benchmark_report(
    result: Phase3RequiredBenchmarkResult,
    *,
    json_path: Path,
    markdown_path: Path,
    agent_kind: str,
    model_path: Path | None = None,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    totals = _totals(result.rows)
    lines = [
        "# Phase 4 Required Benchmark",
        "",
        f"Agent: `{agent_kind}`",
        f"Model: `{model_path.as_posix() if model_path else 'none'}`",
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
        "## Matchups",
        "",
        "| Deck | Rank | Archetype | Opponent | Wins | Losses | Draws | Timeouts | Errors | Win rate |",
        "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result.rows:
        lines.append(
            f"| {row.deck_index} | {row.tournament_rank} | {row.archetype} | {row.opponent} | "
            f"{row.wins} | {row.losses} | {row.draws} | {row.timeouts} | "
            f"{row.errors} | {row.win_rate:.3f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class RecordingRuleAgent:
    def __init__(
        self,
        deck_ids: Sequence[int],
        *,
        card_data: Sequence[Any],
        attack_data: Sequence[Any],
        reward_metadata: dict[str, Any],
    ) -> None:
        self.deck_ids = list(deck_ids)
        self.card_data = card_data
        self.attack_data = attack_data
        self.card_by_id = card_lookup(card_data)
        self.attack_by_id = attack_lookup(attack_data)
        self.reward_metadata = dict(reward_metadata)
        self.frames: list[DecisionFrame] = []
        with contextlib.redirect_stdout(io.StringIO()):
            self.agent = RuleBasedAgent(deck_ids, card_data=card_data, attack_data=attack_data)

    def act(self, observation: Any) -> list[int]:
        selected = self.agent.act(observation)
        frame = make_decision_frame(
            observation,
            deck_ids=self.deck_ids,
            card_by_id=self.card_by_id,
            attack_by_id=self.attack_by_id,
            selected_indices=selected,
            reward_metadata=self.reward_metadata | {"step_index": len(self.frames) + 1},
        )
        if frame is not None:
            self.frames.append(frame)
        return selected


class RecordingPolicyAgent:
    def __init__(
        self,
        agent: Any,
        deck_ids: Sequence[int],
        *,
        card_data: Sequence[Any],
        attack_data: Sequence[Any],
        reward_metadata: dict[str, Any],
    ) -> None:
        self.agent = agent
        self.deck_ids = list(deck_ids)
        self.card_by_id = card_lookup(card_data)
        self.attack_by_id = attack_lookup(attack_data)
        self.reward_metadata = dict(reward_metadata)
        self.frames: list[tuple[DecisionFrame, list[int]]] = []

    def act(self, observation: Any) -> list[int]:
        selected = self.agent.act(observation)
        frame = make_decision_frame(
            observation,
            deck_ids=self.deck_ids,
            card_by_id=self.card_by_id,
            attack_by_id=self.attack_by_id,
            selected_indices=selected,
            reward_metadata=self.reward_metadata | {"step_index": len(self.frames) + 1},
        )
        if frame is not None:
            self.frames.append((frame, list(selected)))
        return selected


def _make_agent(
    agent_kind: str,
    deck_ids: Sequence[int],
    card_data: Sequence[Any],
    attack_data: Sequence[Any],
    *,
    model_path: Path | None = None,
    guidance_rules: Sequence[str] | None = None,
) -> Any:
    if agent_kind == "rule":
        return _quiet_rule_agent(deck_ids, card_data, attack_data)
    rules = default_guidance_rules() if guidance_rules is None else tuple(guidance_rules)
    if agent_kind == "rl":
        rules = ()
    with contextlib.redirect_stdout(io.StringIO()):
        return HybridRlAgent(
            deck_ids,
            card_data=card_data,
            attack_data=attack_data,
            model_path=model_path,
            rule_weight=0.0 if agent_kind == "rl" else 0.35,
            guidance_rules=rules,
        )


def _quiet_rule_agent(
    deck_ids: Sequence[int],
    card_data: Sequence[Any],
    attack_data: Sequence[Any],
) -> RuleBasedAgent:
    with contextlib.redirect_stdout(io.StringIO()):
        return RuleBasedAgent(deck_ids, card_data=card_data, attack_data=attack_data)


def _with_metadata(frame: DecisionFrame, metadata: dict[str, Any]) -> DecisionFrame:
    return DecisionFrame(
        select_type=frame.select_type,
        context=frame.context,
        min_count=frame.min_count,
        max_count=frame.max_count,
        target_count=frame.target_count,
        legal_options=frame.legal_options,
        rule_selected_indices=frame.rule_selected_indices,
        board=frame.board,
        board_image=frame.board_image,
        reward_metadata=frame.reward_metadata | metadata,
        schema_version=frame.schema_version,
    )


def _result_metadata(result: BattleResult, player_index: int) -> dict[str, Any]:
    return {
        "player_index": player_index,
        "winner": result.winner,
        "leader": result.leader,
        "finished": result.finished,
        "prize_counts": list(result.prize_counts) if result.prize_counts is not None else None,
        "error": result.error,
    }


def _agent_outcome(result: BattleResult, player_index: int) -> str:
    if result.error:
        return "draw"
    effective = result.winner if result.winner is not None else result.leader
    if effective is None:
        return "draw"
    return "win" if effective == player_index else "loss"


def _record_row_outcome(
    row: Phase3RequiredBenchmarkRow,
    result: BattleResult,
    *,
    our_is_player0: bool,
) -> None:
    if result.error:
        row.errors += 1
        row.draws += 1
        return
    timeout = result.winner is None and not result.finished
    if timeout:
        row.timeouts += 1
    effective = result.winner if result.winner is not None else result.leader
    if effective is None:
        row.draws += 1
    elif (effective == 0 and our_is_player0) or (effective == 1 and not our_is_player0):
        row.wins += 1
    else:
        row.losses += 1


def _totals(rows: list[Phase3RequiredBenchmarkRow]) -> dict[str, Any]:
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


def _write_json_report(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
