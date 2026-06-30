from __future__ import annotations

import contextlib
import io
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

from ptcg_abc.agent import HybridRlAgent, RuleBasedAgent
from ptcg_abc.agent.phase5_search import Phase5SearchPolicyAgent
from ptcg_abc.agent.phase5_symbolic import Phase5SymbolicPolicyAgent
from ptcg_abc.evaluation import (
    Phase3RequiredDebugGame,
    Phase3RequiredBenchmarkResult,
    Phase3RequiredBenchmarkRow,
    PreparedDeck,
    TOURNAMENT_559_REQUESTED_RANKS,
    TOURNAMENT_559_SOURCE_URL,
    TOURNAMENT_559_SUBSTITUTIONS,
    phase5_league_prepared_decks,
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
from ptcg_abc.rl.phase5_search import RootSearchConfig
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


@dataclass(frozen=True)
class SelfPlaySummary:
    agent: str
    games_requested: int
    games_started: int
    steps: int
    output_path: str
    deck_pool: str
    mode: str
    deck_indices: list[int]
    deck_labels: list[str]
    pair_count: int
    deck_a_index: int | None
    deck_a_label: str | None
    deck_b_index: int | None
    deck_b_label: str | None
    deck_a_wins: int
    deck_b_wins: int
    deck_games: dict[str, int]
    deck_wins: dict[str, int]
    deck_losses: dict[str, int]
    deck_draws: dict[str, int]
    matchups: dict[str, dict[str, int]]
    draws: int
    errors: int
    timeouts: int
    image_size: int
    search_telemetry: dict[str, Any] = field(default_factory=dict)
    trace_path: str | None = None
    trace_records: int = 0
    policy_pool_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SelfPlayDeckPlan:
    mode: str
    deck_indices: list[int]
    deck_labels: list[str]
    pairs: list[tuple[PreparedDeck, PreparedDeck]]
    fixed_deck_a: PreparedDeck | None = None
    fixed_deck_b: PreparedDeck | None = None


@dataclass(frozen=True)
class ProgressionIterationSummary:
    image_size: int
    iteration: int
    selfplay: dict[str, Any]
    update: dict[str, Any]
    evaluation: dict[str, Any]
    model_path: str
    rollout_path: str
    replay_dir: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProgressionExperimentSummary:
    image_size: int
    iterations: int
    summaries: list[ProgressionIterationSummary]
    output_root: str
    dataset_root: str
    model_root: str
    report_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_size": self.image_size,
            "iterations": self.iterations,
            "summaries": [summary.to_dict() for summary in self.summaries],
            "output_root": self.output_root,
            "dataset_root": self.dataset_root,
            "model_root": self.model_root,
            "report_root": self.report_root,
        }


def _ordered_unique_ints(values: Sequence[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


PHASE5_SELFPLAY_DECK_POOL_TOURNAMENT_9 = "tournament-9"
PHASE5_SELFPLAY_DECK_POOL_LEAGUE_13 = "league-13"
PHASE5_SELFPLAY_DECK_POOLS = (
    PHASE5_SELFPLAY_DECK_POOL_TOURNAMENT_9,
    PHASE5_SELFPLAY_DECK_POOL_LEAGUE_13,
)


def phase5_selfplay_prepared_decks(deck_pool: str) -> list[PreparedDeck]:
    if deck_pool == PHASE5_SELFPLAY_DECK_POOL_TOURNAMENT_9:
        return phase3_tournament_559_prepared_decks()
    if deck_pool == PHASE5_SELFPLAY_DECK_POOL_LEAGUE_13:
        return phase5_league_prepared_decks()
    valid = ", ".join(PHASE5_SELFPLAY_DECK_POOLS)
    raise ValueError(
        f"Unknown Phase 5 self-play deck pool {deck_pool!r}; choose one of: {valid}."
    )


def build_selfplay_deck_plan(
    prepared_decks: Sequence[PreparedDeck],
    *,
    deck_a_index: int | None = None,
    deck_b_index: int | None = None,
    selfplay_deck_indices: Sequence[int] | None = None,
) -> SelfPlayDeckPlan:
    decks_by_index = {deck.index: deck for deck in prepared_decks}
    if selfplay_deck_indices:
        if deck_a_index is not None or deck_b_index is not None:
            raise ValueError(
                "Use either selfplay_deck_indices for rotation or deck_a_index/deck_b_index "
                "for a fixed pair, not both."
            )
        requested_indices = _ordered_unique_ints(list(selfplay_deck_indices))
        mode = "rotate"
    elif deck_a_index is None and deck_b_index is None:
        requested_indices = [deck.index for deck in prepared_decks]
        mode = "rotate"
    elif deck_a_index is not None and deck_b_index is not None:
        missing = [index for index in (deck_a_index, deck_b_index) if index not in decks_by_index]
        if missing:
            raise ValueError(f"Unknown prepared deck index for fixed self-play: {missing[0]}.")
        deck_a = decks_by_index[deck_a_index]
        deck_b = decks_by_index[deck_b_index]
        return SelfPlayDeckPlan(
            mode="fixed",
            deck_indices=[deck_a.index, deck_b.index],
            deck_labels=[deck_a.label, deck_b.label],
            pairs=[(deck_a, deck_b)],
            fixed_deck_a=deck_a,
            fixed_deck_b=deck_b,
        )
    else:
        raise ValueError("Set both deck_a_index and deck_b_index, or omit both to rotate decks.")

    if not requested_indices:
        raise ValueError("Self-play deck rotation needs at least one deck.")
    for index in requested_indices:
        if index not in decks_by_index:
            raise ValueError(f"Unknown prepared deck index for self-play rotation: {index}.")
    selected_decks = [decks_by_index[index] for index in requested_indices]
    return SelfPlayDeckPlan(
        mode=mode,
        deck_indices=[deck.index for deck in selected_decks],
        deck_labels=[deck.label for deck in selected_decks],
        pairs=[(deck_a, deck_b) for deck_a in selected_decks for deck_b in selected_decks],
    )


def selfplay_pair_for_game(
    plan: SelfPlayDeckPlan,
    game_index: int,
) -> tuple[PreparedDeck, PreparedDeck]:
    if not plan.pairs:
        raise ValueError("Self-play deck plan has no pairs.")
    return plan.pairs[game_index % len(plan.pairs)]


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
    changed_weight: float = 1.0,
    unchanged_weight: float = 1.0,
    excluded_features: Sequence[str] = (),
    pairwise_changed: bool = False,
    pairwise_margin: float = 0.0,
    pairwise_negatives: str = "baseline",
) -> TrainingSummary:
    frames = read_decision_jsonl(dataset_path)
    model, summary = train_behavior_cloning_model(
        frames,
        epochs=epochs,
        changed_weight=changed_weight,
        unchanged_weight=unchanged_weight,
        excluded_features=excluded_features,
        pairwise_changed=pairwise_changed,
        pairwise_margin=pairwise_margin,
        pairwise_negatives=pairwise_negatives,
    )
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
    changed_weight: float = 1.0,
    unchanged_weight: float = 1.0,
    excluded_features: Sequence[str] = (),
    pairwise_changed: bool = False,
    pairwise_margin: float = 0.0,
    pairwise_negatives: str = "baseline",
) -> Any:
    frames = read_decision_jsonl(dataset_path)
    summary = train_torch_bc_model(
        frames,
        checkpoint_path=checkpoint_path,
        export_model_path=export_model_path,
        epochs=epochs,
        learning_rate=learning_rate,
        changed_weight=changed_weight,
        unchanged_weight=unchanged_weight,
        excluded_features=excluded_features,
        pairwise_changed=pairwise_changed,
        pairwise_margin=pairwise_margin,
        pairwise_negatives=pairwise_negatives,
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
                opponent_deck_ids=benchmark_deck.card_ids,
                sample_dir=sample_dir,
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


def rollout_selfplay_games(
    *,
    sample_dir: Path,
    output_path: Path,
    agent_kind: str = "hybrid",
    model_path: Path | None = None,
    games: int = 1000,
    game_offset: int = 0,
    max_steps: int = 600,
    deck_pool: str = PHASE5_SELFPLAY_DECK_POOL_TOURNAMENT_9,
    deck_a_index: int | None = None,
    deck_b_index: int | None = None,
    selfplay_deck_indices: Sequence[int] | None = None,
    image_size: int = 1024,
    guidance_rules: Sequence[str] | None = None,
    search_config: RootSearchConfig | None = None,
    search_trace_path: Path | None = None,
    search_trace_game_limit: int = 0,
    policy_pool_paths: Sequence[Path] = (),
) -> SelfPlaySummary:
    card_data, attack_data = load_engine_metadata(sample_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    if search_trace_path is not None:
        search_trace_path.parent.mkdir(parents=True, exist_ok=True)
        search_trace_path.write_text("", encoding="utf-8")
    plan = build_selfplay_deck_plan(
        phase5_selfplay_prepared_decks(deck_pool),
        deck_a_index=deck_a_index,
        deck_b_index=deck_b_index,
        selfplay_deck_indices=selfplay_deck_indices,
    )
    games_started = deck_a_wins = deck_b_wins = draws = errors = timeouts = steps_written = 0
    deck_keys = {str(index) for index in plan.deck_indices}
    deck_games = {key: 0 for key in deck_keys}
    deck_wins = {key: 0 for key in deck_keys}
    deck_losses = {key: 0 for key in deck_keys}
    deck_draws = {key: 0 for key in deck_keys}
    matchups: dict[str, dict[str, int]] = {}
    search_telemetry: dict[str, Any] = {}
    trace_records = 0

    for game_index in range(games):
        absolute_game_index = game_offset + game_index
        deck_a, deck_b = selfplay_pair_for_game(plan, absolute_game_index)
        deck_a_is_player0 = absolute_game_index % 2 == 0
        player0_deck = deck_a if deck_a_is_player0 else deck_b
        player1_deck = deck_b if deck_a_is_player0 else deck_a
        pair_key = f"{deck_a.index}-vs-{deck_b.index}"
        matchup = matchups.setdefault(
            pair_key,
            {
                "deck_a_index": deck_a.index,
                "deck_b_index": deck_b.index,
                "games": 0,
                "deck_a_wins": 0,
                "deck_b_wins": 0,
                "draws": 0,
                "errors": 0,
                "timeouts": 0,
            },
        )
        matchup["games"] += 1
        for deck in (player0_deck, player1_deck):
            deck_games[str(deck.index)] = deck_games.get(str(deck.index), 0) + 1
        player0_model_path = _policy_pool_model_path(
            policy_pool_paths,
            absolute_game_index=absolute_game_index,
            player_slot=0,
            default_model_path=model_path,
        )
        player1_model_path = _policy_pool_model_path(
            policy_pool_paths,
            absolute_game_index=absolute_game_index,
            player_slot=1,
            default_model_path=model_path,
        )
        recorder0 = RecordingPolicyAgent(
            _make_agent(
                agent_kind,
                player0_deck.card_ids,
                card_data,
                attack_data,
                model_path=player0_model_path,
                guidance_rules=guidance_rules,
                opponent_deck_ids=player1_deck.card_ids,
                sample_dir=sample_dir,
                search_config=search_config,
            ),
            player0_deck.card_ids,
            card_data=card_data,
            attack_data=attack_data,
            reward_metadata={
                "game_index": absolute_game_index + 1,
                "local_game_index": game_index + 1,
                "deck_index": player0_deck.index,
                "deck_label": player0_deck.label,
                "opponent_deck_index": player1_deck.index,
                "opponent_deck_label": player1_deck.label,
                "collector": "selfplay",
                "agent": agent_kind,
                "image_size": image_size,
                "selfplay_deck_pool": deck_pool,
                "selfplay_mode": plan.mode,
                "selfplay_deck_indices": plan.deck_indices,
                "selfplay_pair_key": pair_key,
                "selfplay_deck_a_index": deck_a.index,
                "selfplay_deck_b_index": deck_b.index,
                "policy_pool_model": (
                    player0_model_path.as_posix() if player0_model_path else None
                ),
            },
        )
        recorder1 = RecordingPolicyAgent(
            _make_agent(
                agent_kind,
                player1_deck.card_ids,
                card_data,
                attack_data,
                model_path=player1_model_path,
                guidance_rules=guidance_rules,
                opponent_deck_ids=player0_deck.card_ids,
                sample_dir=sample_dir,
                search_config=search_config,
            ),
            player1_deck.card_ids,
            card_data=card_data,
            attack_data=attack_data,
            reward_metadata={
                "game_index": absolute_game_index + 1,
                "local_game_index": game_index + 1,
                "deck_index": player1_deck.index,
                "deck_label": player1_deck.label,
                "opponent_deck_index": player0_deck.index,
                "opponent_deck_label": player0_deck.label,
                "collector": "selfplay",
                "agent": agent_kind,
                "image_size": image_size,
                "selfplay_deck_pool": deck_pool,
                "selfplay_mode": plan.mode,
                "selfplay_deck_indices": plan.deck_indices,
                "selfplay_pair_key": pair_key,
                "selfplay_deck_a_index": deck_a.index,
                "selfplay_deck_b_index": deck_b.index,
                "policy_pool_model": (
                    player1_model_path.as_posix() if player1_model_path else None
                ),
            },
        )
        result = run_battle(
            player0_deck.card_ids,
            player1_deck.card_ids,
            sample_dir=sample_dir,
            agent0=recorder0,
            agent1=recorder1,
            card_data=card_data,
            attack_data=attack_data,
            max_steps=max_steps,
        )
        if isinstance(recorder0.agent, Phase5SearchPolicyAgent):
            _accumulate_search_telemetry(search_telemetry, recorder0.agent.search_telemetry())
            if _should_write_selfplay_trace(
                search_trace_path,
                game_index=game_index,
                trace_game_limit=search_trace_game_limit,
            ):
                assert search_trace_path is not None
                trace_records += _write_phase5_search_eval_traces(
                    recorder0.agent,
                    search_trace_path,
                    game_index=absolute_game_index + 1,
                    deck_index=player0_deck.index,
                    deck_label=player0_deck.label,
                    opponent=player1_deck.label,
                )
        if isinstance(recorder1.agent, Phase5SearchPolicyAgent):
            _accumulate_search_telemetry(search_telemetry, recorder1.agent.search_telemetry())
            if _should_write_selfplay_trace(
                search_trace_path,
                game_index=game_index,
                trace_game_limit=search_trace_game_limit,
            ):
                assert search_trace_path is not None
                trace_records += _write_phase5_search_eval_traces(
                    recorder1.agent,
                    search_trace_path,
                    game_index=absolute_game_index + 1,
                    deck_index=player1_deck.index,
                    deck_label=player1_deck.label,
                    opponent=player0_deck.label,
                )
        games_started += int(result.started)
        errors += int(result.error is not None)
        timeouts += int(result.started and not result.finished and result.error is None)
        matchup["errors"] += int(result.error is not None)
        matchup["timeouts"] += int(result.started and not result.finished and result.error is None)
        winner = result.winner if result.winner is not None else result.leader
        if winner is None or result.error:
            draws += 1
            matchup["draws"] += 1
            for deck in (player0_deck, player1_deck):
                deck_draws[str(deck.index)] = deck_draws.get(str(deck.index), 0) + 1
        else:
            slot_a_player_index = 0 if deck_a_is_player0 else 1
            deck_a_won = winner == slot_a_player_index
            winning_deck = player0_deck if winner == 0 else player1_deck
            losing_deck = player1_deck if winner == 0 else player0_deck
            deck_wins[str(winning_deck.index)] = deck_wins.get(str(winning_deck.index), 0) + 1
            deck_losses[str(losing_deck.index)] = deck_losses.get(str(losing_deck.index), 0) + 1
            if deck_a_won:
                deck_a_wins += 1
                matchup["deck_a_wins"] += 1
            else:
                deck_b_wins += 1
                matchup["deck_b_wins"] += 1

        for player_index, recorder in ((0, recorder0), (1, recorder1)):
            final_metadata = _result_metadata(result, player_index) | {
                "player_index": player_index,
                "selfplay_deck_pool": deck_pool,
                "selfplay_mode": plan.mode,
                "selfplay_deck_indices": plan.deck_indices,
                "selfplay_pair_key": pair_key,
                "selfplay_deck_a_index": deck_a.index,
                "selfplay_deck_b_index": deck_b.index,
                "policy_pool_model": (
                    player0_model_path.as_posix()
                    if player_index == 0 and player0_model_path
                    else player1_model_path.as_posix()
                    if player_index == 1 and player1_model_path
                    else None
                ),
            }
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

    return SelfPlaySummary(
        agent=agent_kind,
        games_requested=games,
        games_started=games_started,
        steps=steps_written,
        output_path=str(output_path.as_posix()),
        deck_pool=deck_pool,
        mode=plan.mode,
        deck_indices=plan.deck_indices,
        deck_labels=plan.deck_labels,
        pair_count=len(plan.pairs),
        deck_a_index=plan.fixed_deck_a.index if plan.fixed_deck_a else None,
        deck_a_label=plan.fixed_deck_a.label if plan.fixed_deck_a else None,
        deck_b_index=plan.fixed_deck_b.index if plan.fixed_deck_b else None,
        deck_b_label=plan.fixed_deck_b.label if plan.fixed_deck_b else None,
        deck_a_wins=deck_a_wins,
        deck_b_wins=deck_b_wins,
        deck_games=dict(sorted(deck_games.items(), key=lambda item: int(item[0]))),
        deck_wins=dict(sorted(deck_wins.items(), key=lambda item: int(item[0]))),
        deck_losses=dict(sorted(deck_losses.items(), key=lambda item: int(item[0]))),
        deck_draws=dict(sorted(deck_draws.items(), key=lambda item: int(item[0]))),
        matchups=dict(sorted(matchups.items())),
        draws=draws,
        errors=errors,
        timeouts=timeouts,
        image_size=image_size,
        search_telemetry=_finalize_search_telemetry(search_telemetry),
        trace_path=str(search_trace_path.as_posix()) if search_trace_path is not None else None,
        trace_records=trace_records,
        policy_pool_paths=[path.as_posix() for path in policy_pool_paths],
    )


def run_phase5_league_benchmark(
    *,
    sample_dir: Path,
    agent_kind: str = "phase5-search",
    model_path: Path | None = None,
    games_per_matchup: int = 2,
    max_steps: int = 600,
    debug_limit_per_matchup: int = 0,
    trace_limit: int = 60,
    search_trace_path: Path | None = None,
    search_config: RootSearchConfig | None = None,
) -> Phase3RequiredBenchmarkResult:
    card_data, attack_data = load_engine_metadata(sample_dir)
    league_decks = phase5_league_prepared_decks()
    rows: list[Phase3RequiredBenchmarkRow] = []
    debug_games: list[Phase3RequiredDebugGame] = []
    benchmark_game_index = 0
    if search_trace_path is not None:
        search_trace_path.parent.mkdir(parents=True, exist_ok=True)
        search_trace_path.write_text("", encoding="utf-8")

    for our_deck in league_decks:
        for opponent_deck in league_decks:
            row = Phase3RequiredBenchmarkRow(
                deck_index=our_deck.index,
                deck_label=our_deck.label,
                archetype=our_deck.archetype,
                tournament_rank=our_deck.deck.result.placement_rank,
                opponent=opponent_deck.archetype,
                opponent_deck_label=opponent_deck.label,
                games=games_per_matchup,
            )
            kept_debug_games = 0
            for game_index in range(games_per_matchup):
                benchmark_game_index += 1
                our_is_player0 = game_index % 2 == 0
                keep_debug = (
                    debug_limit_per_matchup > 0
                    and kept_debug_games < debug_limit_per_matchup
                )
                base_our_agent = _make_agent(
                    agent_kind,
                    our_deck.card_ids,
                    card_data,
                    attack_data,
                    model_path=model_path,
                    opponent_deck_ids=opponent_deck.card_ids,
                    sample_dir=sample_dir,
                    search_config=search_config,
                )
                our_agent = (
                    RecordingPolicyAgent(
                        base_our_agent,
                        our_deck.card_ids,
                        card_data=card_data,
                        attack_data=attack_data,
                        reward_metadata={
                            "game_index": game_index + 1,
                            "deck_index": our_deck.index,
                            "deck_label": our_deck.label,
                            "opponent": opponent_deck.archetype,
                            "collector": "phase5_league_benchmark",
                            "agent": agent_kind,
                        },
                        trace_limit=trace_limit,
                    )
                    if keep_debug
                    else base_our_agent
                )
                opponent_agent = _quiet_rule_agent(
                    opponent_deck.card_ids,
                    card_data,
                    attack_data,
                )
                result = run_battle(
                    our_deck.card_ids if our_is_player0 else opponent_deck.card_ids,
                    opponent_deck.card_ids if our_is_player0 else our_deck.card_ids,
                    sample_dir=sample_dir,
                    agent0=our_agent if our_is_player0 else opponent_agent,
                    agent1=opponent_agent if our_is_player0 else our_agent,
                    card_data=card_data,
                    attack_data=attack_data,
                    max_steps=max_steps,
                )
                _record_row_outcome(row, result, our_is_player0=our_is_player0)
                if isinstance(base_our_agent, Phase5SearchPolicyAgent):
                    _accumulate_search_telemetry(
                        row.search_telemetry,
                        base_our_agent.search_telemetry(),
                    )
                    if search_trace_path is not None:
                        _write_phase5_search_eval_traces(
                            base_our_agent,
                            search_trace_path,
                            game_index=benchmark_game_index,
                            deck_index=our_deck.index,
                            deck_label=our_deck.label,
                            opponent=opponent_deck.archetype,
                        )
                if keep_debug and isinstance(our_agent, RecordingPolicyAgent):
                    debug_games.append(
                        Phase3RequiredDebugGame(
                            deck_index=our_deck.index,
                            deck_label=our_deck.label,
                            archetype=our_deck.archetype,
                            tournament_rank=our_deck.deck.result.placement_rank,
                            opponent=opponent_deck.archetype,
                            game_index=game_index + 1,
                            outcome=_outcome_label(
                                result,
                                player_index=0 if our_is_player0 else 1,
                            ),
                            our_player_index=0 if our_is_player0 else 1,
                            steps=result.steps,
                            prize_counts=result.prize_counts,
                            error=result.error,
                            trace=_compact_replay_trace(our_agent.frames),
                        )
                    )
                    kept_debug_games += 1
            row.search_telemetry = _finalize_search_telemetry(row.search_telemetry)
            row.win_rate = row.wins / row.games if row.games else 0.0
            rows.append(row)

    search_telemetry: dict[str, Any] = {}
    for row in rows:
        _accumulate_search_telemetry(search_telemetry, row.search_telemetry)
    return Phase3RequiredBenchmarkResult(
        our_deck_source_url=TOURNAMENT_559_SOURCE_URL,
        requested_ranks=TOURNAMENT_559_REQUESTED_RANKS,
        substitutions=TOURNAMENT_559_SUBSTITUTIONS,
        games_per_matchup=games_per_matchup,
        max_steps=max_steps,
        rows=rows,
        debug_games=debug_games,
        search_telemetry=_finalize_search_telemetry(search_telemetry),
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
    debug_limit_per_matchup: int = 0,
    trace_limit: int = 60,
    search_trace_path: Path | None = None,
    search_config: RootSearchConfig | None = None,
) -> Phase3RequiredBenchmarkResult:
    card_data, attack_data = load_engine_metadata(sample_dir)
    our_decks = phase3_tournament_559_prepared_decks()
    benchmark_decks = required_phase3_prepared_decks(start_index=1)
    rows: list[Phase3RequiredBenchmarkRow] = []
    debug_games: list[Phase3RequiredDebugGame] = []
    benchmark_game_index = 0
    if search_trace_path is not None:
        search_trace_path.parent.mkdir(parents=True, exist_ok=True)
        search_trace_path.write_text("", encoding="utf-8")

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
                benchmark_game_index += 1
                our_is_player0 = game_index % 2 == 0
                keep_debug = debug_limit_per_matchup > 0 and kept_debug_games < debug_limit_per_matchup
                base_our_agent = _make_agent(
                    agent_kind,
                    our_deck.card_ids,
                    card_data,
                    attack_data,
                    model_path=model_path,
                    guidance_rules=guidance_rules,
                    opponent_deck_ids=benchmark_deck.card_ids,
                    sample_dir=sample_dir,
                    search_config=search_config,
                )
                our_agent = (
                    RecordingPolicyAgent(
                        base_our_agent,
                        our_deck.card_ids,
                        card_data=card_data,
                        attack_data=attack_data,
                        reward_metadata={
                            "game_index": game_index + 1,
                            "deck_index": our_deck.index,
                            "deck_label": our_deck.label,
                            "opponent": benchmark_deck.archetype,
                            "collector": "benchmark_replay",
                            "agent": agent_kind,
                        },
                        trace_limit=trace_limit,
                    )
                    if keep_debug
                    else base_our_agent
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
                if isinstance(base_our_agent, Phase5SearchPolicyAgent):
                    _accumulate_search_telemetry(
                        row.search_telemetry,
                        base_our_agent.search_telemetry(),
                    )
                    if search_trace_path is not None:
                        _write_phase5_search_eval_traces(
                            base_our_agent,
                            search_trace_path,
                            game_index=benchmark_game_index,
                            deck_index=our_deck.index,
                            deck_label=our_deck.label,
                            opponent=benchmark_deck.archetype,
                        )
                if keep_debug and isinstance(our_agent, RecordingPolicyAgent):
                    debug_games.append(
                        Phase3RequiredDebugGame(
                            deck_index=our_deck.index,
                            deck_label=our_deck.label,
                            archetype=our_deck.archetype,
                            tournament_rank=our_deck.deck.result.placement_rank,
                            opponent=benchmark_deck.archetype,
                            game_index=game_index + 1,
                            outcome=_outcome_label(
                                result,
                                player_index=0 if our_is_player0 else 1,
                            ),
                            our_player_index=0 if our_is_player0 else 1,
                            steps=result.steps,
                            prize_counts=result.prize_counts,
                            error=result.error,
                            trace=_compact_replay_trace(our_agent.frames),
                        )
                    )
                    kept_debug_games += 1
            row.search_telemetry = _finalize_search_telemetry(row.search_telemetry)
            row.win_rate = row.wins / row.games if row.games else 0.0
            rows.append(row)

    search_telemetry: dict[str, Any] = {}
    for row in rows:
        _accumulate_search_telemetry(search_telemetry, row.search_telemetry)
    search_telemetry = _finalize_search_telemetry(search_telemetry)

    return Phase3RequiredBenchmarkResult(
        our_deck_source_url=TOURNAMENT_559_SOURCE_URL,
        requested_ranks=TOURNAMENT_559_REQUESTED_RANKS,
        substitutions=TOURNAMENT_559_SUBSTITUTIONS,
        games_per_matchup=games_per_matchup,
        max_steps=max_steps,
        rows=rows,
        debug_games=debug_games,
        search_telemetry=search_telemetry,
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
    ]
    if result.search_telemetry:
        telemetry = result.search_telemetry
        lines.extend(
            [
                "## Search Telemetry",
                "",
                f"- Searched decisions: {int(telemetry.get('searched_decisions', 0))}",
                f"- Search-started decisions: {int(telemetry.get('search_started_decisions', 0))}",
                f"- Search-changed decisions: {int(telemetry.get('changed_decisions', 0))}",
                f"- Search change rate: {float(telemetry.get('change_rate', 0.0)):.3f}",
                f"- Search errors: {int(telemetry.get('search_errors', 0))}",
                f"- Search error rate: {float(telemetry.get('search_error_rate', 0.0)):.3f}",
                f"- Candidate probes: {int(telemetry.get('candidate_probes', 0))}",
                f"- Candidate errors: {int(telemetry.get('candidate_errors', 0))}",
                f"- Truncated candidates: {int(telemetry.get('truncated_candidates', 0))}",
                f"- Average search seconds: {float(telemetry.get('avg_search_seconds', 0.0)):.4f}",
                f"- Max search seconds: {float(telemetry.get('max_search_seconds', 0.0)):.4f}",
                "",
                "## Search Telemetry By Matchup",
                "",
                "| Deck | Opponent | Searched | Changed | Search errors | Candidate probes | Candidate errors | Truncated | Avg seconds |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in result.rows:
            row_telemetry = row.search_telemetry
            if not row_telemetry:
                continue
            lines.append(
                f"| {row.deck_index} | {row.opponent} | "
                f"{int(row_telemetry.get('searched_decisions', 0))} | "
                f"{int(row_telemetry.get('changed_decisions', 0))} | "
                f"{int(row_telemetry.get('search_errors', 0))} | "
                f"{int(row_telemetry.get('candidate_probes', 0))} | "
                f"{int(row_telemetry.get('candidate_errors', 0))} | "
                f"{int(row_telemetry.get('truncated_candidates', 0))} | "
                f"{float(row_telemetry.get('avg_search_seconds', 0.0)):.4f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Matchups",
            "",
            "| Deck | Rank | Archetype | Opponent | Wins | Losses | Draws | Timeouts | Errors | Win rate |",
            "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result.rows:
        lines.append(
            f"| {row.deck_index} | {row.tournament_rank} | {row.archetype} | {row.opponent} | "
            f"{row.wins} | {row.losses} | {row.draws} | {row.timeouts} | "
            f"{row.errors} | {row.win_rate:.3f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_image_progression_experiment(
    *,
    sample_dir: Path,
    image_size: int,
    iterations: int = 10,
    selfplay_games: int = 1000,
    eval_games_per_matchup: int = 100,
    max_steps: int = 600,
    deck_a_index: int | None = None,
    deck_b_index: int | None = None,
    selfplay_deck_indices: Sequence[int] | None = None,
    saved_replays_per_matchup: int = 1,
    replay_trace_limit: int = 60,
    update_epochs: int = 1,
    base_model_path: Path | None = None,
    dataset_root: Path = Path("data") / "datasets" / "rl" / "image_progression",
    model_root: Path = Path("models") / "rl" / "image_progression",
    report_root: Path = Path("experiments") / "rl" / "image_progression",
    output_root: Path = Path("experiments") / "rl" / "image_progression",
) -> ProgressionExperimentSummary:
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    size_key = f"image-{image_size}"
    dataset_dir = dataset_root / size_key
    model_dir = model_root / size_key
    report_dir = report_root / size_key
    output_dir = output_root / size_key
    dataset_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_model = base_model_path if base_model_path and base_model_path.exists() else None
    summaries: list[ProgressionIterationSummary] = []
    for iteration in range(1, iterations + 1):
        iteration_key = f"iter-{iteration:02d}"
        rollout_path = dataset_dir / f"{iteration_key}-selfplay.jsonl"
        model_path = model_dir / f"{iteration_key}-model.json"
        update_report_path = report_dir / f"{iteration_key}-update.json"
        eval_json_path = report_dir / f"{iteration_key}-benchmark.json"
        eval_md_path = report_dir / f"{iteration_key}-benchmark.md"
        replay_dir = output_dir / iteration_key / "replays"

        selfplay = rollout_selfplay_games(
            sample_dir=sample_dir,
            output_path=rollout_path,
            agent_kind="hybrid",
            model_path=current_model,
            games=selfplay_games,
            max_steps=max_steps,
            deck_a_index=deck_a_index,
            deck_b_index=deck_b_index,
            selfplay_deck_indices=selfplay_deck_indices,
            image_size=image_size,
        )
        update = train_ppo_from_rollouts(
            rollout_path=rollout_path,
            model_path=current_model or model_path,
            output_path=model_path,
            report_path=update_report_path,
            epochs=update_epochs,
        )
        _attach_model_metadata(
            model_path,
            {
                "experiment": "image_progression",
                "image_size": image_size,
                "iteration": iteration,
                "selfplay_games": selfplay_games,
                "eval_games_per_matchup": eval_games_per_matchup,
                "selfplay_mode": selfplay.mode,
                "selfplay_deck_indices": selfplay.deck_indices,
                "deck_a_index": selfplay.deck_a_index,
                "deck_b_index": selfplay.deck_b_index,
                "board_tensor_note": "Rollout records keep the compact symbolic board tensor; image_size labels the visual replay/model-size sweep.",
            },
        )
        current_model = model_path

        evaluation = run_phase4_required_benchmark(
            sample_dir=sample_dir,
            agent_kind="hybrid",
            model_path=current_model,
            games_per_matchup=eval_games_per_matchup,
            max_steps=max_steps,
            debug_limit_per_matchup=saved_replays_per_matchup,
            trace_limit=replay_trace_limit,
        )
        write_phase4_benchmark_report(
            evaluation,
            json_path=eval_json_path,
            markdown_path=eval_md_path,
            agent_kind=f"hybrid:image-{image_size}:iter-{iteration:02d}",
            model_path=current_model,
        )
        replay_paths = _write_debug_replays(evaluation.debug_games, replay_dir)
        eval_totals = _totals(evaluation.rows) | {
            "report_json": str(eval_json_path.as_posix()),
            "report_md": str(eval_md_path.as_posix()),
            "saved_replays": len(replay_paths),
            "replay_dir": str(replay_dir.as_posix()),
        }
        summary = ProgressionIterationSummary(
            image_size=image_size,
            iteration=iteration,
            selfplay=selfplay.to_dict(),
            update=update.to_dict(),
            evaluation=eval_totals,
            model_path=str(model_path.as_posix()),
            rollout_path=str(rollout_path.as_posix()),
            replay_dir=str(replay_dir.as_posix()),
        )
        summaries.append(summary)
        _write_json_report(
            {
                "image_size": image_size,
                "iterations_completed": iteration,
                "summaries": [item.to_dict() for item in summaries],
            },
            output_dir / "progression_summary.json",
        )

    experiment = ProgressionExperimentSummary(
        image_size=image_size,
        iterations=iterations,
        summaries=summaries,
        output_root=str(output_root.as_posix()),
        dataset_root=str(dataset_root.as_posix()),
        model_root=str(model_root.as_posix()),
        report_root=str(report_root.as_posix()),
    )
    _write_json_report(experiment.to_dict(), output_dir / "progression_summary.json")
    return experiment


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
        trace_limit: int = 0,
    ) -> None:
        self.agent = agent
        self.deck_ids = list(deck_ids)
        self.card_by_id = card_lookup(card_data)
        self.attack_by_id = attack_lookup(attack_data)
        self.reward_metadata = dict(reward_metadata)
        self.trace_limit = trace_limit
        self.frames: list[tuple[DecisionFrame, list[int]]] = []

    def act(self, observation: Any) -> list[int]:
        selected = self.agent.act(observation)
        should_record = self.trace_limit <= 0 or len(self.frames) < self.trace_limit
        if not should_record:
            return selected
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
    opponent_deck_ids: Sequence[int] | None = None,
    sample_dir: Path | None = None,
    search_config: RootSearchConfig | None = None,
) -> Any:
    if agent_kind == "rule":
        return _quiet_rule_agent(deck_ids, card_data, attack_data)
    if agent_kind == "phase5-symbolic":
        return Phase5SymbolicPolicyAgent(
            deck_ids,
            card_data=card_data,
            attack_data=attack_data,
            checkpoint_path=model_path,
        )
    if agent_kind == "phase5-search":
        if opponent_deck_ids is None or sample_dir is None:
            raise ValueError("phase5-search requires opponent_deck_ids and sample_dir.")
        return Phase5SearchPolicyAgent(
            deck_ids,
            opponent_deck_ids=opponent_deck_ids,
            sample_dir=sample_dir,
            card_data=card_data,
            attack_data=attack_data,
            checkpoint_path=model_path,
            search_config=search_config,
        )
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


def _outcome_label(result: BattleResult, *, player_index: int) -> str:
    if result.error:
        return "error"
    timeout = result.winner is None and not result.finished
    effective = result.winner if result.winner is not None else result.leader
    if effective is None:
        return "timeout_draw" if timeout else "draw"
    if effective == player_index:
        return "timeout_win" if timeout else "win"
    return "timeout_loss" if timeout else "loss"


def _compact_replay_trace(
    frames: Sequence[tuple[DecisionFrame, list[int]]],
) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for frame, chosen_indices in frames:
        chosen = set(chosen_indices)
        ranked = sorted(
            frame.legal_options,
            key=lambda action: (action.rule_score, -action.index),
            reverse=True,
        )
        trace.append(
            {
                "turn": frame.board.get("turn"),
                "select_type": frame.select_type,
                "context": frame.context,
                "target_count": frame.target_count,
                "chosen_indices": list(chosen_indices),
                "chosen_options": [
                    _compact_action(action)
                    for action in frame.legal_options
                    if action.index in chosen
                ],
                "top_rule_options": [_compact_action(action) for action in ranked[:5]],
                "board": _compact_board(frame.board),
            }
        )
    return trace


def _compact_action(action: Any) -> dict[str, Any]:
    return {
        "index": action.index,
        "type": action.option_type,
        "card_id": action.card_id,
        "card_name": action.card_name,
        "area": action.area,
        "area_index": action.area_index,
        "target_card_id": action.target_card_id,
        "target_name": action.target_name,
        "target_area": action.target_area,
        "target_index": action.target_index,
        "attack_id": action.attack_id,
        "rule_score": action.rule_score,
    }


def _compact_board(board: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn": board.get("turn"),
        "your_index": board.get("your_index"),
        "my_active": _compact_card_state(board.get("my_active_card")),
        "opponent_active": _compact_card_state(board.get("opponent_active_card")),
        "my_bench": [
            _compact_card_state(card) for card in list(board.get("my_bench_cards", []) or [])
        ],
        "opponent_bench": [
            _compact_card_state(card)
            for card in list(board.get("opponent_bench_cards", []) or [])
        ],
        "my_hand_count": board.get("my_hand_count"),
        "opponent_hand_count": board.get("opponent_hand_count"),
        "my_deck_count": board.get("my_deck_count"),
        "opponent_deck_count": board.get("opponent_deck_count"),
        "my_discard_count": board.get("my_discard_count"),
        "opponent_discard_count": board.get("opponent_discard_count"),
        "my_prizes": board.get("my_prizes"),
        "opponent_prizes": board.get("opponent_prizes"),
        "stadium": _compact_card_state(board.get("stadium_card")),
    }


def _compact_card_state(card: Any) -> dict[str, Any] | None:
    if not isinstance(card, dict) or not card:
        return None
    return {
        "card_id": card.get("card_id"),
        "name": card.get("name"),
        "hp_ratio": card.get("hp_ratio"),
        "damage_ratio": card.get("damage_ratio"),
        "energy_count": card.get("energy_count"),
        "tool_count": card.get("tool_count"),
        "stage": card.get("stage"),
        "is_ex": card.get("is_ex"),
        "is_mega_ex": card.get("is_mega_ex"),
        "asleep": card.get("asleep"),
        "burned": card.get("burned"),
        "confused": card.get("confused"),
        "paralyzed": card.get("paralyzed"),
        "poisoned": card.get("poisoned"),
    }


def _write_debug_replays(
    debug_games: Sequence[Phase3RequiredDebugGame],
    replay_dir: Path,
) -> list[Path]:
    replay_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, game in enumerate(debug_games, start=1):
        path = replay_dir / (
            f"{index:02d}_deck-{game.deck_index}_vs_"
            f"{_safe_slug(game.opponent)}_game-{game.game_index}.json"
        )
        path.write_text(json.dumps(game.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        paths.append(path)
    return paths


def _attach_model_metadata(model_path: Path, metadata: dict[str, Any]) -> None:
    if not model_path.exists():
        return
    model = LinearOptionModel.load(model_path)
    model.metadata.update(metadata)
    model.save(model_path)


def _safe_slug(value: str) -> str:
    chars = []
    for char in value.casefold():
        if char.isalnum():
            chars.append(char)
        elif chars and chars[-1] != "-":
            chars.append("-")
    return "".join(chars).strip("-") or "unknown"


_SEARCH_TELEMETRY_SUM_FIELDS = (
    "searched_decisions",
    "search_started_decisions",
    "changed_decisions",
    "search_errors",
    "candidate_probes",
    "candidate_errors",
    "truncated_candidates",
    "total_search_seconds",
)


def _accumulate_search_telemetry(target: dict[str, Any], source: dict[str, Any]) -> None:
    if not source:
        return
    for field in _SEARCH_TELEMETRY_SUM_FIELDS:
        target[field] = target.get(field, 0) + source.get(field, 0)
    target["max_search_seconds"] = max(
        float(target.get("max_search_seconds", 0.0)),
        float(source.get("max_search_seconds", 0.0)),
    )


def _finalize_search_telemetry(telemetry: dict[str, Any]) -> dict[str, Any]:
    if not telemetry:
        return {}
    searched = int(telemetry.get("searched_decisions", 0) or 0)
    candidate_probes = int(telemetry.get("candidate_probes", 0) or 0)
    changed = int(telemetry.get("changed_decisions", 0) or 0)
    search_errors = int(telemetry.get("search_errors", 0) or 0)
    candidate_errors = int(telemetry.get("candidate_errors", 0) or 0)
    truncated = int(telemetry.get("truncated_candidates", 0) or 0)
    total_seconds = float(telemetry.get("total_search_seconds", 0.0) or 0.0)
    finalized = {
        "searched_decisions": searched,
        "search_started_decisions": int(telemetry.get("search_started_decisions", 0) or 0),
        "changed_decisions": changed,
        "change_rate": changed / searched if searched else 0.0,
        "search_errors": search_errors,
        "search_error_rate": search_errors / searched if searched else 0.0,
        "candidate_probes": candidate_probes,
        "candidate_errors": candidate_errors,
        "candidate_error_rate": candidate_errors / candidate_probes
        if candidate_probes
        else 0.0,
        "truncated_candidates": truncated,
        "truncated_candidate_rate": truncated / candidate_probes
        if candidate_probes
        else 0.0,
        "total_search_seconds": total_seconds,
        "avg_search_seconds": total_seconds / searched if searched else 0.0,
        "max_search_seconds": float(telemetry.get("max_search_seconds", 0.0) or 0.0),
    }
    return finalized


def _should_write_selfplay_trace(
    path: Path | None,
    *,
    game_index: int,
    trace_game_limit: int,
) -> bool:
    if path is None:
        return False
    return trace_game_limit <= 0 or game_index < trace_game_limit


def _write_phase5_search_eval_traces(
    agent: Phase5SearchPolicyAgent,
    path: Path,
    *,
    game_index: int,
    deck_index: int,
    deck_label: str,
    opponent: str,
) -> int:
    if not agent.traces:
        return 0
    records = 0
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for trace in agent.traces:
            enriched = replace(
                trace,
                game_index=game_index,
                deck_index=deck_index,
                deck_label=deck_label,
                opponent=opponent,
            )
            handle.write(json.dumps(enriched.to_dict(), sort_keys=True) + "\n")
            records += 1
    return records


def _policy_pool_model_path(
    policy_pool_paths: Sequence[Path],
    *,
    absolute_game_index: int,
    player_slot: int,
    default_model_path: Path | None,
) -> Path | None:
    if not policy_pool_paths:
        return default_model_path
    if absolute_game_index < 0:
        raise ValueError("absolute_game_index must be non-negative.")
    index = (absolute_game_index * 2 + player_slot) % len(policy_pool_paths)
    return policy_pool_paths[index]


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
