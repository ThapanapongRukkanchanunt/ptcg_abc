from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ptcg_abc.agent.phase5_search import Phase5SearchPolicyAgent
from ptcg_abc.evaluation import (
    Phase3RequiredBenchmarkResult,
    Phase3RequiredBenchmarkRow,
    PreparedDeck,
    TOURNAMENT_559_REQUESTED_RANKS,
    TOURNAMENT_559_SOURCE_URL,
    TOURNAMENT_559_SUBSTITUTIONS,
    phase5_league_prepared_decks,
)
from ptcg_abc.public_agents import (
    LoadedPublicAgent,
    PublicAgentStatus,
    discover_public_agents,
)
from ptcg_abc.rl.dataset import append_trajectory_jsonl
from ptcg_abc.rl.records import DecisionFrame, TrajectoryStep
from ptcg_abc.rl.rewards import reward_from_result_metadata
from ptcg_abc.rl.workflow import (
    RecordingPolicyAgent,
    _accumulate_search_telemetry,
    _finalize_search_telemetry,
    _make_agent,
    _phase5_specialist_model_path,
    _record_row_outcome,
    _result_metadata,
    _with_metadata,
)
from ptcg_abc.simulator import load_engine_metadata, run_battle


@dataclass(frozen=True)
class PublicAgentTrajectorySummary:
    agent: str
    games_requested: int
    games_started: int
    steps: int
    output_path: str
    deck_indices: list[int]
    public_agent_roots: list[str]
    roster_notebook: str | None
    public_agent_keys: list[str] | None
    available_opponents: list[dict[str, Any]]
    unavailable_opponents: list[dict[str, Any]]
    deck_games: dict[str, int]
    deck_wins: dict[str, int]
    deck_losses: dict[str, int]
    deck_draws: dict[str, int]
    opponent_games: dict[str, int]
    opponent_wins: dict[str, int]
    opponent_losses: dict[str, int]
    opponent_draws: dict[str, int]
    matchups: dict[str, dict[str, int]]
    wins: int
    losses: int
    draws: int
    errors: int
    timeouts: int
    search_telemetry: dict[str, Any]
    reward_shaping: dict[str, Any]
    tactical_reward_summary: dict[str, Any]
    specialist_model_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PublicAgentTacticalRewardConfig:
    mode: str = "none"
    attack_bonus: float = 0.10
    attach_bonus: float = 0.06
    missed_attack_penalty: float = -0.10
    missed_attach_penalty: float = -0.06

    def __post_init__(self) -> None:
        if self.mode not in {"none", "basic"}:
            raise ValueError(
                f"Unsupported public-agent tactical reward mode: {self.mode}."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_public_agent_gate(
    rows: Sequence[Phase3RequiredBenchmarkRow],
    *,
    min_win_rate: float = 0.5,
) -> dict[str, Any]:
    opponent_totals = _aggregate_public_gate_rows(
        rows,
        key_fn=lambda row: row.opponent,
        label_fn=lambda row: row.opponent_deck_label or row.opponent,
        min_win_rate=min_win_rate,
    )
    deck_totals = _aggregate_public_gate_rows(
        rows,
        key_fn=lambda row: f"{row.deck_index:02d}",
        label_fn=lambda row: row.deck_label,
        min_win_rate=min_win_rate,
    )
    failing_matchups = [
        {
            "deck_index": row.deck_index,
            "deck_label": row.deck_label,
            "opponent": row.opponent,
            "opponent_deck_label": row.opponent_deck_label,
            "games": row.games,
            "wins": row.wins,
            "losses": row.losses,
            "draws": row.draws,
            "timeouts": row.timeouts,
            "errors": row.errors,
            "win_rate": row.win_rate,
        }
        for row in rows
        if row.games <= 0 or row.win_rate < min_win_rate or row.errors > 0
    ]
    worst_opponent = min(
        opponent_totals,
        key=lambda item: (item["win_rate"], -item["errors"], item["key"]),
        default=None,
    )
    worst_deck = min(
        deck_totals,
        key=lambda item: (item["win_rate"], -item["errors"], item["key"]),
        default=None,
    )
    failing_opponents = [
        item for item in opponent_totals if not item["passed_min_win_rate"]
    ]
    failing_decks = [item for item in deck_totals if not item["passed_min_win_rate"]]
    return {
        "min_win_rate": min_win_rate,
        "passed": not failing_opponents and not failing_decks,
        "strict_matchup_passed": not failing_matchups,
        "opponents": opponent_totals,
        "controlled_decks": deck_totals,
        "worst_opponent": worst_opponent,
        "worst_controlled_deck": worst_deck,
        "failing_opponents": failing_opponents,
        "failing_controlled_decks": failing_decks,
        "failing_matchups": failing_matchups,
    }


def format_public_agent_gate_markdown(summary: dict[str, Any]) -> str:
    min_rate = float(summary.get("min_win_rate", 0.0))
    lines = [
        "## Public-Agent Gate",
        "",
        f"- Minimum aggregate win rate: {min_rate:.3f}",
        f"- Opponent/deck aggregate gate: {'pass' if summary.get('passed') else 'fail'}",
        f"- Strict per-matchup diagnostic: {'pass' if summary.get('strict_matchup_passed') else 'fail'}",
    ]
    worst_opponent = summary.get("worst_opponent")
    if worst_opponent:
        lines.append(
            "- Worst public opponent: "
            f"{worst_opponent['key']} at {worst_opponent['wins']} / "
            f"{worst_opponent['games']} wins ({worst_opponent['win_rate']:.3f})"
        )
    worst_deck = summary.get("worst_controlled_deck")
    if worst_deck:
        lines.append(
            "- Worst controlled deck: "
            f"{worst_deck['key']} {worst_deck['label']} at {worst_deck['wins']} / "
            f"{worst_deck['games']} wins ({worst_deck['win_rate']:.3f})"
        )
    lines.extend(
        [
            "",
            "### Public Opponents",
            "",
            "| Opponent | Wins | Losses | Draws | Timeouts | Errors | Win rate | Gate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in summary.get("opponents", []):
        lines.append(
            f"| {item['key']} | {item['wins']} | {item['losses']} | "
            f"{item['draws']} | {item['timeouts']} | {item['errors']} | "
            f"{item['win_rate']:.3f} | {'pass' if item['passed_min_win_rate'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "### Controlled Decks",
            "",
            "| Deck | Label | Wins | Losses | Draws | Timeouts | Errors | Win rate | Gate |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in summary.get("controlled_decks", []):
        lines.append(
            f"| {item['key']} | {item['label']} | {item['wins']} | "
            f"{item['losses']} | {item['draws']} | {item['timeouts']} | "
            f"{item['errors']} | {item['win_rate']:.3f} | "
            f"{'pass' if item['passed_min_win_rate'] else 'fail'} |"
        )
    if summary.get("failing_matchups"):
        lines.extend(
            [
                "",
                "### Failing Matchups",
                "",
                "| Deck | Opponent | Wins | Games | Win rate | Errors |",
                "| ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in summary["failing_matchups"]:
            lines.append(
                f"| {item['deck_index']} | {item['opponent']} | {item['wins']} | "
                f"{item['games']} | {item['win_rate']:.3f} | {item['errors']} |"
            )
    return "\n".join(lines)


def discover_phase5_public_opponents(
    *,
    sample_dir: Path,
    public_agent_roots: Sequence[Path],
    roster_notebook: Path | None = None,
    include_public: bool = True,
    include_samples: bool = True,
    include_builtin_samples: bool = True,
    public_agent_keys: Sequence[str] | None = None,
) -> tuple[list[LoadedPublicAgent], list[PublicAgentStatus]]:
    opponents, statuses = discover_public_agents(
        roots=public_agent_roots,
        sample_dir=sample_dir,
        roster_notebook=roster_notebook,
        include_public=include_public,
        include_samples=include_samples,
        include_builtin_samples=include_builtin_samples,
    )
    return _filter_public_opponents(opponents, statuses, public_agent_keys)


def run_phase5_public_agent_benchmark(
    *,
    sample_dir: Path,
    public_agent_roots: Sequence[Path],
    roster_notebook: Path | None = None,
    include_public: bool = True,
    include_samples: bool = True,
    include_builtin_samples: bool = True,
    public_agent_keys: Sequence[str] | None = None,
    require_min_opponents: int = 1,
    agent_kind: str = "phase5-full",
    model_path: Path | None = None,
    specialist_model_dir: Path | None = None,
    deck_indices: Sequence[int] | None = None,
    games_per_matchup: int = 2,
    max_steps: int = 600,
    search_config: Any | None = None,
) -> tuple[Phase3RequiredBenchmarkResult, list[PublicAgentStatus]]:
    card_data, attack_data = load_engine_metadata(sample_dir)
    our_decks = _selected_phase5_decks(deck_indices)
    opponents, statuses = discover_phase5_public_opponents(
        sample_dir=sample_dir,
        public_agent_roots=public_agent_roots,
        roster_notebook=roster_notebook,
        include_public=include_public,
        include_samples=include_samples,
        include_builtin_samples=include_builtin_samples,
        public_agent_keys=public_agent_keys,
    )
    if len(opponents) < require_min_opponents:
        raise ValueError(
            f"Only {len(opponents)} public/specialized opponent(s) are available; "
            f"required at least {require_min_opponents}."
        )
    rows: list[Phase3RequiredBenchmarkRow] = []
    aggregate_search: dict[str, Any] = {}
    for our_deck in our_decks:
        our_model_path = _phase5_specialist_model_path(specialist_model_dir, our_deck.index)
        if our_model_path is None:
            our_model_path = model_path
        for opponent in opponents:
            row = Phase3RequiredBenchmarkRow(
                deck_index=our_deck.index,
                deck_label=our_deck.label,
                archetype=our_deck.archetype,
                tournament_rank=our_deck.deck.result.placement_rank,
                opponent=opponent.label,
                opponent_deck_label=_public_opponent_label(opponent),
                games=games_per_matchup,
            )
            for game_index in range(games_per_matchup):
                our_is_player0 = game_index % 2 == 0
                our_agent = _make_agent(
                    agent_kind,
                    our_deck.card_ids,
                    card_data,
                    attack_data,
                    model_path=our_model_path,
                    opponent_deck_ids=opponent.deck_ids,
                    sample_dir=sample_dir,
                    search_config=search_config,
                )
                opponent_agent = opponent.make_agent()
                result = run_battle(
                    our_deck.card_ids if our_is_player0 else opponent.deck_ids,
                    opponent.deck_ids if our_is_player0 else our_deck.card_ids,
                    sample_dir=sample_dir,
                    agent0=our_agent if our_is_player0 else opponent_agent,
                    agent1=opponent_agent if our_is_player0 else our_agent,
                    card_data=card_data,
                    attack_data=attack_data,
                    max_steps=max_steps,
                )
                _record_row_outcome(row, result, our_is_player0=our_is_player0)
                if isinstance(our_agent, Phase5SearchPolicyAgent):
                    _accumulate_search_telemetry(row.search_telemetry, our_agent.search_telemetry())
            row.search_telemetry = _finalize_search_telemetry(row.search_telemetry)
            _accumulate_search_telemetry(aggregate_search, row.search_telemetry)
            row.win_rate = row.wins / row.games if row.games else 0.0
            rows.append(row)
    return (
        Phase3RequiredBenchmarkResult(
            our_deck_source_url=TOURNAMENT_559_SOURCE_URL,
            requested_ranks=TOURNAMENT_559_REQUESTED_RANKS,
            substitutions=TOURNAMENT_559_SUBSTITUTIONS,
            games_per_matchup=games_per_matchup,
            max_steps=max_steps,
            rows=rows,
            debug_games=[],
            search_telemetry=_finalize_search_telemetry(aggregate_search),
        ),
        statuses,
    )


def generate_phase5_public_agent_trajectories(
    *,
    sample_dir: Path,
    public_agent_roots: Sequence[Path],
    output_path: Path,
    roster_notebook: Path | None = None,
    include_public: bool = True,
    include_samples: bool = True,
    include_builtin_samples: bool = True,
    public_agent_keys: Sequence[str] | None = None,
    require_min_opponents: int = 1,
    agent_kind: str = "phase5-rl",
    model_path: Path | None = None,
    specialist_model_dir: Path | None = None,
    deck_indices: Sequence[int] | None = None,
    games_per_matchup: int = 2,
    max_steps: int = 600,
    game_offset: int = 0,
    search_config: Any | None = None,
    overwrite: bool = False,
    outcome_reward_scale: float = 1.0,
    tactical_reward_config: PublicAgentTacticalRewardConfig | None = None,
) -> PublicAgentTrajectorySummary:
    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        raise ValueError(f"Trajectory output already exists at {output_path}.")
    card_data, attack_data = load_engine_metadata(sample_dir)
    our_decks = _selected_phase5_decks(deck_indices)
    opponents, statuses = discover_phase5_public_opponents(
        sample_dir=sample_dir,
        public_agent_roots=public_agent_roots,
        roster_notebook=roster_notebook,
        include_public=include_public,
        include_samples=include_samples,
        include_builtin_samples=include_builtin_samples,
        public_agent_keys=public_agent_keys,
    )
    if len(opponents) < require_min_opponents:
        raise ValueError(
            f"Only {len(opponents)} public/specialized opponent(s) are available; "
            f"required at least {require_min_opponents}."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    games_requested = len(our_decks) * len(opponents) * games_per_matchup
    games_started = steps_written = wins = losses = draws = errors = timeouts = 0
    deck_games = {str(deck.index): 0 for deck in our_decks}
    deck_wins = {str(deck.index): 0 for deck in our_decks}
    deck_losses = {str(deck.index): 0 for deck in our_decks}
    deck_draws = {str(deck.index): 0 for deck in our_decks}
    opponent_games = {opponent.key: 0 for opponent in opponents}
    opponent_wins = {opponent.key: 0 for opponent in opponents}
    opponent_losses = {opponent.key: 0 for opponent in opponents}
    opponent_draws = {opponent.key: 0 for opponent in opponents}
    matchups: dict[str, dict[str, int]] = {}
    aggregate_search: dict[str, Any] = {}
    tactical_config = tactical_reward_config or PublicAgentTacticalRewardConfig()
    tactical_summary = _empty_tactical_reward_summary(tactical_config)

    local_game_index = 0
    for our_deck in our_decks:
        our_model_path = _phase5_specialist_model_path(specialist_model_dir, our_deck.index)
        if our_model_path is None:
            our_model_path = model_path
        for opponent in opponents:
            pair_key = f"{our_deck.index}-vs-{opponent.key}"
            matchup = matchups.setdefault(
                pair_key,
                {
                    "deck_index": our_deck.index,
                    "opponent_key": opponent.key,
                    "games": 0,
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "errors": 0,
                    "timeouts": 0,
                },
            )
            for _ in range(games_per_matchup):
                absolute_game_index = game_offset + local_game_index
                local_game_index += 1
                our_is_player0 = absolute_game_index % 2 == 0
                reward_metadata = {
                    "game_index": absolute_game_index + 1,
                    "local_game_index": local_game_index,
                    "deck_index": our_deck.index,
                    "deck_label": our_deck.label,
                    "opponent_agent_key": opponent.key,
                    "opponent_agent_label": opponent.label,
                    "opponent_source_ref": opponent.source.source_ref,
                    "opponent_deck_label": _public_opponent_label(opponent),
                    "collector": "phase5_public_rule_opponents",
                    "agent": agent_kind,
                    "specialist_model_dir": (
                        specialist_model_dir.as_posix() if specialist_model_dir else None
                    ),
                    "specialist_model_path": (
                        our_model_path.as_posix() if specialist_model_dir and our_model_path else None
                    ),
                }
                recorder = RecordingPolicyAgent(
                    _make_agent(
                        agent_kind,
                        our_deck.card_ids,
                        card_data,
                        attack_data,
                        model_path=our_model_path,
                        opponent_deck_ids=opponent.deck_ids,
                        sample_dir=sample_dir,
                        search_config=search_config,
                    ),
                    our_deck.card_ids,
                    card_data=card_data,
                    attack_data=attack_data,
                    reward_metadata=reward_metadata,
                )
                opponent_agent = opponent.make_agent()
                result = run_battle(
                    our_deck.card_ids if our_is_player0 else opponent.deck_ids,
                    opponent.deck_ids if our_is_player0 else our_deck.card_ids,
                    sample_dir=sample_dir,
                    agent0=recorder if our_is_player0 else opponent_agent,
                    agent1=opponent_agent if our_is_player0 else recorder,
                    card_data=card_data,
                    attack_data=attack_data,
                    max_steps=max_steps,
                )
                games_started += int(result.started)
                errors += int(result.error is not None)
                timeout = int(result.started and not result.finished and result.error is None)
                timeouts += timeout
                deck_games[str(our_deck.index)] += 1
                opponent_games[opponent.key] += 1
                matchup["games"] += 1
                matchup["errors"] += int(result.error is not None)
                matchup["timeouts"] += timeout

                if isinstance(recorder.agent, Phase5SearchPolicyAgent):
                    _accumulate_search_telemetry(aggregate_search, recorder.agent.search_telemetry())

                final_metadata = _result_metadata(
                    result,
                    0 if our_is_player0 else 1,
                ) | {
                    "player_index": 0 if our_is_player0 else 1,
                    "opponent_agent_key": opponent.key,
                    "opponent_agent_label": opponent.label,
                    "opponent_source_ref": opponent.source.source_ref,
                    "opponent_deck_label": _public_opponent_label(opponent),
                    "collector": "phase5_public_rule_opponents",
                }
                reward = reward_from_result_metadata(final_metadata)
                for record in recorder.frames:
                    tactical_reward, tactical_metadata = _tactical_reward_for_frame(
                        record.frame,
                        record.chosen_indices,
                        tactical_config,
                    )
                    _accumulate_tactical_reward(tactical_summary, tactical_metadata)
                    scaled_outcome_reward = float(outcome_reward_scale) * reward
                    step_reward = scaled_outcome_reward + tactical_reward
                    frame = _with_metadata(
                        record.frame,
                        final_metadata
                        | {
                            "outcome_reward": reward,
                            "outcome_reward_scale": float(outcome_reward_scale),
                            "scaled_outcome_reward": scaled_outcome_reward,
                        }
                        | tactical_metadata,
                    )
                    append_trajectory_jsonl(
                        TrajectoryStep(
                            decision=frame,
                            chosen_indices=record.chosen_indices,
                            logprob=record.logprob,
                            value=record.value,
                            reward=step_reward,
                            terminal=result.finished,
                            truncated=not result.finished and result.error is None,
                        ),
                        output_path,
                    )
                    steps_written += 1

                if result.error:
                    draws += 1
                    deck_draws[str(our_deck.index)] += 1
                    opponent_draws[opponent.key] += 1
                    matchup["draws"] += 1
                    continue
                effective = result.winner if result.winner is not None else result.leader
                our_player_index = 0 if our_is_player0 else 1
                if effective is None:
                    draws += 1
                    deck_draws[str(our_deck.index)] += 1
                    opponent_draws[opponent.key] += 1
                    matchup["draws"] += 1
                elif effective == our_player_index:
                    wins += 1
                    deck_wins[str(our_deck.index)] += 1
                    opponent_losses[opponent.key] += 1
                    matchup["wins"] += 1
                else:
                    losses += 1
                    deck_losses[str(our_deck.index)] += 1
                    opponent_wins[opponent.key] += 1
                    matchup["losses"] += 1

    return PublicAgentTrajectorySummary(
        agent=agent_kind,
        games_requested=games_requested,
        games_started=games_started,
        steps=steps_written,
        output_path=output_path.as_posix(),
        deck_indices=[deck.index for deck in our_decks],
        public_agent_roots=[root.as_posix() for root in public_agent_roots],
        roster_notebook=roster_notebook.as_posix() if roster_notebook else None,
        public_agent_keys=_normalized_public_agent_keys(public_agent_keys),
        available_opponents=[opponent.to_status().to_dict() for opponent in opponents],
        unavailable_opponents=[
            status.to_dict() for status in statuses if status.status != "available"
        ],
        deck_games=dict(sorted(deck_games.items(), key=lambda item: int(item[0]))),
        deck_wins=dict(sorted(deck_wins.items(), key=lambda item: int(item[0]))),
        deck_losses=dict(sorted(deck_losses.items(), key=lambda item: int(item[0]))),
        deck_draws=dict(sorted(deck_draws.items(), key=lambda item: int(item[0]))),
        opponent_games=dict(sorted(opponent_games.items())),
        opponent_wins=dict(sorted(opponent_wins.items())),
        opponent_losses=dict(sorted(opponent_losses.items())),
        opponent_draws=dict(sorted(opponent_draws.items())),
        matchups=dict(sorted(matchups.items())),
        wins=wins,
        losses=losses,
        draws=draws,
        errors=errors,
        timeouts=timeouts,
        search_telemetry=_finalize_search_telemetry(aggregate_search),
        reward_shaping={
            "outcome_reward_scale": float(outcome_reward_scale),
            "tactical_reward": tactical_config.to_dict(),
        },
        tactical_reward_summary=_finalize_tactical_reward_summary(tactical_summary),
        specialist_model_dir=specialist_model_dir.as_posix() if specialist_model_dir else None,
    )


def write_public_agent_status_report(
    path: Path,
    statuses: Sequence[PublicAgentStatus],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([status.to_dict() for status in statuses], indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_public_agent_trajectory_report(
    path: Path,
    summary: PublicAgentTrajectorySummary,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def _selected_phase5_decks(deck_indices: Sequence[int] | None) -> list[PreparedDeck]:
    decks = phase5_league_prepared_decks()
    if not deck_indices:
        return decks
    by_index = {deck.index: deck for deck in decks}
    selected: list[PreparedDeck] = []
    for index in deck_indices:
        if index not in by_index:
            raise ValueError(f"Unknown Phase 5 deck index: {index}.")
        if index not in [deck.index for deck in selected]:
            selected.append(by_index[index])
    return selected


def _tactical_reward_for_frame(
    frame: DecisionFrame,
    chosen_indices: Sequence[int],
    config: PublicAgentTacticalRewardConfig,
) -> tuple[float, dict[str, Any]]:
    selected = {int(index) for index in chosen_indices}
    selected_actions = [
        action for action in frame.legal_options if int(action.index) in selected
    ]
    attack_available = any(action.option_type == "ATTACK" for action in frame.legal_options)
    attach_available = any(action.option_type == "ATTACH" for action in frame.legal_options)
    attack_taken = any(action.option_type == "ATTACK" for action in selected_actions)
    attach_taken = any(action.option_type == "ATTACH" for action in selected_actions)
    end_selected = any(action.option_type == "END" for action in selected_actions)
    empty_selection = not selected_actions

    reward = 0.0
    missed_attack = False
    missed_attach = False
    if config.mode == "basic":
        if attack_taken:
            reward += config.attack_bonus
        elif attack_available and (end_selected or empty_selection):
            reward += config.missed_attack_penalty
            missed_attack = True

        if attach_taken:
            reward += config.attach_bonus
        elif attach_available and (attack_taken or end_selected or empty_selection):
            reward += config.missed_attach_penalty
            missed_attach = True

    metadata = {
        "tactical_reward_mode": config.mode,
        "tactical_step_reward": reward,
        "tactical_attack_available": attack_available,
        "tactical_attack_taken": attack_taken,
        "tactical_attach_available": attach_available,
        "tactical_attach_taken": attach_taken,
        "tactical_end_selected": end_selected,
        "tactical_empty_selection": empty_selection,
        "tactical_missed_attack": missed_attack,
        "tactical_missed_attach": missed_attach,
    }
    return reward, metadata


def _empty_tactical_reward_summary(
    config: PublicAgentTacticalRewardConfig,
) -> dict[str, Any]:
    return {
        "mode": config.mode,
        "steps": 0,
        "reward_sum": 0.0,
        "attack_available": 0,
        "attack_taken": 0,
        "attach_available": 0,
        "attach_taken": 0,
        "end_selected": 0,
        "empty_selection": 0,
        "missed_attack": 0,
        "missed_attach": 0,
    }


def _accumulate_tactical_reward(
    summary: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    summary["steps"] += 1
    summary["reward_sum"] += float(metadata.get("tactical_step_reward", 0.0))
    for field, key in (
        ("attack_available", "tactical_attack_available"),
        ("attack_taken", "tactical_attack_taken"),
        ("attach_available", "tactical_attach_available"),
        ("attach_taken", "tactical_attach_taken"),
        ("end_selected", "tactical_end_selected"),
        ("empty_selection", "tactical_empty_selection"),
        ("missed_attack", "tactical_missed_attack"),
        ("missed_attach", "tactical_missed_attach"),
    ):
        summary[field] += int(bool(metadata.get(key, False)))


def _finalize_tactical_reward_summary(summary: dict[str, Any]) -> dict[str, Any]:
    output = dict(summary)
    steps = int(output.get("steps", 0) or 0)
    output["avg_reward"] = (
        float(output.get("reward_sum", 0.0)) / steps if steps else 0.0
    )
    for key in (
        "attack_taken",
        "missed_attack",
        "attach_taken",
        "missed_attach",
        "end_selected",
    ):
        denominator_key = (
            "attack_available" if "attack" in key else "attach_available"
        )
        if key == "end_selected":
            denominator_key = "steps"
        denominator = int(output.get(denominator_key, 0) or 0)
        output[f"{key}_rate"] = (
            float(output.get(key, 0)) / denominator if denominator else 0.0
        )
    return output


def _filter_public_opponents(
    opponents: Sequence[LoadedPublicAgent],
    statuses: Sequence[PublicAgentStatus],
    public_agent_keys: Sequence[str] | None,
) -> tuple[list[LoadedPublicAgent], list[PublicAgentStatus]]:
    selected = _normalized_public_agent_keys(public_agent_keys)
    if not selected:
        return list(opponents), list(statuses)
    selected_set = set(selected)
    known_keys = {opponent.key for opponent in opponents} | {
        status.source.key for status in statuses
    }
    unknown = [key for key in selected if key not in known_keys]
    if unknown:
        raise ValueError(f"Unknown public-agent key(s): {', '.join(unknown)}.")
    return (
        [opponent for opponent in opponents if opponent.key in selected_set],
        [status for status in statuses if status.source.key in selected_set],
    )


def _normalized_public_agent_keys(
    public_agent_keys: Sequence[str] | None,
) -> list[str] | None:
    if not public_agent_keys:
        return None
    selected: list[str] = []
    for key in public_agent_keys:
        clean = str(key).strip()
        if clean and clean not in selected:
            selected.append(clean)
    return selected or None


def _public_opponent_label(opponent: LoadedPublicAgent) -> str:
    source = opponent.source
    return f"{source.label} / {source.source_ref}"


def _aggregate_public_gate_rows(
    rows: Sequence[Phase3RequiredBenchmarkRow],
    *,
    key_fn: Callable[[Phase3RequiredBenchmarkRow], str],
    label_fn: Callable[[Phase3RequiredBenchmarkRow], str],
    min_win_rate: float,
) -> list[dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = key_fn(row)
        item = aggregates.setdefault(
            key,
            {
                "key": key,
                "label": label_fn(row),
                "games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "timeouts": 0,
                "errors": 0,
            },
        )
        item["games"] += row.games
        item["wins"] += row.wins
        item["losses"] += row.losses
        item["draws"] += row.draws
        item["timeouts"] += row.timeouts
        item["errors"] += row.errors
    for item in aggregates.values():
        item["win_rate"] = item["wins"] / item["games"] if item["games"] else 0.0
        item["passed_min_win_rate"] = (
            item["games"] > 0
            and item["win_rate"] >= min_win_rate
            and item["errors"] == 0
        )
    return [aggregates[key] for key in sorted(aggregates)]
