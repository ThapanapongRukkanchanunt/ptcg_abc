from __future__ import annotations

import json
from html import escape
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ptcg_abc.agent.phase5_search import Phase5SearchPolicyAgent
from ptcg_abc.evaluation import (
    Phase3RequiredBenchmarkResult,
    Phase3RequiredBenchmarkRow,
    Phase3RequiredDebugGame,
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
    _compact_replay_trace,
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
    controlled_public_agent_key: str | None = None
    controlled_deck_index: int | None = None
    policy_epsilon: float | None = None
    policy_seed: int | None = None
    teacher_agent: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PublicControlledDeck:
    index: int
    label: str
    archetype: str
    tournament_rank: int
    card_ids: list[int]
    source_key: str
    source_ref: str


@dataclass(frozen=True)
class PublicAgentTacticalRewardConfig:
    mode: str = "none"
    attack_bonus: float = 0.10
    attach_bonus: float = 0.06
    missed_attack_penalty: float = -0.10
    missed_attach_penalty: float = -0.06
    fractional_prize_weight: float = 1.0
    fractional_opponent_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in {
            "none",
            "basic",
            "fractional-prize",
            "basic-fractional-prize",
        }:
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
    controlled_public_agent_key: str | None = None,
    controlled_deck_index: int = 101,
    agent_kind: str = "phase5-full",
    model_path: Path | None = None,
    specialist_model_dir: Path | None = None,
    deck_indices: Sequence[int] | None = None,
    games_per_matchup: int = 2,
    max_steps: int = 600,
    search_config: Any | None = None,
    search_trace_path: Path | None = None,
    search_trace_game_limit: int = 0,
    policy_epsilon: float = 0.0,
    replay_output_dir: Path | None = None,
    saved_win_replays: int = 0,
    saved_loss_replays: int = 0,
    replay_trace_limit: int = 120,
) -> tuple[Phase3RequiredBenchmarkResult, list[PublicAgentStatus]]:
    card_data, attack_data = load_engine_metadata(sample_dir)
    if search_trace_path is not None:
        search_trace_path.parent.mkdir(parents=True, exist_ok=True)
        search_trace_path.write_text("", encoding="utf-8")
    our_decks = _selected_controlled_decks(
        sample_dir=sample_dir,
        public_agent_roots=public_agent_roots,
        roster_notebook=roster_notebook,
        include_public=include_public,
        include_samples=include_samples,
        include_builtin_samples=include_builtin_samples,
        deck_indices=deck_indices,
        controlled_public_agent_key=controlled_public_agent_key,
        controlled_deck_index=controlled_deck_index,
    )
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
    debug_games: list[Phase3RequiredDebugGame] = []
    aggregate_search: dict[str, Any] = {}
    saved_replay_counts = {"win": 0, "loss": 0}
    if replay_output_dir is not None:
        replay_output_dir.mkdir(parents=True, exist_ok=True)
    for our_deck in our_decks:
        our_model_path = _phase5_specialist_model_path(
            specialist_model_dir,
            _controlled_deck_index(our_deck),
        )
        if our_model_path is None:
            our_model_path = model_path
        for opponent in opponents:
            row = Phase3RequiredBenchmarkRow(
                deck_index=_controlled_deck_index(our_deck),
                deck_label=_controlled_deck_label(our_deck),
                archetype=_controlled_deck_archetype(our_deck),
                tournament_rank=_controlled_deck_rank(our_deck),
                opponent=opponent.label,
                opponent_deck_label=_public_opponent_label(opponent),
                games=games_per_matchup,
            )
            for game_index in range(games_per_matchup):
                our_is_player0 = game_index % 2 == 0
                reward_metadata = {
                    "game_index": game_index + 1,
                    "deck_index": row.deck_index,
                    "deck_label": row.deck_label,
                    "opponent_agent_key": opponent.key,
                    "opponent_agent_label": opponent.label,
                    "opponent_source_ref": opponent.source.source_ref,
                    "opponent_deck_label": _public_opponent_label(opponent),
                    "collector": "phase5_public_rule_opponents_eval",
                    "agent": agent_kind,
                    "specialist_model_dir": (
                        specialist_model_dir.as_posix() if specialist_model_dir else None
                    ),
                    "specialist_model_path": (
                        our_model_path.as_posix()
                        if specialist_model_dir and our_model_path
                        else None
                    ),
                    "controlled_public_agent_key": controlled_public_agent_key,
                    "policy_epsilon": float(policy_epsilon),
                }
                base_agent = _make_agent(
                    agent_kind,
                    _controlled_deck_ids(our_deck),
                    card_data,
                    attack_data,
                    model_path=our_model_path,
                    opponent_deck_ids=opponent.deck_ids,
                    sample_dir=sample_dir,
                    search_config=search_config,
                    policy_epsilon=policy_epsilon,
                )
                should_capture = _should_capture_public_replay(
                    replay_output_dir,
                    saved_replay_counts=saved_replay_counts,
                    saved_win_replays=saved_win_replays,
                    saved_loss_replays=saved_loss_replays,
                )
                our_agent = (
                    RecordingPolicyAgent(
                        base_agent,
                        _controlled_deck_ids(our_deck),
                        card_data=card_data,
                        attack_data=attack_data,
                        reward_metadata=reward_metadata,
                        trace_limit=replay_trace_limit,
                    )
                    if should_capture
                    else base_agent
                )
                opponent_agent = opponent.make_agent()
                result = run_battle(
                    _controlled_deck_ids(our_deck) if our_is_player0 else opponent.deck_ids,
                    opponent.deck_ids if our_is_player0 else _controlled_deck_ids(our_deck),
                    sample_dir=sample_dir,
                    agent0=our_agent if our_is_player0 else opponent_agent,
                    agent1=opponent_agent if our_is_player0 else our_agent,
                    card_data=card_data,
                    attack_data=attack_data,
                    max_steps=max_steps,
                )
                _record_row_outcome(row, result, our_is_player0=our_is_player0)
                search_agent = _search_agent_from_public_agent(our_agent)
                if search_agent is not None:
                    _accumulate_search_telemetry(row.search_telemetry, search_agent.search_telemetry())
                    if _should_write_public_search_trace(
                        search_trace_path,
                        game_index=game_index,
                        trace_game_limit=search_trace_game_limit,
                    ):
                        assert search_trace_path is not None
                        _write_public_search_traces(
                            search_agent,
                            search_trace_path,
                            game_index=game_index,
                            deck_index=row.deck_index,
                            deck_label=row.deck_label,
                            opponent=opponent.label,
                            opponent_key=opponent.key,
                            result=result,
                            our_player_index=0 if our_is_player0 else 1,
                        )
                if isinstance(our_agent, RecordingPolicyAgent):
                    replay = _public_replay_debug_game(
                        row,
                        result,
                        our_agent=our_agent,
                        game_index=game_index,
                        our_player_index=0 if our_is_player0 else 1,
                    )
                    if _maybe_write_public_replay(
                        replay,
                        replay_output_dir,
                        saved_replay_counts=saved_replay_counts,
                        saved_win_replays=saved_win_replays,
                        saved_loss_replays=saved_loss_replays,
                    ):
                        debug_games.append(replay)
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
            debug_games=debug_games,
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
    controlled_public_agent_key: str | None = None,
    controlled_deck_index: int = 101,
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
    policy_epsilon: float = 0.0,
    policy_seed: int | None = None,
    teacher_agent_kind: str | None = None,
) -> PublicAgentTrajectorySummary:
    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        raise ValueError(f"Trajectory output already exists at {output_path}.")
    if teacher_agent_kind is not None and teacher_agent_kind != "rule":
        raise ValueError(f"Unsupported trajectory teacher agent: {teacher_agent_kind}.")
    card_data, attack_data = load_engine_metadata(sample_dir)
    our_decks = _selected_controlled_decks(
        sample_dir=sample_dir,
        public_agent_roots=public_agent_roots,
        roster_notebook=roster_notebook,
        include_public=include_public,
        include_samples=include_samples,
        include_builtin_samples=include_builtin_samples,
        deck_indices=deck_indices,
        controlled_public_agent_key=controlled_public_agent_key,
        controlled_deck_index=controlled_deck_index,
    )
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
    deck_games = {str(_controlled_deck_index(deck)): 0 for deck in our_decks}
    deck_wins = {str(_controlled_deck_index(deck)): 0 for deck in our_decks}
    deck_losses = {str(_controlled_deck_index(deck)): 0 for deck in our_decks}
    deck_draws = {str(_controlled_deck_index(deck)): 0 for deck in our_decks}
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
        deck_index = _controlled_deck_index(our_deck)
        deck_label = _controlled_deck_label(our_deck)
        our_card_ids = _controlled_deck_ids(our_deck)
        our_model_path = _phase5_specialist_model_path(specialist_model_dir, deck_index)
        if our_model_path is None:
            our_model_path = model_path
        for opponent in opponents:
            pair_key = f"{deck_index}-vs-{opponent.key}"
            matchup = matchups.setdefault(
                pair_key,
                {
                    "deck_index": deck_index,
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
                    "deck_index": deck_index,
                    "deck_label": deck_label,
                    "opponent_agent_key": opponent.key,
                    "opponent_agent_label": opponent.label,
                    "opponent_source_ref": opponent.source.source_ref,
                    "opponent_deck_label": _public_opponent_label(opponent),
                    "collector": "phase5_public_rule_opponents",
                    "agent": agent_kind,
                    "teacher_agent": teacher_agent_kind,
                    "controlled_public_agent_key": controlled_public_agent_key,
                    "policy_epsilon": float(policy_epsilon),
                    "policy_seed": (
                        int(policy_seed) + absolute_game_index
                        if policy_seed is not None
                        else None
                    ),
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
                        our_card_ids,
                        card_data,
                        attack_data,
                        model_path=our_model_path,
                        opponent_deck_ids=opponent.deck_ids,
                        sample_dir=sample_dir,
                        search_config=search_config,
                        policy_epsilon=policy_epsilon,
                        policy_seed=(
                            int(policy_seed) + absolute_game_index
                            if policy_seed is not None
                            else None
                        ),
                    ),
                    our_card_ids,
                    card_data=card_data,
                    attack_data=attack_data,
                    reward_metadata=reward_metadata,
                    target_agent=(
                        _make_agent(
                            teacher_agent_kind,
                            our_card_ids,
                            card_data,
                            attack_data,
                        )
                        if teacher_agent_kind is not None
                        else None
                    ),
                    target_agent_kind=teacher_agent_kind,
                )
                opponent_agent = opponent.make_agent()
                result = run_battle(
                    our_card_ids if our_is_player0 else opponent.deck_ids,
                    opponent.deck_ids if our_is_player0 else our_card_ids,
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
                deck_games[str(deck_index)] += 1
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
                records = list(recorder.frames)
                our_player_index = 0 if our_is_player0 else 1
                for record_index, record in enumerate(records):
                    next_frame = (
                        records[record_index + 1].frame
                        if record_index + 1 < len(records)
                        else None
                    )
                    tactical_reward, tactical_metadata = _tactical_reward_for_frame(
                        record.frame,
                        record.chosen_indices,
                        tactical_config,
                        post_action_board=record.post_action_board,
                        next_frame=next_frame,
                        final_prize_counts=result.prize_counts,
                        player_index=our_player_index,
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
                    deck_draws[str(deck_index)] += 1
                    opponent_draws[opponent.key] += 1
                    matchup["draws"] += 1
                    continue
                effective = result.winner if result.winner is not None else result.leader
                our_player_index = 0 if our_is_player0 else 1
                if effective is None:
                    draws += 1
                    deck_draws[str(deck_index)] += 1
                    opponent_draws[opponent.key] += 1
                    matchup["draws"] += 1
                elif effective == our_player_index:
                    wins += 1
                    deck_wins[str(deck_index)] += 1
                    opponent_losses[opponent.key] += 1
                    matchup["wins"] += 1
                else:
                    losses += 1
                    deck_losses[str(deck_index)] += 1
                    opponent_wins[opponent.key] += 1
                    matchup["losses"] += 1

    return PublicAgentTrajectorySummary(
        agent=agent_kind,
        games_requested=games_requested,
        games_started=games_started,
        steps=steps_written,
        output_path=output_path.as_posix(),
        deck_indices=[_controlled_deck_index(deck) for deck in our_decks],
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
        controlled_public_agent_key=controlled_public_agent_key,
        controlled_deck_index=controlled_deck_index if controlled_public_agent_key else None,
        policy_epsilon=(
            float(policy_epsilon)
            if agent_kind in {"phase5-epsilon", "phase5-epsilon-mixture"}
            else None
        ),
        policy_seed=policy_seed,
        teacher_agent=teacher_agent_kind,
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


def _selected_controlled_decks(
    *,
    sample_dir: Path,
    public_agent_roots: Sequence[Path],
    roster_notebook: Path | None,
    include_public: bool,
    include_samples: bool,
    include_builtin_samples: bool,
    deck_indices: Sequence[int] | None,
    controlled_public_agent_key: str | None,
    controlled_deck_index: int,
) -> list[Any]:
    if not controlled_public_agent_key:
        return list(_selected_phase5_decks(deck_indices))
    if deck_indices:
        raise ValueError(
            "Use either --deck-index for Phase 5 league decks or "
            "--controlled-public-agent-key for a public/sample controlled deck, not both."
        )
    opponents, statuses = discover_phase5_public_opponents(
        sample_dir=sample_dir,
        public_agent_roots=public_agent_roots,
        roster_notebook=roster_notebook,
        include_public=include_public,
        include_samples=include_samples,
        include_builtin_samples=include_builtin_samples,
        public_agent_keys=[controlled_public_agent_key],
    )
    if len(opponents) != 1:
        errors = [
            f"{status.source.key}: {status.error or status.status}"
            for status in statuses
            if status.source.key == controlled_public_agent_key
        ]
        detail = "; ".join(errors) if errors else "not available"
        raise ValueError(
            f"Controlled public/sample deck {controlled_public_agent_key!r} is {detail}."
        )
    source = opponents[0]
    return [
        PublicControlledDeck(
            index=int(controlled_deck_index),
            label=source.label,
            archetype=source.label,
            tournament_rank=int(controlled_deck_index),
            card_ids=list(source.deck_ids),
            source_key=source.key,
            source_ref=source.source.source_ref,
        )
    ]


def _controlled_deck_index(deck: Any) -> int:
    return int(deck.index)


def _controlled_deck_label(deck: Any) -> str:
    return str(deck.label)


def _controlled_deck_archetype(deck: Any) -> str:
    return str(deck.archetype)


def _controlled_deck_rank(deck: Any) -> int:
    if isinstance(deck, PublicControlledDeck):
        return int(deck.tournament_rank)
    return int(deck.deck.result.placement_rank)


def _controlled_deck_ids(deck: Any) -> list[int]:
    return list(deck.card_ids)


def _search_agent_from_public_agent(agent: Any) -> Phase5SearchPolicyAgent | None:
    if isinstance(agent, Phase5SearchPolicyAgent):
        return agent
    wrapped = getattr(agent, "agent", None)
    if isinstance(wrapped, Phase5SearchPolicyAgent):
        return wrapped
    return None


def _should_capture_public_replay(
    replay_output_dir: Path | None,
    *,
    saved_replay_counts: dict[str, int],
    saved_win_replays: int,
    saved_loss_replays: int,
) -> bool:
    if replay_output_dir is None:
        return False
    return (
        saved_replay_counts.get("win", 0) < max(0, saved_win_replays)
        or saved_replay_counts.get("loss", 0) < max(0, saved_loss_replays)
    )


def _public_replay_debug_game(
    row: Phase3RequiredBenchmarkRow,
    result: Any,
    *,
    our_agent: RecordingPolicyAgent,
    game_index: int,
    our_player_index: int,
) -> Phase3RequiredDebugGame:
    return Phase3RequiredDebugGame(
        deck_index=row.deck_index,
        deck_label=row.deck_label,
        archetype=row.archetype,
        tournament_rank=row.tournament_rank,
        opponent=row.opponent,
        game_index=game_index + 1,
        outcome=_public_game_outcome(result, our_player_index=our_player_index),
        our_player_index=our_player_index,
        steps=int(getattr(result, "steps", 0) or 0),
        prize_counts=getattr(result, "prize_counts", None),
        error=getattr(result, "error", None),
        trace=_compact_replay_trace(our_agent.frames),
    )


def _maybe_write_public_replay(
    replay: Phase3RequiredDebugGame,
    replay_output_dir: Path | None,
    *,
    saved_replay_counts: dict[str, int],
    saved_win_replays: int,
    saved_loss_replays: int,
) -> bool:
    if replay_output_dir is None:
        return False
    outcome = "win" if replay.outcome == "win" else "loss" if replay.outcome == "loss" else ""
    if not outcome:
        return False
    limit = max(0, saved_win_replays if outcome == "win" else saved_loss_replays)
    if saved_replay_counts.get(outcome, 0) >= limit:
        return False
    saved_replay_counts[outcome] = saved_replay_counts.get(outcome, 0) + 1
    stem = (
        f"deck-{replay.deck_index:02d}-vs-{_slug(replay.opponent)}-"
        f"game-{replay.game_index:04d}-{outcome}"
    )
    json_path = replay_output_dir / f"{stem}.json"
    html_path = replay_output_dir / f"{stem}.html"
    payload = replay.to_dict() | {
        "visualizer_url": None,
        "html_path": html_path.as_posix(),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    html_path.write_text(_public_replay_html(payload), encoding="utf-8")
    return True


def _public_replay_html(payload: dict[str, Any]) -> str:
    title = (
        f"Deck {payload.get('deck_index')} vs {payload.get('opponent')} "
        f"{payload.get('outcome')}"
    )
    rows = []
    for step, record in enumerate(payload.get("trace", []), start=1):
        chosen = ", ".join(
            _action_label(action) for action in record.get("chosen_options", [])
        )
        top_rule = "<br>".join(
            escape(_action_label(action)) for action in record.get("top_rule_options", [])
        )
        board = record.get("board", {})
        board_text = escape(json.dumps(board, sort_keys=True))
        rows.append(
            "<tr>"
            f"<td>{step}</td>"
            f"<td>{escape(str(record.get('turn', '')))}</td>"
            f"<td>{escape(str(record.get('select_type', '')))}</td>"
            f"<td>{escape(chosen)}</td>"
            f"<td>{top_rule}</td>"
            f"<td><code>{board_text}</code></td>"
            "</tr>"
        )
    return "\n".join(
        [
            "<!doctype html>",
            "<html><head><meta charset=\"utf-8\">",
            f"<title>{escape(title)}</title>",
            "<style>",
            "body{font-family:system-ui,Arial,sans-serif;margin:24px;line-height:1.4}",
            "table{border-collapse:collapse;width:100%;font-size:13px}",
            "th,td{border:1px solid #ddd;padding:6px;vertical-align:top}",
            "th{background:#f5f5f5;text-align:left}",
            "code{white-space:pre-wrap;font-size:12px}",
            "</style></head><body>",
            f"<h1>{escape(title)}</h1>",
            "<ul>",
            f"<li>Outcome: {escape(str(payload.get('outcome')))}</li>",
            f"<li>Steps: {escape(str(payload.get('steps')))}</li>",
            f"<li>Prize counts: {escape(str(payload.get('prize_counts')))}</li>",
            "</ul>",
            "<table><thead><tr><th>#</th><th>Turn</th><th>Select</th>"
            "<th>Chosen</th><th>Top Rule Options</th><th>Board</th></tr></thead><tbody>",
            *rows,
            "</tbody></table></body></html>",
        ]
    )


def _action_label(action: dict[str, Any]) -> str:
    pieces = [str(action.get("type", ""))]
    if action.get("card_name"):
        pieces.append(str(action["card_name"]))
    elif action.get("card_id") is not None:
        pieces.append(f"card {action['card_id']}")
    if action.get("target_name"):
        pieces.append(f"-> {action['target_name']}")
    elif action.get("target_card_id") is not None:
        pieces.append(f"-> card {action['target_card_id']}")
    return " ".join(piece for piece in pieces if piece)


def _slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "-" for char in value]
    return "-".join(part for part in "".join(chars).split("-") if part) or "replay"


def _should_write_public_search_trace(
    path: Path | None,
    *,
    game_index: int,
    trace_game_limit: int,
) -> bool:
    if path is None:
        return False
    return trace_game_limit <= 0 or game_index < trace_game_limit


def _write_public_search_traces(
    agent: Phase5SearchPolicyAgent,
    path: Path,
    *,
    game_index: int,
    deck_index: int,
    deck_label: str,
    opponent: str,
    opponent_key: str,
    result: Any,
    our_player_index: int,
) -> int:
    if not agent.traces:
        return 0
    outcome = _public_game_outcome(result, our_player_index=our_player_index)
    records = 0
    error = getattr(result, "error", None)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for trace in agent.traces:
            payload = trace.to_dict()
            payload.update(
                {
                    "game_index": int(game_index),
                    "deck_index": int(deck_index),
                    "deck_label": deck_label,
                    "opponent": opponent,
                    "opponent_key": opponent_key,
                    "our_player_index": int(our_player_index),
                    "game_outcome": outcome,
                    "winner": getattr(result, "winner", None),
                    "leader": getattr(result, "leader", None),
                    "finished": bool(getattr(result, "finished", False)),
                    "started": bool(getattr(result, "started", False)),
                    "error": str(error) if error is not None else None,
                }
            )
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            records += 1
    return records


def _public_game_outcome(result: Any, *, our_player_index: int) -> str:
    if getattr(result, "error", None):
        return "error"
    if bool(getattr(result, "started", False)) and not bool(getattr(result, "finished", False)):
        return "timeout"
    effective = getattr(result, "winner", None)
    if effective is None:
        effective = getattr(result, "leader", None)
    if effective is None:
        return "draw"
    return "win" if int(effective) == int(our_player_index) else "loss"


def _tactical_reward_for_frame(
    frame: DecisionFrame,
    chosen_indices: Sequence[int],
    config: PublicAgentTacticalRewardConfig,
    *,
    post_action_board: dict[str, Any] | None = None,
    next_frame: DecisionFrame | None = None,
    final_prize_counts: Sequence[int] | None = None,
    player_index: int | None = None,
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
    basic_reward = 0.0
    missed_attack = False
    missed_attach = False
    if config.mode in {"basic", "basic-fractional-prize"}:
        if attack_taken:
            basic_reward += config.attack_bonus
        elif attack_available and (end_selected or empty_selection):
            basic_reward += config.missed_attack_penalty
            missed_attack = True

        if attach_taken:
            basic_reward += config.attach_bonus
        elif attach_available and (attack_taken or end_selected or empty_selection):
            basic_reward += config.missed_attach_penalty
            missed_attach = True

    fractional_reward, fractional_metadata = _fractional_prize_reward_for_frame(
        frame,
        post_action_board=post_action_board,
        next_frame=next_frame,
        final_prize_counts=final_prize_counts,
        player_index=player_index,
        config=config,
    )
    reward = basic_reward + fractional_reward
    metadata = {
        "tactical_reward_mode": config.mode,
        "tactical_step_reward": reward,
        "tactical_basic_reward": basic_reward,
        "tactical_attack_available": attack_available,
        "tactical_attack_taken": attack_taken,
        "tactical_attach_available": attach_available,
        "tactical_attach_taken": attach_taken,
        "tactical_end_selected": end_selected,
        "tactical_empty_selection": empty_selection,
        "tactical_missed_attack": missed_attack,
        "tactical_missed_attach": missed_attach,
    }
    metadata.update(fractional_metadata)
    return reward, metadata


def _fractional_prize_reward_for_frame(
    frame: DecisionFrame,
    *,
    post_action_board: dict[str, Any] | None,
    next_frame: DecisionFrame | None,
    final_prize_counts: Sequence[int] | None,
    player_index: int | None,
    config: PublicAgentTacticalRewardConfig,
) -> tuple[float, dict[str, Any]]:
    metadata: dict[str, Any] = {
        "tactical_fractional_prize_before": None,
        "tactical_fractional_prize_after": None,
        "tactical_fractional_prize_delta": 0.0,
        "tactical_fractional_opponent_prize_before": None,
        "tactical_fractional_opponent_prize_after": None,
        "tactical_fractional_opponent_prize_delta": 0.0,
        "tactical_fractional_prize_reward": 0.0,
        "tactical_fractional_prize_weight": float(config.fractional_prize_weight),
        "tactical_fractional_opponent_weight": float(config.fractional_opponent_weight),
        "tactical_fractional_prize_has_after_board": False,
        "tactical_fractional_prize_after_source": "none",
    }
    if config.mode not in {"fractional-prize", "basic-fractional-prize"}:
        return 0.0, metadata

    after_board: dict[str, Any] | None = None
    after_source = "none"
    if post_action_board is not None:
        after_board = post_action_board
        after_source = "post-action"
    elif next_frame is not None:
        after_board = next_frame.board
        after_source = "next-frame"
    else:
        after_board = _board_with_final_prizes(
            frame.board,
            final_prize_counts=final_prize_counts,
            player_index=player_index,
        )
        if after_board is not None:
            after_source = "final-prizes"
    if after_board is None:
        return 0.0, metadata

    before_self = _fractional_prize_progress(frame.board, perspective="my")
    after_self = _fractional_prize_progress(after_board, perspective="my")
    before_opp = _fractional_prize_progress(frame.board, perspective="opponent")
    after_opp = _fractional_prize_progress(after_board, perspective="opponent")
    self_delta = after_self - before_self
    opponent_delta = after_opp - before_opp
    reward = float(config.fractional_prize_weight) * (
        self_delta - float(config.fractional_opponent_weight) * opponent_delta
    )

    metadata.update(
        {
            "tactical_fractional_prize_before": before_self,
            "tactical_fractional_prize_after": after_self,
            "tactical_fractional_prize_delta": self_delta,
            "tactical_fractional_opponent_prize_before": before_opp,
            "tactical_fractional_opponent_prize_after": after_opp,
            "tactical_fractional_opponent_prize_delta": opponent_delta,
            "tactical_fractional_prize_reward": reward,
            "tactical_fractional_prize_has_after_board": True,
            "tactical_fractional_prize_after_source": after_source,
        }
    )
    return reward, metadata


def _board_with_final_prizes(
    board: dict[str, Any],
    *,
    final_prize_counts: Sequence[int] | None,
    player_index: int | None,
) -> dict[str, Any] | None:
    if final_prize_counts is None or player_index is None:
        return None
    if len(final_prize_counts) < 2 or int(player_index) not in {0, 1}:
        return None
    output = dict(board)
    output["my_prizes"] = int(final_prize_counts[int(player_index)])
    output["opponent_prizes"] = int(final_prize_counts[1 - int(player_index)])
    return output


def _fractional_prize_progress(
    board: dict[str, Any],
    *,
    perspective: str,
) -> float:
    if perspective == "my":
        remaining_prizes = board.get("my_prizes", 6)
        target_active = board.get("opponent_active_card")
        target_bench = board.get("opponent_bench_cards", [])
    elif perspective == "opponent":
        remaining_prizes = board.get("opponent_prizes", 6)
        target_active = board.get("my_active_card")
        target_bench = board.get("my_bench_cards", [])
    else:
        raise ValueError(f"Unsupported fractional prize perspective: {perspective}.")

    progress = 6.0 - _clamp_float(_safe_float(remaining_prizes, default=6.0), 0.0, 6.0)
    cards: list[Any] = []
    if isinstance(target_active, dict) and target_active:
        cards.append(target_active)
    if isinstance(target_bench, list):
        cards.extend(card for card in target_bench if isinstance(card, dict) and card)
    for card in cards:
        progress += _card_fractional_prize_progress(card)
    return progress


def _card_fractional_prize_progress(card: dict[str, Any]) -> float:
    max_hp = _safe_float(card.get("max_hp"), default=0.0)
    if max_hp <= 0:
        return 0.0
    hp = _clamp_float(_safe_float(card.get("hp"), default=max_hp), 0.0, max_hp)
    damage_fraction = _clamp_float((max_hp - hp) / max_hp, 0.0, 1.0)
    return _card_prize_value(card) * damage_fraction


def _card_prize_value(card: dict[str, Any]) -> float:
    if bool(card.get("is_mega_ex", False)):
        return 3.0
    if bool(card.get("is_ex", False)):
        return 2.0
    return 1.0


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _empty_tactical_reward_summary(
    config: PublicAgentTacticalRewardConfig,
) -> dict[str, Any]:
    return {
        "mode": config.mode,
        "steps": 0,
        "reward_sum": 0.0,
        "basic_reward_sum": 0.0,
        "fractional_prize_reward_sum": 0.0,
        "fractional_prize_delta_sum": 0.0,
        "fractional_opponent_prize_delta_sum": 0.0,
        "fractional_after_board": 0,
        "fractional_after_board_sources": {},
        "fractional_prize_weight": float(config.fractional_prize_weight),
        "fractional_opponent_weight": float(config.fractional_opponent_weight),
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
    summary["basic_reward_sum"] += float(metadata.get("tactical_basic_reward", 0.0))
    summary["fractional_prize_reward_sum"] += float(
        metadata.get("tactical_fractional_prize_reward", 0.0)
    )
    summary["fractional_prize_delta_sum"] += float(
        metadata.get("tactical_fractional_prize_delta", 0.0)
    )
    summary["fractional_opponent_prize_delta_sum"] += float(
        metadata.get("tactical_fractional_opponent_prize_delta", 0.0)
    )
    summary["fractional_after_board"] += int(
        bool(metadata.get("tactical_fractional_prize_has_after_board", False))
    )
    source = str(metadata.get("tactical_fractional_prize_after_source", "none"))
    source_counts = summary.setdefault("fractional_after_board_sources", {})
    source_counts[source] = int(source_counts.get(source, 0)) + 1
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
    output["avg_basic_reward"] = (
        float(output.get("basic_reward_sum", 0.0)) / steps if steps else 0.0
    )
    output["avg_fractional_prize_reward"] = (
        float(output.get("fractional_prize_reward_sum", 0.0)) / steps
        if steps
        else 0.0
    )
    output["avg_fractional_prize_delta"] = (
        float(output.get("fractional_prize_delta_sum", 0.0)) / steps if steps else 0.0
    )
    output["avg_fractional_opponent_prize_delta"] = (
        float(output.get("fractional_opponent_prize_delta_sum", 0.0)) / steps
        if steps
        else 0.0
    )
    output["fractional_after_board_rate"] = (
        float(output.get("fractional_after_board", 0)) / steps if steps else 0.0
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
