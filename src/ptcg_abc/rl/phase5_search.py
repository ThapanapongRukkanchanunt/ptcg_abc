from __future__ import annotations

import contextlib
import glob
import io
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from ptcg_abc.agent.rule_based import (
    RuleBasedAgent,
    _card_type_name,
    _get,
    select_option_indices,
)
from ptcg_abc.evaluation import phase3_tournament_559_prepared_decks, required_phase3_prepared_decks
from ptcg_abc.rl.dataset import append_decision_jsonl
from ptcg_abc.rl.featurizer import attack_lookup, card_lookup, make_decision_frame, summarize_board
from ptcg_abc.rl.records import DecisionFrame
from ptcg_abc.simulator import BattleResult, _with_sample_submission_on_path, load_engine_metadata, run_battle


PHASE5_SEARCH_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RootSearchConfig:
    """Small bounded one-turn root-search configuration for Phase 5 data generation."""

    top_k: int = 4
    max_rollout_steps: int = 30
    min_candidates: int = 2
    rule_prior_weight: float = 0.08
    damage_weight: float = 0.025
    self_damage_weight: float = 0.015
    prize_weight: float = 3.0
    opponent_prize_weight: float = 2.5
    setup_weight: float = 0.15
    hand_weight: float = 0.02
    terminal_win_score: float = 100.0
    terminal_loss_score: float = -100.0
    truncated_penalty: float = 0.25
    tactical_score_weight: float = 1.0
    normalize_tactical_score: bool = False
    policy_prior_weight: float = 0.0
    neural_action_value_weight: float = 0.0
    neural_tactical_weight: float = 0.0
    leaf_state_value_weight: float = 0.0
    root_select_types: tuple[str, ...] = ("MAIN",)
    root_contexts: tuple[str, ...] = ("MAIN",)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["root_select_types"] = list(self.root_select_types)
        payload["root_contexts"] = list(self.root_contexts)
        return payload


@dataclass(frozen=True)
class HiddenStateSample:
    your_deck: list[int]
    your_prize: list[int]
    opponent_deck: list[int]
    opponent_prize: list[int]
    opponent_hand: list[int]
    opponent_active: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateEvaluation:
    indices: list[int]
    option_index: int | None
    option_type: str
    card_name: str
    attack_id: int | None
    rule_score: float
    rule_rank: int
    rollout_steps: int = 0
    terminal: bool = False
    turn_ended: bool = False
    truncated: bool = False
    error: str | None = None
    tactical_score: float = 0.0
    tactical_score_prior: float = 0.0
    tactical_score_component: float = 0.0
    leaf_state_value: float = 0.0
    leaf_state_value_prior: float = 0.0
    leaf_state_value_score: float = 0.0
    rule_prior: float = 0.0
    policy_score: float = 0.0
    policy_prior: float = 0.0
    neural_action_value: float = 0.0
    neural_action_value_prior: float = 0.0
    neural_tactical_score: float = 0.0
    neural_tactical_prior: float = 0.0
    prior_score: float = 0.0
    combined_score: float = 0.0
    end_turn: int | None = None
    end_result: int | None = None
    end_prize_counts: list[int] | None = None
    damage_delta: float = 0.0
    self_damage_delta: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RootSearchDecisionTrace:
    schema_version: int
    game_index: int
    step_index: int
    deck_index: int
    deck_label: str
    opponent: str
    player_index: int
    turn: int
    select_type: str
    context: str
    baseline_indices: list[int]
    search_indices: list[int]
    changed: bool
    search_started: bool
    search_error: str | None
    hidden_counts: dict[str, int]
    candidates: list[CandidateEvaluation]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "game_index": self.game_index,
            "step_index": self.step_index,
            "deck_index": self.deck_index,
            "deck_label": self.deck_label,
            "opponent": self.opponent,
            "player_index": self.player_index,
            "turn": self.turn,
            "select_type": self.select_type,
            "context": self.context,
            "baseline_indices": list(self.baseline_indices),
            "search_indices": list(self.search_indices),
            "changed": self.changed,
            "search_started": self.search_started,
            "search_error": self.search_error,
            "hidden_counts": dict(self.hidden_counts),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "config": dict(self.config),
        }


@dataclass(frozen=True)
class SearchDataSummary:
    games_requested: int
    game_offset: int
    shard_index: int
    shard_count: int
    games_started: int
    decisions: int
    searched_decisions: int
    changed_decisions: int
    candidate_probes: int
    probe_errors: int
    truncated_rollouts: int
    output_path: str
    trace_path: str
    append: bool
    errors: int
    timeouts: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchMergeSummary:
    decision_files: int
    trace_files: int
    decision_records: int
    trace_records: int
    output_path: str
    trace_path: str
    manifest_path: str | None
    input_paths: list[str]
    trace_input_paths: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OneTurnRootSearchAgent:
    """Collect search-improved labels with bounded Search API probes.

    The agent keeps the Phase 4 rule agent as the rollout/default policy. For eligible
    root decisions it evaluates the top rule-prior candidates by simulating each first
    action to the end of the current turn, then returns the best scored candidate.
    """

    def __init__(
        self,
        deck_ids: Sequence[int],
        *,
        opponent_deck_ids: Sequence[int],
        sample_dir: Path,
        card_data: Sequence[Any],
        attack_data: Sequence[Any],
        reward_metadata: dict[str, Any] | None = None,
        config: RootSearchConfig | None = None,
    ) -> None:
        if len(deck_ids) != 60:
            raise ValueError(f"OneTurnRootSearchAgent needs a 60-card deck, got {len(deck_ids)}.")
        if len(opponent_deck_ids) != 60:
            raise ValueError(
                f"OneTurnRootSearchAgent needs a 60-card opponent deck, got {len(opponent_deck_ids)}."
            )
        self.deck_ids = list(deck_ids)
        self.opponent_deck_ids = list(opponent_deck_ids)
        self.sample_dir = Path(sample_dir)
        self.card_data = card_data
        self.attack_data = attack_data
        self.card_by_id = card_lookup(card_data)
        self.attack_by_id = attack_lookup(attack_data)
        self.reward_metadata = dict(reward_metadata or {})
        self.config = config or RootSearchConfig()
        self.frames: list[DecisionFrame] = []
        self.traces: list[RootSearchDecisionTrace] = []
        with contextlib.redirect_stdout(io.StringIO()):
            self.rule_agent = RuleBasedAgent(
                self.deck_ids,
                card_data=card_data,
                attack_data=attack_data,
            )
        self._search_begin, self._search_step, self._search_end = _load_search_api(self.sample_dir)

    def act(self, observation: Any) -> list[int]:
        baseline = _valid_indices(self.rule_agent.act(observation), _option_count(observation))
        frame = make_decision_frame(
            observation,
            deck_ids=self.deck_ids,
            card_by_id=self.card_by_id,
            attack_by_id=self.attack_by_id,
            selected_indices=baseline,
            reward_metadata=self._base_metadata(applied=False, baseline=baseline, search=baseline),
        )
        if frame is None:
            return baseline if _get(observation, "select") is not None else list(self.deck_ids)

        selected = list(baseline)
        trace: RootSearchDecisionTrace | None = None
        if self._should_search(frame):
            selected, trace = self._search_decision(observation, frame, baseline)
            self.traces.append(trace)

        changed = selected != baseline
        metadata = self._base_metadata(
            applied=trace is not None and trace.search_started,
            baseline=baseline,
            search=selected,
            changed=changed,
            search_error=trace.search_error if trace is not None else None,
        )
        improved_frame = _replace_frame_selection(frame, selected, metadata)
        self.frames.append(improved_frame)
        return selected

    def _should_search(self, frame: DecisionFrame) -> bool:
        return (
            frame.select_type in self.config.root_select_types
            and frame.context in self.config.root_contexts
            and frame.target_count == 1
            and len(frame.legal_options) >= self.config.min_candidates
            and self.config.top_k > 0
        )

    def _search_decision(
        self,
        observation: Any,
        frame: DecisionFrame,
        baseline: list[int],
    ) -> tuple[list[int], RootSearchDecisionTrace]:
        candidates = _candidate_evaluations(frame, baseline, top_k=self.config.top_k)
        search_started = False
        search_error: str | None = None
        hidden: HiddenStateSample | None = None
        try:
            hidden = sample_hidden_state(
                observation,
                own_deck_ids=self.deck_ids,
                opponent_deck_ids=self.opponent_deck_ids,
                card_by_id=self.card_by_id,
            )
            root_state = self._search_begin(
                observation,
                hidden.your_deck,
                hidden.your_prize,
                hidden.opponent_deck,
                hidden.opponent_prize,
                hidden.opponent_hand,
                hidden.opponent_active,
                False,
            )
            search_started = True
            root_search_id = int(_get(root_state, "searchId", 0) or 0)
            for candidate in candidates:
                self._evaluate_candidate(
                    root_search_id=root_search_id,
                    candidate=candidate,
                    root_observation=observation,
                    root_frame=frame,
                )
        except Exception as exc:
            search_error = f"{type(exc).__name__}: {exc}"
        finally:
            if search_started:
                try:
                    self._search_end()
                except Exception:
                    pass

        _score_candidates(candidates, self.config)
        selected = _best_candidate_indices(candidates, baseline)
        trace = RootSearchDecisionTrace(
            schema_version=PHASE5_SEARCH_SCHEMA_VERSION,
            game_index=int(self.reward_metadata.get("game_index", 0) or 0),
            step_index=len(self.frames) + 1,
            deck_index=int(self.reward_metadata.get("deck_index", 0) or 0),
            deck_label=str(self.reward_metadata.get("deck_label", "")),
            opponent=str(self.reward_metadata.get("opponent", "")),
            player_index=int(frame.board.get("your_index", 0) or 0),
            turn=int(frame.board.get("turn", 0) or 0),
            select_type=frame.select_type,
            context=frame.context,
            baseline_indices=list(baseline),
            search_indices=list(selected),
            changed=selected != baseline,
            search_started=search_started,
            search_error=search_error,
            hidden_counts=_hidden_counts(hidden),
            candidates=candidates,
            config=self.config.to_dict(),
        )
        return selected, trace

    def _evaluate_candidate(
        self,
        *,
        root_search_id: int,
        candidate: CandidateEvaluation,
        root_observation: Any,
        root_frame: DecisionFrame,
    ) -> None:
        root_current = _get(root_observation, "current")
        root_turn = int(_get(root_current, "turn", 0) or 0)
        root_player = int(root_frame.board.get("your_index", 0) or 0)
        root_metrics = _board_metrics(root_frame.board, player_index=root_player)
        state = None
        try:
            state = self._search_step(root_search_id, candidate.indices)
            candidate.rollout_steps = 1
            final_observation = _get(state, "observation")
            for _ in range(max(0, self.config.max_rollout_steps - 1)):
                current = _get(final_observation, "current")
                if _is_terminal(current):
                    candidate.terminal = True
                    break
                if _turn_has_ended(current, root_turn):
                    candidate.turn_ended = True
                    break
                select = _get(final_observation, "select")
                if select is None:
                    break
                acting_player = int(_get(current, "yourIndex", root_player) or root_player)
                deck_ids = self.deck_ids if acting_player == root_player else self.opponent_deck_ids
                choice = select_option_indices(
                    select,
                    current=current,
                    card_by_id=self.card_by_id,
                    attack_by_id=self.attack_by_id,
                    deck_ids=deck_ids,
                )
                state = self._search_step(int(_get(state, "searchId", 0) or 0), list(choice))
                candidate.rollout_steps += 1
                final_observation = _get(state, "observation")

            final_current = _get(final_observation, "current")
            if _is_terminal(final_current):
                candidate.terminal = True
            if _turn_has_ended(final_current, root_turn):
                candidate.turn_ended = True
            candidate.truncated = not candidate.terminal and not candidate.turn_ended
            end_board = _summarize_board_for_player(
                final_current,
                player_index=root_player,
                card_by_id=self.card_by_id,
            )
            end_metrics = _board_metrics(end_board, player_index=root_player)
            candidate.tactical_score = _tactical_score(
                root_metrics,
                end_metrics,
                final_current,
                player_index=root_player,
                config=self.config,
                truncated=candidate.truncated,
            )
            leaf_value = self._candidate_leaf_state_value(
                final_observation,
                root_player=root_player,
            )
            if leaf_value is not None:
                candidate.leaf_state_value = float(leaf_value)
            candidate.end_turn = int(_get(final_current, "turn", 0) or 0)
            result = _get(final_current, "result", None)
            candidate.end_result = int(result) if result is not None else None
            candidate.end_prize_counts = [
                int(end_metrics["my_prizes"]),
                int(end_metrics["opponent_prizes"]),
            ]
            candidate.damage_delta = float(end_metrics["opponent_damage"] - root_metrics["opponent_damage"])
            candidate.self_damage_delta = float(end_metrics["my_damage"] - root_metrics["my_damage"])
        except Exception as exc:
            candidate.error = f"{type(exc).__name__}: {exc}"
            candidate.tactical_score = -999.0

    def _candidate_leaf_state_value(
        self,
        final_observation: Any,
        *,
        root_player: int,
    ) -> float | None:
        return None

    def _base_metadata(
        self,
        *,
        applied: bool,
        baseline: Sequence[int],
        search: Sequence[int],
        changed: bool = False,
        search_error: str | None = None,
    ) -> dict[str, Any]:
        return self.reward_metadata | {
            "collector": "phase5_search",
            "phase5_search_applied": applied,
            "phase5_baseline_indices": list(baseline),
            "phase5_search_indices": list(search),
            "phase5_search_changed": changed,
            "phase5_search_error": search_error,
        }


def generate_search_improved_data(
    *,
    sample_dir: Path,
    output_path: Path,
    trace_path: Path,
    games: int = 2,
    game_offset: int = 0,
    shard_index: int = 0,
    shard_count: int = 1,
    max_steps: int = 80,
    append: bool = False,
    config: RootSearchConfig | None = None,
) -> SearchDataSummary:
    if games < 0:
        raise ValueError("games must be non-negative.")
    if game_offset < 0:
        raise ValueError("game_offset must be non-negative.")
    if shard_count <= 0:
        raise ValueError("shard_count must be positive.")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count.")
    card_data, attack_data = load_engine_metadata(sample_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if not append:
        output_path.write_text("", encoding="utf-8")
        trace_path.write_text("", encoding="utf-8")
    our_decks = phase3_tournament_559_prepared_decks()
    benchmark_decks = required_phase3_prepared_decks(start_index=1)
    matchups = [(our, benchmark) for our in our_decks for benchmark in benchmark_decks]
    search_config = config or RootSearchConfig()
    games_started = decisions = searched = changed = probes = probe_errors = truncated = 0
    errors = timeouts = 0

    for local_game_index in range(games):
        absolute_game_index = phase5_absolute_game_index(
            local_game_index,
            game_offset=game_offset,
            shard_index=shard_index,
            shard_count=shard_count,
        )
        our_deck, benchmark_deck, our_is_player0 = phase5_matchup_for_game_index(
            matchups,
            absolute_game_index,
        )
        search_agent = OneTurnRootSearchAgent(
            our_deck.card_ids,
            opponent_deck_ids=benchmark_deck.card_ids,
            sample_dir=sample_dir,
            card_data=card_data,
            attack_data=attack_data,
            reward_metadata={
                "game_index": absolute_game_index + 1,
                "local_game_index": local_game_index + 1,
                "game_offset": game_offset,
                "shard_index": shard_index,
                "shard_count": shard_count,
                "deck_index": our_deck.index,
                "deck_label": our_deck.label,
                "opponent": benchmark_deck.archetype,
            },
            config=search_config,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            benchmark_agent = RuleBasedAgent(
                benchmark_deck.card_ids,
                card_data=card_data,
                attack_data=attack_data,
            )
        result = run_battle(
            our_deck.card_ids if our_is_player0 else benchmark_deck.card_ids,
            benchmark_deck.card_ids if our_is_player0 else our_deck.card_ids,
            sample_dir=sample_dir,
            agent0=search_agent if our_is_player0 else benchmark_agent,
            agent1=benchmark_agent if our_is_player0 else search_agent,
            card_data=card_data,
            attack_data=attack_data,
            max_steps=max_steps,
        )
        games_started += int(result.started)
        errors += int(result.error is not None)
        timeouts += int(result.started and not result.finished and result.error is None)
        final_metadata = _result_metadata(result, player_index=0 if our_is_player0 else 1)
        for frame in search_agent.frames:
            append_decision_jsonl(_with_metadata(frame, final_metadata), output_path)
            decisions += 1
        for trace in search_agent.traces:
            _append_trace_jsonl(trace, trace_path)
            searched += 1
            changed += int(trace.changed)
            probes += len(trace.candidates)
            probe_errors += sum(1 for candidate in trace.candidates if candidate.error is not None)
            truncated += sum(1 for candidate in trace.candidates if candidate.truncated)

    return SearchDataSummary(
        games_requested=games,
        game_offset=game_offset,
        shard_index=shard_index,
        shard_count=shard_count,
        games_started=games_started,
        decisions=decisions,
        searched_decisions=searched,
        changed_decisions=changed,
        candidate_probes=probes,
        probe_errors=probe_errors,
        truncated_rollouts=truncated,
        output_path=str(output_path.as_posix()),
        trace_path=str(trace_path.as_posix()),
        append=append,
        errors=errors,
        timeouts=timeouts,
    )


def phase5_absolute_game_index(
    local_game_index: int,
    *,
    game_offset: int = 0,
    shard_index: int = 0,
    shard_count: int = 1,
) -> int:
    if local_game_index < 0:
        raise ValueError("local_game_index must be non-negative.")
    if game_offset < 0:
        raise ValueError("game_offset must be non-negative.")
    if shard_count <= 0:
        raise ValueError("shard_count must be positive.")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count.")
    return game_offset + shard_index + local_game_index * shard_count


def phase5_matchup_for_game_index(
    matchups: Sequence[tuple[Any, Any]],
    absolute_game_index: int,
) -> tuple[Any, Any, bool]:
    if not matchups:
        raise ValueError("Phase 5 search generation needs at least one matchup.")
    if absolute_game_index < 0:
        raise ValueError("absolute_game_index must be non-negative.")
    our_deck, benchmark_deck = matchups[absolute_game_index % len(matchups)]
    return our_deck, benchmark_deck, absolute_game_index % 2 == 0


def merge_search_data(
    *,
    decision_inputs: Sequence[str | Path],
    trace_inputs: Sequence[str | Path],
    output_path: Path,
    trace_path: Path,
    manifest_path: Path | None = None,
) -> SearchMergeSummary:
    decision_paths = _expand_input_paths(decision_inputs)
    trace_paths = _expand_input_paths(trace_inputs)
    decision_records = _merge_jsonl_paths(decision_paths, output_path)
    trace_records = _merge_jsonl_paths(trace_paths, trace_path)
    summary = SearchMergeSummary(
        decision_files=len(decision_paths),
        trace_files=len(trace_paths),
        decision_records=decision_records,
        trace_records=trace_records,
        output_path=str(output_path.as_posix()),
        trace_path=str(trace_path.as_posix()),
        manifest_path=str(manifest_path.as_posix()) if manifest_path is not None else None,
        input_paths=[path.as_posix() for path in decision_paths],
        trace_input_paths=[path.as_posix() for path in trace_paths],
    )
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(summary.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return summary


def sample_hidden_state(
    observation: Any,
    *,
    own_deck_ids: Sequence[int],
    opponent_deck_ids: Sequence[int],
    card_by_id: dict[int, Any],
) -> HiddenStateSample:
    current = _get(observation, "current")
    select = _get(observation, "select")
    players = list(_get(current, "players", []) or [])
    your_index = int(_get(current, "yourIndex", 0) or 0)
    opponent_index = 1 - your_index
    mine = players[your_index] if 0 <= your_index < len(players) else None
    opponent = players[opponent_index] if 0 <= opponent_index < len(players) else None

    own_remaining = _remaining_deck_cards(
        own_deck_ids,
        _visible_card_ids_for_player(current, your_index),
    )
    opponent_remaining = _remaining_deck_cards(
        opponent_deck_ids,
        _visible_card_ids_for_player(current, opponent_index),
    )
    own_pool = list(own_remaining)
    opponent_pool = list(opponent_remaining)
    your_deck_count = int(_get(mine, "deckCount", 0) or 0)
    your_prize_count = len(list(_get(mine, "prize", []) or []))
    opponent_deck_count = int(_get(opponent, "deckCount", 0) or 0)
    opponent_prize_count = len(list(_get(opponent, "prize", []) or []))
    opponent_hand_count = int(_get(opponent, "handCount", 0) or 0)

    if _get(select, "deck") is not None:
        your_deck = []
    else:
        your_deck = _take_from_pool(own_pool, your_deck_count, fallback=own_deck_ids)
    your_prize = _known_or_sampled_zone(
        _get(mine, "prize", []),
        own_pool,
        your_prize_count,
        fallback=own_deck_ids,
    )
    opponent_deck = _take_from_pool(
        opponent_pool,
        opponent_deck_count,
        fallback=opponent_deck_ids,
    )
    opponent_prize = _known_or_sampled_zone(
        _get(opponent, "prize", []),
        opponent_pool,
        opponent_prize_count,
        fallback=opponent_deck_ids,
    )
    opponent_hand = _known_or_sampled_zone(
        _get(opponent, "hand", []),
        opponent_pool,
        opponent_hand_count,
        fallback=opponent_deck_ids,
    )
    opponent_active = _opponent_facedown_active(
        opponent,
        opponent_pool,
        card_by_id=card_by_id,
    )
    return HiddenStateSample(
        your_deck=your_deck,
        your_prize=your_prize,
        opponent_deck=opponent_deck,
        opponent_prize=opponent_prize,
        opponent_hand=opponent_hand,
        opponent_active=opponent_active,
    )


def _load_search_api(sample_dir: Path) -> tuple[Any, Any, Any]:
    previous_path = _with_sample_submission_on_path(sample_dir)
    try:
        from cg.api import search_begin, search_end, search_step
    finally:
        sys.path = previous_path
    return search_begin, search_step, search_end


def _expand_input_paths(inputs: Sequence[str | Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for value in inputs:
        pattern = str(value)
        matches = [Path(match) for match in glob.glob(pattern)]
        if not matches:
            candidate = Path(pattern)
            if candidate.exists():
                matches = [candidate]
        for path in sorted(matches, key=lambda item: item.as_posix()):
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
    if not paths:
        raise FileNotFoundError("No JSONL shard inputs matched the requested paths or globs.")
    return paths


def _merge_jsonl_paths(paths: Sequence[Path], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for path in paths:
            with path.open("r", encoding="utf-8") as source:
                for line in source:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    output.write(stripped + "\n")
                    records += 1
    return records


def _candidate_evaluations(
    frame: DecisionFrame,
    baseline: Sequence[int],
    *,
    top_k: int,
) -> list[CandidateEvaluation]:
    ranked = sorted(
        frame.legal_options,
        key=lambda action: (action.rule_score, -action.index),
        reverse=True,
    )
    ordered_indices: list[int] = []
    for index in baseline:
        if 0 <= index < len(frame.legal_options) and index not in ordered_indices:
            ordered_indices.append(int(index))
    for action in ranked:
        if len(ordered_indices) >= top_k:
            break
        if action.index not in ordered_indices:
            ordered_indices.append(action.index)

    candidates: list[CandidateEvaluation] = []
    for index in ordered_indices:
        action = frame.legal_options[index]
        candidates.append(
            CandidateEvaluation(
                indices=[index],
                option_index=index,
                option_type=action.option_type,
                card_name=action.card_name,
                attack_id=action.attack_id,
                rule_score=action.rule_score,
                rule_rank=action.rule_rank,
            )
        )
    return candidates


def _score_candidates(candidates: Sequence[CandidateEvaluation], config: RootSearchConfig) -> None:
    successful = [candidate for candidate in candidates if candidate.error is None]
    tactical_score_norm = _normalized_candidate_values(successful, "tactical_score")
    leaf_value_norm = _normalized_candidate_values(successful, "leaf_state_value")
    rule_norm = _normalized_candidate_values(successful, "rule_score")
    policy_norm = _normalized_candidate_values(successful, "policy_score")
    q_norm = _normalized_candidate_values(successful, "neural_action_value")
    tactical_norm = _normalized_candidate_values(successful, "neural_tactical_score")
    for candidate in candidates:
        if candidate.error is not None:
            candidate.tactical_score_prior = 0.0
            candidate.tactical_score_component = candidate.tactical_score
            candidate.leaf_state_value_prior = 0.0
            candidate.leaf_state_value_score = 0.0
            candidate.rule_prior = 0.0
            candidate.policy_prior = 0.0
            candidate.neural_action_value_prior = 0.0
            candidate.neural_tactical_prior = 0.0
            candidate.prior_score = 0.0
            candidate.combined_score = candidate.tactical_score
            continue
        candidate.tactical_score_prior = tactical_score_norm.get(id(candidate), 0.0)
        candidate.leaf_state_value_prior = leaf_value_norm.get(id(candidate), 0.0)
        candidate.rule_prior = rule_norm.get(id(candidate), 0.0)
        candidate.policy_prior = policy_norm.get(id(candidate), 0.0)
        candidate.neural_action_value_prior = q_norm.get(id(candidate), 0.0)
        candidate.neural_tactical_prior = tactical_norm.get(id(candidate), 0.0)
        tactical_input = (
            candidate.tactical_score_prior
            if config.normalize_tactical_score
            else candidate.tactical_score
        )
        candidate.tactical_score_component = (
            config.tactical_score_weight * tactical_input
        )
        candidate.leaf_state_value_score = (
            config.leaf_state_value_weight * candidate.leaf_state_value_prior
        )
        candidate.prior_score = (
            config.rule_prior_weight * candidate.rule_prior
            + config.policy_prior_weight * candidate.policy_prior
            + config.neural_action_value_weight * candidate.neural_action_value_prior
            + config.neural_tactical_weight * candidate.neural_tactical_prior
        )
        candidate.combined_score = (
            candidate.tactical_score_component
            + candidate.leaf_state_value_score
            + candidate.prior_score
        )


def _normalized_candidate_values(
    candidates: Sequence[CandidateEvaluation],
    field_name: str,
) -> dict[int, float]:
    values = [float(getattr(candidate, field_name, 0.0) or 0.0) for candidate in candidates]
    if not values:
        return {}
    low = min(values)
    high = max(values)
    if high == low:
        return {id(candidate): 0.0 for candidate in candidates}
    return {
        id(candidate): (float(getattr(candidate, field_name, 0.0) or 0.0) - low) / (high - low)
        for candidate in candidates
    }


def _best_candidate_indices(
    candidates: Sequence[CandidateEvaluation],
    baseline: Sequence[int],
) -> list[int]:
    successful = [candidate for candidate in candidates if candidate.error is None]
    if not successful:
        return list(baseline)
    best = max(
        successful,
        key=lambda candidate: (
            candidate.combined_score,
            -candidate.rule_rank,
            -(candidate.option_index if candidate.option_index is not None else 9999),
        ),
    )
    return list(best.indices)


def _tactical_score(
    root: dict[str, float],
    end: dict[str, float],
    current: Any,
    *,
    player_index: int,
    config: RootSearchConfig,
    truncated: bool,
) -> float:
    result = _get(current, "result", -1)
    if result is not None and int(result) in {0, 1}:
        return config.terminal_win_score if int(result) == player_index else config.terminal_loss_score
    prizes_taken = root["my_prizes"] - end["my_prizes"]
    opponent_prizes_taken = root["opponent_prizes"] - end["opponent_prizes"]
    damage_delta = end["opponent_damage"] - root["opponent_damage"]
    self_damage_delta = end["my_damage"] - root["my_damage"]
    bench_delta = end["my_bench_count"] - root["my_bench_count"]
    hand_delta = end["my_hand_count"] - root["my_hand_count"]
    score = (
        config.prize_weight * prizes_taken
        - config.opponent_prize_weight * opponent_prizes_taken
        + config.damage_weight * damage_delta
        - config.self_damage_weight * self_damage_delta
        + config.setup_weight * bench_delta
        + config.hand_weight * hand_delta
    )
    if truncated:
        score -= config.truncated_penalty
    return float(score)


def _board_metrics(board: dict[str, Any], *, player_index: int) -> dict[str, float]:
    return {
        "my_prizes": float(board.get("my_prizes", 0) or 0),
        "opponent_prizes": float(board.get("opponent_prizes", 0) or 0),
        "my_damage": _total_damage(board.get("my_active_card"))
        + sum(_total_damage(card) for card in list(board.get("my_bench_cards", []) or [])),
        "opponent_damage": _total_damage(board.get("opponent_active_card"))
        + sum(_total_damage(card) for card in list(board.get("opponent_bench_cards", []) or [])),
        "my_bench_count": float(board.get("my_bench_count", 0) or 0),
        "my_hand_count": float(board.get("my_hand_count", 0) or 0),
    }


def _total_damage(card: Any) -> float:
    if not isinstance(card, dict) or not card:
        return 0.0
    hp_ratio = float(card.get("hp_ratio", 1.0) or 0.0)
    max_hp = float(card.get("max_hp", 0.0) or 0.0)
    damage = card.get("damage")
    if damage is not None:
        return float(damage)
    return max(0.0, (1.0 - hp_ratio) * max_hp)


def _visible_card_ids_for_player(current: Any, player_index: int) -> list[int]:
    players = list(_get(current, "players", []) or [])
    player = players[player_index] if 0 <= player_index < len(players) else None
    ids: list[int] = []
    if player is not None:
        for pokemon in list(_get(player, "active", []) or []):
            ids.extend(_pokemon_card_ids(pokemon))
        for pokemon in list(_get(player, "bench", []) or []):
            ids.extend(_pokemon_card_ids(pokemon))
        for card in list(_get(player, "discard", []) or []):
            ids.extend(_card_id_list(card))
        hand = _get(player, "hand", None)
        if hand is not None:
            for card in list(hand or []):
                ids.extend(_card_id_list(card))
        for card in list(_get(player, "prize", []) or []):
            ids.extend(_card_id_list(card))
    for card in list(_get(current, "stadium", []) or []):
        if int(_get(card, "playerIndex", -1) or -1) == player_index:
            ids.extend(_card_id_list(card))
    looking = _get(current, "looking", None)
    if looking is not None:
        for card in list(looking or []):
            if int(_get(card, "playerIndex", -1) or -1) == player_index:
                ids.extend(_card_id_list(card))
    return ids


def _pokemon_card_ids(pokemon: Any) -> list[int]:
    if pokemon is None:
        return []
    ids = _card_id_list(pokemon)
    for area_name in ("energyCards", "tools", "preEvolution"):
        for card in list(_get(pokemon, area_name, []) or []):
            ids.extend(_card_id_list(card))
    return ids


def _card_id_list(card: Any) -> list[int]:
    if card is None:
        return []
    card_id = _get(card, "id", None)
    if card_id is None:
        card_id = _get(card, "cardId", None)
    return [int(card_id)] if card_id is not None else []


def _remaining_deck_cards(deck_ids: Sequence[int], visible_ids: Sequence[int]) -> list[int]:
    counts = Counter(int(card_id) for card_id in deck_ids)
    for card_id in visible_ids:
        if counts[int(card_id)] > 0:
            counts[int(card_id)] -= 1
    remaining: list[int] = []
    for card_id in deck_ids:
        if counts[int(card_id)] > 0:
            remaining.append(int(card_id))
            counts[int(card_id)] -= 1
    return remaining


def _take_cards(cards: Sequence[int], count: int) -> list[int]:
    if count <= 0:
        return []
    if len(cards) >= count:
        return [int(card_id) for card_id in cards[:count]]
    if not cards:
        return []
    padded = list(cards)
    while len(padded) < count:
        padded.append(int(cards[len(padded) % len(cards)]))
    return [int(card_id) for card_id in padded[:count]]


def _take_from_pool(pool: list[int], count: int, *, fallback: Sequence[int] = ()) -> list[int]:
    if count <= 0:
        return []
    selected = [int(card_id) for card_id in pool[:count]]
    del pool[: min(count, len(pool))]
    if len(selected) >= count:
        return selected
    selected.extend(_take_cards(fallback or selected, count - len(selected)))
    return selected[:count]


def _known_or_sampled_zone(
    zone: Any,
    remaining: list[int],
    count: int,
    *,
    fallback: Sequence[int] = (),
) -> list[int]:
    known: list[int] = []
    for card in list(zone or []):
        ids = _card_id_list(card)
        if ids:
            known.append(ids[0])
    if len(known) >= count:
        return known[:count]
    sampled = _take_from_pool(remaining, count - len(known), fallback=fallback)
    return known + sampled


def _opponent_facedown_active(
    opponent: Any,
    remaining: Sequence[int],
    *,
    card_by_id: dict[int, Any],
) -> list[int]:
    active = list(_get(opponent, "active", []) or [])
    if not active or active[0] is not None:
        return []
    for card_id in remaining:
        card = card_by_id.get(int(card_id))
        if card is not None and _card_type_name(card) == "POKEMON":
            return [int(card_id)]
    return [int(remaining[0])] if remaining else []


def _replace_frame_selection(
    frame: DecisionFrame,
    selected_indices: Sequence[int],
    metadata: dict[str, Any],
) -> DecisionFrame:
    return DecisionFrame(
        select_type=frame.select_type,
        context=frame.context,
        min_count=frame.min_count,
        max_count=frame.max_count,
        target_count=frame.target_count,
        legal_options=frame.legal_options,
        rule_selected_indices=list(selected_indices),
        board=frame.board,
        board_image=frame.board_image,
        reward_metadata=frame.reward_metadata | metadata,
        schema_version=frame.schema_version,
    )


def _summarize_board_for_player(
    current: Any,
    *,
    player_index: int,
    card_by_id: dict[int, Any],
) -> dict[str, Any]:
    return summarize_board(
        _PerspectiveCurrent(current, player_index),
        card_by_id=card_by_id,
    )


class _PerspectiveCurrent:
    def __init__(self, current: Any, player_index: int) -> None:
        self._current = current
        self.yourIndex = player_index

    def __getattr__(self, name: str) -> Any:
        return getattr(self._current, name)


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


def _result_metadata(result: BattleResult, *, player_index: int) -> dict[str, Any]:
    return {
        "player_index": player_index,
        "winner": result.winner,
        "leader": result.leader,
        "finished": result.finished,
        "prize_counts": list(result.prize_counts) if result.prize_counts is not None else None,
        "error": result.error,
    }


def _append_trace_jsonl(trace: RootSearchDecisionTrace, path: Path) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(trace.to_dict(), sort_keys=True) + "\n")


def _option_count(observation: Any) -> int:
    select = _get(observation, "select")
    return len(list(_get(select, "option", []) or [])) if select is not None else 0


def _valid_indices(values: Sequence[int], size: int) -> list[int]:
    output: list[int] = []
    for value in values:
        index = int(value)
        if 0 <= index < size and index not in output:
            output.append(index)
    return output


def _is_terminal(current: Any) -> bool:
    result = _get(current, "result", -1)
    return result is not None and int(result) != -1


def _turn_has_ended(current: Any, root_turn: int) -> bool:
    return int(_get(current, "turn", root_turn) or root_turn) != root_turn


def _hidden_counts(hidden: HiddenStateSample | None) -> dict[str, int]:
    if hidden is None:
        return {}
    return {
        "your_deck": len(hidden.your_deck),
        "your_prize": len(hidden.your_prize),
        "opponent_deck": len(hidden.opponent_deck),
        "opponent_prize": len(hidden.opponent_prize),
        "opponent_hand": len(hidden.opponent_hand),
        "opponent_active": len(hidden.opponent_active),
    }
