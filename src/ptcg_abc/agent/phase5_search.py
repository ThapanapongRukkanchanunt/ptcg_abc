from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

from ptcg_abc.agent.phase5_symbolic import (
    Phase5SymbolicPolicyAgent,
    _target_count,
    _valid_indices,
)
from ptcg_abc.rl.featurizer import attack_lookup, card_lookup, make_decision_frame
from ptcg_abc.rl.phase5_search import (
    PHASE5_SEARCH_SCHEMA_VERSION,
    CandidateEvaluation,
    OneTurnRootSearchAgent,
    RootSearchConfig,
    RootSearchDecisionTrace,
    _best_candidate_indices,
    _hidden_counts,
    _load_search_api,
    _score_candidates,
    sample_hidden_state,
)


class Phase5SearchPolicyAgent(Phase5SymbolicPolicyAgent):
    """Phase 5 online policy-plus-root-search agent for evaluation.

    The direct symbolic policy supplies the root prior and candidate set. The
    competition Search API then probes each candidate to the end of the current
    turn using the same tactical scoring code as Phase 5 search-data generation.
    """

    def __init__(
        self,
        deck_ids: Sequence[int],
        *,
        opponent_deck_ids: Sequence[int],
        sample_dir: Path | str,
        card_data: Sequence[Any] = (),
        attack_data: Sequence[Any] = (),
        checkpoint_path: Path | str | None = None,
        device: str | None = None,
        search_config: RootSearchConfig | None = None,
    ) -> None:
        if len(opponent_deck_ids) != 60:
            raise ValueError(
                "Phase5SearchPolicyAgent needs a 60-card opponent deck, "
                f"got {len(opponent_deck_ids)}."
            )
        super().__init__(
            deck_ids,
            card_data=card_data,
            attack_data=attack_data,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        self.opponent_deck_ids = list(opponent_deck_ids)
        self.sample_dir = Path(sample_dir)
        self.config = search_config or RootSearchConfig()
        self.card_by_id = card_lookup(card_data)
        self.attack_by_id = attack_lookup(attack_data)
        self.traces: list[RootSearchDecisionTrace] = []
        self.search_decisions = 0
        self.changed_decisions = 0
        self.search_errors = 0
        self.search_started_decisions = 0
        self.candidate_probes = 0
        self.candidate_errors = 0
        self.truncated_candidates = 0
        self.search_elapsed_seconds = 0.0
        self.max_search_elapsed_seconds = 0.0
        self._search_begin, self._search_step, self._search_end = _load_search_api(
            self.sample_dir
        )

    def act(self, observation: Any) -> list[int]:
        select = _get(observation, "select")
        if select is None:
            return list(self.deck_ids)

        state = self.state_adapter.parse(observation)
        legal_actions = self.legal_adapter.parse(observation)
        if not legal_actions:
            return self.fallback_agent.act(observation)
        self._reset_turn_context_if_needed(state)
        self.memory.observe(state)
        belief = self.memory.belief_state(state, own_deck_ids=self.deck_ids)
        encoded = self.encoder.encode(state, legal_actions, belief)
        target_count = _target_count(encoded.legal_action_mask, legal_actions)
        if target_count <= 0:
            return []

        model_outputs = self._score_model_outputs(encoded)
        scores = model_outputs["action_logits"]
        baseline_positions = self._rank_policy_positions(
            encoded,
            legal_actions,
            scores,
        )[:target_count]
        baseline = _valid_indices(
            [
                encoded.legal_action_indices[position]
                for position in baseline_positions
                if encoded.legal_action_indices[position] >= 0
            ],
            len(legal_actions),
        )
        if len(baseline) < max(0, legal_actions[0].min_count):
            baseline = self.fallback_agent.act(observation)

        frame = make_decision_frame(
            observation,
            deck_ids=self.deck_ids,
            card_by_id=self.card_by_id,
            attack_by_id=self.attack_by_id,
            selected_indices=baseline,
            reward_metadata={
                "collector": "phase5_search_eval",
                "phase5_policy_indices": list(baseline),
            },
        )
        selected = list(baseline)
        if frame is not None and self._should_search(frame):
            start = time.perf_counter()
            selected, trace = self._search_decision(
                observation,
                frame,
                baseline,
                encoded,
                legal_actions,
                scores,
                model_outputs,
            )
            elapsed = time.perf_counter() - start
            self.traces.append(trace)
            self.search_decisions += 1
            self.search_started_decisions += int(trace.search_started)
            self.changed_decisions += int(selected != baseline)
            self.search_errors += int(trace.search_error is not None)
            self.candidate_probes += len(trace.candidates)
            self.candidate_errors += sum(
                1 for candidate in trace.candidates if candidate.error is not None
            )
            self.truncated_candidates += sum(
                1 for candidate in trace.candidates if candidate.truncated
            )
            self.search_elapsed_seconds += elapsed
            self.max_search_elapsed_seconds = max(self.max_search_elapsed_seconds, elapsed)

        selected = _valid_indices(selected, len(legal_actions))
        selected_positions = self._positions_for_indices(encoded, selected)
        self._observe_selected_actions(encoded.legal_action_features, selected_positions)
        return selected

    def search_telemetry(self) -> dict[str, Any]:
        avg_seconds = (
            self.search_elapsed_seconds / self.search_decisions
            if self.search_decisions
            else 0.0
        )
        return {
            "searched_decisions": self.search_decisions,
            "search_started_decisions": self.search_started_decisions,
            "changed_decisions": self.changed_decisions,
            "change_rate": self.changed_decisions / self.search_decisions
            if self.search_decisions
            else 0.0,
            "search_errors": self.search_errors,
            "search_error_rate": self.search_errors / self.search_decisions
            if self.search_decisions
            else 0.0,
            "candidate_probes": self.candidate_probes,
            "candidate_errors": self.candidate_errors,
            "candidate_error_rate": self.candidate_errors / self.candidate_probes
            if self.candidate_probes
            else 0.0,
            "truncated_candidates": self.truncated_candidates,
            "truncated_candidate_rate": self.truncated_candidates / self.candidate_probes
            if self.candidate_probes
            else 0.0,
            "total_search_seconds": self.search_elapsed_seconds,
            "avg_search_seconds": avg_seconds,
            "max_search_seconds": self.max_search_elapsed_seconds,
        }

    def _rank_policy_positions(
        self,
        encoded: Any,
        legal_actions: Sequence[Any],
        scores: Sequence[float],
    ) -> list[int]:
        positions = [
            position
            for position, index in enumerate(encoded.legal_action_indices)
            if index >= 0 and encoded.legal_action_mask[position] > 0
        ]
        return sorted(
            positions,
            key=lambda pos: (
                scores[pos],
                (
                    legal_actions[encoded.legal_action_indices[pos]].rule_score
                    if self.use_rule_tiebreak
                    else 0.0
                ),
                -legal_actions[encoded.legal_action_indices[pos]].local_index,
            ),
            reverse=True,
        )

    def _positions_for_indices(self, encoded: Any, indices: Sequence[int]) -> list[int]:
        position_by_index = {
            action_index: position
            for position, action_index in enumerate(encoded.legal_action_indices)
            if action_index >= 0
        }
        positions: list[int] = []
        for index in indices:
            position = position_by_index.get(index)
            if position is not None:
                positions.append(position)
        return positions

    def _should_search(self, frame: Any) -> bool:
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
        frame: Any,
        baseline: list[int],
        encoded: Any,
        legal_actions: Sequence[Any],
        scores: Sequence[float],
        model_outputs: dict[str, Any] | None = None,
    ) -> tuple[list[int], RootSearchDecisionTrace]:
        candidates = self._policy_candidates(
            frame,
            baseline,
            encoded,
            legal_actions,
            scores,
            model_outputs,
        )
        search_started = False
        search_error: str | None = None
        hidden = None
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
                OneTurnRootSearchAgent._evaluate_candidate(
                    self,
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

        if search_error is not None:
            selected = list(baseline)
        else:
            _score_candidates(candidates, self.config)
            selected = _best_candidate_indices(candidates, baseline)
        trace = RootSearchDecisionTrace(
            schema_version=PHASE5_SEARCH_SCHEMA_VERSION,
            game_index=len(self.traces) + 1,
            step_index=len(self.traces) + 1,
            deck_index=0,
            deck_label="",
            opponent="",
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

    def _policy_candidates(
        self,
        frame: Any,
        baseline: Sequence[int],
        encoded: Any,
        legal_actions: Sequence[Any],
        scores: Sequence[float],
        model_outputs: dict[str, Any] | None = None,
    ) -> list[CandidateEvaluation]:
        ordered_indices: list[int] = []
        for index in baseline:
            if 0 <= index < len(frame.legal_options) and index not in ordered_indices:
                ordered_indices.append(int(index))
        for position in self._rank_policy_positions(encoded, legal_actions, scores):
            if len(ordered_indices) >= self.config.top_k:
                break
            index = int(encoded.legal_action_indices[position])
            if 0 <= index < len(frame.legal_options) and index not in ordered_indices:
                ordered_indices.append(index)

        position_by_index = {
            action_index: position
            for position, action_index in enumerate(encoded.legal_action_indices)
            if action_index >= 0
        }
        candidates: list[CandidateEvaluation] = []
        for policy_rank, index in enumerate(ordered_indices, start=1):
            action = frame.legal_options[index]
            position = position_by_index.get(index)
            policy_score = float(scores[position]) if position is not None else 0.0
            action_q = (
                float(model_outputs["action_q"][position])
                if model_outputs is not None and position is not None
                else 0.0
            )
            tactical_score = (
                float(model_outputs["tactical_score"][position])
                if model_outputs is not None and position is not None
                else 0.0
            )
            candidates.append(
                CandidateEvaluation(
                    indices=[index],
                    option_index=index,
                    option_type=action.option_type,
                    card_name=action.card_name,
                    attack_id=action.attack_id,
                    rule_score=policy_score,
                    rule_rank=policy_rank,
                    policy_score=policy_score,
                    neural_action_value=action_q,
                    neural_tactical_score=tactical_score,
                )
            )
        return candidates


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)
