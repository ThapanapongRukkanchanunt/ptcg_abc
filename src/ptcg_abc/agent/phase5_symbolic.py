from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any, Sequence

from ptcg_abc.agent.rule_based import RuleBasedAgent
from ptcg_abc.rl.phase5_adapters import GameMemory, LegalOptionAdapter, StateAdapter
from ptcg_abc.rl.phase5_encoder import Phase5SymbolicEncoder
from ptcg_abc.rl.phase5_policy import (
    AlphaStarTurnPolicy,
    PHASE5_POLICY_CHECKPOINT_FORMAT,
    Phase5PolicyConfig,
    Phase5PolicyUnavailable,
    TORCH_AVAILABLE,
)


class Phase5SymbolicPolicyAgent:
    """Direct policy agent backed by the Phase 5 symbolic torch checkpoint."""

    def __init__(
        self,
        deck_ids: Sequence[int],
        *,
        card_data: Sequence[Any] = (),
        attack_data: Sequence[Any] = (),
        checkpoint_path: Path | str | None = None,
        device: str | None = None,
        use_rule_tiebreak: bool = True,
    ) -> None:
        if len(deck_ids) != 60:
            raise ValueError(
                f"Phase5SymbolicPolicyAgent needs a 60-card deck, got {len(deck_ids)}."
            )
        if checkpoint_path is None:
            raise ValueError("Phase5SymbolicPolicyAgent requires checkpoint_path.")
        if not TORCH_AVAILABLE:
            raise Phase5PolicyUnavailable(
                "PyTorch is not installed. Evaluate Phase 5 symbolic checkpoints "
                "on ERAWAN or install the rl extra locally."
            )

        import torch

        self.deck_ids = list(deck_ids)
        self.checkpoint_path = Path(checkpoint_path)
        self.card_data = card_data
        self.attack_data = attack_data
        self.torch = torch
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.use_rule_tiebreak = use_rule_tiebreak
        self.state_adapter = StateAdapter(card_data=card_data)
        self.legal_adapter = LegalOptionAdapter(
            deck_ids=deck_ids,
            card_data=card_data,
            attack_data=attack_data,
        )
        self.memory = GameMemory()
        self._turn_key: tuple[int, int] | None = None
        self._previous_actions: list[list[float]] = []

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != PHASE5_POLICY_CHECKPOINT_FORMAT:
            raise ValueError(
                f"Unsupported Phase 5 symbolic checkpoint: {checkpoint.get('format')}"
            )
        config = Phase5PolicyConfig(**checkpoint["config"])
        encoder_config = checkpoint.get("encoder", {})
        self.max_previous_actions = int(encoder_config.get("max_previous_actions", 16))
        self.encoder = Phase5SymbolicEncoder(
            max_entities=int(encoder_config.get("max_entities", 96)),
            max_actions=int(encoder_config.get("max_actions", 128)),
        )
        self.model = AlphaStarTurnPolicy(config)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        with contextlib.redirect_stdout(io.StringIO()):
            self.fallback_agent = RuleBasedAgent(
                self.deck_ids,
                card_data=card_data,
                attack_data=attack_data,
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

        scores = self._score_encoded(encoded)
        encoded_positions = [
            position
            for position, index in enumerate(encoded.legal_action_indices)
            if index >= 0 and encoded.legal_action_mask[position] > 0
        ]
        ranked_positions = sorted(
            encoded_positions,
            key=lambda pos: (
                scores[pos],
                legal_actions[pos].rule_score if self.use_rule_tiebreak else 0.0,
                -legal_actions[pos].local_index,
            ),
            reverse=True,
        )
        selected_positions = ranked_positions[:target_count]
        selected = [
            encoded.legal_action_indices[position]
            for position in selected_positions
            if encoded.legal_action_indices[position] >= 0
        ]
        if len(selected) < max(0, legal_actions[0].min_count):
            selected = self.fallback_agent.act(observation)
        selected = _valid_indices(selected, len(legal_actions))
        self._observe_selected_actions(encoded.legal_action_features, selected_positions)
        return selected

    def _score_encoded(self, encoded: Any) -> list[float]:
        torch = self.torch
        previous_x, previous_mask = self._previous_action_tensors(
            action_dim=len(encoded.legal_action_features[0])
        )
        with torch.no_grad():
            output = self.model(
                torch.tensor([encoded.global_features], dtype=torch.float32, device=self.device),
                torch.tensor([encoded.entity_features], dtype=torch.float32, device=self.device),
                torch.tensor([encoded.entity_mask], dtype=torch.float32, device=self.device),
                torch.tensor(
                    [encoded.legal_action_features],
                    dtype=torch.float32,
                    device=self.device,
                ),
                torch.tensor([encoded.legal_action_mask], dtype=torch.float32, device=self.device),
                previous_x,
                previous_mask,
            )
        return [float(value) for value in output["action_logits"][0].detach().cpu().tolist()]

    def _previous_action_tensors(self, *, action_dim: int) -> tuple[Any, Any]:
        rows = [list(row) for row in self._previous_actions[-self.max_previous_actions :]]
        mask = [1.0] * len(rows)
        rows.extend([[0.0] * action_dim for _ in range(self.max_previous_actions - len(rows))])
        mask.extend([0.0] * (self.max_previous_actions - len(mask)))
        return (
            self.torch.tensor([rows], dtype=self.torch.float32, device=self.device),
            self.torch.tensor([mask], dtype=self.torch.float32, device=self.device),
        )

    def _reset_turn_context_if_needed(self, state: Any) -> None:
        key = (state.your_index, state.turn)
        if key != self._turn_key:
            self._turn_key = key
            self._previous_actions = []

    def _observe_selected_actions(
        self,
        action_features: Sequence[Sequence[float]],
        selected_positions: Sequence[int],
    ) -> None:
        for position in selected_positions:
            if 0 <= position < len(action_features):
                self._previous_actions.append(list(action_features[position]))
        if len(self._previous_actions) > self.max_previous_actions:
            self._previous_actions = self._previous_actions[-self.max_previous_actions :]


def _target_count(mask: Sequence[float], legal_actions: Sequence[Any]) -> int:
    if not legal_actions:
        return 0
    available = int(sum(1 for value in mask if value > 0))
    target = int(getattr(legal_actions[0], "target_count", 1) or 1)
    return max(0, min(target, available))


def _valid_indices(values: Sequence[int], size: int) -> list[int]:
    output: list[int] = []
    for value in values:
        index = int(value)
        if 0 <= index < size and index not in output:
            output.append(index)
    return output


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)
