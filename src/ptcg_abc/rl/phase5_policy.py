from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
    _TORCH_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on local torch install health.
    torch = None
    nn = None
    TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = exc

PHASE5_POLICY_CHECKPOINT_FORMAT = "ptcg_abc_phase5_alphastar_turn_policy_v1"


class Phase5PolicyUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class Phase5PolicyConfig:
    global_dim: int
    entity_dim: int
    action_dim: int
    d_model: int = 128
    nhead: int = 4
    transformer_layers: int = 2
    feedforward_dim: int = 256
    turn_hidden_dim: int = 128
    dropout: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


if TORCH_AVAILABLE:
    class AlphaStarTurnPolicy(nn.Module):
        """Small AlphaStar-inspired legal-action model for Phase 5.

        The model encodes symbolic entities with a transformer core and scores the
        current legal action set. The policy is autoregressive at turn level: a
        previous-action sequence is encoded by a GRU and conditions the next-action
        logits. The simulator still owns legality and supplies the legal choices.
        """

        def __init__(self, config: Phase5PolicyConfig) -> None:
            super().__init__()
            self.config = config
            self.global_encoder = nn.Sequential(
                nn.Linear(config.global_dim, config.d_model),
                nn.GELU(),
                nn.LayerNorm(config.d_model),
            )
            self.entity_encoder = nn.Sequential(
                nn.Linear(config.entity_dim, config.d_model),
                nn.GELU(),
                nn.LayerNorm(config.d_model),
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
            )
            self.entity_core = nn.TransformerEncoder(
                encoder_layer,
                num_layers=config.transformer_layers,
            )
            self.action_encoder = nn.Sequential(
                nn.Linear(config.action_dim, config.d_model),
                nn.GELU(),
                nn.LayerNorm(config.d_model),
            )
            self.turn_core = nn.GRU(
                input_size=config.d_model,
                hidden_size=config.turn_hidden_dim,
                batch_first=True,
            )
            self.policy_head = nn.Sequential(
                nn.Linear(config.d_model * 2 + config.turn_hidden_dim, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 1),
            )
            self.value_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 1),
            )

        def forward(
            self,
            global_x: Any,
            entity_x: Any,
            entity_mask: Any,
            action_x: Any,
            action_mask: Any,
            previous_action_x: Any | None = None,
            previous_action_mask: Any | None = None,
        ) -> dict[str, Any]:
            state_embedding = self.encode_state(global_x, entity_x, entity_mask)
            action_embedding = self.action_encoder(action_x)
            turn_embedding = self.encode_turn_context(
                previous_action_x,
                previous_action_mask,
                batch_size=action_x.shape[0],
                device=action_x.device,
            )
            expanded_state = state_embedding.unsqueeze(1).expand(-1, action_x.shape[1], -1)
            expanded_turn = turn_embedding.unsqueeze(1).expand(-1, action_x.shape[1], -1)
            logits = self.policy_head(
                torch.cat([expanded_state, action_embedding, expanded_turn], dim=-1)
            ).squeeze(-1)
            logits = logits.masked_fill(action_mask <= 0, -1.0e9)
            return {
                "action_logits": logits,
                "state_value": self.value_head(state_embedding).squeeze(-1),
                "state_embedding": state_embedding,
                "turn_embedding": turn_embedding,
            }

        def encode_state(self, global_x: Any, entity_x: Any, entity_mask: Any) -> Any:
            global_token = self.global_encoder(global_x).unsqueeze(1)
            entity_tokens = self.entity_encoder(entity_x)
            tokens = torch.cat([global_token, entity_tokens], dim=1)
            global_mask = torch.ones(
                (entity_mask.shape[0], 1),
                dtype=entity_mask.dtype,
                device=entity_mask.device,
            )
            token_mask = torch.cat([global_mask, entity_mask], dim=1)
            encoded = self.entity_core(tokens, src_key_padding_mask=token_mask <= 0)
            return encoded[:, 0, :]

        def encode_turn_context(
            self,
            previous_action_x: Any | None,
            previous_action_mask: Any | None = None,
            *,
            batch_size: int,
            device: Any,
        ) -> Any:
            if previous_action_x is None or previous_action_x.shape[1] == 0:
                return torch.zeros(batch_size, self.config.turn_hidden_dim, device=device)
            previous_embedding = self.action_encoder(previous_action_x)
            output, hidden = self.turn_core(previous_embedding)
            if previous_action_mask is not None:
                lengths = previous_action_mask.sum(dim=1).long().clamp(min=0)
                if int(lengths.max().item()) == 0:
                    return torch.zeros(batch_size, self.config.turn_hidden_dim, device=device)
                gather_index = (lengths - 1).clamp(min=0)
                gathered = output[torch.arange(batch_size, device=device), gather_index]
                return gathered * (lengths > 0).float().unsqueeze(1)
            return hidden[-1]

        def checkpoint_payload(self) -> dict[str, Any]:
            return {
                "format": PHASE5_POLICY_CHECKPOINT_FORMAT,
                "config": self.config.to_dict(),
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in self.state_dict().items()
                },
            }

else:

    class AlphaStarTurnPolicy:
        def __init__(self, config: Phase5PolicyConfig) -> None:
            reason = f" Import failed with: {_TORCH_IMPORT_ERROR}" if _TORCH_IMPORT_ERROR else ""
            raise Phase5PolicyUnavailable(
                "PyTorch is not installed. Train Phase 5 policy models on ERAWAN "
                f"or install the rl extra locally.{reason}"
            )


def make_phase5_policy_config(
    *,
    global_dim: int,
    entity_dim: int,
    action_dim: int,
    d_model: int = 128,
) -> Phase5PolicyConfig:
    return Phase5PolicyConfig(
        global_dim=global_dim,
        entity_dim=entity_dim,
        action_dim=action_dim,
        d_model=d_model,
    )


__all__ = [
    "AlphaStarTurnPolicy",
    "PHASE5_POLICY_CHECKPOINT_FORMAT",
    "Phase5PolicyConfig",
    "Phase5PolicyUnavailable",
    "TORCH_AVAILABLE",
    "make_phase5_policy_config",
]
