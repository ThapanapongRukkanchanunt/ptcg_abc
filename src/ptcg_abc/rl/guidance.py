from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ptcg_abc.rl.records import ActionFrame, DecisionFrame


@dataclass(frozen=True)
class GuidanceDecision:
    forced_indices: set[int] = field(default_factory=set)
    blocked_indices: set[int] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)


def evaluate_guidance(
    frame: DecisionFrame,
    *,
    enabled_rules: Iterable[str] | None = None,
) -> GuidanceDecision:
    enabled = set(enabled_rules or default_guidance_rules())
    forced: set[int] = set()
    blocked: set[int] = set()
    notes: list[str] = []
    if "force_prize_attacks" in enabled:
        attack = _best_clear_prize_attack(frame.legal_options)
        if attack is not None:
            forced.add(attack.index)
            notes.append("force_prize_attacks")
    if "block_bad_optional" in enabled and frame.min_count == 0:
        for action in frame.legal_options:
            if _looks_like_bad_optional(action, frame):
                blocked.add(action.index)
        if blocked:
            notes.append("block_bad_optional")
    if "prefer_rule_top" in enabled and not forced:
        best = max(frame.legal_options, key=lambda action: (action.rule_score, -action.index), default=None)
        if best is not None and best.rule_score > 50000:
            forced.add(best.index)
            notes.append("prefer_rule_top")
    return GuidanceDecision(
        forced_indices=forced - blocked,
        blocked_indices=blocked,
        notes=notes,
    )


def default_guidance_rules() -> tuple[str, ...]:
    return (
        "force_prize_attacks",
        "block_bad_optional",
        "prefer_rule_top",
    )


def _best_clear_prize_attack(actions: list[ActionFrame]) -> ActionFrame | None:
    attacks = [
        action
        for action in actions
        if action.option_type == "ATTACK" and action.rule_score >= 12000
    ]
    return max(attacks, key=lambda action: (action.rule_score, -action.index), default=None)


def _looks_like_bad_optional(action: ActionFrame, frame: DecisionFrame) -> bool:
    if action.rule_score > 0:
        return False
    if frame.context in {"SETUP_BENCH_POKEMON", "TO_BENCH", "TO_FIELD", "NOT_MOVE"}:
        return action.option_type in {"CARD", "PLAY", "ABILITY"}
    if frame.context in {"DISCARD", "TO_DECK", "TO_DECK_BOTTOM"}:
        return True
    return action.option_type in {"DISCARD", "RETREAT"}
