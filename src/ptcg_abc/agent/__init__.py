from ptcg_abc.agent.random_agent import RandomAgent
from ptcg_abc.agent.rule_based import RuleBasedAgent, score_legal_options, select_option_indices

__all__ = [
    "HybridRlAgent",
    "RandomAgent",
    "RuleBasedAgent",
    "score_legal_options",
    "select_option_indices",
]


def __getattr__(name: str):
    if name == "HybridRlAgent":
        from ptcg_abc.agent.hybrid_rl import HybridRlAgent

        return HybridRlAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
