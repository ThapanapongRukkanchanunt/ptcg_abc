from ptcg_abc.agent.random_agent import RandomAgent
from ptcg_abc.agent.rule_based import RuleBasedAgent, score_legal_options, select_option_indices

__all__ = [
    "HybridRlAgent",
    "Phase5SearchPolicyAgent",
    "Phase5SymbolicPolicyAgent",
    "RandomAgent",
    "RuleBasedAgent",
    "score_legal_options",
    "select_option_indices",
]


def __getattr__(name: str):
    if name == "HybridRlAgent":
        from ptcg_abc.agent.hybrid_rl import HybridRlAgent

        return HybridRlAgent
    if name == "Phase5SymbolicPolicyAgent":
        from ptcg_abc.agent.phase5_symbolic import Phase5SymbolicPolicyAgent

        return Phase5SymbolicPolicyAgent
    if name == "Phase5SearchPolicyAgent":
        from ptcg_abc.agent.phase5_search import Phase5SearchPolicyAgent

        return Phase5SearchPolicyAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
