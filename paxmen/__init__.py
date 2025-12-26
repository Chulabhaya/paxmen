"""PaxMen: A JAX-based multi-agent reinforcement learning environment.

This module provides a cooperative dot-eating environment in a hub-and-corridors maze
for 4-6 agents, implemented using JAX for efficient JIT compilation and vectorization.
"""

from paxmen.paxmen_env import (
    PaxMen,
    PaxMenCTRolloutManager,
    State,
    visualize_paxmen_state,
)

__all__ = [
    "PaxMen",
    "PaxMenCTRolloutManager",
    "State",
    "visualize_paxmen_state",
]
