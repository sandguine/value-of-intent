"""Models that use backbone networks."""

from .actor_critic import ActorCritic
from .cpc_module import CPCModule

__all__ = ["ActorCritic", "CPCModule"]
