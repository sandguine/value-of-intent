"""Contrastive Predictive Coding network implementation."""

from typing import Dict, Type, Union
import jax.numpy as jnp
import flax.linen as nn
import distrax
from flax.linen.initializers import constant, orthogonal

from .base import BaseCPCNetwork
from .backbones.cnn import CNN
from .backbones.ff import FeedForward
from .backbones.rnn import RNN

class CPCNetwork(BaseCPCNetwork):
    """CPC network with contrastive predictive coding."""
    
    def __init__(
        self,
        action_dim: int,
        backbone_cls: Type[Union[CNN, FeedForward, RNN]],
        backbone_config: Dict,
        projection_dim: int = 128,
        gru_hidden_dim: int = 256,
        future_steps: int = 3,
        temperature: float = 0.1,
        activation: str = "tanh"
    ):
        super().__init__(
            action_dim=action_dim,
            backbone_cls=backbone_cls,
            backbone_config=backbone_config,
            projection_dim=projection_dim,
            gru_hidden_dim=gru_hidden_dim,
            future_steps=future_steps,
            temperature=temperature,
            activation=activation
        ) 