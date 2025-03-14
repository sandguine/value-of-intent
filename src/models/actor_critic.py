"""Actor-Critic model that can use either CNN or FF backbone."""

import jax.numpy as jnp
import flax.linen as nn
import distrax
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, Dict, Union, Type

from .backbones.cnn import CNN
from .backbones.ff import FeedForward
from .backbones.rnn import RNN

class ActorCritic(nn.Module):
    """Actor-Critic architecture that can use either CNN or FF backbone"""
    action_dim: int
    backbone_cls: Type[Union[CNN, FeedForward]]
    backbone_config: Dict
    
    def setup(self):
        # Shared backbone network
        self.backbone = self.backbone_cls(**self.backbone_config)
        
        # Actor head
        self.actor = nn.Dense(
            features=self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )
        
        # Critic head
        self.critic = nn.Dense(
            features=1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )
    
    def __call__(self, x, return_features=False):
        """Forward pass through the actor-critic model.
        
        Args:
            x: Input observations
            return_features: If True, also return backbone features for CPC
            
        Returns:
            pi: Categorical distribution over actions
            value: Value function estimates
            features: (Optional) Backbone features if return_features=True
        """
        # Get features from backbone
        features = self.backbone(x)
        
        # Actor head: policy distribution
        actor_out = self.actor(features)
        pi = distrax.Categorical(logits=actor_out)
        
        # Critic head: value function
        critic_out = self.critic(features)
        value = jnp.squeeze(critic_out, axis=-1)
        
        if return_features:
            return pi, value, features
        return pi, value 