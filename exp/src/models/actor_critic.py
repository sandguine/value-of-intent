import jax.numpy as jnp
import flax.linen as nn
import distrax
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Dict, Union, Type, Optional, Tuple

from .backbones.cnn import CNN
from .backbones.ff import FeedForward
from .backbones.rnn import RNN

class ActorCritic(nn.Module):
    action_dim: int
    backbone_cls: Type[Union[CNN, FeedForward, RNN]]
    backbone_config: Dict
    activation: str = "tanh"

    def setup(self):
        self.backbone = self.backbone_cls(**self.backbone_config)
        self.actor = nn.Dense(
            features=self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )
        self.critic = nn.Dense(
            features=1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )

    def __call__(
        self, 
        x, 
        carry=None, 
        return_features=False
    ):
        # Handle activation
        act_fn = nn.relu if self.activation == "relu" else nn.tanh
        
        # Get features from backbone
        if isinstance(self.backbone, RNN):
            features, carry = self.backbone(x, carry)
        else:
            features = self.backbone(x)

        # Actor head
        actor_out = self.actor(features)
        pi = distrax.Categorical(logits=actor_out)

        # Critic head
        critic_out = self.critic(features)
        value = jnp.squeeze(critic_out, axis=-1)

        if return_features:
            return pi, value, features
        return pi, value
