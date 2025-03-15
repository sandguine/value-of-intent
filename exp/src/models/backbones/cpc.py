import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from flax.linen.dtypes import DTypes
from distrax import Categorical
from typing import Sequence, Dict, Type

# Follow CPC Original Paper (Oord et al., 2018) closely
class CPCNetwork(nn.Module):
    action_dim: int
    backbone_cls: Type[nn.Module]
    backbone_config: Dict
    projection_dim: int = 128
    gru_hidden_dim: int = 256
    future_steps: int = 3  # Predict 3 steps ahead
    temperature: float = 0.1

    def setup(self):
        self.encoder = self.backbone_cls(**self.backbone_config)
        self.gru = nn.GRUCell(features=self.gru_hidden_dim)
        self.projection_head = nn.Dense(self.projection_dim, kernel_init=orthogonal(1.0))

        self.actor = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))
        self.critic = nn.Dense(1, kernel_init=orthogonal(1.0))

        # Prediction heads for future states
        self.future_predictors = [
            nn.Dense(self.projection_dim, kernel_init=orthogonal(1.0))
            for _ in range(self.future_steps)
        ]

    def __call__(self, x, h_state=None, return_features=False):
        # Encode current observation
        z = self.encoder(x)
        
        # Initialize GRU state if None
        if h_state is None:
            h_state = self.gru.initialize_carry(jax.random.PRNGKey(0), z.shape[0], self.gru_hidden_dim)
        
        # Update GRU state
        h_next = self.gru(h_state, z)
        
        # Project to lower dimension for CPC
        projected = self.projection_head(h_next)
        
        # Policy and value outputs
        logits = self.actor(h_next)
        pi = Categorical(logits=logits)
        value = self.critic(h_next).squeeze(-1)

        if return_features:
            return pi, value, z, projected, h_next
        
        return pi, value