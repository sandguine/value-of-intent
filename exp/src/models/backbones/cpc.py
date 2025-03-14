import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from flax.linen.dtypes import DTypes
from distrax import Categorical
from typing import Sequence

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
        self.gru = nn.GRUCell()
        self.projection_head = nn.Dense(self.projection_dim, kernel_init=orthogonal())

        self.actor = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))
        self.critic = nn.Dense(1, kernel_init=orthogonal())

        # Prediction heads for future latents (W_k in CPC)
        self.future_predictors = [
            nn.Dense(self.projection_dim, kernel_init=orthogonal())
            for _ in range(self.future_steps)
        ]

    def __call__(self, x, h_state, return_features=False):
        z = self.encoder(x)
        h_state, h_next = self.gru(z, h_state)
        projected = self.projection_head(h_state)

        logits = self.actor(h_state)
        value = self.critic(h_state).squeeze(-1)

        pi = distrax.Categorical(logits=logits)

        return pi, value, projected, h_next