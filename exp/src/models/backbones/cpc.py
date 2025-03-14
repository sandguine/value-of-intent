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

    def setup(self):
        self.encoder = self.backbone_cls(**self.backbone_config)
        self.gru = nn.GRUCell()
        self.projection_head = nn.Dense(self.projection_dim, kernel_init=orthogonal())

        self.actor = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))
        self.critic = nn.Dense(1, kernel_init=orthogonal())

    def __call__(self, x, h_state, return_features=False):
        z = self.encoder(x)
        h_state, h_next = self.gru(z, h_prev)
        projected = self.projection_head(h)

        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)

        pi = distrax.Categorical(logits=logits)

        return pi, value, projected, h_next
