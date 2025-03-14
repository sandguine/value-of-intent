"""Feedforward backbone networks."""

import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, Dict

class FeedForward(nn.Module):
    """Feedforward network backbone"""
    hidden_layers: Sequence[int]  # Features in hidden layers
    activation: str = "tanh"

    def setup(self):
        self.act_fn = nn.relu if self.activation == "relu" else nn.tanh
        
        # Create layers
        self.layers = []
        for features in self.hidden_layers:
            self.layers.append(
                nn.Dense(
                    features,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0)
                )
            )

    @nn.compact
    def __call__(self, x):
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.reshape((x.shape[0], -1))
            
        # Forward pass through layers
        for layer in self.layers:
            x = layer(x)
            x = self.act_fn(x)
            
        return x 