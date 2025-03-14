"""CNN backbone networks."""

import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence

class CNN(nn.Module):
    """CNN module for processing visual observations"""
    features: Sequence[int]  # Number of features per layer
    kernel_sizes: Sequence[Sequence[int]]  # Kernel sizes per layer
    dense_features: int  # Features in final dense layer
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        act_fn = nn.relu if self.activation == "relu" else nn.tanh
        
        # Apply convolution layers
        for features, kernel_size in zip(self.features, self.kernel_sizes):
            x = nn.Conv(
                features=features,
                kernel_size=kernel_size,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0)
            )(x)
            x = act_fn(x)
        
        # Flatten output
        x = x.reshape((x.shape[0], -1))
        
        # Final dense layer
        x = nn.Dense(
            features=self.dense_features,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = act_fn(x)
        
        return x 