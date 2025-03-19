# backbones/rnn.py

"""RNN backbone networks."""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, Tuple, Optional, Union
import functools


class ScannedRNN(nn.Module):
    hidden_size: int

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        ins, resets = x
        new_carry = self.initialize_carry(ins.shape[0])

        carry = jnp.where(resets[:, None], new_carry, carry)

        gru_cell = nn.GRUCell(
            features=self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )
        new_carry, y = gru_cell(carry, ins)
        return new_carry, y

    def initialize_carry(self, batch_size):
        # No submodule initialization here; just zeros initialization.
        return jnp.zeros((batch_size, self.hidden_size))



class RNN(nn.Module):
    """Multi-layer RNN backbone for processing sequential data."""

    hidden_sizes: Sequence[int]
    dense_features: int
    activation: str = "tanh"

    def setup(self):
        self.rnn_layers = [
            ScannedRNN(hidden_size=hidden_size)
            for hidden_size in self.hidden_sizes
        ]
        self.layer_norm = nn.LayerNorm()
        self.dense = nn.Dense(
            features=self.dense_features,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )

    def __call__(self, x, carry=None):
        obs, resets = x
        
        if carry is None:
            carry = [
                layer.initialize_carry(obs.shape[0]) for layer in self.rnn_layers
            ]

        new_carries = []
        features = obs
        for layer, state in zip(self.rnn_layers, carry):
            state, features = layer(state, (features, resets))
            features = self.layer_norm(features)
            new_carries.append(state)

        features = self.dense(features)
        features = nn.relu(features) if self.activation == "relu" else nn.tanh(features)

        return features, new_carries