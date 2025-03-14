"""RNN backbone networks."""

import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence

class RNN(nn.Module):
    """RNN module for processing sequential data"""
    hidden_sizes: Sequence[int]  # Hidden layer sizes
    dense_features: int  # Features in final dense layer
    activation: str = "tanh"
    
    @nn.compact
    def __call__(self, x, carry=None):
        # Expected input shape: (batch_size, seq_len, feature_dim)
        act_fn = nn.relu if self.activation == "relu" else nn.tanh
        
        # Initialize carry state if not provided
        if carry is None:
            carry = [None] * len(self.hidden_sizes)
        
        # Process sequence through RNN layers
        current_carry = []
        for i, hidden_size in enumerate(self.hidden_sizes):
            # Create and apply RNN cell
            rnn = nn.GRUCell(
                features=hidden_size,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0)
            )
            
            # Process sequence
            carry_i = carry[i]
            outputs = []
            for t in range(x.shape[1]):  # iterate over sequence length
                x_t = x[:, t, :]
                carry_i, output = rnn(carry_i, x_t)
                outputs.append(output)
            
            # Store carry state
            current_carry.append(carry_i)
            
            # Stack outputs and apply activation
            x = jnp.stack(outputs, axis=1)  # (batch_size, seq_len, hidden_size)
            x = act_fn(x)
        
        # Final dense layer on the last timestep
        x = x[:, -1, :]  # Take last timestep
        x = nn.Dense(
            features=self.dense_features,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = act_fn(x)
        
        return x, current_carry

    def initialize_carry(self, batch_size):
        """Initialize carry state for RNN."""
        return [None] * len(self.hidden_sizes) 