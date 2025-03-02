"""
Upperbound transformer model for overcooked.
"""

# Core imports for JAX machine learning
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import flax

# Environment and visualization imports
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer

# Configuration and logging imports
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

# Results saving imports
import os
import pickle
from datetime import datetime
from pathlib import Path
import traceback

# Plotting imports
import matplotlib.pyplot as plt

# Helper imports
from functools import partial

class TransformerFeatureExtractor(nn.Module):
    """Minimal Transformer Encoder for PPO."""
    num_layers: int = 2  # Number of Transformer blocks
    model_dim: int = 64  # Hidden dimension
    num_heads: int = 4   # Attention heads

    def setup(self):
        """Define Transformer Encoder layers."""
        self.encoder_blocks = [
            nn.SelfAttention(
                num_heads=self.num_heads,
                qkv_features=self.model_dim,
                kernel_init=orthogonal(np.sqrt(2))
            ) for _ in range(self.num_layers)
        ]

        # Final projection layer after Transformer processing
        self.final_dense = nn.Dense(features=self.model_dim, kernel_init=orthogonal(np.sqrt(2)))

    def __call__(self, x):
        """Apply Transformer Encoder to input."""
        for block in self.encoder_blocks:
            x = block(x)  # Apply self-attention layers
        x = self.final_dense(x)  # Final feature projection
        return x

class ActorCritic(nn.Module):
    """Actor-Critic model using Transformer feature extractor."""
    action_dim: int
    activation: str = "tanh"

    def setup(self):
        """Define Transformer-based Actor-Critic architecture."""
        self.act_fn = nn.relu if self.activation == "relu" else nn.tanh

        # Replace CNN with Transformer feature extractor
        self.feature_extractor = TransformerFeatureExtractor()

        # Actor network
        self.actor_hidden = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="actor_hidden"
        )
        self.actor_out = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor_output"
        )

        # Critic network
        self.critic_hidden = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="critic_hidden"
        )
        self.critic_out = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic_output"
        )

    @nn.compact
    def __call__(self, x):
        """Forward pass of the Transformer-based Actor-Critic model."""
        # Extract high-level features using Transformer
        features = self.feature_extractor(x)

        # Actor head
        actor_hidden = self.actor_hidden(features)
        actor_hidden = self.act_fn(actor_hidden)
        actor_logits = self.actor_out(actor_hidden)
        pi = distrax.Categorical(logits=actor_logits)

        # Critic head
        critic_hidden = self.critic_hidden(features)
        critic_hidden = self.act_fn(critic_hidden)
        critic_value = self.critic_out(critic_hidden)

        return pi, jnp.squeeze(critic_value, axis=-1)
    
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray

