"""
This is a variation of intent inference fron CPC loss and transformer model.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax

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
from typing import Dict, Any, Optional
import yaml

# Results saving imports
import os
import pickle
from datetime import datetime
from pathlib import Path
import sys

class CPCModule(nn.Module):
    """Contrastive Predictive Coding module for learning representations"""
    latent_dim: int = 64
    future_steps: int = 3
    hidden_size: int = 256  # Added for autoregressive model

    def setup(self):
        # Autoregressive model (GRU for context creation)
        self.gru = nn.GRUCell(features=self.hidden_size)
        
        # Projection for context vector
        self.context_proj = nn.Dense(
            features=self.latent_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )
        
        # Predictive transformations for each future step
        self.Wk = [
            nn.Dense(
                features=self.latent_dim,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0)
            ) for k in range(self.future_steps)
        ]

    def __call__(self, z_t, z_future, temperature=0.1):
        batch_size = z_t.shape[0]
        
        # Create context through autoregressive model
        c_t = self.gru(z_t)  # (batch_size, hidden_size)
        c_t = self.context_proj(c_t)  # (batch_size, latent_dim)
        
        total_loss = 0
        for k in range(self.future_steps):
            # Predict future latent representations
            pred_future = self.Wk[k](c_t)  # (batch_size, latent_dim)
            
            # Compute similarity scores
            similarity = jnp.einsum('bd,nd->bn', pred_future, z_future[:, k])  # (batch_size, batch_size)
            similarity = similarity / temperature
            
            # Positive samples are on the diagonal
            labels = jnp.arange(batch_size)
            
            # InfoNCE loss
            loss = optax.softmax_cross_entropy_with_integer_labels(similarity, labels)
            total_loss += loss.mean()
            
        return total_loss / self.future_steps

# Example usage in training loop:
'''
# Assume observations shape: (batch_size, seq_len, obs_dim)
z_embeddings = encoder(observations)  # (batch_size, seq_len, latent_dim)

# Current timestep embeddings
z_t = z_embeddings[:, :-self.future_steps]  # (batch_size, seq_len-future_steps, latent_dim)

# Future timestep embeddings
z_future = jnp.stack([
    z_embeddings[:, i+1:i+1+self.future_steps] 
    for i in range(z_embeddings.shape[1] - self.future_steps)
], axis=1)  # (batch_size, seq_len-future_steps, future_steps, latent_dim)

cpc_loss = cpc_module(z_t, z_future)
total_loss = task_loss + λ * cpc_loss  # λ is a hyperparameter
'''