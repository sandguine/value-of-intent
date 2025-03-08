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
    latent_dim: int = 64
    future_steps: int = 3

    def setup(self):
        self.Wk = [self.param(f"W_{k}", orthogonal(), (self.latent_dim, self.latent_dim)) for k in range(self.future_steps)]

    def __call__(self, z_t, c_t, z_future):
        losses = []
        for k in range(self.future_steps):
            logits = jnp.einsum('bi,ij,bj->b', z_future[:, k, :], self.Wk[k], c_t)
            labels = jnp.arange(logits.shape[0])
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            losses.append(loss)
        return jnp.mean(jnp.stack(losses))

# right after collecting trajectories via jax.lax.scan in _update_step
# Assume traj_batch.obs shape: (T, num_envs, seq_len, obs_dim)
'''
z_embeddings = network.apply(train_state.params, traj_batch.obs.reshape(-1, seq_len, obs_dim), method=network.feature_extractor)
c_context = z_embeddings[:, -1, :]  # Context at final timestep
z_future = jnp.stack([jnp.roll(z_embeddings, shift=-k-1, axis=1)[:, -1, :] for k in range(future_steps)], axis=1)

cpc_loss = cpc_module(z_embeddings[:, -1, :], c_context, z_future)
'''