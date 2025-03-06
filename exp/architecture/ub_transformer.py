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

# might need rewrite
def get_rollout(train_state: TrainState, config: dict, save_dir=None):
    """
    Generate a single episode rollout for Transformer-based Upper Bound PPO visualization.
    Both agents are learning and use the Transformer-based policy.
    """

    # Initialize environment
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    # Initialize Transformer-based network
    network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    # Initialize observation input shape for Transformer (sequence-based)
    init_x = jnp.zeros((1, 1) + env.observation_space().shape)  # Add batch & sequence dim
    network.init(key_a, init_x)
    network_params = train_state.params

    done = False

    # Reset environment and initialize tracking lists
    obs, state = env.reset(key_r)
    state_seq = [state]
    rewards = []
    shaped_rewards = []

    # Run episode until completion
    while not done:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        # Stack and reshape observations for Transformer processing
        obs_batch = jnp.stack([obs[a] for a in env.agents])  # Stack multi-agent observations
        obs_batch = obs_batch.reshape((1, 1, *obs_batch.shape[1:]))  # Add sequence dim

        # Get actions from Transformer-based policy for BOTH agents
        pi, _ = network.apply(network_params, obs_batch)
        actions = pi.sample(seed=key_a0)

        # Convert actions to env format
        env_act = {agent: actions[i].squeeze() for i, agent in enumerate(env.agents)}

        # Step environment forward
        obs, state, reward, done, info = env.step(key_s, state, env_act)
        done = done["__all__"]
        rewards.append(reward['agent_0'])
        shaped_rewards.append(info["shaped_reward"]['agent_0'])

        state_seq.append(state)

    # Plot rewards for visualization (same as FF version)
    plt.plot(rewards, label="reward", color='C0')
    plt.plot(shaped_rewards, label="shaped_reward", color='C1')
    plt.legend()
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title("Episode Reward and Shaped Reward Progression")
    plt.grid()
    
    # Save the reward plot
    reward_plot_path = os.path.join(save_dir, "reward_plot.png") if save_dir else "reward_plot.png"
    plt.savefig(reward_plot_path)
    plt.show()
    plt.close()

    return state_seq


def make_train(config):
    # Initialize environment
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    # Calculate key training parameters
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"] 
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    
    env = LogWrapper(env, replace_info=False)
    
    def linear_schedule(count):
        """Learning rate annealing schedule"""
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac
    
    # Schedule for annealing reward shaping
    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.,
        end_value=0.,
        transition_steps=config["REW_SHAPING_HORIZON"]
    )

    # This is the main training loop where the training starts.
    # It initializes network with: correct number of parameters, optimizer, and learning rate annealing.
    def train(rng):
        """Main training loop"""
        # Initialize network and optimizer
        network = ActorCritic(
            env.action_space().n,
            activation=config["ACTIVATION"]
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space().shape)
        
        init_x = init_x.flatten()
        # print("init_x shape:", init_x.shape)
        
        network_params = network.init(_rng, init_x)
        
        # Setup optimizer with optional learning rate annealing
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        
        # Initialize environment states
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        
        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            # This function handle single environment step and collets transitions
            
            # These lines need to be moved once settled
            from collections import deque
            # Define sequence length for Transformer
            SEQUENCE_LENGTH = config["SEQUENCE_LENGTH"]

            def _env_step(runner_state, obs_buffer, unused):
                train_state, env_state, last_obs, update_step, rng = runner_state

                # Maintain rolling buffer of past observations
                if len(obs_buffer) < SEQUENCE_LENGTH:
                    # Not enough history -> pad with last_obs
                    obs_seq = jnp.repeat(last_obs[None, :], SEQUENCE_LENGTH, axis=0)
                else:
                    # Enough history -> use real sequence
                    obs_buffer.append(last_obs)  # Append latest observation
                    obs_buffer.popleft()  # Remove oldest to maintain fixed length
                    obs_seq = jnp.stack(list(obs_buffer), axis=1)

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(obs_seq, env.agents, config["NUM_ACTORS"])  # Adapted for Transformer
                pi, value = network.apply(train_state.params, obs_batch)  # Transformer processes sequential input
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )

                info["reward"] = reward["agent_0"]

                current_timestep = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                reward = jax.tree_util.tree_map(lambda x, y: x + y * rew_shaping_anneal(current_timestep), 
                                                reward, info["shaped_reward"])

                transition = Transition(
                    batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                )
                runner_state = (train_state, env_state, obsv, update_step, rng)
                
                return runner_state, obs_buffer, (transition, info)