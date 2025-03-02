"""
Implementation of Transformer-based Actor-Critic model for PPO.
"""

# Core imports for JAX machine learning
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
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
from typing import Dict, Any, Optional
import yaml

# Results saving imports
import os
import pickle
from datetime import datetime
from pathlib import Path
import sys

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

def get_rollout(train_state: TrainState, agent_1_params: dict, config: dict, save_dir=None):
    """
    Generate a single episode rollout for visualization (Lower Bound Version for Transformer PPO).
    Applies Transformer policy only to agent_0 while using a fixed policy for agent_1.
    """

    # Initialize environment
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    # Initialize Transformer-based network
    network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    # Initialize observation input shape
    init_x = jnp.zeros((1,) + env.observation_space().shape).flatten()
    network.init(key_a, init_x)

    # Ensure using the first seed of the first parallel env is selected if multiple seeds are present
    agent_1_params = {
        "params": jax.tree_util.tree_map(lambda x: x[0], agent_1_params["params"])
    }

    # Retrieve network parameters
    network_params_agent_0 = train_state.params  # Trainable Transformer policy
    network_params_agent_1 = agent_1_params  # Fixed pretrained policy

    done = False
    obs, state = env.reset(key_r)
    state_seq = [state]
    rewards = []
    shaped_rewards = []

    # Run episode until completion
    while not done:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        # Flatten observations to match Transformer input expectations
        obs = {k: v.flatten() for k, v in obs.items()}
        obs_agent_1 = obs["agent_1"][None, ...]  # Batch dim for agent_1
        obs_agent_0 = obs["agent_0"][None, ...]  # Batch dim for agent_0

        # Agent 1 (fixed partner) action using preloaded parameters
        pi_1, _ = network.apply(network_params_agent_1, obs_agent_1)  # Fixed policy
        action_1 = pi_1.sample(seed=key_a1)[0]  # Sample action

        # Agent 0 (learning agent) action using Transformer-based policy
        pi_0, _ = network.apply(network_params_agent_0, obs_agent_0)  # Transformer policy
        action_0 = pi_0.sample(seed=key_a0)[0]  # Sample action

        actions = {"agent_0": action_0, "agent_1": action_1}

        # Step environment forward
        obs, state, reward, done, info = env.step(key_s, state, actions)
        done = done["__all__"]  # Ensure proper episode termination tracking

        # Track rewards
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

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def save_training_results(save_dir, out, config, prefix=""):
    os.makedirs(save_dir, exist_ok=True)

    # Helper function to check if an object is pickleable
    def is_pickleable(obj):
        try:
            pickle.dumps(obj)
            return True
        except Exception:
            return False
    
    # Helper function to convert JAX arrays to numpy and filter unpickleable objects
    def process_tree(tree):
        def convert_and_filter(x):
            # First convert any JAX arrays to numpy
            if isinstance(x, (jax.Array, np.ndarray)):
                x = np.array(x)
            # Then verify it can be pickled
            return x if is_pickleable(x) else None
        
        return jax.tree_util.tree_map(convert_and_filter, tree)
    
    # Convert the entire output to numpy-compatible format
    numpy_out = jax.tree_util.tree_map(
        lambda x: np.array(x) if isinstance(x, (jax.Array, np.ndarray)) else x,
        jax.device_get(out)
    )
    
    # Use pickling to validate and filter objects
    pickle_safe_out = {}
    for key, value in numpy_out.items():
        try:
            # Test pickling to validate
            pickle.dumps(value)
            pickle_safe_out[key] = value
        except Exception as e:
            print(f"Warning: Skipping unpickleable key '{key}' in output due to: {str(e)}")

    # Save the entire training output in pickle and npz formats
    pickle_out_path = os.path.join(save_dir, f"complete_out.pkl")
    with open(pickle_out_path, 'wb') as f:
        pickle.dump(pickle_safe_out, f)
    # print(f"Saved complete training output in pickle format: {pickle_out_path}")

    # Save the complete output in npz format (for compatibility)
    npz_out_path = os.path.join(save_dir, f"complete_out.npz")
    np.savez(npz_out_path, **pickle_safe_out)
    # print(f"Saved complete training output in npz format: {npz_out_path}")

    # Collect seed-specific parameters into a single dictionary
    all_seeds_params = {}
    for seed_idx in range(config["NUM_SEEDS"]):
        try:
            # Only process runner_state if it exists and has content
            if "runner_state" not in out or not out["runner_state"]:
                print(f"Warning: No runner_state found for seed {seed_idx}")
                continue
                
            # Extract seed-specific state
            train_state = jax.tree_util.tree_map(
                lambda x: x[seed_idx] if x is not None else None,
                out["runner_state"][0]
            )
            
            # Process each component of the state separately
            processed_state = {}
            
            # Handle parameters if they exist
            if hasattr(train_state, 'params'):
                processed_params = process_tree(train_state.params)
                if processed_params is not None:
                    processed_state['params'] = processed_params['params']
            
            # Handle step count if it exists
            if hasattr(train_state, 'step'):
                try:
                    processed_state['step'] = np.array(train_state.step)
                except Exception as e:
                    print(f"Warning: Could not process step for seed {seed_idx}: {str(e)}")
            
            # Handle metrics if they exist
            if "metrics" in out:
                processed_metrics = process_tree(
                    jax.tree_util.tree_map(
                        lambda x: x[seed_idx] if isinstance(x, (jax.Array, np.ndarray)) else x,
                        out["metrics"]
                    )
                )
                if processed_metrics:
                    processed_state['metrics'] = processed_metrics
            
            # Only save non-empty states
            if processed_state:
                all_seeds_params[f"seed_{seed_idx}"] = processed_state
            
        except Exception as e:
            print(f"Warning: Could not process seed {seed_idx} due to: {str(e)}")
            continue
    
    # Save seed-specific parameters if we have any
    if all_seeds_params:
        # Save as pickle first (our validation format)
        pickle_seeds_path = os.path.join(save_dir, f"all_seeds_params.pkl")
        with open(pickle_seeds_path, 'wb') as f:
            pickle.dump(all_seeds_params, f)
        print(f"Saved all seed-specific parameters in pickle format: {pickle_seeds_path}")
        
        # Then save as npz for compatibility
        npz_seeds_path = os.path.join(save_dir, f"all_seeds_params.npz")
        np.savez(npz_seeds_path, **all_seeds_params)
        print(f"Saved all seed-specific parameters in npz format: {npz_seeds_path}")
    else:
        print("Warning: No seed-specific parameters were successfully processed")

def load_training_results(load_dir, load_type="params", config=None):
    key = jax.random.PRNGKey(0) 
    key, subkey = jax.random.split(key)

    if load_type == "params":
        pickle_path = os.path.join(load_dir, f"all_seeds_params.pkl")
        if os.path.exists(pickle_path):
            print("Loading params from pickle format...")
            with open(pickle_path, 'rb') as f:
                all_params = pickle.load(f)
            
            num_seeds = len(all_params.keys())
            print("num seeds", num_seeds)

            all_params = flax.core.freeze(all_params)
            num_envs = config["NUM_ENVS"]

            sampled_indices = jax.random.choice(subkey, num_seeds, shape=(num_envs,), replace=False) # Sample equivalent to NUM_ENVS

            print("sampled_indices", sampled_indices)

            sampled_params_list = [{'params': all_params[f'seed_{i}']['params']} for i in sampled_indices]

            # Extract 16 sampled parameter sets
            sampled_params = jax.tree_util.tree_map(
                lambda *x: jnp.stack(x, axis=0), *sampled_params_list
            )

            print("Successfully loaded pretrained model.")
            print("Loaded params type:", type(sampled_params))  # Should be <FrozenDict>
            print("Shape of sampled_params:", jax.tree_util.tree_map(lambda x: x.shape, sampled_params))

            return sampled_params
                
    elif load_type == "complete":
        pickle_path = os.path.join(load_dir, f"complete_out.pkl")
        if os.path.exists(pickle_path):
            print("Loading complete output from pickle format...")
            with open(pickle_path, 'rb') as f:
                out = pickle.load(f)
                # Convert numpy arrays to JAX arrays
                return jax.tree_util.tree_map(
                    lambda x: jax.numpy.array(x) if isinstance(x, np.ndarray) else x,
                    out
                )
    
    raise FileNotFoundError(f"No saved {load_type} found in {load_dir}")

def create_visualization(train_state, agent_1_params, config, filename, save_dir=None, agent_view_size=5):
    """Helper function to create and save visualization"""
    if not isinstance(config, dict):
        config = OmegaConf.to_container(config, resolve=True)
    
    # Get the rollout
    state_seq = get_rollout(train_state, agent_1_params, config, save_dir)
    
    # Create visualization
    viz = OvercookedVisualizer()
    viz.animate(state_seq, agent_view_size=agent_view_size, filename=filename)

def make_train(config):
    # Initialize environment
    dims = config["DIMS"]
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    # Verify dimensions match what we validated in main
    assert np.prod(env.observation_space().shape) == dims["base_obs_dim"], "Observation dimension mismatch"
    assert env.action_space().n == dims["action_dim"], "Action dimension mismatch"

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
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac
    
    # Schedule for annealing reward shaping
    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.,
        end_value=0.,
        transition_steps=config["REW_SHAPING_HORIZON"]
    )

    def train(rng):
        network = ActorCritic(
            action_dim=dims["action_dim"],
            activation=config["ACTIVATION"]
        )

        # Initialize seeds
        rng, _rng = jax.random.split(rng)
        _rng_agent_0 = jax.random.split(_rng)

        # Initialize networks with correct dimensions from config
        init_x_agent_0 = jnp.zeros(dims["base_obs_dim"])
        
        network_params_agent_0 = network.init(_rng_agent_0, init_x_agent_0)

        def create_optimizer(config):
            if config["ANNEAL_LR"]:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),  # First transformation
                    optax.adam(learning_rate=linear_schedule, eps=1e-5)  # Second transformation
                )
            else:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(config["LR"], eps=1e-5)
                )
            return tx
        
        tx_agent_0 = create_optimizer(config)

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params_agent_0,
            tx=tx_agent_0
        )

        
        
        
        
    