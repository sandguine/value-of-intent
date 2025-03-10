"""
Implementation of Transformer-based Actor-Critic model for PPO.

Agent_0 is a Transformer-based learning agent. Agent_1 is fixed with pretrained FF or CNN policies.
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

import flax.linen as nn
import jax.numpy as jnp
import jax
from flax.linen.initializers import orthogonal, constant

class TransformerFeatureExtractor(nn.Module):
    """Minimal Transformer Encoder for PPO with sequence processing."""
    num_layers: int = 2  # Number of Transformer blocks
    model_dim: int = 64  # Hidden dimension
    num_heads: int = 4   # Attention heads
    seq_len: int = 5     # Number of timesteps to consider

    def setup(self):
        """Define Transformer Encoder layers."""
        # Projection layer to map input features (base_obs_dim) to model_dim
        self.input_projection = nn.Dense(self.model_dim, kernel_init=orthogonal(jnp.sqrt(2)), name="input_projection")

        # Transformer encoder layers
        self.encoder_blocks = [
            nn.SelfAttention(
                num_heads=self.num_heads,
                qkv_features=self.model_dim,
                kernel_init=orthogonal(jnp.sqrt(2))
            ) for _ in range(self.num_layers)
        ]

        # Final dense layer for feature extraction
        self.final_dense = nn.Dense(features=self.model_dim, kernel_init=orthogonal(jnp.sqrt(2)))

    def create_positional_encoding(self, seq_len, model_dim):
        """Create sinusoidal positional encoding for sequence processing."""
        positions = jnp.arange(seq_len)[:, None]  # Shape (seq_len, 1)
        div_term = jnp.exp(jnp.arange(0, model_dim, 2) * -(jnp.log(10000.0) / model_dim))
        pos_enc = jnp.zeros((seq_len, model_dim))

        pos_enc = pos_enc.at[:, 0::2].set(jnp.sin(positions * div_term))
        pos_enc = pos_enc.at[:, 1::2].set(jnp.cos(positions * div_term))
        return pos_enc

    @nn.compact
    def __call__(self, x):
        """Apply Transformer Encoder to input."""
        # Ensure input is projected to match `model_dim`
        x = self.input_projection(x)  # Shape: (batch, seq_len, model_dim)

        # Compute positional encoding on-the-fly
        pos_enc = self.create_positional_encoding(self.seq_len, self.model_dim)

        # Add position information
        x = x + pos_enc

        # Apply Transformer layers
        for block in self.encoder_blocks:
            x = block(x)  # Apply self-attention layers

        x = self.final_dense(x)  # Final feature projection
        return x

class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"
    seq_len: int = 5  # Sequence length for Transformer

    def setup(self):
        """Define Transformer-based Actor-Critic architecture."""
        self.act_fn = nn.relu if self.activation == "relu" else nn.tanh

        # Feature extractor: Transformer-based
        self.feature_extractor = TransformerFeatureExtractor(seq_len=self.seq_len)

        # Actor network
        self.actor_hidden = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="actor_hidden")
        self.actor_out = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor_output")

        # Critic network
        self.critic_hidden = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="critic_hidden")
        self.critic_out = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic_output")

    @nn.compact
    def __call__(self, x):
        """Forward pass of the Transformer-based Actor-Critic model."""
        # Extract features using Transformer
        features = self.feature_extractor(x)  # Shape (B, T, model_dim)

        # Take the last timestep's features
        last_feature = features[:, -1, :]  # Shape (B, model_dim)

        # Actor head
        actor_hidden = self.actor_hidden(last_feature)
        actor_hidden = self.act_fn(actor_hidden)
        actor_logits = self.actor_out(actor_hidden)
        pi = distrax.Categorical(logits=actor_logits)

        # Critic head
        critic_hidden = self.critic_hidden(last_feature)
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
    network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"], seq_len=config["SEQ_LEN"])
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    # Initialize observation input shape
    init_x = jnp.zeros((1,) + env.observation_space().shape).flatten()
    network_params_agent_0 = network.init(key_a, init_x)

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

    # Initialize observation sequence buffer for Transformer agent
    seq_len = 5  # Match Transformer sequence length
    obs_history_0 = jnp.roll(obs_history_0, shift=-1, axis=1)  # Shift all observations left
    obs_history_0 = obs_history_0.at[:, -1, :].set(last_obs['agent_0'])  # Add newest observation

    # Run episode until completion
    while not done:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        # Flatten observations to match Transformer input expectations
        obs = {k: v.flatten() for k, v in obs.items()}
        obs_agent_1 = obs["agent_1"][None, ...]  # Batch dim for agent_1

        # Agent 1 (fixed partner) action using preloaded parameters
        pi_1, _ = network.apply(network_params_agent_1, obs_agent_1)  # Fixed policy
        action_1 = pi_1.sample(seed=key_a1)[0]  # Sample action

        # Shift observation history for agent_0 BEFORE selecting action
        obs_history_0 = jnp.roll(obs_history_0, shift=-1, axis=0)
        obs_history_0 = obs_history_0.at[-1, :].set(obs["agent_0"])

        # Use updated observation sequence
        pi_0, _ = network.apply(network_params_agent_0, obs_history_0[None, ...])  # Transformer policy
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

            num_envs = config["NUM_ENVS"]

            sampled_indices = jax.random.choice(subkey, num_seeds, shape=(num_envs,), replace=False) # Sample equivalent to NUM_ENVS

            print("sampled_indices", sampled_indices)

            # Freeze only the parameters, not the entire dictionary
            sampled_params_list = [{'params': flax.core.freeze(all_params[f'seed_{i}']['params'])} for i in sampled_indices]

            # Stack the sampled parameter sets correctly
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
    # state_seq = get_rollout(train_state, agent_1_params, config, save_dir)
    
    # Create visualization
    # viz = OvercookedVisualizer()
    # viz.animate(state_seq, agent_view_size=agent_view_size, filename=filename)

def make_train(config):
    # Initialize environment
    dims = config["DIMS"]
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    seq_len = config["SEQ_LEN"]

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

    def train(rng, pretrained_params, seq_len):
        network = ActorCritic(
            action_dim=dims["action_dim"],  # Use dimension from config
            activation=config["ACTIVATION"],
            seq_len=seq_len
        )

        # Initialize seeds
        rng, _rng = jax.random.split(rng)
        _rng_agent_0, _rng_agent_1 = jax.random.split(_rng)  # Split for two networks

        # Initialize networks with correct dimensions from config
        init_x_agent_0 = jnp.zeros((1, seq_len, dims["base_obs_dim"]))  # Shape (1, 5, 520)
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

        # Create separate optimizer chains for each agent and only keep the one for agent_0 since this is the learning agent
        tx_agent_0 = create_optimizer(config)

        # Create train state for agent_0 only
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params_agent_0,
            tx=tx_agent_0
        )
        
        # Initialize environment states
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        
        # TRAIN LOOP
        def _update_step(runner_state, unused, seq_len, pretrained_params):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused, seq_len, pretrained_params):
                train_state, env_state, last_obs, update_step, rng, obs_history_0 = runner_state
                rng, rng_action_1, rng_action_0, rng_step = jax.random.split(rng, 4)

                num_envs = last_obs['agent_1'].shape[0]  # Should be 16

                agent_1_obs = last_obs['agent_1'].reshape(num_envs, -1)  # Shape: (520,)
                rng_action_1_split = jax.random.split(rng_action_1, num_envs)

                # Vectorized application across all environments
                agent_1_action = jax.vmap(
                    lambda params, obs, rng: network.apply(params, obs)[0].sample(seed=rng),
                    in_axes=(0, 0, 0)
                )(pretrained_params, agent_1_obs, rng_action_1_split)

                # Initialize history buffer at the start of rollout
                if update_step == 0:  
                    obs_history_0 = jnp.roll(obs_history_0, shift=-1, axis=1)  # Shift all observations left
                obs_history_0 = obs_history_0.at[:, -1, :].set(last_obs['agent_0'])  # Add newest observation

                # Shift observation history for agent_0 BEFORE selecting action
                obs_history_0 = jnp.roll(obs_history_0, shift=-1, axis=1)
                obs_history_0 = obs_history_0.at[:, -1, :].set(last_obs['agent_0'])

                # Apply agent_0 policy using trainable parameters
                agent_0_pi, agent_0_value = network.apply(train_state.params, obs_history_0)
                agent_0_action = agent_0_pi.sample(seed=rng_action_0)
                agent_0_log_prob = agent_0_pi.log_prob(agent_0_action)

                # Step the environment
                actions = {"agent_0": agent_0_action, "agent_1": agent_1_action}
                next_obs, next_env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    jax.random.split(rng_step, num_envs),
                    env_state,
                    actions,
                )

                # Create transition
                transition = Transition(
                    done=done["agent_0"],
                    action=agent_0_action,
                    value=agent_0_value,
                    reward=reward["agent_0"],
                    log_prob=agent_0_log_prob,
                    obs=obs_history_0,
                )

                # Pass obs_history_0 forward for next step
                runner_state = (train_state, next_env_state, next_obs, update_step, rng, obs_history_0)
                return runner_state, (transition, info)

            runner_state, (traj_batch, info) = jax.lax.scan(
                lambda state, unused: _env_step(state, unused, seq_len, pretrained_params),  
                runner_state, 
                None, 
                config["NUM_STEPS"]
            )
            
            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, update_step, rng, obs_history_0 = runner_state
            
            # Ensure sequence length matches Transformer expectation
            # last_obs_agent0 = last_obs['agent_0'].reshape(last_obs['agent_0'].shape[0], seq_len, -1)
            last_obs_agent0 = last_obs['agent_0'].reshape(last_obs['agent_0'].shape[0], seq_len, -1)

            # Compute last value for advantage calculation
            _, last_val = network.apply(train_state.params, last_obs_agent0)

            # calculate_gae itself didn't need to be changed because we can use the same advantage function for both agents
            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
        
                    # Calculate delta and GAE per agent
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    #print(f"calculated gae shape: {gae.shape}, value: {gae}")
                    
                    return (gae, value), gae

                # Initialize with agent-specific last value
                init_gae = jnp.zeros_like(last_val)
                init_value = last_val

                # Calculate advantages for an agent
                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (init_gae, init_value),
                    traj_batch,
                    reverse=True,
                    unroll=16
                )
                
                return advantages, advantages + traj_batch.value

            # Calculate advantages 
            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused, seq_len, config, network):
                def _update_minbatch(train_state, batch_info, seq_len, config, network):
                    print("\nStarting minibatch update...")

                    # Unpack batch_info which now contains only agent_0 data
                    agent_0_data = batch_info['agent_0']
                    traj_batch = agent_0_data['traj']
                    advantages = agent_0_data['advantages']
                    targets = agent_0_data['targets']

                    def _loss_fn(params, traj_batch, gae, seq_len, targets, config, network):
                        """Calculate loss for agent_0."""

                        # Ensure batch size is correctly inferred
                        batch_size = traj_batch.obs.shape[0] // seq_len if traj_batch.obs.shape[0] % seq_len == 0 else traj_batch.obs.shape[0]
                        # obs_seq = traj_batch.obs.reshape(batch_size, seq_len, traj_batch.obs.shape[-1])
                        obs_seq = traj_batch.obs.reshape(-1, seq_len, traj_batch.obs.shape[-1])

                        # Forward pass through Transformer
                        pi, value = network.apply(params, obs_seq)
                        log_prob = pi.log_prob(traj_batch.action)

                        # Ensure targets match the sequence format
                        if targets.ndim == 1:
                            targets = targets[:, None]  # Ensure at least (batch_size, 1)
                        targets_seq = targets.reshape(batch_size, seq_len, -1)

                        # Value loss calculation
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets_seq)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets_seq)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # Normalize GAE correctly along the sequence dimension
                        if seq_len > 1:
                            gae = (gae - gae.mean(axis=1, keepdims=True)) / (gae.std(axis=1, keepdims=True) + 1e-8)

                        # PPO Actor Loss (Clipped Ratio)
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        loss_actor = -jnp.minimum(
                            ratio * gae,
                            jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        ).mean()

                        # Entropy regularization (Exploration bonus)
                        entropy = pi.entropy().mean()

                        # Compute total loss
                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )

                        return total_loss, {
                            'value_loss': value_loss,
                            'actor_loss': loss_actor,
                            'entropy': entropy,
                            'total_loss': total_loss
                        }

                    # Compute gradients for agent_0
                    grad_fn_0 = jax.value_and_grad(lambda p: _loss_fn(p, traj_batch, advantages, seq_len, targets, config, network), has_aux=True)
                    (loss_0, aux_0), grads_0 = grad_fn_0(train_state.params)

                    # Compute gradient norms
                    grad_norm_0 = optax.global_norm(grads_0)

                    # Update only agent_0
                    train_state = train_state.apply_gradients(grads=grads_0)

                    return train_state, (loss_0, aux_0)
                
                train_state, traj_batch, advantages, targets, obs_history_0, rng = update_state
                rng, _rng = jax.random.split(rng)

                # Calculate total batch size and minibatch size
                batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
                minibatch_size = batch_size // config["NUM_MINIBATCHES"]

                # Ensure batch size is evenly divisible
                assert batch_size % config["NUM_MINIBATCHES"] == 0, "Steps * Envs must be divisible by number of minibatches"

                # Reshape function that handles the different observation sizes
                def reshape_agent_data(agent_dict):
                    def reshape_field(x, field_name):
                        if not isinstance(x, (dict, jnp.ndarray)):
                            return x
                        return x.reshape(batch_size, -1) if field_name == 'obs' else x.reshape(batch_size)

                    return {
                        'traj': Transition(**{
                            field: reshape_field(getattr(agent_dict['traj'], field), field)
                            for field in agent_dict['traj']._fields
                        }),
                        'advantages': agent_dict['advantages'].reshape(batch_size),
                        'targets': agent_dict['targets'].reshape(batch_size),
                        'obs_history_0': agent_dict['obs_history_0'].reshape(batch_size, seq_len, -1)  # Fix for Transformer input
                    }

                # Reshape trajectory data
                agent_data = {
                    "agent_0": {"traj": traj_batch, "advantages": advantages, "targets": targets}
                }
                agent_data = {agent: reshape_agent_data(data) for agent, data in agent_data.items()}

                # Shuffle data
                permutation = jax.random.permutation(_rng, batch_size)
                agent_data = {
                    agent: {
                        'traj': Transition(**{
                            field: jnp.take(getattr(data['traj'], field), permutation, axis=0)
                            for field in data['traj']._fields
                        }),
                        'advantages': jnp.take(data['advantages'], permutation, axis=0),
                        'targets': jnp.take(data['targets'], permutation, axis=0),
                        'obs_history_0': jnp.take(data['obs_history_0'], permutation, axis=0)
                    }
                    for agent, data in agent_data.items()
                }
                
                # Minibatch function
                def create_minibatches(data):
                    return {
                        'traj': Transition(**{
                            field: getattr(data["traj"], field).reshape((config["NUM_MINIBATCHES"], -1) + getattr(data["traj"], field).shape[1:])
                            for field in data["traj"]._fields
                        }),
                        'advantages': data["advantages"].reshape((config["NUM_MINIBATCHES"], -1)),
                        'targets': data["targets"].reshape((config["NUM_MINIBATCHES"], -1)),
                        'obs_history_0': data["obs_history_0"].reshape((config["NUM_MINIBATCHES"], -1, seq_len, -1))
                    }

                # Create minibatches
                minibatches = {agent: create_minibatches(data) for agent, data in agent_data.items()}
                assert advantages.shape[0] % config["NUM_MINIBATCHES"] == 0, "Minibatch size is incorrect!"

                # Perform minibatch updates
                train_state, total_loss = jax.lax.scan(
                    lambda state, data: _update_minbatch(state, data, seq_len, config, network),
                    train_state,
                    minibatches
                )

                return (train_state, traj_batch, advantages, targets, rng), total_loss

            update_state = (train_state, traj_batch, advantages, targets, obs_history_0, rng)
            update_state, loss_info = jax.lax.scan(
                lambda state, _: _update_epoch(state, _, seq_len, config, network),
                update_state,
                None,
                config["UPDATE_EPOCHS"]
            )

            train_state = update_state[0]

            # Extract training metrics
            metric = info
            current_timestep = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            metric["shaped_reward"] = metric["shaped_reward"]["agent_0"]
            metric["shaped_reward_annealed"] = metric["shaped_reward"]*rew_shaping_anneal(current_timestep)

            rng = update_state[-1]

            def callback(metric):
                wandb.log(
                    metric
                )
            update_step = update_step + 1
            metric = jax.tree.map(lambda x: x.mean(), metric)
            metric["update_step"] = update_step
            metric["env_step"] = update_step*config["NUM_STEPS"]*config["NUM_ENVS"]
            jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, update_step, rng, obs_history_0)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        # Initialize observation history for agent_0
        initial_obs_history_0 = jnp.tile(obsv['agent_0'][:, None, :], (1, seq_len, 1))
        runner_state = (train_state, env_state, obsv, 0, _rng, initial_obs_history_0)
        runner_state, metrics = jax.lax.scan(
            lambda state, unused: _update_step(state, unused, seq_len, pretrained_params),  
            runner_state,
            None,
            config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train

@hydra.main(version_base=None, config_path="config", config_name="adapt_asymm")
def main(config):

    # Validate config
    required_keys = ["ENV_NAME", "ENV_KWARGS"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    # Process config
    config = OmegaConf.to_container(config)
    layout_name = config["ENV_KWARGS"]["layout"]
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]

    # Create environment using JaxMARL framework
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    # Get environment dimensions
    base_obs_shape = env.observation_space().shape
    base_obs_dim = int(np.prod(base_obs_shape))
    action_dim = int(env.action_space().n)
    augmented_obs_dim = base_obs_dim + action_dim

    # Validate dimensions
    assert base_obs_dim > 0, f"Invalid base observation dimension: {base_obs_dim}"
    assert action_dim > 0, f"Invalid action dimension: {action_dim}"
    assert augmented_obs_dim > base_obs_dim, "Augmented dim must be larger than base dim"

    # Store dimensions in config for easy access
    config["DIMS"] = {
        "base_obs_shape": base_obs_shape,
        "base_obs_dim": base_obs_dim,
        "action_dim": action_dim,
        "augmented_obs_dim": augmented_obs_dim
    }

    # Initialize wandb logging
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "FF", "Adaptability", "Oracle", "Transforrmer"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f'tf_ippo_oc_{layout_name}'
    )

    print("\nVerifying config before rollout:")
    print("Config keys:", config.keys())
    if "DIMS" in config:
        print("Found dimensions:")
        for key, value in config["DIMS"].items():
            print(f"  {key}: {value}")
    else:
        raise ValueError("DIMS not found in config - check dimension initialization")

    # Create a new directory for the results
    # Process layout configuration
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date = datetime.now().strftime("%Y%m%d")
    model_dir_name = f"tf_ippo_oc_{layout_name}_{timestamp}"
    save_dir = os.path.join(
        "saved_models", 
        date,
        layout_name, 
        f"{model_dir_name}_{config['SEED']}"
    )
    os.makedirs(save_dir, exist_ok=True)

    # Setup random seeds and training
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])    

    # Load pretrained parameters once at this level
    pretrained_params = load_training_results(config["LOAD_PATH"], load_type="params", config=config)

    # Start training
    train_fn = make_train(config)
    train_jit = jax.jit(lambda rng: train_fn(rng, pretrained_params, config["SEQ_LEN"]))
    out = jax.vmap(train_jit)(rngs)

    # Generate and save visualization
    try:
        train_state = jax.tree_util.tree_map(lambda x: x[0], out["runner_state"][0])
        viz_base_name = f"lb_ippo_oc_{layout_name}"
        viz_filename = os.path.join(save_dir, f'{viz_base_name}_{config["SEED"]}.gif')
        create_visualization(train_state, pretrained_params, config, viz_filename, save_dir)
    except Exception as e:  
        print(f"Error generating visualization: {e}")
        traceback.print_exc()

    # Save parameters and results
    save_training_results(save_dir, out, config, prefix="tf_ippo_oc_")
    np.savez(os.path.join(save_dir, "tf_metrics.npz"), 
            **{key: np.array(value) for key, value in out["metrics"].items()})
    
    with open(os.path.join(save_dir, "config.pkl"), 'wb') as f:
        pickle.dump(config, f)
    
    # Plot and save learning curves
    rewards = out["metrics"]["returned_episode_returns"].reshape((config["NUM_SEEDS"], -1))

    # Calculate mean and standard deviation of rewards across seeds
    reward_mean = rewards.mean(0)
    reward_std = rewards.std(0)
    reward_std_err = reward_std / np.sqrt(config["NUM_SEEDS"])

    if config["NUM_SEEDS"] > 1:
        # Log individual seed rewards
        for update_step in range(rewards.shape[1]):
            log_data = {"Update_Step": update_step}

            for seed_idx in range(config["NUM_SEEDS"]):
                log_data[f"Seeds/Seed_{seed_idx}/Returned_Episode_Returns"] = rewards[seed_idx, update_step]

            wandb.log(log_data)
        
        # Save learning curve locally
        plt.figure()
        plt.plot(reward_mean, label="Average Across All Seeds", color='black', linewidth=2)
        plt.fill_between(range(len(reward_mean)), 
                        reward_mean - reward_std_err,
                        reward_mean + reward_std_err,
                        alpha=0.2, color='gray', label="Mean ± Std Err")
        for seed_idx in range(config["NUM_SEEDS"]):
            plt.plot(rewards[seed_idx], alpha=0.7)
        plt.xlabel("Update Step")
        plt.ylabel("Returned Episode Returns")
        plt.title("Per-Seed Performance on Returned Episode Returns")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid()
        
    else:
        # Save learning curve locally
        plt.figure()
        plt.plot(reward_mean, label="Mean Reward")
        plt.xlabel("Update Step")
        plt.ylabel("Returned Episode Returns")
        plt.title("Lower Bound Learning Curve")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid()

    learning_curve_name = f"tf_ippo_oc_learning_curve"
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{learning_curve_name}.png'))
    plt.close()

if __name__ == "__main__":
    main()