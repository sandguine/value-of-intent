"""
This is the code to test whether the oracle information on partner's next action is helpful for adaptability to different fixed policies.
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
import traceback

# Results saving imports
import os
import pickle
from datetime import datetime
from pathlib import Path

# Plotting imports
import matplotlib.pyplot as plt

# Helper imports
from functools import partial

class ActorCritic(nn.Module):
    action_dim: Sequence[int]  # Dimension of action space
    activation: str = "tanh"   # Activation function to use
    history_length: int = 5    # Number of past observations to use for prediction

    def setup(self):
        # Store activation function
        self.act_fn = nn.relu if self.activation == "relu" else nn.tanh

        # Initialize dense layers with consistent naming
        self.actor_dense1 = nn.Dense(
            64, 
            kernel_init=orthogonal(np.sqrt(2)), 
            bias_init=constant(0.0),
            name="actor_dense1"
        )
        self.actor_dense2 = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_dense2"
        )
        self.actor_out = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_out"
        )

        # Critic network layers
        self.critic_dense1 = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_dense1"
        )
        self.critic_dense2 = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_dense2"
        )
        self.critic_out = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_out"
        )

        # CPC components
        self.cpc_encoder = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="cpc_encoder")
        self.cpc_predictor = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="cpc_predictor")
        self.cpc_projection = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="cpc_projection")

    def compute_cpc_loss(self, encoded_history, predicted_future, future_obs):
        # Project future observations
        future_proj = self.cpc_projection(future_obs)
        
        # Normalize embeddings
        encoded_norm = encoded_history / (jnp.linalg.norm(encoded_history, axis=-1, keepdims=True) + 1e-8)
        predicted_norm = predicted_future / (jnp.linalg.norm(predicted_future, axis=-1, keepdims=True) + 1e-8)
        future_norm = future_proj / (jnp.linalg.norm(future_proj, axis=-1, keepdims=True) + 1e-8)
        
        # Compute InfoNCE loss
        logits = jnp.matmul(predicted_norm, future_norm.T)
        labels = jnp.arange(logits.shape[0])
        cpc_loss = jax.nn.sparse_categorical_crossentropy(logits, labels)
        
        return cpc_loss.mean()

    @nn.compact
    def __call__(self, x, historical_obs=None):
        # Expected input dimension is the last dimension of the input tensor
        expected_dim = x.shape[-1] if len(x.shape) > 1 else x.shape[0]
        # print(f"Expected input dim: {expected_dim}")

        # Actor network
        actor = self.actor_dense1(x)
        actor = self.act_fn(actor)
        actor = self.actor_dense2(actor)
        actor = self.act_fn(actor)
        actor = self.actor_out(actor)
        pi = distrax.Categorical(logits=actor)

        # Critic network
        critic = self.critic_dense1(x)
        critic = self.act_fn(critic)
        critic = self.critic_dense2(critic)
        critic = self.act_fn(critic)
        critic = self.critic_out(critic)

        # CPC forward pass if historical observations are provided
        if historical_obs is not None:
            # Encode historical observations
            encoded_history = self.cpc_encoder(historical_obs)
            encoded_history = self.act_fn(encoded_history)
            
            # Predict future representation
            predicted_future = self.cpc_predictor(encoded_history)
            predicted_future = self.act_fn(predicted_future)
            
            return pi, jnp.squeeze(critic, axis=-1), predicted_future

        return pi, jnp.squeeze(critic, axis=-1), None

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    historical_obs: jnp.ndarray  # Add historical observations

def get_rollout(train_state, agent_1_params, config, save_dir=None):
    # Initialize environment
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    # Initialize network
    network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    # Initialize observation input shape
    init_x = jnp.zeros((1,) + env.observation_space().shape).flatten()
    network.init(key_a, init_x)

    # Ensure using the first seed of first parallel env is selected if multiple seeds are present
    agent_1_params = {
        "params": jax.tree_util.tree_map(lambda x: x[0], agent_1_params["params"])
    }

    # Initialize agent_0 parameters (train_state) and retreive agent_1 parameters (pretrained)
    network_params_agent_0 = train_state.params
    network_params_agent_1 = agent_1_params

    done = False
    obs, state = env.reset(key_r)
    state_seq = [state]
    rewards = []
    shaped_rewards = []

    # Run episode until completion
    while not done:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        # Flatten observations for network input
        obs = {k: v.flatten() for k, v in obs.items()}
        obs_agent_1 = obs["agent_1"][None, ...]
        obs_agent_0 = obs["agent_0"][None, ...]

        # Agent 1, fixed partner, action sampling
        pi_1, _ = network.apply(network_params_agent_1, obs["agent_1"])
        action_1 = jnp.squeeze(pi_1.sample(seed=key_a1))

        # Augment Agent 0's observation with Agent 1's action (oracle lookahead)
        one_hot_action = jax.nn.one_hot(action_1, env.action_space().n)[None, ...]
        augmented_obs_agent_0 = jnp.concatenate([obs_agent_0, one_hot_action], axis=1)

        # Agent 0 (learning agent) action from augmented observation
        pi_0, _ = network.apply(network_params_agent_0, augmented_obs_agent_0)
        action_0 = jnp.squeeze(pi_0.sample(seed=key_a0))

        actions = {"agent_0": action_0, "agent_1": action_1}

        # Step environment forward
        obs, state, reward, done, info = env.step(key_s, state, actions)
        done = done["__all__"]

        # Track rewards
        rewards.append(reward['agent_0'])
        shaped_rewards.append(info["shaped_reward"]['agent_0'])
        state_seq.append(state)

    # Plot rewards for visualization
    plt.plot(rewards, label="reward", color='C0')
    plt.plot(shaped_rewards, label="shaped_reward", color='C1')
    plt.legend()
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title("Episode Reward and Shaped Reward Progression")
    plt.grid()
    if save_dir:
        reward_plot_path = os.path.join(save_dir, "reward_plot.png")
    else:
        reward_plot_path = "reward_plot.png"
    plt.savefig(reward_plot_path)
    plt.show()
    plt.close()

    return state_seq

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1)) # This reshapes the observation space to a 3D array with the shape (num_actors, num_envs, -1)
    return {a: x[i] for i, a in enumerate(agent_list)} # This returns the observation space for the agents

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

            sampled_indices = jax.random.choice(subkey, num_seeds, shape=(num_envs,), replace=False)
            print("sampled_indices", sampled_indices)

            sampled_params_list = [{'params': all_params[f'seed_{i}']['params']} for i in sampled_indices]

            # Extract 16 sampled parameter sets
            sampled_params = jax.tree_util.tree_map(
                lambda *x: jnp.stack(x, axis=0), *sampled_params_list
            )

            print("Successfully loaded pretrained model.")
            print("Loaded params type:", type(sampled_params))  # Should be <FrozenDict>
            # print("Shape of sampled_params:", jax.tree_util.tree_map(lambda x: x.shape, sampled_params))

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

    def train(rng, pretrained_params):
        # Initialize network with fixed action dimension
        network = ActorCritic(
            action_dim=dims["action_dim"],  # Use dimension from config
            activation=config["ACTIVATION"]
        )

        # Initialize seeds
        rng, _rng = jax.random.split(rng)
        _rng_agent_0, _rng_agent_1 = jax.random.split(_rng)  # Split for two networks _rng_agent_1 is unused

        # Initialize networks with correct dimensions from config
        init_x_agent_0 = jnp.zeros(dims["augmented_obs_dim"])  # Agent 0 gets augmented obs
        
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
        def _update_step(runner_state, unused, pretrained_params):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused, pretrained_params):
                train_state, env_state, last_obs, update_step, rng = runner_state
                rng, rng_action_1, rng_action_0, rng_step = jax.random.split(rng, 4)

                # Extract correct agent_1 parameters per environment
                num_envs = last_obs['agent_1'].shape[0]

                # Get historical observations for agent 0
                agent_0_obs = last_obs['agent_0'].reshape(num_envs, -1)
                historical_obs = last_obs.get('agent_0_history', jnp.zeros((num_envs, config["HISTORY_LENGTH"], -1)))
                
                # Update history by shifting and adding new observation
                historical_obs = jnp.roll(historical_obs, shift=-1, axis=1)
                historical_obs = historical_obs.at[:, -1].set(agent_0_obs)

                # Agent 1 action sampling (fixed partner)
                agent_1_obs = last_obs['agent_1'].reshape(num_envs, -1)
                rng_action_1_split = jax.random.split(rng_action_1, num_envs)
                agent_1_action = jax.vmap(
                    lambda params, obs, rng: network.apply(params, obs)[0].sample(seed=rng),
                    in_axes=(0, 0, 0)
                )(pretrained_params, agent_1_obs, rng_action_1_split)

                # Agent 0 action sampling with CPC
                agent_0_pi, agent_0_value, predicted_future = network.apply(
                    train_state.params, 
                    agent_0_obs,
                    historical_obs
                )
                agent_0_action = agent_0_pi.sample(seed=rng_action_0)
                agent_0_log_prob = agent_0_pi.log_prob(agent_0_action)

                # Step environment
                actions = {"agent_0": agent_0_action, "agent_1": agent_1_action}
                next_obs, next_env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    jax.random.split(rng_step, num_envs),
                    env_state,
                    actions,
                )

                # Add history to next observations
                next_obs['agent_0_history'] = historical_obs

                # Create transition
                transition = Transition(
                    done=done["agent_0"],
                    action=agent_0_action,
                    value=agent_0_value,
                    reward=reward["agent_0"],
                    log_prob=agent_0_log_prob,
                    obs=agent_0_obs,
                    historical_obs=historical_obs,
                )

                runner_state = (train_state, next_env_state, next_obs, update_step, rng)
                return runner_state, (transition, info)

            runner_state, (traj_batch, info) = jax.lax.scan(
                lambda state, unused: _env_step(state, unused, pretrained_params),  
                runner_state, 
                None, 
                config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, update_step, rng = runner_state
            
            # Calculate last values for agent_0
            # For agent_0, need to include agent_1's last action in observation
            one_hot_last_action = jax.nn.one_hot(traj_batch.action[-1], env.action_space().n)
            last_obs_agent0 = last_obs['agent_0'].reshape(last_obs['agent_0'].shape[0], -1)
            last_obs_agent0_augmented = jnp.concatenate([last_obs_agent0, one_hot_last_action], axis=-1)
            _, agent_0_last_val = network.apply(train_state.params, last_obs_agent0_augmented)
            # print("agent_0_last_val shape:", agent_0_last_val.shape)

            # Combine values for advantage calculation
            _, last_val = network.apply(train_state.params, last_obs_agent0_augmented)

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
                    # print(f"delta shape: {delta.shape}, value: {delta}")

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
            def _update_epoch(update_state, unused, config):
                def _update_minbatch(train_state, batch_info, config):
                    print("\nStarting minibatch update...")
                    # Unpack batch_info which now contains separate agent data
                    agent_0_data = batch_info['agent_0']
                    
                    # print("Minibatch shapes:")
                    # print("Agent 0 data:", jax.tree_util.tree_map(lambda x: x.shape, agent_0_data))

                    traj_batch = agent_0_data['traj']
                    advantages = agent_0_data['advantages']
                    targets = agent_0_data['targets']

                    def _loss_fn(params, traj_batch, gae, targets, config):
                        """Calculate loss for agent_0."""
                        pi, value, predicted_future = network.apply(params, traj_batch.obs, traj_batch.historical_obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # Value loss calculation
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # Actor loss calculation
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor = -jnp.minimum(
                            ratio * gae,
                            jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        ).mean()

                        # CPC loss calculation
                        cpc_loss = network.compute_cpc_loss(
                            traj_batch.historical_obs,
                            predicted_future,
                            traj_batch.obs
                        )

                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                            + config["CPC_COEF"] * cpc_loss
                        )

                        return total_loss, {
                            'value_loss': value_loss,
                            'actor_loss': loss_actor,
                            'cpc_loss': cpc_loss,
                            'entropy': entropy,
                            'total_loss': total_loss
                        }

                    # Compute gradients for agent 0
                    grad_fn_0 = jax.value_and_grad(lambda p: _loss_fn(p, traj_batch, advantages, targets, config), has_aux=True)
                    (loss_0, aux_0), grads_0 = grad_fn_0(train_state.params)

                    # Compute gradient norms correctly
                    grad_norm_0 = optax.global_norm(grads_0)

                    # Update only agent_0
                    train_state = train_state.apply_gradients(grads=grads_0)

                    return train_state, (loss_0, aux_0)

                train_state, traj_batch, advantages, targets, rng = update_state
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
                        'targets': agent_dict['targets'].reshape(batch_size)
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
                        'targets': jnp.take(data['targets'], permutation, axis=0)
                    }
                    for agent, data in agent_data.items()
                }

                # Minibatch function
                def create_minibatches(data):
                    return {
                        'traj': Transition(**{
                            field: getattr(data["traj"], field).reshape((config["NUM_MINIBATCHES"], -1) + getattr(data["traj"], field).shape[1:])
                            for field in data["traj"]._fields  # Use data["traj"]
                        }),
                        'advantages': data["advantages"].reshape((config["NUM_MINIBATCHES"], -1)),
                        'targets': data["targets"].reshape((config["NUM_MINIBATCHES"], -1))
                    }

                # Create minibatches
                minibatches = {agent: create_minibatches(data) for agent, data in agent_data.items()}
                assert advantages.shape[0] % config["NUM_MINIBATCHES"] == 0, "Minibatch size is incorrect!"

                # Perform minibatch updates
                train_state, total_loss = jax.lax.scan(
                    lambda state, data: _update_minbatch(state, data, config),  
                    train_state, 
                    minibatches
                )

                return (train_state, traj_batch, advantages, targets, rng), total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                lambda state, _: _update_epoch(state, _, config),
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

            runner_state = (train_state, env_state, last_obs, update_step, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, 0, _rng)
        runner_state, metrics = jax.lax.scan(
            lambda state, unused: _update_step(state, unused, pretrained_params),  
            runner_state,
            None,
            config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train

@hydra.main(version_base=None, config_path="config", config_name="coord_ring")
def main(config):
    print("\nConfig Debug:")
    print("Raw config content:", config)
    print("Config type:", type(config))
    print("Config composition:", hydra.core.hydra_config.HydraConfig.get().runtime.config_sources)
    
    # Hydra's config path
    print("\nHydra Config Info:")
    print(f"Config Path: {hydra.utils.get_original_cwd()}/config")
    
    #print current working directory
    print(f"Current Directory: {os.getcwd()}")
    
    #print absolute path to config
    config_path = os.path.abspath(os.path.join(hydra.utils.get_original_cwd(), "config"))
    print(f"Absolute Config Path: {config_path}")

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
        tags=["IPPO", "FF", "Adaptability", "Oracle"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f'op_{layout_name}'
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
    model_dir_name = f"op_{layout_name}_{timestamp}"
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
    train_jit = jax.jit(lambda rng: train_fn(rng, pretrained_params))
    out = jax.vmap(train_jit)(rngs)

    # Generate and save visualization
    try:
        train_state = jax.tree_util.tree_map(lambda x: x[0], out["runner_state"][0])
        viz_base_name = f"op_{layout_name}"
        viz_filename = os.path.join(save_dir, f'{viz_base_name}_{config["SEED"]}.gif')
        create_visualization(train_state, pretrained_params, config, viz_filename, save_dir)
    except Exception as e:  
        print(f"Error generating visualization: {e}")
        traceback.print_exc()

    # Save parameters and results
    save_training_results(save_dir, out, config, prefix="op_")
    np.savez(os.path.join(save_dir, "op_metrics.npz"), 
            **{key: np.array(value) for key, value in out["metrics"].items()})
    
    with open(os.path.join(save_dir, "config.pkl"), 'wb') as f:
        pickle.dump(config, f)

    print(f"Training results saved to: {save_dir}")
    
    print('** Saving Results **')
    print("Original shape:", out["metrics"]["returned_episode_returns"].shape)
    print("Raw values before mean:", out["metrics"]["returned_episode_returns"][:10])  # First 10 values

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
    
    learning_curve_name = f"op_learning_curve"
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{learning_curve_name}.png'))
    plt.close()

if __name__ == "__main__":
    main()