"""
[1] Only need to load the parameters from pre-trained of the partner agent.
[1] Everytime network.apply is being called is when we need to passed along either trainstate.params or pretrained parameters from fixed partners.
    [1] Basically, just used the load_training_results for network.apply agent_1 to load and sample properly, sample equal to numbers of NUM_ENVS.
    [0] Pretty sure that in get_rollout, the while not done is to apply to each seed multiple times so we could just used from sampled. <- Double check this
[1] Network initialization need to be done only for agent_0.
[1] Update advantages, train states etc. only on self.
    [1] As for fixed just keep using the fix params no updates.
    [0] Double check the size of dictionary. We can use agent_0 data directly or wrapped under dict if dimension mismatched.
[1] There shouldn't be much changed at all to this version except for loading the save params from pre-trained.
[1] Need to get the save and visualization functions correctly from baseline.
[0] Need to plot to wandb correctly, either each seed individually or find the right aggregating methods for all seeds.
    [0] Honestly, perhaps look individually since this might be relevant to policy coming from different SECs
        [0] This could also reveal characteristics of the partner policy or self that make it s.t. total reward is higher.
        [0] This might reveal info behaviorally that may have been previously neglected.
[1] Maybe look and compare to the oracle_shared as well on the batchify and unbatchify. -> handle this a bit differently through _env_step instead
[0] Double check in the all dictionary elements that we can use agent_0 directly 
    [0] i.e. train_state can be directly equal to train_state_agent_0
    [1] auxilary lost, don't need combined_aux and agent_0 aux is enough

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
import random
# Plotting imports
import matplotlib.pyplot as plt

# Helper imports
from functools import partial
import random

class ActorCritic(nn.Module):
    """Neural network architecture implementing both policy (actor) and value function (critic)

    Attributes:
        action_dim: Dimension of action space
        activation: Activation function to use (either "relu" or "tanh")
    """
    action_dim: Sequence[int]  # Dimension of action space
    activation: str = "tanh"   # Activation function to use

    def setup(self):
        """Initialize layers and activation function.
        This runs once when the model is created.
        """
        #print("Setup method called")

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

    @nn.compact
    def __call__(self, x):
        """Forward pass of the network.
        
        Args:
            x: Input tensor with shape (batch_size, input_dim)
               where input_dim is either base_obs_dim or base_obs_dim + action_dim
               
        Returns:
            Tuple of (action distribution, value estimate)
        """
        # # Print debug information about input shape
        print("Network input x shape:", x.shape)
        print("ActorCritic input shape:", x.shape)
        
        # Expected input dimension is the last dimension of the input tensor
        expected_dim = x.shape[-1] if len(x.shape) > 1 else x.shape[0]
        print(f"Expected input dim: {expected_dim}")

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

        return pi, jnp.squeeze(critic, axis=-1)

class Transition(NamedTuple):
    """Container for storing experience transitions

    Attributes:
        done: Episode termination flag
        action: Actual action taken by the agent
        value: Value function estimate
        reward: Reward received
        log_prob: Log probability of action
        obs: Observation
    """
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray

def get_rollout(train_state, agent_1_params, is_shared_params, config, save_dir=None):
    """Generate a single episode rollout for visualization.
    
    Runs a single episode in the environment using the current policy networks to generate
    actions. Used for visualizing agent behavior during training.
    
    Args:
        train_state: Current training state containing network parameters
        agent_1_params: Pretrained parameters for partner agent
        is_shared_params: Boolean flag for shared vs. separate parameters
        config: Dictionary containing environment and training configuration
        
    Returns:
        Episode trajectory data including states, rewards, and shaped rewards
    """
    if "DIMS" not in config:
        raise ValueError("Config is missing DIMS dictionary. Check that dimensions were properly initialized in main()")
    
    dims = config["DIMS"]

    print("\nRollout Dimensions:")
    print(f"Base observation shape: {dims['base_obs_shape']} -> {dims['base_obs_dim']}")
    print(f"Action dimension: {dims['action_dim']}")
    print(f"Augmented observation dim: {dims['augmented_obs_dim']}\n")

    # Initialize environment
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    # Verify dimensions
    assert np.prod(env.observation_space().shape) == dims["base_obs_dim"], \
        "Observation dimension mismatch in rollout"
    assert env.action_space().n == dims["action_dim"], \
        "Action dimension mismatch in rollout"

    # Initialize network
    network = ActorCritic(
        action_dim=dims["action_dim"],
        activation=config["ACTIVATION"]
    )
    
    # Initialize seeds
    key = jax.random.PRNGKey(0)
    key, key_a, key_r = jax.random.split(key, 3)

    # Reset environment before using obs
    obs, state = env.reset(key_r)

    state_seq = [state]
    rewards = []
    shaped_rewards = []

    done = False

    while not done:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        # Process partner agent (Baseline, no augmentation)
        agent_1_obs = obs["agent_1"].flatten()

        if is_shared_params:
            pi_1, _ = network.apply(agent_1_params, agent_1_obs)
        else:
            pi_1, _ = network.apply(agent_1_params, agent_1_obs)
        
        action_1 = pi_1.sample(seed=key_a1)

        # Process learning agent (Oracle: Augmented, Baseline: Raw)
        agent_0_obs = obs["agent_0"].flatten()

        if config["AGENT_TYPE"] == "oracle":
            one_hot_action = jax.nn.one_hot(action_1, env.action_space().n)
            agent_0_obs = jnp.concatenate([agent_0_obs, one_hot_action])

        pi_0, _ = network.apply(train_state.params, agent_0_obs)
        action_0 = pi_0.sample(seed=key_a0)

        actions = {
            "agent_0": action_0,
            "agent_1": action_1
        }

        obs, state, reward, done, info = env.step(key_s, state, actions)
        done = done["__all__"]

        rewards.append(reward['agent_0'])
        shaped_rewards.append(info["shaped_reward"]['agent_0'])

        state_seq.append(state)

    # Plot rewards
    import matplotlib.pyplot as plt

    plt.plot(rewards, label="reward")
    plt.plot(shaped_rewards, label="shaped_reward")
    plt.legend()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "reward_plot.png"))
    plt.close()

    return state_seq

def batchify(x: dict, agent_list, num_actors):
    """Converts individual agent observations into a batched tensor."""
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    """
    Splits a batched tensor of observations into individual agent observations.

    Args:
        x (jnp.ndarray): A batched tensor of shape `(num_actors, num_envs, -1)`.
        agent_list (list): List of agent identifiers corresponding to the observations.
        num_envs (int): Number of parallel environments.
        num_actors (int): Number of actors (e.g., agents or agent-environment pairs).

    Returns:
        dict: A dictionary mapping agent identifiers to their respective observations.

    Purpose:
    - Converts batched observations back into individual agent-specific observations 
    for environment interactions.
    - Maintains compatibility between network outputs and environment inputs.
    """
    x = x.reshape((num_actors, num_envs, -1)) # This reshapes the observation space to a 3D array with the shape (num_actors, num_envs, -1)
    return {a: x[i] for i, a in enumerate(agent_list)} # This returns the observation space for the agents

def validate_batching(pretrained_params, transitions, config):
    """Validate correct batching and partner assignment."""
    # Check partner parameter consistency
    assert len(pretrained_params) == config["NUM_ENVS"], \
        f"Mismatch in partner parameters: {len(env_partner_params)} vs {config['NUM_ENVS']} environments"
    
    # Validate transition shapes
    expected_batch_shape = (config["NUM_ENVS"],)
    assert transitions.done.shape == expected_batch_shape, \
        f"Wrong done shape: {transitions.done.shape} vs {expected_batch_shape}"
    assert transitions.action.shape == expected_batch_shape, \
        f"Wrong action shape: {transitions.action.shape} vs {expected_batch_shape}"
    
    # Validate observation augmentation
    expected_obs_shape = (config["NUM_ENVS"], config["DIMS"]["augmented_obs_dim"])
    assert transitions.obs.shape == expected_obs_shape, \
        f"Wrong observation shape: {transitions.obs.shape} vs {expected_obs_shape}"
    
    # Print first few steps for manual inspection
    print("\nBatching Validation:")
    print(f"Number of environments: {config['NUM_ENVS']}")
    print(f"Partner parameter sets: {len(env_partner_params)}")
    print(f"Observation shape: {transitions.obs.shape}")
    print(f"Action shape: {transitions.action.shape}")
    
    return True

def save_training_results(save_dir, out, config, prefix=""):
    """
    Save both complete training output and seed-specific parameters in JAX and pickle/npz formats.

    Args:
        save_dir: Directory to save results
        out: Complete output from jax.vmap(train_jit)(rngs)
        config: Config containing NUM_SEEDS
        prefix: Optional prefix for filenames
    """
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
    pickle_out_path = os.path.join(save_dir, f"{prefix}complete_out.pkl")
    with open(pickle_out_path, 'wb') as f:
        pickle.dump(pickle_safe_out, f)
    print(f"Saved complete training output in pickle format: {pickle_out_path}")

    # Save the complete output in npz format (for compatibility)
    npz_out_path = os.path.join(save_dir, f"{prefix}complete_out.npz")
    np.savez(npz_out_path, **pickle_safe_out)
    print(f"Saved complete training output in npz format: {npz_out_path}")

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
                    processed_state['params'] = processed_params
            
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
        pickle_seeds_path = os.path.join(save_dir, f"params.pkl")
        with open(pickle_seeds_path, 'wb') as f:
            pickle.dump(all_seeds_params, f)
        print(f"Saved all seed-specific parameters in pickle format: {pickle_seeds_path}")
        
        # Then save as npz for compatibility
        npz_seeds_path = os.path.join(save_dir, f"{prefix}all_seeds_params.npz")
        np.savez(npz_seeds_path, **all_seeds_params)
        print(f"Saved all seed-specific parameters in npz format: {npz_seeds_path}")
    else:
        print("Warning: No seed-specific parameters were successfully processed")

def load_training_results(load_dir, load_type="params", config=None):
    """
    Load training results from pickle format
    
    Args:
        load_dir: Directory containing saved files
        prefix: Prefix used in filenames
        load_type: Either "params" or "complete" to load just params or complete output
    Returns:
        Loaded data converted to JAX arrays where appropriate
    """
    key = jax.random.PRNGKey(0) 
    key, subkey = jax.random.split(key)

    if load_type == "params":
        pickle_path = os.path.join(load_dir, f"params.pkl")
        if os.path.exists(pickle_path):
            print("Loading params from pickle format...")
            with open(pickle_path, 'rb') as f:
                all_params = pickle.load(f)

            # Ensure correct structure
            if not isinstance(all_params, dict) or "params" not in all_params:
                raise ValueError(f"Invalid parameter structure in {pickle_path}. Expected a dict with 'params' key.")

            # Convert parameters to JAX-compatible format
            all_params = jax.tree_util.tree_map(jnp.array, all_params)
            
            num_seeds = len(all_params)
            seed_idx = jax.random.randint(subkey, shape=(), minval=0, maxval=num_seeds)
            sampled_params = jax.tree_util.tree_map(lambda x: x[seed_idx], all_params)
            sampled_params = flax.core.freeze(sampled_params)

            # print(sampled_params)

            print("Successfully loaded pretrained model.")
            print("Loaded params type:", type(sampled_params))  # Should be <FrozenDict>
            print("Keys in params:", sampled_params.keys())
            # print("Shape of sampled_params:", jax.tree_map(lambda x: x.shape, sampled_params))
            
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

# def load_pretrained_params(path, is_complete=False):
#     """
#     Simple function to load pretrained parameters from a file.
    
#     Args:
#         path: Direct path to the parameter file
#         is_complete: Whether to load complete training output
#     """
#     # Debug print
#     print(f"Attempting to load from path: {path}")
#     print(f"Path type: {type(path)}")
#     load_path = os.path.join(path, "params.pkl")

#     if isinstance(path, dict):
#         print("Path contents:", path)
#         raise TypeError("Path is a dictionary instead of a string path")
        
#     if not os.path.exists(str(path)):
#         raise FileNotFoundError(f"No file found at: {path}")
        
#     with open(path, 'rb') as f:
#         data = pickle.load(f)
        
#     return jax.tree_util.tree_map(
#         lambda x: jax.numpy.array(x) if isinstance(x, np.ndarray) else x,
#         data
#     )

# Almost correct, but not quite
# def prepare_training_params(config):
#     load_path = os.path.join(config["LOAD_PATH"], "params.pkl")

#     with open(load_path, 'rb') as f:
#         loaded_params = pickle.load(f)  # Directly loads FrozenDict

#     # Ensure JAX-compatible format
#     loaded_params = jax.tree_util.tree_map(jnp.array, loaded_params)

#     # Create train state
#     network = ActorCritic(
#         action_dim=config["DIMS"]["action_dim"], 
#         activation=config["ACTIVATION"]
#     )

#     tx = optax.chain(
#         optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
#         optax.adam(config["LR"], eps=1e-5)
#     )

#     fixed_train_state = TrainState.create(
#         apply_fn=network.apply,
#         params=loaded_params,  # Directly use the loaded params
#         tx=tx
#     )

#     return fixed_train_state

# Close to correct
# def prepare_training_params(config):
#     """Loads and prepares pre-trained parameters for JAX training."""
#     load_path = os.path.join(config["LOAD_PATH"], "params.pkl")

#     # Load parameters
#     with open(load_path, 'rb') as f:
#         all_params = pickle.load(f)

#     # If multiple seeds exist, sample `NUM_ENVS` parameter sets
#     if isinstance(all_params, dict) and len(all_params) > 1:
#         print(f"Sampling {config['NUM_ENVS']} parameter sets from 100 available seeds.")
#         sampled_seeds = random.sample(list(all_params.keys()), config["NUM_ENVS"])

#         # Stack parameters across `NUM_ENVS`
#         params = {k: jnp.stack([all_params[seed][k] for seed in sampled_seeds], axis=0)
#                   for k in all_params[sampled_seeds[0]]}
#     else:
#         print("Warning: No multiple seeds detected, using saved parameters as-is.")
#         params = all_params

#     # Convert parameters to JAX-compatible format
#     params = jax.tree_util.tree_map(jnp.array, params)

#     return params

# def prepare_training_params(config):
#     load_path = os.path.join(config["LOAD_PATH"], "params.pkl")
    
#     with open(load_path, 'rb') as f:
#         loaded_params = pickle.load(f)
    
#     if isinstance(loaded_params, dict) and 'params' in loaded_params:
#         # Keep parameters in their original Flax format
#         params = loaded_params['params']
#         # Don't stack the parameters - keep them as a single set
#         print(f"Loaded pretrained parameters for {config['NUM_ENVS']} environments")
#         return params
    
#     raise ValueError(f"Unexpected parameter structure in {load_path}")

# def prepare_training_params(config):
#     """Load and prepare pretrained parameters for NUM_ENVS environments."""
#     load_path = os.path.join(config["LOAD_PATH"], "params.pkl")
    
#     with open(load_path, 'rb') as f:
#         loaded_params = pickle.load(f)
    
#     print("Loaded parameters structure:", type(loaded_params))
    
#     if isinstance(loaded_params, dict) and 'params' in loaded_params:
#         # Get the single parameter set
#         params = loaded_params['params']
        
#         # Stack the parameters NUM_ENVS times using JAX operations
#         stacked_params = jax.tree_map(
#             lambda x: jnp.stack([x] * config["NUM_ENVS"]),
#             params
#         )
        
#         print(f"Prepared parameters for {config['NUM_ENVS']} environments")
#         return stacked_params
    
#     raise ValueError(f"Unexpected parameter structure in {load_path}")

# def prepare_training_params(config):
#     """Simplified partner parameter loading"""
#     load_path = os.path.join(config["LOAD_PATH"], "params.pkl")
    
#     with open(load_path, 'rb') as f:
#         loaded_params = pickle.load(f)

#     # Debug prints
#     print("Type of loaded_params:", type(loaded_params))
#     print("Keys if dict:", loaded_params.keys() if isinstance(loaded_params, dict) else "Not a dict")
#     print("Full structure:", loaded_params)
    
#     # Extract all available parameter sets
#     param_sets = []
#     if isinstance(loaded_params, dict):
#         for seed_key in loaded_params:
#             if seed_key.startswith('seed_'):
#                 param_sets.append(loaded_params[seed_key]['params'])
#     else:
#         param_sets = [loaded_params]
    
#     num_params = len(param_sets)
#     print(f"Found {num_params} parameter sets")
    
#     if num_params == 0:
#         raise ValueError("No valid parameter sets found in loaded file")
    
#     # If we need more parameter sets than available, cycle through existing ones
#     if config["NUM_ENVS"] > num_params:
#         param_sets = param_sets * (config["NUM_ENVS"] // num_params + 1)
    
#     # Generate indices that are guaranteed to be in range
#     rng = jax.random.PRNGKey(config["SEED"])
#     indices = jax.random.randint(rng, (config["NUM_ENVS"],), 0, len(param_sets))
    
#     selected_params = [param_sets[i] for i in indices]
#     print(f"Selected {len(selected_params)} parameter sets for {config['NUM_ENVS']} environments")
    
#     return selected_params

# def prepare_training_params(config):
#     """Load and prepare parameters for multiple parallel environments."""
#     loaded_data = load_pretrained_params(config["LOAD_PATH"])
    
#     # Get number of available parameter sets
#     if isinstance(loaded_data, dict) and 'seed_0' in loaded_data:
#         param_sets = [loaded_data[f'seed_{i}']['params'] 
#                      for i in range(len(loaded_data)) 
#                      if f'seed_{i}' in loaded_data]
#     else:
#         param_sets = [loaded_data]
        
#     # Ensure we have enough parameter sets
#     if len(param_sets) < config["NUM_ENVS"]:
#         print(f"Warning: Only {len(param_sets)} parameter sets available for {config['NUM_ENVS']} environments")
#         # Cycle through parameters if needed
#         param_sets = param_sets * (config["NUM_ENVS"] // len(param_sets) + 1)
        
#     # Sample NUM_ENVS partners randomly
#     rng = jax.random.PRNGKey(config["SEED"])
#     indices = jax.random.permutation(rng, len(param_sets))[:config["NUM_ENVS"]]
    
#     # Create a fixed mapping of environment to partner parameters
#     env_partner_params = {
#         i: param_sets[idx] for i, idx in enumerate(indices)
#     }
    
#     return env_partner_params

def create_visualization(train_state, config, filename, save_dir=None, agent_view_size=5):
    """Helper function to create and save visualization"""
    # Ensure we have a clean filename
    base_name = os.path.splitext(os.path.basename(filename))[0]
    clean_filename = f"{base_name}.gif"  # Force .gif extension
    
    # Get the rollout
    state_seq = get_rollout(train_state, config, save_dir)
    
    # Create visualization
    viz = OvercookedVisualizer()
    
    # Save with clean filename
    if save_dir:
        clean_filename = os.path.join(save_dir, clean_filename)
    viz.animate(state_seq, agent_view_size=agent_view_size, filename=clean_filename)

def create_safe_filename(base_name, config, timestamp=None):
    """Creates a safe filename for saving visualizations and plots"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract just the layout name without the full configuration
    layout_name = config["ENV_KWARGS"].get("layout_name", "default")
    if isinstance(layout_name, (dict, jax.Array, np.ndarray)):
        layout_name = "custom_layout"
    
    # Generate a safe filename
    safe_name = f"{base_name}_{layout_name}_{timestamp}"
    
    # Remove any unsafe characters
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in "_-")
    
    return safe_name

def make_train(config):
    """Creates the main training function for IPPO with the given configuration.
    
    This function sets up the training environment, networks, and optimization process
    for training multiple agents using Independent PPO (IPPO). It handles:
    - Environment initialization and wrapping
    - Network architecture setup for both agents
    - Learning rate scheduling and reward shaping annealing
    - Training loop configuration including batch sizes and update schedules
    
    Args:
        config: Dictionary containing training hyperparameters and environment settings
               including:
               - DIMS: Environment dimensions
               - ENV_NAME: Name of environment to train in
               - ENV_KWARGS: Environment configuration parameters
               - NUM_ENVS: Number of parallel environments
               - NUM_STEPS: Number of steps per training iteration
               - TOTAL_TIMESTEPS: Total environment steps to train for
               - Learning rates, batch sizes, and other optimization parameters
               
    Returns:
        train: The main training function that takes an RNG key and executes the full
               training loop, returning the trained agent policies
    """
    
    # Initialize environment
    dims = config["DIMS"]
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    # Verify dimensions match what we validated in main
    assert np.prod(env.observation_space().shape) == dims["base_obs_dim"], "Observation dimension mismatch"
    assert env.action_space().n == dims["action_dim"], "Action dimension mismatch"

    # Load pretrained parameters once at this level
    pretrained_params = load_training_results(config["LOAD_PATH"], load_type="params", config=config)

    # Calculate key training parameters
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"] 
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # Configuration printing
    print("Initializing training with config:")
    print(f"NUM_ENVS: {config['NUM_ENVS']}")
    print(f"NUM_STEPS: {config['NUM_STEPS']}")
    print(f"NUM_UPDATES: {config['NUM_UPDATES']}")
    print(f"NUM_MINIBATCHES: {config['NUM_MINIBATCHES']}")
    print(f"TOTAL_TIMESTEPS: {config['TOTAL_TIMESTEPS']}")
    print(f"ENV_NAME: {config['ENV_NAME']}")
    print(f"DIMS: {config['DIMS']}")
    
    env = LogWrapper(env, replace_info=False)
    
    def linear_schedule(count):
        """Linear learning rate annealing schedule that decays over training.
        
        Calculates a learning rate multiplier that decreases linearly from 1.0 to 0.0
        over the course of training. Used to gradually reduce the learning rate to help
        convergence.
        
        Args:
            count: Current training step count used to calculate progress through training
        
        Returns:
            float: The current learning rate after applying the annealing schedule,
                  calculated as: base_lr * (1 - training_progress)
        """
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac
    
    # Schedule for annealing reward shaping
    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.,
        end_value=0.,
        transition_steps=config["REW_SHAPING_HORIZON"]
    )

    def train(rng):
        """Main training loop for Independent PPO (IPPO) algorithm.
        
        Implements the core training loop for training multiple agents using IPPO.
        Handles network initialization, environment setup, and training iteration.
        
        Args:
            rng: JAX random number generator key for reproducibility
            
        Returns:
            Tuple containing:
            - Final trained network parameters for both agents
            - Training metrics and statistics
            - Environment states from training
            
        The training process:
        1. Initializes separate policy networks for each agent
        2. Collects experience in parallel environments
        3. Updates policies using PPO with independent value functions
        4. Tracks and logs training metrics
        """
        # Shapes we're initializing with
        #print("Action space:", env.action_space().n)
        #print("Observation space shape:", env.observation_space().shape)

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
            """Creates an optimizer chain for training each agent's neural network.
            
            The optimizer chain consists of:
            1. Gradient clipping using global norm
            2. Adam optimizer with either:
            - Annealed learning rate that decays linearly over training
            - Fixed learning rate specified in config
            
            Args:
                config: Dictionary containing optimization parameters like:
                    - ANNEAL_LR: Whether to use learning rate annealing
                    - MAX_GRAD_NORM: Maximum gradient norm for clipping
                    - LR: Base learning rate
                    
            Returns:
                optax.GradientTransformation: The composed optimizer chain
            """
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

        # Store agent_1's pretrained params separately
        # pretrained_params = load_pretrained_params(config)
        # agent_1_params = pretrained_params  # From load_and_process_pretrained
        
        # Initialize environment states
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        
        # TRAIN LOOP
        def _update_step(runner_state, unused):
            """Executes a single training update step in the IPPO algorithm.
            
            This function performs one complete update iteration including:
            1. Collecting trajectories by running the current policy in the environment
            2. Computing advantages and returns
            3. Updating both agents' neural networks using PPO
            
            The update handles both agents (agent_0 and agent_1) separately, with agent_0
            receiving augmented observations that include agent_1's actions.
            
            Args:
                runner_state: Tuple containing:
                    - train_state: Current training state with network parameters
                    - env_state: Current environment state
                    - last_obs: Previous observations from environment
                    - update_step: Current training iteration number
                    - rng: Random number generator state
                unused: Placeholder parameter for JAX scan compatibility
                
            Returns:
                Tuple containing:
                    - Updated runner_state
                    - Metrics dictionary with training statistics
            """
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused, dims, pretrained_params):
                train_state, env_state, last_obs, update_step, rng = runner_state
                rng, step_rng = jax.random.split(rng)

                # Separate observations by environment
                batch_size = last_obs['agent_0'].shape[0]  # Should equal NUM_ENVS
                print(f"batch_size: {batch_size}")
                
                # Process each environment with its fixed partner
                def process_single_env(env_idx, obs_and_rng):
                    obs, env_rng = obs_and_rng

                    # Process partner agent (Baseline, fixed)
                    partner_obs = obs['agent_1']
                    print("partner_obs.shape BEFORE vmap:", partner_obs.shape)
                    partner_obs = partner_obs.transpose(1, 0, 2).reshape(-1, 520)  # Ensure correct batching
                    print("partner_obs.shape BEFORE vmap:", partner_obs.shape)  # Should be (NUM_ENVS, 520)

                    pi_partner, _ = jax.vmap(network.apply, in_axes=(None, 0))(
                        pretrained_params, 
                        partner_obs
                    )
                    print("pi_partner.shape AFTER vmap:", pi_partner.batch_shape)  # Should be (NUM_ENVS,)

                    partner_action = pi_partner.sample(seed=env_rng)

                    # Process learning agent (Oracle)
                    agent_obs = obs['agent_0'].reshape(-1)  # Flatten to (NUM_ENVS, base_obs_dim)
                    partner_action_onehot = jax.nn.one_hot(partner_action, dims["action_dim"])
                    agent_obs_aug = jnp.concatenate([agent_obs, partner_action_onehot], axis=-1)

                    pi_agent, value_agent = network.apply(train_state.params, agent_obs_aug)
                    agent_action = pi_agent.sample(seed=env_rng)

                    return {
                        'agent_0': {
                            'action': agent_action,
                            'value': value_agent,
                            'log_prob': pi_agent.log_prob(agent_action),
                            'obs_aug': agent_obs_aug
                        },
                        'agent_1': {
                            'action': partner_action
                        }
                    }

                # Generate per-environment RNGs
                env_rngs = jax.random.split(step_rng, batch_size)
                
                # Process all environments in parallel
                env_outputs = jax.vmap(process_single_env)(
                    jnp.arange(batch_size),
                    (last_obs, env_rngs)
                )

                # Collect actions for environment step
                actions = {
                    'agent_0': env_outputs['agent_0']['action'],
                    'agent_1': env_outputs['agent_1']['action']
                }

                # Debug action shapes before stepping environment
                print("Actions shape before env.step() - agent_0:", actions["agent_0"].shape)
                print("Actions shape before env.step() - agent_1:", actions["agent_1"].shape)

                # Step environments
                next_obs, next_state, rewards, dones, infos = jax.vmap(env.step)(
                    jax.random.split(rng, batch_size), 
                    env_state, 
                    actions
                )

                # Create transition for learning agent only
                transition = Transition(
                    done=dones['agent_0'],
                    action=env_outputs['agent_0']['action'],
                    value=env_outputs['agent_0']['value'],
                    reward=rewards['agent_0'],
                    log_prob=env_outputs['agent_0']['log_prob'],
                    obs=env_outputs['agent_0']['obs_aug']
                )

                # Debug expected shapes before validation
                print(f"Expected NUM_ENVS: {config['NUM_ENVS']}")
                print(f"pretrained_params length: {len(pretrained_params)}")  # Should be NUM_ENVS
                print(f"transitions.obs shape: {transition.obs.shape}")  # Should be (NUM_ENVS, augmented_obs_dim)

                # Verify that the batching is correct
                validation_passed = validate_batching(pretrained_params, transition, config)
                if not validation_passed:
                    raise ValueError("Batching validation failed!")

                runner_state = (train_state, next_state, next_obs, update_step, rng)
                return runner_state, (transition, infos)


            runner_state, (traj_batch, info, processed_obs) = jax.lax.scan(
                partial(_env_step, dims=dims, pretrained_params=pretrained_params), 
                runner_state, None, config["NUM_STEPS"]
            )
            
            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, update_step, rng = runner_state
            
            # Calculate last values for agent_0
            # For agent_0, need to include agent_1's last action in observation
            one_hot_last_action = jax.nn.one_hot(traj_batch.action[-1, 1], env.action_space().n)
            last_obs_agent0 = last_obs['agent_0'].reshape(last_obs['agent_0'].shape[0], -1)
            last_obs_agent0_augmented = jnp.concatenate([last_obs_agent0, one_hot_last_action], axis=-1)
            _, agent_0_last_val = network.apply(train_state['agent_0'].params, last_obs_agent0_augmented)
            print("agent_0_last_val shape:", agent_0_last_val.shape)

            # Combine values for advantage calculation
            _, last_val = network.apply(train_state.params, last_obs_agent0_augmented)

            # calculate_gae itself didn't need to be changed because we can use the same advantage function for both agents
            def _calculate_gae(traj_batch, last_val):
                """Calculate Generalized Advantage Estimation (GAE) for trajectories.
                
                This function computes the GAE for a given trajectory batch and last value,
                which are used to estimate the advantage of each action in the trajectory.

                Args:
                    traj_batch: Trajectory batch containing transitions
                    last_val: Last value estimates for the trajectory
                    
                Returns:
                    Tuple containing:
                        - Advantages for the trajectory
                        - Returns (advantages + value estimates)
                """
                print(f"\nGAE Calculation Debug:")
                print("traj_batch types:", jax.tree_map(lambda x: x.dtype, traj_batch))
                print(f"traj_batch shapes:", jax.tree_map(lambda x: x.shape, traj_batch))
                print("last_val types:", jax.tree_map(lambda x: x.dtype, last_val))
                print(f"last_val shape: {last_val.shape}")
                

                def _get_advantages(gae_and_next_value, transition):
                    """Calculate GAE and returns for a single transition.
                    
                    This function computes the GAE and returns for a single transition,
                    which are used to update the policy and value functions.
                    
                    Args:
                        gae_and_next_value: Tuple containing current GAE and next value
                        transition: Single transition containing data for one step
                    
                    Returns:
                        Tuple containing:
                            - Updated GAE and next value
                            - Calculated GAE
                    """
                    gae, next_value = gae_and_next_value
                    
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
        
                     # Calculate delta and GAE per agent
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    print(f"delta shape: {delta.shape}, value: {delta}")

                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    print(f"calculated gae shape: {gae.shape}, value: {gae}")
                    
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

                # Calculate returns (advantages + value estimates)
                print(f"\nFinal shapes:")
                print(f"advantages shape: {advantages.shape}")
                print(f"returns shape: {(advantages + traj_batch.value).shape}")
                return advantages, advantages + traj_batch.value

            # Calculate advantages 
            advantages, targets = _calculate_gae(traj_batch, last_val)
            print("advantages shape:", advantages.shape)
            print("targets shape:", targets.shape)
            print("traj_batch value shape:", traj_batch.value.shape)
            print("traj_batch reward shape:", traj_batch.reward.shape)
            print("traj_batch data types:", jax.tree_map(lambda x: x.dtype, traj_batch))

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                """
                Performs a complete training epoch for both agents.
                
                Args:
                    update_state: Tuple containing (train_state, traj_batch, advantages, targets, rng)
                    unused: Placeholder for scan compatibility
                    
                Returns:
                    Updated state and loss information
                """
                def _update_minbatch(train_state, batch_info):
                    """Updates network parameters using a minibatch of experience.
                    
                    Args:
                        train_state: Current training state containing both agents' parameters
                        batch_info: Tuple of (traj_batch, advantages, targets) for both agents
                        
                    Returns:
                        Updated training state and loss information
                    """
                    print("\nStarting minibatch update...")
                    # Unpack batch_info which now contains separate agent data
                    agent_0_data, agent_1_data = batch_info['agent_0'], batch_info['agent_1']
                    
                    print("Minibatch shapes:")
                    print("Agent 0 data:", jax.tree_map(lambda x: x.shape, agent_0_data))
                    print("Agent 1 data:", jax.tree_map(lambda x: x.shape, agent_1_data))


                    # Loss function itself didn't need to be changed because we can use the same loss function for both agents
                    # def _loss_fn(params, traj_batch, gae, targets, agent_type):
                    #     """Calculate loss for a single agent.
                        
                    #     This function computes the loss for a single agent, which is used
                    #     to update the policy and value functions.
                        
                    #     Args:
                    #         params: Network parameters for the agent
                    #         traj_batch: Trajectory batch containing transitions
                    #         gae: Generalized Advantage Estimation (GAE) for the trajectory
                    #         targets: Target values (advantages + value estimates) for the trajectory
                        
                    #     Returns:
                    #         Tuple containing:
                    #             - Total loss for the agent
                    #             - Auxiliary loss information (value loss, actor loss, entropy)
                    #     """
                    #     # RERUN NETWORK
                    #     print(f"\nCalculating losses for {agent_type}...")
                    #     print(f"Input obs shape: {traj_batch.obs.shape}")
                    #     pi, value = network.apply(params, traj_batch.obs)
                    #     print(f"Network outputs - pi shape: {pi.batch_shape}, value shape: {value.shape}")
                    #     log_prob = pi.log_prob(traj_batch.action)
                    #     print(f"Log prob shape: {log_prob.shape}")

                    #     # CALCULATE VALUE LOSS
                    #     value_pred_clipped = traj_batch.value + (
                    #         value - traj_batch.value
                    #     ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    #     value_losses = jnp.square(value - targets)
                    #     value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    #     value_loss = (
                    #         0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    #     )

                    #     # CALCULATE ACTOR LOSS
                    #     ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    #     print(f"Importance ratio shape: {ratio.shape}")
                    #     gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    #     print(f"Normalized GAE shape: {gae.shape}")
                    #     loss_actor1 = ratio * gae
                    #     loss_actor2 = (
                    #         jnp.clip(
                    #             ratio,
                    #             1.0 - config["CLIP_EPS"],
                    #             1.0 + config["CLIP_EPS"],
                    #         )
                    #         * gae
                    #     )
                    #     loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    #     loss_actor = loss_actor.mean()
                    #     entropy = pi.entropy().mean()

                    #     total_loss = (
                    #         loss_actor
                    #         + config["VF_COEF"] * value_loss
                    #         - config["ENT_COEF"] * entropy
                    #     )
                    #     loss_info = {
                    #         'value_loss': value_loss,
                    #         'actor_loss': loss_actor,
                    #         'entropy': entropy,
                    #         'total_loss': total_loss,
                    #         'grad_norm': None  # Will be filled later
                    #     }
                        
                    #     print(f"\nLoss breakdown for {agent_type}:")
                    #     for k, v in loss_info.items():
                    #         if v is not None:
                    #             print(f"{k}: {v}")
                        
                    #     return total_loss, loss_info
                    def _loss_fn(params, traj_batch, gae, targets):
                        """Calculate loss for agent_0."""
                        pi, value = network.apply(params, traj_batch.obs)
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
                        
                        entropy = pi.entropy().mean()
                        
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

                    # Create separate loss functions for each agent, and only keep the agent_0 since we don't need agent_1
                    loss_fn_agent_0 = partial(_loss_fn, agent_type='agent_0')
                    # loss_fn_agent_1 = partial(_loss_fn, agent_type='agent_1')

                    # Create gradient function, same as before
                    grad_fn_0 = jax.value_and_grad(loss_fn_agent_0, has_aux=True)
                    # grad_fn_1 = jax.value_and_grad(loss_fn_agent_1, has_aux=True)
    
                    # Compute gradients for agent 0
                    (loss_0, aux_0), grads_0 = grad_fn_0(
                        train_state.params,
                        traj_batch,
                        advantages,
                        targets
                    )

                    print("\nGradient stats:")
                    print(f"Grad norm agent_0: {optax.global_norm(grads_0)}")
                    # print(f"Grad norm agent_1: {optax.global_norm(grads_1)}")
                    
                    # Update only agent_0
                    train_state = train_state.apply_gradients(grads=grads_0)
                    
                    return train_state, (loss_0, aux_0)

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                # Calculate total batch size and minibatch size
                batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
                config["MINIBATCH_SIZE"] = batch_size // config["NUM_MINIBATCHES"]

                # Verify that the data can be evenly split into minibatches
                assert (
                    batch_size % config["NUM_MINIBATCHES"] == 0
                ), "Steps * Envs must be divisible by number of minibatches"               # Shape: (128, 16)

                #print("\nBatch processing, Pre-reshape diagnostics:")
                #print("agent_0 obs structure:", jax.tree_map(lambda x: x.shape if hasattr(x, 'shape') else type(x), agent_data['agent_0']['traj'].obs))
                #print("agent_1 obs structure:", jax.tree_map(lambda x: x.shape if hasattr(x, 'shape') else type(x), agent_data['agent_1']['traj'].obs))
                #print("Advantages shape:", advantages.shape)
                #print("Targets shape:", targets.shape)

                # Reshape function that handles the different observation sizes
                def reshape_agent_data(agent_dict):
                    """Reshape trajectory data for a single agent.
                    
                    This function reshapes the trajectory data for a single agent,
                    ensuring that the observations are reshaped correctly while keeping
                    the features dimension intact.
                    
                    Args:
                        agent_dict: Dictionary containing trajectory data for an agent
                    
                    Returns:
                        Dictionary containing reshaped trajectory data
                    """
                    def reshape_field(x, field_name):
                        """Reshape a single field of the trajectory data.
                        
                        This function reshapes a single field of the trajectory data,
                        ensuring that the observations are reshaped correctly while keeping
                        the features dimension intact.

                        Args:
                            x: Data to be reshaped
                            field_name: Name of the field to be reshaped
                        
                        Returns:
                            Reshaped data
                        """
                        if not isinstance(x, (dict, jnp.ndarray)):
                            return x
                            
                        if field_name == 'obs':
                            # Keep the features dimension intact, only combine timesteps and envs
                            #print(f"Reshaping {field_name} from {x.shape} to {(batch_size, -1)}")
                            return x.reshape(batch_size, -1)
                        else:
                            # For other fields, flatten to (batch_size,)
                            #print(f"Reshaping {field_name} from {x.shape} to {(batch_size,)}")
                            return x.reshape(batch_size)
                            
                    return {
                        'traj': Transition(
                            **{field: reshape_field(getattr(agent_dict['traj'], field), field)
                               for field in agent_dict['traj']._fields}
                        ),
                        'advantages': agent_dict['advantages'].reshape(batch_size),
                        'targets': agent_dict['targets'].reshape(batch_size)
                    }

                # Reshape each agent's data
                agent_data = {
                    agent: reshape_agent_data(data)
                    for agent, data in agent_data.items()
                }

                # After reshaping:
                print("\nPost-reshape diagnostics:")
                print("Reshaped agent_0 obs:", jax.tree_map(lambda x: x.shape if hasattr(x, 'shape') else type(x), agent_data['agent_0']['traj'].obs))
                # print("Reshaped agent_1 obs:", jax.tree_map(lambda x: x.shape if hasattr(x, 'shape') else type(x), agent_data['agent_1']['traj'].obs))

                # Create permutation and shuffle
                permutation = jax.random.permutation(_rng, batch_size)
                agent_data = {
                    agent: {
                        'traj': Transition(
                            **{field: jnp.take(getattr(data['traj'], field), permutation, axis=0)
                               for field in data['traj']._fields}
                        ),
                        'advantages': jnp.take(data['advantages'], permutation, axis=0),
                        'targets': jnp.take(data['targets'], permutation, axis=0)
                    }
                    for agent, data in agent_data.items()
                }

                print("\nShuffled batch structure:", jax.tree_map(lambda x: x.shape if hasattr(x, 'shape') else type(x), agent_data))

                # Create minibatches
                def create_minibatches(data):
                    """Create minibatches from trajectory data.
                    
                    This function divides the trajectory data into smaller minibatches,
                    which are used for training the policy and value networks.
                    
                    Args:
                        data: Dictionary containing trajectory data for an agent
                    
                    Returns:
                        Dictionary containing minibatched trajectory data
                    """
                    return {
                        'traj': Transition(
                            **{field: jnp.reshape(getattr(data['traj'], field), 
                                                [config["NUM_MINIBATCHES"], -1] + list(getattr(data['traj'], field).shape[1:]))
                               for field in data['traj']._fields}
                        ),
                        'advantages': data['advantages'].reshape(config["NUM_MINIBATCHES"], -1),
                        'targets': data['targets'].reshape(config["NUM_MINIBATCHES"], -1)
                    }

                minibatches = {
                    agent: create_minibatches(data)
                    for agent, data in agent_data.items()
                }

                print("\nMinibatches structure:", jax.tree_map(lambda x: x.shape if hasattr(x, 'shape') else type(x), minibatches))

                # Update networks using minibatches
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )

                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = info
            current_timestep = update_step*config["NUM_STEPS"]*config["NUM_ENVS"]
            metric["shaped_reward"] = metric["shaped_reward"]["agent_0"]
            metric["shaped_reward_annealed"] = metric["shaped_reward"]*rew_shaping_anneal(current_timestep)
            
            rng = update_state[-1]

            def callback(metric):
                """Log training metrics to wandb.
                
                This function logs the training metrics to wandb, which are used for
                monitoring and analysis during training.

                Args:
                    metric: Training metrics to be logged
                """
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
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train

@hydra.main(version_base=None, config_path="config", config_name="adaptability")
def main(config):
    """Main entry point for training
    
    Args:
        config: Hydra configuration object containing training parameters
        
    Returns:
        Training results and metrics
    
    Raises:
        ValueError: If the environment dimensions are invalid
    """
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
        name=f'adaptability_ff_ippo_oc_{config["ENV_KWARGS"]["layout"]}'
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
    model_dir_name = create_safe_filename("adaptability_ff_ippo_oc", config, timestamp)
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
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)

    # Save parameters and results
    save_training_results(save_dir, out, config, prefix="adaptability_")
    np.savez(os.path.join(save_dir, "metrics.npz"), 
             **{key: np.array(value) for key, value in out["metrics"].items()})
    
    with open(os.path.join(save_dir, "config.pkl"), 'wb') as f:
        pickle.dump(config, f)

    print(f"Training results saved to: {save_dir}")

    # Generate and save visualization
    train_state = jax.tree_util.tree_map(lambda x: x[0], out["runner_state"][0])
    viz_base_name = create_safe_filename("adaptability", config, timestamp)
    viz_filename = os.path.join(save_dir, f'{viz_base_name}_{config["SEED"]}.gif')
    create_visualization(train_state, config, viz_filename, save_dir)
    
    print('** Saving Results **')
    print("Original shape:", out["metrics"]["returned_episode_returns"].shape)
    print("Raw values before mean:", out["metrics"]["returned_episode_returns"][:10])  # First 10 values

    # Plot and save learning curves
    rewards = out["metrics"]["returned_episode_returns"].reshape((config["NUM_SEEDS"], -1))
    print("Shape of rewards:", rewards.shape)
    print("Values per seed:")
    for i in range(5):
        print(f"Seed {i}:", rewards[i])
    reward_mean = rewards.mean(0)
    print("First few values of reward_mean:", reward_mean[:5])
    print("Check for NaN:", np.isnan(reward_mean).any())
    print("Range of values:", np.min(reward_mean), np.max(reward_mean))
    reward_std = rewards.std(0) / np.sqrt(config["NUM_SEEDS"])
    
    plt.figure()
    plt.plot(reward_mean)
    plt.fill_between(range(len(reward_mean)), 
                    reward_mean - reward_std,
                    reward_mean + reward_std,
                    alpha=0.2)
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    
    learning_curve_name = create_safe_filename(f"{config['ENV_NAME']}_learning_curve", config, timestamp)
    plt.savefig(os.path.join(save_dir, f'{learning_curve_name}.png'))
    plt.close()

if __name__ == "__main__":
    main()