"""
This implementation builds upon the PureJaxRL PPO framework, adapting it for multi-agent
settings and incorporating enhancements specific to Overcooked-style environments.
"""

# Essential imports for JAX-based computation, multi-agent reinforcement learning (JAXMARL), and environment handling (Gymnasium).
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

"""
Actor-Critic Class for PPO.

This class defines a shared neural network architecture for policy optimization, 
consisting of separate actor and critic networks:

- The **actor network** predicts action probabilities using a two-layer fully connected 
  architecture, followed by a categorical distribution.
- The **critic network** predicts the value function using a similar two-layer structure, 
  outputting a single scalar value.

**Key Features**:
- Implements orthogonal kernel initialization (`np.sqrt(2)` scaling for hidden layers, 
  `0.01` for output layers), ensuring stable gradient updates.
- Zero-initialized biases (`constant(0.0)`) simplify optimization at early stages.
- Supports flexible activation functions (`tanh` or `relu`), defaulting to `tanh`.

**Why Orthogonal Initialization?**
Orthogonal initialization helps maintain variance in activations, ensuring better 
gradient flow during backpropagation. The scaling factor (`np.sqrt(2)` for hidden layers) 
balances this variance based on network depth.

**Activation Function Choices**:
- `tanh`: Bounded output with smoother gradients, particularly useful for smaller 
  architectures or environments requiring stable learning.
- `relu`: Unbounded output, often better for deeper networks but prone to exploding gradients.

**Outputs**:
1. `distrax.Categorical`: Encapsulates action probabilities for the actor network.
2. Scalar value estimate: Represents the critic's state-value prediction.

This modular design reduces computational overhead and ensures alignment between the 
actor and critic, promoting efficient policy learning.
"""

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
        #print("ActorCritic input shape:", x.shape)
        
        # Expected input dimension is the last dimension of the input tensor
        expected_dim = x.shape[-1] if len(x.shape) > 1 else x.shape[0]
        #print(f"Expected input dim: {expected_dim}")

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
    """
    A NamedTuple to encapsulate all necessary information from a single environment step.

    This class represents a transition in reinforcement learning, capturing data 
    required for policy optimization and training the agent. Transitions are 
    collected into batches during training to compute advantages and update policies.

    Fields:
        - `done` (jnp.ndarray): A flag indicating whether the episode has terminated 
        (True) or continues (False).
        - `action` (jnp.ndarray): The action taken by the agent during this step.
        - `value` (jnp.ndarray): The value function estimate for the observed state.
        - `reward` (jnp.ndarray): The reward received for taking the specified action.
        - `log_prob` (jnp.ndarray): The log-probability of the action under the current 
        policy, used for advantage estimation and loss computation.
        - `obs` (jnp.ndarray): The observation of the environment at this step.

    Purpose:
    - Encapsulates all key components of a single step in the environment, making 
    it easy to process, batch, and use for training updates.
    - Designed to work seamlessly with JAX operations for efficiency.
    """
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray

def get_rollout(train_state, config):
    """
    Generates a single episode rollout for visualization and debugging.

    This function simulates a full episode in the environment using the current 
    trained policy. It collects and returns the sequence of environment states 
    while visualizing rewards to provide insights into the agent's performance.

    Args:
        train_state (TrainState): Contains the trained model parameters and 
                                optimizer state.
        config (dict): Configuration dictionary with environment and training 
                    settings.

    Returns:
        List: A sequence of environment states representing the episode.

    Steps:
    1. **Initialize the Environment and Network**:
    - Creates the specified environment using `jaxmarl.make`.
    - Prepares the actor-critic network with the action space dimensions and 
        activation function.

    2. **Setup Random Number Generators**:
    - Splits the RNG key into separate keys for environment resets and action 
        sampling, ensuring reproducibility.

    3. **Run the Rollout Loop**:
    - Repeatedly interacts with the environment:
        - Obtains observations, samples actions from the trained policy, and steps 
        the environment.
        - Collects rewards, shaped rewards, and the sequence of environment states.

    4. **Plot and Save Reward Trends**:
    - Visualizes the episode's reward and shaped reward trends using matplotlib.

    Purpose:
    - Helps analyze how well the agent performs during an episode.
    - Visualizes reward progression, providing insights into policy effectiveness.

    Example:
    ```python
    state_seq = get_rollout(train_state, config)
    # state_seq now contains all states for the episode, which can be visualized 
    # using OvercookedVisualizer or other tools.
    """
    # Initialize the environment
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"]) 

    # Initialize network
    network = ActorCritic(action_dim=env.action_space().n, activation=config["ACTIVATION"])

    # Initialize the random number generator for the environment, reset, and action selection
    # We split the key into three separate keys for the environment, reset, and action selection 
    # because we need to generate different random numbers for each of these operations for reproducibility
    # We split into these three parts because these parts are independent of each other and the main core part of the code
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3) 

    # Initialize environment and tracking variables
    obs, state = env.reset(key_r) # This resets the environment with the key_r
    state_seq = [state] # This initializes the state sequence with the initial state
    rewards = [] # This initializes the rewards list
    shaped_rewards = [] # This initializes the shaped rewards list 

    done = False # This is the done flag

    # This is the main loop that runs the episode
    while not done:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4) # This splits the key into four separate keys for the action selection and the environment step

        # Process agent_1 (uses base observation)
        agent_1_obs = obs["agent_1"].flatten()  # Base observation
        print("Agent 1 observation shape:", agent_1_obs.shape)
        agent_1_aug_obs = process_observations(agent_1_obs, None, config)
        print("Agent 1 augmented observation shape:", agent_1_aug_obs.shape)
        pi_1, _ = network.apply(train_state.params, agent_1_aug_obs)
        action_1 = pi_1.sample(seed=key_a1)

        # Process agent_0 with augmented observation
        agent_0_obs = obs["agent_0"].flatten()  # Base observation
        print("Agent 0 observation shape:", agent_0_obs.shape)
        agent_0_aug_obs = process_observations(agent_0_obs, action_1, config)
        print("Agent 0 augmented observation shape:", agent_0_aug_obs.shape)
        pi_0, _ = network.apply(train_state.params, agent_0_aug_obs)
        action_0 = pi_0.sample(seed=key_a0)

        # Get actions
        actions = {"agent_0": action_0, "agent_1": action_1}

        # STEP ENV
        obs, state, reward, done, info = env.step(key_s, state, actions) # This steps the environment with the key_s and the actions    
        done = done["__all__"] # This extracts the done flag from the environment step

        # Track rewards
        rewards.append(reward['agent_0']) # This appends the reward for the first agent to the rewards list
        shaped_rewards.append(info["shaped_reward"]['agent_0']) # This appends the shaped reward for the first agent to the shaped rewards list

        state_seq.append(state) # This appends the state to the state sequence

    # Plot the reward and shaped reward over the course of the episode
    from matplotlib import pyplot as plt

    plt.plot(rewards, label="reward") # This plots the reward over the course of the episode
    plt.plot(shaped_rewards, label="shaped_reward") # This plots the shaped reward over the course of the episode
    plt.legend() # This adds a legend to the plot
    plt.savefig("reward.png") # This saves the plot as a png file
    plt.show() # This shows the plot

    return state_seq # This returns the state sequence

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

def process_observations(last_obs, action=None, config=None):
    """Helper function to process observations for both agents"""
    # If input is already flattened array (like in get_rollout)
    if isinstance(last_obs, jnp.ndarray):
        # Just add zero vector for agent_1's action space
        zero_vector = jnp.zeros(config["DIMS"]["action_dim"])
        return jnp.concatenate([last_obs, zero_vector])
    
    # If input is dictionary (like in training)
    agent_1_obs = last_obs['agent_1'].reshape(last_obs['agent_1'].shape[0], -1)
    agent_0_obs = last_obs['agent_0'].reshape(last_obs['agent_0'].shape[0], -1)
    
    zero_vector = jnp.zeros((agent_1_obs.shape[0], config["DIMS"]["action_dim"]))
    agent_1_processed = jnp.concatenate([agent_1_obs, zero_vector], axis=-1)
    
    if action is not None:
        one_hot_action = jax.vmap(jax.nn.one_hot, in_axes=(0, None))(action, config["DIMS"]["action_dim"])
        agent_0_processed = jnp.concatenate([agent_0_obs, one_hot_action], axis=-1)
    else:
        agent_0_processed = jnp.concatenate([agent_0_obs, zero_vector], axis=-1)
    
    return {"agent_0": agent_0_processed, "agent_1": agent_1_processed}

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
        pickle_seeds_path = os.path.join(save_dir, f"{prefix}all_seeds_params.pkl")
        with open(pickle_seeds_path, 'wb') as f:
            pickle.dump(all_seeds_params, f)
        print(f"Saved all seed-specific parameters in pickle format: {pickle_seeds_path}")
        
        # Then save as npz for compatibility
        npz_seeds_path = os.path.join(save_dir, f"{prefix}all_seeds_params.npz")
        np.savez(npz_seeds_path, **all_seeds_params)
        print(f"Saved all seed-specific parameters in npz format: {npz_seeds_path}")
    else:
        print("Warning: No seed-specific parameters were successfully processed")
    
def load_training_results(load_dir, prefix="", load_type="params"):
    """
    Load training results from pickle format
    
    Args:
        load_dir: Directory containing saved files
        prefix: Prefix used in filenames
        load_type: Either "params" or "complete" to load just params or complete output
    Returns:
        Loaded data converted to JAX arrays where appropriate
    """
    if load_type == "params":
        pickle_path = os.path.join(load_dir, f"{prefix}params_pickle.pkl")
        if os.path.exists(pickle_path):
            print("Loading params from pickle format...")
            with open(pickle_path, 'rb') as f:
                params = pickle.load(f)
                # Convert numpy arrays to JAX arrays
                return jax.tree_util.tree_map(
                    lambda x: jax.numpy.array(x) if isinstance(x, np.ndarray) else x,
                    params
                )
                
    elif load_type == "complete":
        pickle_path = os.path.join(load_dir, f"{prefix}complete_out_pickle.pkl")
        if os.path.exists(pickle_path):
            print("Loading complete output from pickle format...")
            with open(pickle_path, 'rb') as f:
                out = pickle.load(f)
                # Convert numpy arrays to JAX arrays
                return jax.tree_util.tree_map(
                    lambda x: jax.numpy.array(x) if isinstance(x, np.ndarray) else x,
                    out
                )
    
    raise FileNotFoundError(f"No saved {load_type} found in {load_dir} with prefix {prefix}")

def create_visualization(train_state, config, filename, agent_view_size=5):
    """Helper function to create and save visualization"""
    state_seq = get_rollout(train_state, config)
    viz = OvercookedVisualizer()
    viz.animate(state_seq, agent_view_size=agent_view_size, filename=filename)

def make_train(config):
    """
    Implements a linear decay for the learning rate over the course of training.

    Purpose:
    - Ensures a high learning rate at the start for fast exploration and optimization.
    - Gradually reduces the learning rate to stabilize training as the agent converges.

    Mechanism:
    - Progress is measured in terms of completed updates (`count`), normalized by 
    the total number of updates (`NUM_UPDATES`).
    - The learning rate is scaled linearly based on this progress:
    `lr = initial_lr * (1.0 - progress_fraction)`
    """
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"]) # This makes the environment

    # Initialize environment
    dims = config["DIMS"]

    # Verify dimensions match what we validated in main
    assert np.prod(env.observation_space().shape) == dims["base_obs_dim"], "Observation dimension mismatch"
    assert env.action_space().n == dims["action_dim"], "Action dimension mismatch"

    # Calculate key training parameters
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"] # This sets the number of actors to the number of agents times the number of environments
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"] 
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    
    env = LogWrapper(env, replace_info=False) # This wraps the environment with the LogWrapper to log the environment information
    
    # This is the learning rate schedule
    # Its purpose is to progressively reduce the learning rate during training based on how far along the training process is.
    # It does this by calculating the fraction of the total number of updates that have been completed and then multiplying it by the initial learning rate.
    # This creates a linear decrease in the learning rate as the training progresses.
    def linear_schedule(count):
        """
        Implements a linear learning rate scheduler.

        The learning rate decays linearly from its initial value (config["LR"]) 
        to 0 over the course of training. This is useful for stabilizing training 
        as the model or agent converges. The decay is based on the training 
        progress, which is measured in terms of updates.

        Args:
            count (int): The current step or minibatch count, used to calculate 
                        the fraction of training progress.

        Returns:
            float: The scaled learning rate for the current step.

        Explanation:
        - Training is divided into multiple updates, with each update consisting 
        of several minibatches and epochs (defined by config["NUM_MINIBATCHES"] 
        and config["UPDATE_EPOCHS"]).
        - The fraction of progress is calculated by dividing the completed updates 
        by the total number of updates (config["NUM_UPDATES"]).
        - The learning rate is scaled linearly based on this progress:
            frac = 1.0 - (progress / total_updates)
            lr = config["LR"] * frac
        - At the start of training (progress = 0), the learning rate is at its 
        maximum value (config["LR"]). As training approaches the final update, 
        the learning rate gradually reduces to zero.
        """
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac
    
    # This is the reward shaping annealing schedule
    """
    This is the reward shaping annealing schedule
    It is used to gradually reduce the reward shaping during training based on how far along the training process is.
    It does this by calculating the fraction of the total number of updates that have been completed and then multiplying it 
    by the initial reward shaping value.
    This creates a linear decrease in the reward shaping as the training progresses.
    """
    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.,
        end_value=0.,
        transition_steps=config["REW_SHAPING_HORIZON"]
    )

    # This is the train function
    def train(rng):
        """
        Main training function that initializes the network, environment, and training state,
        and orchestrates the training loop.

        Args:
            rng (jax.random.PRNGKey): Random number generator key for reproducibility.

        Returns:
            dict: A dictionary containing the final runner state and training metrics.

        Purpose:
            - To initialize and train a reinforcement learning agent using a proximal policy optimization (PPO)-like algorithm.
            - Implements a multi-agent training framework where observations, actions, and rewards are processed collaboratively.

        Steps:
            1. Initialize the Actor-Critic network with appropriate activation and parameters.
            2. Initialize the environment for multiple parallel instances.
            3. Execute the training loop, which alternates between collecting trajectories, calculating advantages, and updating the network.
            4. Log metrics for performance monitoring.

        Why this design:
            - Combines high-level modularity for network initialization, environment management, and training steps.
            - Uses `jax.lax.scan` for efficient loop-based execution on hardware accelerators (e.g., GPUs/TPUs).
        """

        # INIT NETWORK
        init_x = jnp.zeros(config["DIMS"]["augmented_obs_dim"]) # This initializes the observation space with dummy values of zeros
        print("init_x shape:", init_x.shape)
        init_x = init_x.flatten() # This flattens the observation space to a 1D array for the network input
        print("init_x shape after flattening:", init_x.shape)
        network = ActorCritic(action_dim=env.action_space().n, activation=config["ACTIVATION"])

        rng, _rng = jax.random.split(rng) # This splits the key into two separate keys for the network initialization
        network_params = network.init(_rng, init_x) # This initializes the network with the key_a and the flattened observation space
        
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

        # INIT ENV
        rng, _rng = jax.random.split(rng) # This splits the key into two separate keys for the environment reset
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"]) # This splits the key into the number of environments
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng) # This resets the environment with the key_r and the number of environments
        
        # TRAIN LOOP
        # This is the train loop
        def _update_step(runner_state, unused):
            """
            Performs a single update step in the training loop, including trajectory collection,
            advantage calculation, and network updates.

            Args:
                runner_state (tuple): The current state of the runner, including train state, 
                                    environment state, last observation, update step, and RNG.
                unused: Placeholder for compatibility with `jax.lax.scan`.

            Returns:
                tuple: Updated runner state and metrics from the current step.

            Purpose:
                - To manage and integrate all substeps of a training iteration.
                - Coordinates trajectory collection, GAE-based advantage computation, and policy updates.

            Steps:
                1. Collect trajectories by interacting with the environment.
                2. Compute Generalized Advantage Estimation (GAE) for training.
                3. Perform multiple optimization epochs over minibatches of collected data.
            """
            # COLLECT TRAJECTORIES
            # This is the environment step function
            def _env_step(runner_state, unused):
                """
                Executes a single environment step for all agents using a shared policy.

                Args:
                    runner_state (tuple): Contains the current training state, environment state,
                                        observations, update step, and RNG.
                    unused: Placeholder for JAX scan compatibility.

                Returns:
                    Updated runner_state and transition information for the current step.
                """
                train_state, env_state, last_obs, update_step, rng = runner_state

                # Split RNG for action sampling and environment step
                rng, rng_action_1, rng_action_0, rng_step = jax.random.split(rng, 4)

                # Process agent_1 first (uses zero vector for action part)
                processed_obs = process_observations(last_obs, action=None, config=config)
                pi_1, value_1 = network.apply(train_state.params, processed_obs["agent_1"])
                action_1 = pi_1.sample(seed=rng_action_1)
                log_prob_1 = pi_1.log_prob(action_1)

                # Process agent_0 with agent_1's action
                processed_obs = process_observations(last_obs, action=action_1, config=config)
                pi_0, value_0 = network.apply(train_state.params, processed_obs["agent_0"])
                action_0 = pi_0.sample(seed=rng_action_0)
                log_prob_0 = pi_0.log_prob(action_0)

                # Step environment
                actions = {"agent_0": action_0, "agent_1": action_1}
                values = {"agent_0": value_0, "agent_1": value_1}
                log_probs = {"agent_0": log_prob_0, "agent_1": log_prob_1}

                next_obs, next_env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    jax.random.split(rng_step, config["NUM_ENVS"]), env_state, actions
                )
    
                # Create transition with explicit reshaping
                transition = Transition(
                    done=batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action=batchify(actions, env.agents, config["NUM_ACTORS"]).squeeze(),
                    value=batchify(values, env.agents, config["NUM_ACTORS"]).squeeze(),
                    reward=batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob=batchify(log_probs, env.agents, config["NUM_ACTORS"]).squeeze(),
                    obs=batchify(processed_obs, env.agents, config["NUM_ACTORS"]).squeeze()
                )

                # Update Runner State
                runner_state = (train_state, next_env_state, next_obs, update_step, rng)
                return runner_state, (transition, info)

            runner_state, (traj_batch, info) = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"] # This scans the environment step function over the number of steps
            )
            
            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, update_step, rng = runner_state

            # Split RNG for action sampling and environment step
            rng, rng_action_1, rng_action_0, rng_step = jax.random.split(rng, 4)

            # We need to process the observations for both agents because we need to get the value function for the last observation
            # This is separate from the environment step function because we need to get the value function for the last observation.
            # And inside the environment step function we only process the observations for the current step not this last one.
            
            # Process agent_1 first
            processed_obs_agent_1 = process_observations(last_obs, action=None, config=config)
            pi_1, value_1 = network.apply(train_state.params, processed_obs_agent_1["agent_1"])
            action_1 = pi_1.sample(seed=rng_action_1)
            log_prob_1 = pi_1.log_prob(action_1)

            # Process agent_0 with agent_1's action
            processed_obs_agent_0 = process_observations(last_obs, action=action_1, config=config)
            pi_0, value_0 = network.apply(train_state.params, processed_obs_agent_0["agent_0"])
            action_0 = pi_0.sample(seed=rng_action_0)
            log_prob_0 = pi_0.log_prob(action_0)

            # Batching the observations for the network
            obs = {
                "agent_0": processed_obs_agent_0["agent_0"],
                "agent_1": processed_obs_agent_1["agent_1"]
            }
            obs_batch = batchify(obs, env.agents, config["NUM_ACTORS"])

            # getting the value function for the last observation
            _, last_val = network.apply(train_state.params, obs_batch)

            # This is the advantage calculation function
            def _calculate_gae(traj_batch, last_val):
                """
                Computes Generalized Advantage Estimation (GAE) for training.

                Args:
                    traj_batch (Transition): A batch of trajectories collected from the environment.
                    last_val (jnp.ndarray): The value function estimate for the last observation.

                Returns:
                    tuple: Computed advantages and value targets for training.
                
                Purpose:
                    - To calculate the advantage function for training the policy network.
                    - Smoothens advantage estimation with a combination of rewards and value predictions.
                
                Steps:
                    1. Compute the temporal difference (TD) error at each step.
                    2. Accumulate the TD error over multiple timesteps, applying a decay factor.
                    3. Compute the GAE by combining the TD error with the value function estimate.
                    4. Compute the targets by adding the GAE to the value function estimate.
                """
                def _get_advantages(gae_and_next_value, transition):
                    """
                    Helper function for GAE computation to calculate advantages iteratively.

                    Args:
                        gae_and_next_value (tuple): GAE accumulator and next value.
                        transition (Transition): Current transition data.

                    Returns:
                        tuple: Updated GAE accumulator and computed advantage for the step.

                    Purpose:
                        - To compute the temporal difference (TD) error and update the GAE estimate.
                    """
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )

                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)
            
            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                # This function updates the network using the collected trajectories, 
                # advantages, and value targets for a single epoch.
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    print("Minibatch shapes:")
                    print(f"traj_batch: {jax.tree_map(lambda x: x.shape, traj_batch)}")
                    print(f"advantages: {advantages.shape}")
                    print(f"targets: {targets.shape}")

                    def _loss_fn(params, traj_batch, gae, targets):
                        print("\nCalculating losses...")
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        print(f"Network outputs - pi shape: {pi.batch_shape}, value shape: {value.shape}")
                        log_prob = pi.log_prob(traj_batch.action)
                        print(f"Log prob shape: {log_prob.shape}")

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        print(f"Value loss: {value_loss}")

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        print(f"Importance ratio shape: {ratio.shape}")
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        print(f"Normalized GAE shape: {gae.shape}")
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()
                        print(f"Actor loss: {loss_actor}, Entropy: {entropy}")

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        print(f"Total loss: {total_loss}")
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    print("\nGradient stats:")
                    print(f"Grad norm: {optax.global_norm(grads)}")
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                print("Minibatches structure:", jax.tree_map(lambda x: x.shape, minibatches))
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
                """
                Logs training metrics using a callback mechanism (e.g., to WandB).

                Args:
                    metric (dict): Training metrics to be logged.

                Purpose:
                    - To monitor training progress and performance metrics in real-time.
                    - Facilitates debugging and performance visualization.
                """
                wandb.log(
                    metric
                )
            update_step = update_step + 1
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
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

@hydra.main(version_base=None, config_path="config", config_name="base_config")
def main(config):
    """Main entry point for training with shared oracle setup.
    
    Args:
        config: Hydra configuration object containing training parameters.
    """
    # Debug and print Hydra config
    print("\nConfig Debug:")
    print("Raw config content:", config)
    print("Config type:", type(config))
    print("Config composition:", hydra.core.hydra_config.HydraConfig.get().runtime.config_sources)
    
    print("\nHydra Config Info:")
    print(f"Config Path: {hydra.utils.get_original_cwd()}/config")
    print(f"Config Name: base_config")
    print(f"Current Directory: {os.getcwd()}")

    # Absolute path to the config
    config_path = os.path.abspath(os.path.join(hydra.utils.get_original_cwd(), "config"))
    print(f"Absolute Config Path: {config_path}")

    # Convert Hydra config to a Python dictionary
    config = OmegaConf.to_container(config)
    layout_name = config["ENV_KWARGS"]["layout"]
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]

    # Validate environment dimensions and prepare training setup
    print("\nSetting up the environment...")
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    base_obs_shape = env.observation_space().shape
    base_obs_dim = int(np.prod(base_obs_shape))
    action_dim = int(env.action_space().n)
    augmented_obs_dim = base_obs_dim + action_dim

    # Validate dimensions
    assert base_obs_dim > 0, f"Invalid base observation dimension: {base_obs_dim}"
    assert action_dim > 0, f"Invalid action dimension: {action_dim}"
    assert augmented_obs_dim > base_obs_dim, "Augmented dim must be larger than base dim"

    # Store dimensions in config
    config["DIMS"] = {
        "base_obs_shape": base_obs_shape,
        "base_obs_dim": base_obs_dim,
        "action_dim": action_dim,
        "augmented_obs_dim": augmented_obs_dim
    }

    # Initialize wandb for shared oracle setup
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "FF", "SharedParams", "Oracle"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f'oracle_shared_ff_ippo_overcooked_{layout_name}'
    )

    # Create a new directory for the results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date = datetime.now().strftime("%Y%m%d")
    save_dir = os.path.join(
        "saved_models", 
        date,
        f"{layout_name}", 
        f"oracle_shared_ff_ippo_overcooked_{layout_name}_{timestamp}_{config['SEED']}"
    )
    os.makedirs(save_dir, exist_ok=True)

    # Setup random seeds and training
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])    
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)

    # Save parameters and results
    save_training_results(save_dir, out, config, prefix="oracle_shared_ff_ippo_overcooked_")

    metrics_dict = {key: np.array(value) for key, value in out["metrics"].items()}

    metrics_path = os.path.join(save_dir, "metrics.npz")
    np.savez(metrics_path, **metrics_dict)

    # Save the configuration
    config_path = os.path.join(save_dir, "config.pkl")
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)

    print(f"Training results saved to: {save_dir}")

    # Generate and save visualization
    train_state = jax.tree_util.tree_map(lambda x: x[0], out["runner_state"][0])
    viz_filename = os.path.join(save_dir, f'oracle_shared_ff_ippo_overcooked_{layout_name}_{timestamp}_{config["SEED"]}.gif')
    create_visualization(train_state, config, viz_filename)
    
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
    reward_std = rewards.std(0) / np.sqrt(config["NUM_SEEDS"])  # standard error
    
    plt.figure()
    plt.plot(reward_mean)
    plt.fill_between(range(len(reward_mean)), reward_mean - reward_std, reward_mean + reward_std, alpha=0.2)
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    
    filename = f'{config["ENV_NAME"]}_cramped_room_new'
    plt.savefig(os.path.join(save_dir, f'{filename}.png'))
    plt.close()

if __name__ == "__main__":
    main()