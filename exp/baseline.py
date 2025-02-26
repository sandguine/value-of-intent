"""
Implementation of Independent PPO (IPPO) for multi-agent environments.
Based on PureJaxRL's PPO implementation but adapted for multi-agent scenarios.
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
        # print("Network input x shape:", x.shape)
        # print("ActorCritic input shape:", x.shape)
        
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
    
def get_rollout(train_state, config, save_dir=None):
    """Generate a single episode rollout for visualization"""
    # Initialize environment
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    # Initialize network
    network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    # Initialize observation
    init_x = jnp.zeros(env.observation_space().shape)
    init_x = init_x.flatten()

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

        # Flatten observations for network input
        obs = {k: v.flatten() for k, v in obs.items()}

        # Get actions from policy for both agents
        pi_0, _ = network.apply(network_params, obs["agent_0"])
        pi_1, _ = network.apply(network_params, obs["agent_1"])

        actions = {
            "agent_0": pi_0.sample(seed=key_a0), 
            "agent_1": pi_1.sample(seed=key_a1)
        }

        # Step environment forward
        obs, state, reward, done, info = env.step(key_s, state, actions)
        done = done["__all__"]
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
    """Convert dict of agent observations to batched array"""
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    """Convert batched array back to dict of agent observations"""
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

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

def load_training_results(load_dir, load_type="params"):
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
        pickle_path = os.path.join(load_dir, f"params.pkl")
        if os.path.exists(pickle_path):
            # print("Loading params from pickle format...")
            with open(pickle_path, 'rb') as f:
                params = pickle.load(f)
                # Convert numpy arrays to JAX arrays
                return jax.tree_util.tree_map(
                    lambda x: jax.numpy.array(x) if isinstance(x, np.ndarray) else x,
                    params
                )
                
    elif load_type == "complete":
        pickle_path = os.path.join(load_dir, f"complete_out.pkl")
        if os.path.exists(pickle_path):
            # print("Loading complete output from pickle format...")
            with open(pickle_path, 'rb') as f:
                out = pickle.load(f)
                # Convert numpy arrays to JAX arrays
                return jax.tree_util.tree_map(
                    lambda x: jax.numpy.array(x) if isinstance(x, np.ndarray) else x,
                    out
                )
    
    raise FileNotFoundError(f"No saved {load_type} found in {load_dir}")

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

def load_sweep_config(path: str) -> Dict[Any, Any]:
    with open(path, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['method', 'metric', 'parameters']
    for field in required_fields:
        if field not in sweep_config:
            raise ValueError(f"Sweep config missing required field: {field}")
            
    return sweep_config
    
def make_train(config):
    """Creates the main training function with the given config"""
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
    
    # Learning rate and reward shaping annealing schedules
    # The learning rate is annealed linearly over the course of training because
    # if the learning rate is too high, the model can diverge.
    # By making the learning rate decay linearly, we can ensure that the model can converge.
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
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, update_step, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                pi, value = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                
                env_act = {k:v.flatten() for k,v in env_act.items()}
                
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    rng_step, env_state, env_act
                )

                info["reward"] = reward["agent_0"]

                current_timestep = update_step*config["NUM_STEPS"]*config["NUM_ENVS"]
                reward = jax.tree_util.tree_map(lambda x,y: x+y*rew_shaping_anneal(current_timestep), reward, info["shaped_reward"])

                transition = Transition(
                    batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                )
                runner_state = (train_state, env_state, obsv, update_step, rng)
                return runner_state, (transition, info)

            runner_state, (traj_batch, info) = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            
            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, update_step, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            _, last_val = network.apply(train_state.params, last_obs_batch)

            # This function calculates the advantage for each transition in the trajectory (basically, policy optimization).
            # It returns the advantages and value targets.
            def _calculate_gae(traj_batch, last_val):
                
                # This function calculates the advantage for each transition in the trajectory.
                def _get_advantages(gae_and_next_value, transition):
                    # Unpack the carried state (previous GAE and next state's value)
                    gae, next_value = gae_and_next_value
                    # Get current transition info
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )

                    # Calculate TD error (temporal difference)
                    # δt = rt + γV(st+1) - V(st)
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value

                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    # print(f"calculated gae shape: {gae.shape}, value: {gae}")

                    # Return the updated GAE and the next state's value
                    return (gae, value), gae

                # Use scan to process the trajectory backwards
                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val), # Initial GAE and the final value
                    traj_batch, # Sequence of transitions
                    reverse=True, # Process the trajectory backwards
                    unroll=16, # Unroll optimization
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)
            
            # UPDATE NETWORK
            # This function performs multiple optimization steps on the collected trajectories.
            def _update_epoch(update_state, unused):
                # This function updates the network using the collected trajectories, 
                # advantages, and value targets for a single epoch.
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # print("\nCalculating losses...")
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        # print(f"Network outputs - pi shape: {pi.batch_shape}, value shape: {value.shape}")
                        log_prob = pi.log_prob(traj_batch.action)
                        # print(f"Log prob shape: {log_prob.shape}")

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        # print(f"Value loss: {value_loss}")

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        # print(f"Importance ratio shape: {ratio.shape}")
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        # print(f"Normalized GAE shape: {gae.shape}")
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
                        # print(f"Actor loss: {loss_actor}, Entropy: {entropy}")

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        # print(f"Total loss: {total_loss}")
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    # print("\nGradient stats:")
                    # print(f"Grad norm: {optax.global_norm(grads)}")
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
                # print("Minibatches structure:", jax.tree_map(lambda x: x.shape, minibatches))
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
def main(hydra_config):
    """
    Main entry point for training that handles path setup, configuration management,
    and training execution.
    """
    # Set up Python path by adding project root to system path
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        
    # Process base configuration
    config = OmegaConf.to_container(hydra_config)
    layout_name = config["ENV_KWARGS"]["layout"]

    # First convert Hydra config to a plain dictionary
    config = OmegaConf.to_container(hydra_config)
    
    # Initialize wandb first
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "FF", "Baseline", "Oracle", "Upper Bound"],
        config=config,  # Pass initial config
        mode=config["WANDB_MODE"],
        name=f'ub_ippo_oc_{config["ENV_KWARGS"]["layout"]}',
        # settings=wandb.Settings(start_method="thread"),
        # settings=wandb.Settings(code_dir=".", _disable_stats=True),
    )
    
    # Process layout configuration
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date = datetime.now().strftime("%Y%m%d")
    model_dir_name = f"ub_ippo_oc_{layout_name}_{timestamp}"
    save_dir = os.path.join(
        "saved_models", 
        date,
        layout_name, 
        f"{model_dir_name}_{config['SEED']}"
    )
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize and run training
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])    
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)
    
    # Save results and generate visualizations
    save_training_results(save_dir, out, config, prefix="ub_ippo_oc_")
    np.savez(os.path.join(save_dir, "metrics.npz"), **{key: np.array(value) for key, value in out["metrics"].items()})
    
    with open(os.path.join(save_dir, "config.pkl"), 'wb') as f:
        pickle.dump(config, f)
        
    # Create and save visualization
    train_state = jax.tree_util.tree_map(lambda x: x[0], out["runner_state"][0])
    viz_base_name = f"ub_ippo_oc_{layout_name}_{timestamp}"
    viz_filename = os.path.join(save_dir, f'{viz_base_name}_{config["SEED"]}.gif')
    create_visualization(train_state, config, viz_filename, save_dir)
    
    # Plot and save learning curves
    rewards = out["metrics"]["returned_episode_returns"].reshape((config["NUM_SEEDS"], -1))

    # Calculate mean and standard deviation of rewards across seeds
    reward_mean = rewards.mean(0)
    reward_std = rewards.std(0)
    reward_std_err = reward_std / np.sqrt(config["NUM_SEEDS"])

    # Log individual seed rewards
    for update_step in range(rewards.shape[1]):
        log_data = {"Update_Step": update_step}
        log_data["Rewards/Mean"] = reward_mean[update_step]
        log_data["Rewards/Upper_Bound"] = reward_mean[update_step] + reward_std_err[update_step]
        log_data["Rewards/Lower_Bound"] = reward_mean[update_step] - reward_std_err[update_step]

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
        plt.plot(rewards[seed_idx], label=f'Seed {seed_idx}', alpha=0.7)
    plt.xlabel("Update Step")
    plt.ylabel("Returned Episode Returns")
    plt.title("Per-Seed Performance on Returned Episode Returns")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    
    learning_curve_name = f"ub_ippo_oc_{config['ENV_NAME']}_learning_curve"
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{learning_curve_name}.png'))
    plt.close()

if __name__ == "__main__":
    main()