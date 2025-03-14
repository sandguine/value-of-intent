"""
Implementation of PPO with CNN architecture and Contrastive Predictive Coding (CPC).
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
import traceback

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

class CNN(nn.Module):
    """CNN module with CPC embedding layer"""
    activation: str = "tanh"
    embedding_dim: int = 128  # New parameter for CPC embedding dimension

    @nn.compact
    def __call__(self, x):
        # Select activation function
        act_fn = nn.relu if self.activation == "relu" else nn.tanh
        
        # CNN layers (same as before)
        x = nn.Conv(features=32, kernel_size=(5, 5),
                   kernel_init=orthogonal(np.sqrt(2)),
                   bias_init=constant(0.0))(x)
        x = act_fn(x)
        
        x = nn.Conv(features=32, kernel_size=(3, 3),
                   kernel_init=orthogonal(np.sqrt(2)),
                   bias_init=constant(0.0))(x)
        x = act_fn(x)
        
        x = nn.Conv(features=32, kernel_size=(3, 3),
                   kernel_init=orthogonal(np.sqrt(2)),
                   bias_init=constant(0.0))(x)
        x = act_fn(x)
        
        # Flatten output
        x = x.reshape((x.shape[0], -1))
        
        # Dense layer
        x = nn.Dense(features=64,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0))(x)
        x = act_fn(x)
        
        # New CPC embedding layer
        embedding = nn.Dense(features=self.embedding_dim,
                           kernel_init=orthogonal(np.sqrt(2)),
                           bias_init=constant(0.0))(x)
        
        return x, embedding

class ActorCritic(nn.Module):
    """Actor-Critic network with CPC embedding"""
    action_dim: Sequence[int]
    activation: str = "tanh"
    embedding_dim: int = 128

    def setup(self):
        self.cnn = CNN(activation=self.activation, embedding_dim=self.embedding_dim)
        
        # Actor network layers
        self.actor_dense = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)))
        self.actor_out = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))
        
        # Critic network layers
        self.critic_dense = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)))
        self.critic_out = nn.Dense(1, kernel_init=orthogonal(1.0))
        
        self.act_fn = nn.relu if self.activation == "relu" else nn.tanh

    def __call__(self, x):
        features, embedding = self.cnn(x)
        
        # Actor head
        actor = self.actor_dense(features)
        actor = self.act_fn(actor)
        actor = self.actor_out(actor)
        pi = distrax.Categorical(logits=actor)
        
        # Critic head
        critic = self.critic_dense(features)
        critic = self.act_fn(critic)
        critic = self.critic_out(critic)
        
        return pi, jnp.squeeze(critic, axis=-1), embedding

def compute_cpc_loss(embeddings, config):
    """Compute CPC loss using InfoNCE"""
    batch_size = embeddings.shape[0]
    seq_len = embeddings.shape[1]
    
    # Normalize embeddings
    embeddings = embeddings / jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
    
    # Prepare positive and negative samples
    # Use current embedding to predict k steps ahead
    k_steps = config["CPC_PREDICTION_STEPS"]
    total_loss = 0.0
    
    for k in range(1, k_steps + 1):
        # Prepare context and targets
        context = embeddings[:, :-k]  # Current embeddings
        targets = embeddings[:, k:]   # Future embeddings
        
        # Compute similarity scores
        similarity = jnp.einsum('bte,bse->bts', context, targets)
        
        # Create positive and negative masks
        pos_mask = jnp.eye(similarity.shape[1])
        neg_mask = 1.0 - pos_mask
        
        # Compute InfoNCE loss
        pos_similarity = similarity * pos_mask
        neg_similarity = similarity * neg_mask
        
        # Temperature scaling
        temperature = config["CPC_TEMPERATURE"]
        logits = similarity / temperature
        
        # Compute loss using cross-entropy
        labels = jnp.eye(similarity.shape[1])
        loss = optax.softmax_cross_entropy(logits, labels)
        
        total_loss += loss.mean()
    
    return total_loss / k_steps

class Transition(NamedTuple):
    """Container for storing experience transitions"""
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    embedding: jnp.ndarray  # Added embedding field

def get_rollout(train_state, agent_1_params, config, save_dir=None):
    """Generate a single episode rollout for visualization"""
    # Initialize environment
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    # Initialize network
    network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    done = False
    obs, state = env.reset(key_r)
    state_seq = [state]
    rewards = []
    shaped_rewards = []

    # Run episode until completion
    while not done:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        # Process observations for CNN input (no flattening)
        obs_agent_0 = obs["agent_0"][None, ...]
        obs_agent_1 = obs["agent_1"][None, ...]

        # Get actions from policies
        pi_1, _ = network.apply(agent_1_params, obs_agent_1)
        action_1 = pi_1.sample(seed=key_a1)[0]

        pi_0, _ = network.apply(train_state.params, obs_agent_0)
        action_0 = pi_0.sample(seed=key_a0)[0]

        actions = {"agent_0": action_0, "agent_1": action_1}

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
    plt.close()

    return state_seq

def batchify(x: dict, agent_list, num_actors):
    """Convert dict of agent observations to batched array"""
    x = jnp.stack([x[a] for a in agent_list])
    return x  # No reshape for CNN input

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    """Convert batched array back to dict of agent observations"""
    x = x.reshape((num_actors, num_envs) + x.shape[2:])  # Preserve spatial dimensions
    return {a: x[i] for i, a in enumerate(agent_list)}

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

    def train(rng, pretrained_params):
        """Main training loop"""
        # Initialize network and optimizer
        network = ActorCritic(
            env.action_space().n, 
            activation=config["ACTIVATION"],
            embedding_dim=config["EMBEDDING_DIM"]
        )
        rng, _rng = jax.random.split(rng)
        
        # Initialize with proper observation shape for CNN
        init_x = jnp.zeros((1,) + env.observation_space().shape)
        network_params = network.init(_rng, init_x)

        # Setup optimizer with optional learning rate annealing
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5)
            )

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
        def _update_step(runner_state, unused, pretrained_params):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused, pretrained_params):
                train_state, env_state, last_obs, update_step, rng = runner_state
                rng, rng_action_1, rng_action_0, rng_step = jax.random.split(rng, 4)

                # Extract correct agent_1 parameters per environment
                num_envs = last_obs['agent_1'].shape[0]
                agent_1_obs = last_obs['agent_1']  # Keep spatial dimensions for CNN
                rng_action_1_split = jax.random.split(rng_action_1, num_envs)

                # Vectorized application across all environments
                agent_1_action = jax.vmap(
                    lambda params, obs, rng: network.apply(params, obs[None, ...])[0].sample(seed=rng),
                    in_axes=(0, 0, 0)
                )(pretrained_params, agent_1_obs, rng_action_1_split)

                # Agent 0: Process observations for CNN
                agent_0_obs = last_obs['agent_0']  # Keep spatial dimensions

                # Apply agent_0 policy using trainable parameters
                agent_0_pi, agent_0_value, agent_0_embedding = network.apply(train_state.params, agent_0_obs)
                agent_0_action = agent_0_pi.sample(seed=rng_action_0)
                agent_0_log_prob = agent_0_pi.log_prob(agent_0_action)

                # Step the environment
                actions = {"agent_0": agent_0_action, "agent_1": agent_1_action}
                next_obs, next_env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    jax.random.split(rng_step, num_envs),
                    env_state,
                    actions,
                )

                # Create transition with embedding
                transition = Transition(
                    done=done["agent_0"],
                    action=agent_0_action,
                    value=agent_0_value,
                    reward=reward["agent_0"],
                    log_prob=agent_0_log_prob,
                    obs=agent_0_obs,
                    embedding=agent_0_embedding,
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
            
            # Calculate last values for agent_0 (the learning agent)
            last_obs_agent0 = last_obs['agent_0']  # Keep spatial dimensions for CNN
            _, agent_0_last_val = network.apply(train_state.params, last_obs_agent0)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.done, transition.value, transition.reward
                    
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, agent_0_last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused, config):
                def _update_minibatch(train_state, batch_info, config):
                    agent_0_data = batch_info['agent_0']
                    traj_batch = agent_0_data['traj']
                    advantages = agent_0_data['advantages']
                    targets = agent_0_data['targets']

                    def _loss_fn(params, traj_batch, gae, targets, config):
                        pi, value, embeddings = network.apply(params, traj_batch.obs)
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

                        # CPC loss calculation
                        cpc_loss = compute_cpc_loss(embeddings, config)

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                            + config["CPC_COEF"] * cpc_loss
                        )

                        return total_loss, {
                            'value_loss': value_loss,
                            'actor_loss': loss_actor,
                            'entropy': entropy,
                            'cpc_loss': cpc_loss,
                            'total_loss': total_loss
                        }

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (loss, aux), grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets, config
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (loss, aux)

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
                assert batch_size % config["NUM_MINIBATCHES"] == 0

                # Prepare batched data while preserving spatial dimensions for CNN
                def prepare_batch_data(data):
                    return {
                        'traj': Transition(**{
                            field: getattr(data['traj'], field)
                            for field in data['traj']._fields
                        }),
                        'advantages': data['advantages'],
                        'targets': data['targets']
                    }

                agent_data = {
                    "agent_0": {"traj": traj_batch, "advantages": advantages, "targets": targets}
                }
                agent_data = {agent: prepare_batch_data(data) for agent, data in agent_data.items()}

                # Create minibatches
                def create_minibatches(data):
                    return {
                        'traj': Transition(**{
                            field: getattr(data["traj"], field).reshape(
                                (config["NUM_MINIBATCHES"], -1) + getattr(data["traj"], field).shape[1:]
                            )
                            for field in data["traj"]._fields
                        }),
                        'advantages': data["advantages"].reshape((config["NUM_MINIBATCHES"], -1)),
                        'targets': data["targets"].reshape((config["NUM_MINIBATCHES"], -1))
                    }

                minibatches = {agent: create_minibatches(data) for agent, data in agent_data.items()}

                train_state, total_loss = jax.lax.scan(
                    lambda state, data: _update_minibatch(state, data, config),
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
                wandb.log(metric)

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

def save_training_results(save_dir, out, config, prefix=""):
    """Save training results to specified directory"""
    os.makedirs(save_dir, exist_ok=True)

    def is_pickleable(obj):
        try:
            pickle.dumps(obj)
            return True
        except Exception:
            return False
    
    def process_tree(tree):
        def convert_and_filter(x):
            if isinstance(x, (jax.Array, np.ndarray)):
                x = np.array(x)
            return x if is_pickleable(x) else None
        
        return jax.tree_util.tree_map(convert_and_filter, tree)
    
    # Convert outputs to numpy format
    numpy_out = jax.tree_util.tree_map(
        lambda x: np.array(x) if isinstance(x, (jax.Array, np.ndarray)) else x,
        jax.device_get(out)
    )
    
    # Filter and save pickleable objects
    pickle_safe_out = {}
    for key, value in numpy_out.items():
        try:
            pickle.dumps(value)
            pickle_safe_out[key] = value
        except Exception as e:
            print(f"Warning: Skipping unpickleable key '{key}' due to: {str(e)}")

    # Save complete output
    pickle_out_path = os.path.join(save_dir, f"complete_out.pkl")
    with open(pickle_out_path, 'wb') as f:
        pickle.dump(pickle_safe_out, f)

    npz_out_path = os.path.join(save_dir, f"complete_out.npz")
    np.savez(npz_out_path, **pickle_safe_out)

    # Process seed-specific parameters
    all_seeds_params = {}
    for seed_idx in range(config["NUM_SEEDS"]):
        try:
            if "runner_state" not in out or not out["runner_state"]:
                print(f"Warning: No runner_state found for seed {seed_idx}")
                continue
                
            train_state = jax.tree_util.tree_map(
                lambda x: x[seed_idx] if x is not None else None,
                out["runner_state"][0]
            )
            
            processed_state = {}
            
            if hasattr(train_state, 'params'):
                processed_params = process_tree(train_state.params)
                if processed_params is not None:
                    processed_state['params'] = processed_params['params']
            
            if hasattr(train_state, 'step'):
                try:
                    processed_state['step'] = np.array(train_state.step)
                except Exception as e:
                    print(f"Warning: Could not process step for seed {seed_idx}: {str(e)}")
            
            if "metrics" in out:
                processed_metrics = process_tree(
                    jax.tree_util.tree_map(
                        lambda x: x[seed_idx] if isinstance(x, (jax.Array, np.ndarray)) else x,
                        out["metrics"]
                    )
                )
                if processed_metrics:
                    processed_state['metrics'] = processed_metrics
            
            if processed_state:
                all_seeds_params[f"seed_{seed_idx}"] = processed_state
            
        except Exception as e:
            print(f"Warning: Could not process seed {seed_idx} due to: {str(e)}")
            continue
    
    if all_seeds_params:
        pickle_seeds_path = os.path.join(save_dir, f"all_seeds_params.pkl")
        with open(pickle_seeds_path, 'wb') as f:
            pickle.dump(all_seeds_params, f)
        
        npz_seeds_path = os.path.join(save_dir, f"all_seeds_params.npz")
        np.savez(npz_seeds_path, **all_seeds_params)
    else:
        print("Warning: No seed-specific parameters were successfully processed")

def load_training_results(load_dir, load_type="params", config=None):
    """Load training results from specified directory"""
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
            sampled_params = jax.tree_util.tree_map(
                lambda *x: jnp.stack(x, axis=0), *sampled_params_list
            )

            print("Successfully loaded pretrained model.")
            print("Loaded params type:", type(sampled_params))
            print("Shape of sampled_params:", jax.tree_util.tree_map(lambda x: x.shape, sampled_params))

            return sampled_params
                
    elif load_type == "complete":
        pickle_path = os.path.join(load_dir, f"complete_out.pkl")
        if os.path.exists(pickle_path):
            print("Loading complete output from pickle format...")
            with open(pickle_path, 'rb') as f:
                out = pickle.load(f)
                return jax.tree_util.tree_map(
                    lambda x: jax.numpy.array(x) if isinstance(x, np.ndarray) else x,
                    out
                )
    
    raise FileNotFoundError(f"No saved {load_type} found in {load_dir}")

def create_visualization(train_state, agent_1_params, config, filename, save_dir=None, agent_view_size=5):
    """Create and save visualization of agent behavior"""
    if not isinstance(config, dict):
        config = OmegaConf.to_container(config, resolve=True)
    
    state_seq = get_rollout(train_state, agent_1_params, config, save_dir)
    viz = OvercookedVisualizer()
    viz.animate(state_seq, agent_view_size=agent_view_size, filename=filename)

@hydra.main(version_base=None, config_path="config", config_name="adapt_asymm")
def main(config):
    """Main entry point for training"""
    # Set up Python path
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Process configuration
    config = OmegaConf.to_container(config)
    layout_name = config["ENV_KWARGS"]["layout"]
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]

    # Add CPC-specific configuration if not present
    if "CPC_COEF" not in config:
        config["CPC_COEF"] = 1.0
    if "CPC_PREDICTION_STEPS" not in config:
        config["CPC_PREDICTION_STEPS"] = 3
    if "CPC_TEMPERATURE" not in config:
        config["CPC_TEMPERATURE"] = 0.1
    if "EMBEDDING_DIM" not in config:
        config["EMBEDDING_DIM"] = 128

    # Initialize wandb
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "CNN", "Adaptability", "Oracle", "Lower Bound"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f'lb_cnn_{layout_name}'
    )

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date = datetime.now().strftime("%Y%m%d")
    model_dir_name = f"lb_cnn_{layout_name}_{timestamp}"
    save_dir = os.path.join(
        "saved_models", 
        date,
        layout_name, 
        f"{model_dir_name}_{config['SEED']}"
    )
    os.makedirs(save_dir, exist_ok=True)

    # Load pretrained parameters
    pretrained_params = load_training_results(config["LOAD_PATH"], load_type="params", config=config)

    # Initialize and run training
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = jax.jit(lambda rng: make_train(config)(rng, pretrained_params))
    out = jax.vmap(train_jit)(rngs)

    # Save results and generate visualizations
    save_training_results(save_dir, out, config, prefix="lb_cnn_")
    np.savez(os.path.join(save_dir, "lb_cnn_metrics.npz"), 
            **{key: np.array(value) for key, value in out["metrics"].items()})

    with open(os.path.join(save_dir, "config.pkl"), 'wb') as f:
        pickle.dump(config, f)

    # Create visualization
    try:
        train_state = jax.tree_util.tree_map(lambda x: x[0], out["runner_state"][0])
        viz_base_name = f"lb_cnn_{layout_name}"
        viz_filename = os.path.join(save_dir, f'{viz_base_name}_{config["SEED"]}.gif')
        create_visualization(train_state, pretrained_params, config, viz_filename, save_dir)
    except Exception as e:
        print(f"Error generating visualization: {e}")
        traceback.print_exc()

    # Plot learning curves
    rewards = out["metrics"]["returned_episode_returns"].reshape((config["NUM_SEEDS"], -1))
    reward_mean = rewards.mean(0)
    reward_std = rewards.std(0)
    reward_std_err = reward_std / np.sqrt(config["NUM_SEEDS"])

    # Log individual seed rewards
    for update_step in range(rewards.shape[1]):
        log_data = {"Update_Step": update_step}
        for seed_idx in range(config["NUM_SEEDS"]):
            log_data[f"Seeds/Seed_{seed_idx}/Returned_Episode_Returns"] = rewards[seed_idx, update_step]
        wandb.log(log_data)

    # Save learning curve
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

    learning_curve_name = f"lb_cnn_learning_curve"
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{learning_curve_name}.png'))
    plt.close()

if __name__ == "__main__":
    main()
