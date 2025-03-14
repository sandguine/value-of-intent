"""
Implementation of PPO with CPC Loss.
"""

# Core imports for JAX machine learning
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Dict, Type
from flax.training.train_state import TrainState
import distrax
import flax

# Environment imports
import jaxmarl
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from jaxmarl.wrappers.baselines import LogWrapper

# Configuration and logging imports
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

# Results saving imports
import os
import pickle
from datetime import datetime
from pathlib import Path
import sys

# Plotting imports
import matplotlib.pyplot as plt

# Local imports
from src.utils.data import get_network, batchify, unbatchify
from src.utils.io import save_training_results, load_training_results
from src.utils.viz import create_visualization, plot_learning_curves

class Transition(NamedTuple):
    """Container for storing experience transitions with CPC support"""
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    latent_features: jnp.ndarray  # Features from backbone network
    projected_features: jnp.ndarray  # Features after projection head
    future_features: jnp.ndarray  # Sequence of future latent features

class CPCProjectionHead(nn.Module):
    """Projection head for CPC features"""
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, x):
        # Removed ReLU to avoid gradient bottlenecks
        return nn.Dense(
            features=self.hidden_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )(x)

def compute_cpc_loss(current_features, future_features, temperature=0.1):
    """Compute CPC loss using current features to predict future observations.
    
    Args:
        current_features: shape (batch_size, feature_dim)
        future_features: shape (batch_size, num_future_steps, feature_dim)
        temperature: scaling factor for similarity scores
    """
    # Normalize features
    current_features = current_features / (jnp.linalg.norm(current_features, axis=-1, keepdims=True) + 1e-8)
    future_features = future_features / (jnp.linalg.norm(future_features, axis=-1, keepdims=True) + 1e-8)
    
    # Compute similarity scores - (batch_size, batch_size, num_future_steps)
    similarity = jnp.einsum('bd,bfd->bf', current_features, future_features) / temperature
    
    # Create positive labels (diagonal indices)
    batch_size = current_features.shape[0]
    labels = jnp.arange(batch_size)
    
    # Compute InfoNCE loss for each future step
    losses = []
    for t in range(future_features.shape[1]):
        t_similarity = similarity[:, t]  # (batch_size, batch_size)
        loss = -jnp.mean(
            jax.nn.log_softmax(t_similarity)[jnp.arange(batch_size)]
        )
        losses.append(loss)
    
    return jnp.mean(jnp.array(losses))

def get_rollout(train_state, agent_1_params, config, save_dir=None):
    """Generate a single episode rollout for visualization.
    
    Args:
        train_state: Training state for agent_0 (learning agent)
        agent_1_params: Fixed pretrained parameters for agent_1
        config: Configuration dictionary
        save_dir: Optional directory to save visualization files
    """
    # Initialize environment
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    network, projection_head = get_network(config, env.action_space().n)
    
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    done = False
    obs, state = env.reset(key_r)
    state_seq = [state]
    rewards = []
    shaped_rewards = []
    
    while not done:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        # Process observations based on architecture
        if config["ARCHITECTURE"].lower() == "cnn":
            obs_batch = {
                'agent_0': obs['agent_0'][None, ...],  # Keep spatial dimensions for CNN
                'agent_1': obs['agent_1'][None, ...]
            }
        elif config["ARCHITECTURE"].lower() == "rnn":
            # For RNN, we need to maintain the sequence dimension
            # Assuming obs shape is (seq_len, *feature_dims)
            obs_batch = {
                'agent_0': obs['agent_0'][None, :, :],  # Add batch dim but keep sequence dim
                'agent_1': obs['agent_1'][None, :, :]
            }
        else:  # feedforward case
            obs_batch = {
                'agent_0': obs['agent_0'].flatten()[None, ...],
                'agent_1': obs['agent_1'].flatten()[None, ...]
            }

        # Get actions from policies
        # Agent 1 (fixed partner) uses pretrained parameters
        pi_1, _ = network.apply(agent_1_params, obs_batch['agent_1'])
        action_1 = pi_1.sample(seed=key_a1)[0]

        # Agent 0 (learning agent) uses current training parameters
        pi_0, _ = network.apply(train_state.params, obs_batch['agent_0'])
        action_0 = pi_0.sample(seed=key_a0)[0]

        actions = {"agent_0": action_0, "agent_1": action_1}

        # Step environment forward
        obs, state, reward, done, info = env.step(key_s, state, actions)
        done = done["__all__"]
        rewards.append(reward['agent_0'])
        shaped_rewards.append(info["shaped_reward"]['agent_0'])

        state_seq.append(state)

    # Plot rewards
    plt.figure()
    plt.plot(rewards, label="reward", color='C0')
    plt.plot(shaped_rewards, label="shaped_reward", color='C1')
    plt.legend()
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title("Episode Reward and Shaped Reward Progression")
    plt.grid()
    
    if save_dir:
        reward_plot_path = os.path.join(save_dir, "reward_plot.png")
        plt.savefig(reward_plot_path)
    plt.close()

    return state_seq

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
        network, projection_head = get_network(config, env.action_space().n)
        rng, _rng = jax.random.split(rng)
        
        # Initialize with proper observation shape based on architecture
        if config["ARCHITECTURE"].lower() == "cnn":
            init_x = jnp.zeros((1,) + env.observation_space().shape)
        else:
            init_x = jnp.zeros((1, np.prod(env.observation_space().shape)))
            
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

                # Process observations and extract features
                if config["ARCHITECTURE"].lower() == "cnn":
                    obs_batch = {
                        'agent_0': last_obs['agent_0'],
                        'agent_1': last_obs['agent_1']
                    }
                else:
                    obs_batch = {
                        'agent_0': last_obs['agent_0'].reshape(last_obs['agent_0'].shape[0], -1),
                        'agent_1': last_obs['agent_1'].reshape(last_obs['agent_1'].shape[0], -1)
                    }

                # Get features and policy outputs for agent_0
                pi_0, value_0, latent_features = network.apply(
                    train_state.params.network,
                    obs_batch['agent_0'],
                    return_features=True
                )
                projected_features = projection_head.apply(
                    train_state.params.projection_head,
                    latent_features
                )

                # Get actions
                action_0 = pi_0.sample(seed=rng_action_0)
                log_prob_0 = pi_0.log_prob(action_0)

                # Agent 1 uses pretrained parameters
                pi_1, _ = network.apply(pretrained_params, obs_batch['agent_1'])
                action_1 = pi_1.sample(seed=rng_action_1)

                actions = {"agent_0": action_0, "agent_1": action_1}

                # Step environment
                next_obs, next_env_state, reward, done, info = jax.vmap(env.step)(
                    jax.random.split(rng_step, config["NUM_ENVS"]),
                    env_state,
                    actions
                )

                # Get future features for CPC
                future_obs_sequence = next_obs['agent_0']  # Shape: (batch, future_steps, obs_dim)
                _, _, future_latent = jax.vmap(lambda x: network.apply(
                    train_state.params.network,
                    x,
                    return_features=True
                ))(future_obs_sequence)
                
                future_projected = jax.vmap(lambda x: projection_head.apply(
                    train_state.params.projection_head,
                    x
                ))(future_latent)

                transition = Transition(
                    done=done["agent_0"],
                    action=action_0,
                    value=value_0,
                    reward=reward["agent_0"],
                    log_prob=log_prob_0,
                    obs=obs_batch['agent_0'],
                    latent_features=latent_features,
                    projected_features=projected_features,
                    future_features=future_projected
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

            # Process last observations for value calculation
            if config["ARCHITECTURE"].lower() == "cnn":
                last_obs_agent0 = last_obs['agent_0']
            else:
                last_obs_agent0 = last_obs['agent_0'].reshape(last_obs['agent_0'].shape[0], -1)

            # Get last value for advantage calculation
            _, last_val = network.apply(train_state.params.network, last_obs_agent0)

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

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minibatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        """Compute combined PPO and CPC loss"""
                        network_params, projection_params = params
                        
                        # Get policy outputs and features
                        pi, value, latent = network.apply(
                            network_params,
                            traj_batch.obs,
                            return_features=True
                        )
                        log_prob = pi.log_prob(traj_batch.action)
                        
                        # Project features
                        projected = projection_head.apply(
                            projection_params,
                            latent
                        )
                        
                        # Standard PPO losses
                        value_loss = 0.5 * jnp.square(value - targets).mean()
                        
                        # Actor loss with clipping
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        actor_loss1 = ratio * gae
                        actor_loss2 = jnp.clip(ratio, 1 - config["CLIP_EPS"], 1 + config["CLIP_EPS"]) * gae
                        actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()
                        
                        # Entropy bonus
                        entropy = pi.entropy().mean()
                        
                        # CPC loss
                        cpc_loss = compute_cpc_loss(
                            projected,
                            traj_batch.future_features,
                            config["CPC_TEMPERATURE"]
                        )
                        
                        # Combine losses
                        total_loss = (
                            actor_loss +
                            config["VF_COEF"] * value_loss -
                            config["ENT_COEF"] * entropy +
                            config["CPC_COEF"] * cpc_loss
                        )
                        
                        return total_loss, (value_loss, actor_loss, entropy, cpc_loss)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        (train_state.params.network, train_state.params.projection_head), traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                
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
                
                train_state, total_loss = jax.lax.scan(
                    _update_minibatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            
            # Log metrics
            metric = info
            current_timestep = update_step*config["NUM_STEPS"]*config["NUM_ENVS"]
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

@hydra.main(version_base=None, config_path="config", config_name="cpc")
def main(config):
    # Add CPC-specific config
    config.update({
        "CPC_COEF": 0.1,  # Weight for CPC loss
        "CPC_TEMPERATURE": 0.1,  # Temperature for similarity scaling
        "CPC_FUTURE_STEPS": 8,  # Number of future steps to predict
        "CPC_PROJECTION_DIM": 128,  # Dimension of projected features
    })
    
    # Set up Python path
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        
    # Process configuration
    config = OmegaConf.to_container(config)
    layout_name = config["ENV_KWARGS"]["layout"]

    # Initialize wandb
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", config["ARCHITECTURE"].upper(), "Adaptability", "Oracle", "Lower Bound"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f'lb_{config["ARCHITECTURE"]}_{layout_name}',
    )
    
    # Process layout configuration
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date = datetime.now().strftime("%Y%m%d")
    model_dir_name = f"lb_{config['ARCHITECTURE']}_{layout_name}_{timestamp}"
    save_dir = os.path.join(
        "saved_models", 
        date,
        layout_name, 
        f"{model_dir_name}_{config['SEED']}"
    )
    os.makedirs(save_dir, exist_ok=True)

    # Load pretrained parameters for agent_1
    pretrained_params = load_training_results(config["LOAD_PATH"], load_type="params", config=config)
    
    # Initialize and run training
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])    
    train_jit = jax.jit(lambda rng: make_train(config)(rng, pretrained_params))
    out = jax.vmap(train_jit)(rngs)
    
    # Save results and generate visualizations
    save_training_results(save_dir, out, config)
    
    # Create visualization
    train_state = jax.tree_util.tree_map(lambda x: x[0], out["runner_state"][0])
    viz_filename = os.path.join(save_dir, f'lb_{config["ARCHITECTURE"]}_{layout_name}_{config["SEED"]}.gif')
    create_visualization(train_state, pretrained_params, config, viz_filename, save_dir)
    
    # Plot learning curves
    rewards = out["metrics"]["returned_episode_returns"].reshape((config["NUM_SEEDS"], -1))
    plot_learning_curves(rewards, config, save_dir)

    # Log individual seed rewards
    for update_step in range(rewards.shape[1]):
        log_data = {"Update_Step": update_step}
        for seed_idx in range(config["NUM_SEEDS"]):
            log_data[f"Seeds/Seed_{seed_idx}/Returned_Episode_Returns"] = rewards[seed_idx, update_step]
        wandb.log(log_data)

if __name__ == "__main__":
    main()
