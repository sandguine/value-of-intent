import jax.numpy as jnp
import jax
import jaxmarl
import os
import matplotlib.pyplot as plt
import numpy as np
from src.models.actor_critic import ActorCritic
from src.models.backbones.cnn import CNN
from src.models.backbones.rnn import RNN
from src.models.backbones.ff import FeedForward

def get_network(config, action_dim):
    """Factory function to create the appropriate network based on config.
    
    In the lower bound implementation, this network is used for both:
    - The learning agent (agent_0) with trainable parameters
    - The fixed agent (agent_1) with pretrained parameters from upper bound
    """
    if config["ARCHITECTURE"].lower() == "cnn":
        return ActorCritic(
            action_dim=action_dim,
            backbone_cls=CNN,
            backbone_config={
                "features": config["CNN_CONFIG"]["features"],
                "kernel_sizes": config["CNN_CONFIG"]["kernel_sizes"],
                "dense_features": config["CNN_CONFIG"]["dense_features"],
                "activation": config["ACTIVATION"]
            }
        )
    elif config["ARCHITECTURE"].lower() == "rnn":
        return ActorCritic(
            action_dim=action_dim,
            backbone_cls=RNN,
            backbone_config={
                "hidden_sizes": config["RNN_CONFIG"]["hidden_sizes"],
                "dense_features": config["RNN_CONFIG"]["dense_features"],
                "activation": config["ACTIVATION"]
            }
        )
    elif config["ARCHITECTURE"].lower() == "ff":
        return ActorCritic(
            action_dim=action_dim,
            backbone_cls=FeedForward,
            backbone_config={
                "hidden_layers": config["FF_CONFIG"]["hidden_layers"],
                "activation": config["ACTIVATION"]
            }
        )
    else:
        raise ValueError(f"Unknown architecture: {config['ARCHITECTURE']}")

def batchify(x: dict, agent_list, num_actors):
    """Convert dict of agent observations to batched array"""
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    """Convert batched array back to dict of agent observations"""
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def process_observations(obs, agent_list, num_actors, obs_shape, config):
    """Process observations based on architecture type.
    
    Args:
        obs: Raw observations from environment
        agent_list: List of agent names
        num_actors: Number of actors (agents * num_envs)
        obs_shape: Shape of observation space
        config: Configuration dictionary containing architecture info
        
    Returns:
        Processed observations in the format expected by the network
    """
    if config["ARCHITECTURE"].lower() == "cnn":
        return jnp.stack([obs[a] for a in agent_list]).reshape(-1, *obs_shape)
    elif config["ARCHITECTURE"].lower() in ["rnn", "ff"]:
        return batchify({k: v.flatten() for k, v in obs.items()}, agent_list, num_actors)
    else:
        raise ValueError(f"Unsupported architecture: {config['ARCHITECTURE']}")

def create_initial_obs(obs_shape, config):
    """Create initial observation tensor based on architecture type.
    
    Args:
        obs_shape: Shape of observation space
        config: Configuration dictionary containing architecture info
        
    Returns:
        Initial observation tensor for network initialization
    """
    if config["ARCHITECTURE"].lower() == "rnn":
        obs_dim = int(np.prod(obs_shape))
        return (
            jnp.zeros((1, obs_dim)),          # obs
            jnp.zeros((1,), dtype=bool)       # resets
        )
    elif config["ARCHITECTURE"].lower() == "ff":
        obs_dim = int(np.prod(obs_shape))
        return jnp.zeros((1, obs_dim))
    elif config["ARCHITECTURE"].lower() == "cnn":
        return jnp.zeros((1,) + obs_shape)
    else:
        raise ValueError(f"Unsupported architecture: {config['ARCHITECTURE']}")

def process_observations_asymmetric(obs, config):
    """Process observations for lower bound and CPC implementations.
    
    Args:
        obs: Raw observations from environment
        config: Configuration dictionary containing architecture info
        
    Returns:
        Dictionary containing processed observations for agent_0 and agent_1
    """
    if config["ARCHITECTURE"].lower() == "cnn":
        return {
            'agent_0': obs['agent_0'],  # Keep spatial dimensions for CNN
            'agent_1': obs['agent_1']
        }
    elif config["ARCHITECTURE"].lower() == "rnn":
        # For RNN, we need to maintain the sequence dimension
        # Assuming obs shape is (seq_len, *feature_dims)
        return {
            'agent_0': obs['agent_0'][None, :, :],  # Add batch dim but keep sequence dim
            'agent_1': obs['agent_1'][None, :, :]
        }
    else:  # feedforward case
        return {
            'agent_0': obs['agent_0'].flatten()[None, ...],
            'agent_1': obs['agent_1'].flatten()[None, ...]
        }