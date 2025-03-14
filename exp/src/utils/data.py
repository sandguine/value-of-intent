import jax.numpy as jnp
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