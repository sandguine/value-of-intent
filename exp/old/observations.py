"""Unified observation processing for different architectures and training scenarios."""

from typing import Dict, Tuple, Union
import jax
import jax.numpy as jnp
import numpy as np

class ObservationProcessor:
    """Handles observation processing for different architectures and training scenarios."""
    
    def __init__(self, config: Dict, obs_shape: Tuple[int, ...]):
        self.config = config
        self.obs_shape = obs_shape
        self.architecture = config["ARCHITECTURE"].lower()
        
    def process_batch(
        self,
        obs: Dict[str, jnp.ndarray],
        agent_list: list,
        num_actors: int
    ) -> jnp.ndarray:
        """Process observations for batched training."""
        if self.architecture == "cnn":
            return self._process_cnn(obs, agent_list)
        elif self.architecture in ["rnn", "ff"]:
            return self._process_sequential(obs, agent_list, num_actors)
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
            
    def process_asymmetric(
        self,
        obs: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """Process observations for asymmetric training scenarios."""
        if self.architecture == "cnn":
            return {
                'agent_0': obs['agent_0'][None, ...],
                'agent_1': obs['agent_1'][None, ...]
            }
        elif self.architecture == "rnn":
            return {
                'agent_0': obs['agent_0'][None, :, :],
                'agent_1': obs['agent_1'][None, :, :]
            }
        else:  # feedforward case
            return {
                'agent_0': obs['agent_0'].flatten()[None, ...],
                'agent_1': obs['agent_1'].flatten()[None, ...]
            }
            
    def create_initial_obs(self) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Create initial observation tensor based on architecture."""
        if self.architecture == "rnn":
            obs_dim = int(np.prod(self.obs_shape))
            return (
                jnp.zeros((1, obs_dim)),          # obs
                jnp.zeros((1,), dtype=bool)       # resets
            )
        elif self.architecture == "ff":
            obs_dim = int(np.prod(self.obs_shape))
            return jnp.zeros((1, obs_dim))
        elif self.architecture == "cnn":
            return jnp.zeros((1,) + self.obs_shape)
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
            
    def _process_cnn(
        self,
        obs: Dict[str, jnp.ndarray],
        agent_list: list
    ) -> jnp.ndarray:
        """Process observations for CNN architecture."""
        return jnp.stack([obs[a] for a in agent_list]).reshape(-1, *self.obs_shape)
        
    def _process_sequential(
        self,
        obs: Dict[str, jnp.ndarray],
        agent_list: list,
        num_actors: int
    ) -> jnp.ndarray:
        """Process observations for RNN or FF architectures."""
        x = jnp.stack([obs[a] for a in agent_list])
        return x.reshape((num_actors, -1))
        
    def process_future_sequence(
        self,
        obs: Dict[str, jnp.ndarray],
        future_steps: int
    ) -> jnp.ndarray:
        """Process observations for future sequence prediction (CPC)."""
        if self.architecture == "cnn":
            return obs['agent_0']  # Shape: (batch, future_steps, obs_dim)
        elif self.architecture in ["rnn", "ff"]:
            return obs['agent_0'].reshape(-1, future_steps, -1)
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}") 