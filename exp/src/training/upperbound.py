"""Upper bound training implementation."""

from typing import Dict, NamedTuple, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState

from .base import BaseTrainer, TrainingState
from ..models.actor_critic import ActorCritic
from ..utils.observations import ObservationProcessor

class Transition(NamedTuple):
    """Container for storing experience transitions."""
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray

class UpperBoundTrainer(BaseTrainer):
    """Trainer for upper bound case (independent PPO)."""
    
    def __init__(self, config: Dict, env):
        super().__init__(
            config=config,
            env=env,
            network_cls=ActorCritic,
            backbone_cls=self._get_backbone_cls(config),
            backbone_config=self._get_backbone_config(config)
        )
        self.obs_processor = ObservationProcessor(config, env.observation_space().shape)
        
    def _get_backbone_cls(self, config: Dict) -> type:
        """Get backbone class based on config."""
        arch = config["ARCHITECTURE"].lower()
        if arch == "cnn":
            from ..models.backbones.cnn import CNN
            return CNN
        elif arch == "rnn":
            from ..models.backbones.rnn import RNN
            return RNN
        elif arch == "ff":
            from ..models.backbones.ff import FeedForward
            return FeedForward
        else:
            raise ValueError(f"Unknown architecture: {arch}")
            
    def _get_backbone_config(self, config: Dict) -> Dict:
        """Get backbone configuration based on config."""
        arch = config["ARCHITECTURE"].lower()
        if arch == "cnn":
            return config["CNN_CONFIG"]
        elif arch == "rnn":
            return config["RNN_CONFIG"]
        elif arch == "ff":
            return config["FF_CONFIG"]
        else:
            raise ValueError(f"Unknown architecture: {arch}")
            
    def _env_step(
        self,
        training_state: TrainingState,
        unused
    ) -> Tuple[TrainingState, Tuple]:
        """Execute one environment step."""
        train_state, env_state, last_obs, update_step, rng = training_state
        rng, rng_action, rng_step = jax.random.split(rng, 3)
        
        # Process observations
        obs_batch = self.obs_processor.process_batch(
            last_obs,
            self.env.agents,
            self.num_actors
        )
        
        # Get policy outputs
        pi, value = self.network_cls.apply(train_state.params, obs_batch)
        action = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(action)
        
        # Convert actions to environment format
        env_act = {
            k: v.flatten()
            for k, v in self._unbatchify(action, self.env.agents, self.config["NUM_ENVS"]).items()
        }
        
        # Step environment
        next_obs, next_env_state, reward, done, info = jax.vmap(self.env.step)(
            jax.random.split(rng_step, self.config["NUM_ENVS"]),
            env_state,
            env_act
        )
        
        # Create transition
        transition = Transition(
            done=self._batchify(done, self.env.agents, self.num_actors).squeeze(),
            action=action,
            value=value,
            reward=self._batchify(reward, self.env.agents, self.num_actors).squeeze(),
            log_prob=log_prob,
            obs=obs_batch
        )
        
        # Update training state
        training_state = TrainingState(
            train_state=train_state,
            env_state=next_env_state,
            last_obs=next_obs,
            update_step=update_step,
            rng=rng
        )
        
        return training_state, (transition, info)
        
    def _update_epoch(
        self,
        update_state: Tuple,
        unused
    ) -> Tuple[Tuple, jnp.ndarray]:
        """Execute one update epoch."""
        train_state, traj_batch, advantages, targets, rng = update_state
        rng, _rng = jax.random.split(rng)
        
        # Prepare minibatches
        batch_size = self.minibatch_size * self.config["NUM_MINIBATCHES"]
        permutation = jax.random.permutation(_rng, batch_size)
        
        batch = (traj_batch, advantages, targets)
        batch = jax.tree_util.tree_map(
            lambda x: x.reshape((batch_size,) + x.shape[2:]),
            batch
        )
        shuffled_batch = jax.tree_util.tree_map(
            lambda x: jnp.take(x, permutation, axis=0),
            batch
        )
        minibatches = jax.tree_util.tree_map(
            lambda x: jnp.reshape(
                x,
                [self.config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
            ),
            shuffled_batch
        )
        
        # Update network
        train_state, total_loss = jax.lax.scan(
            self._update_minibatch,
            train_state,
            minibatches
        )
        
        update_state = (train_state, traj_batch, advantages, targets, rng)
        return update_state, total_loss
        
    def _update_minibatch(
        self,
        train_state: TrainState,
        batch_info: Tuple
    ) -> Tuple[TrainState, jnp.ndarray]:
        """Update network on a single minibatch."""
        traj_batch, advantages, targets = batch_info
        
        def _loss_fn(params, traj_batch, gae, targets):
            pi, value = self.network_cls.apply(params, traj_batch.obs)
            log_prob = pi.log_prob(traj_batch.action)
            
            # Value loss
            value_pred_clipped = traj_batch.value + (
                value - traj_batch.value
            ).clip(-self.config["CLIP_EPS"], self.config["CLIP_EPS"])
            value_losses = jnp.square(value - targets)
            value_losses_clipped = jnp.square(value_pred_clipped - targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
            
            # Actor loss
            ratio = jnp.exp(log_prob - traj_batch.log_prob)
            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
            loss_actor1 = ratio * gae
            loss_actor2 = jnp.clip(
                ratio,
                1.0 - self.config["CLIP_EPS"],
                1.0 + self.config["CLIP_EPS"]
            ) * gae
            loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
            entropy = pi.entropy().mean()
            
            total_loss = (
                loss_actor
                + self.config["VF_COEF"] * value_loss
                - self.config["ENT_COEF"] * entropy
            )
            return total_loss, (value_loss, loss_actor, entropy)
            
        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        total_loss, grads = grad_fn(
            train_state.params,
            traj_batch,
            advantages,
            targets
        )
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, total_loss
        
    def _update_step(
        self,
        runner_state: TrainingState,
        unused
    ) -> Tuple[TrainingState, Dict]:
        """Execute one update step."""
        # Collect trajectories
        runner_state, (traj_batch, info) = jax.lax.scan(
            self._env_step,
            runner_state,
            None,
            self.config["NUM_STEPS"]
        )
        
        # Calculate advantages
        train_state, env_state, last_obs, update_step, rng = runner_state
        last_obs_batch = self.obs_processor.process_batch(
            last_obs,
            self.env.agents,
            self.num_actors
        )
        _, last_val = self.network_cls.apply(train_state.params, last_obs_batch)
        
        advantages, targets = self._calculate_gae(traj_batch, last_val)
        
        # Update network
        update_state = (train_state, traj_batch, advantages, targets, rng)
        update_state, loss_info = jax.lax.scan(
            self._update_epoch,
            update_state,
            None,
            self.config["UPDATE_EPOCHS"]
        )
        train_state = update_state[0]
        
        # Log metrics
        metric = info
        current_timestep = update_step * self.config["NUM_STEPS"] * self.config["NUM_ENVS"]
        metric["shaped_reward"] = metric["shaped_reward"]["agent_0"]
        metric["shaped_reward_annealed"] = metric["shaped_reward"] * self.rew_shaping_schedule(current_timestep)
        
        # Update runner state
        runner_state = TrainingState(
            train_state=train_state,
            env_state=env_state,
            last_obs=last_obs,
            update_step=update_step + 1,
            rng=update_state[-1]
        )
        
        return runner_state, metric
        
    def _batchify(self, x: Dict, agent_list: list, num_actors: int) -> jnp.ndarray:
        """Convert dict of agent data to batched array."""
        x = jnp.stack([x[a] for a in agent_list])
        return x.reshape((num_actors, -1))
        
    def _unbatchify(
        self,
        x: jnp.ndarray,
        agent_list: list,
        num_envs: int
    ) -> Dict:
        """Convert batched array back to dict of agent data."""
        x = x.reshape((len(agent_list), num_envs, -1))
        return {a: x[i] for i, a in enumerate(agent_list)} 