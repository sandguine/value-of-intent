"""CPC training implementation."""

from typing import Dict, NamedTuple, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState

from .base import BaseTrainer, TrainingState
from ..models.cpc import CPCNetwork
from ..utils.observations import ObservationProcessor

class Transition(NamedTuple):
    """Container for storing experience transitions."""
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    features: jnp.ndarray
    future_obs: jnp.ndarray

class CPCTrainer(BaseTrainer):
    """Trainer for CPC case (PPO with contrastive predictive coding)."""
    
    def __init__(self, config: Dict, env):
        super().__init__(
            config=config,
            env=env,
            network_cls=CPCNetwork,
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
        rng, rng_action, rng_step, rng_future = jax.random.split(rng, 4)
        
        # Process observations
        obs_batch = self.obs_processor.process_batch(last_obs)
        
        # Get actions and features from policy
        pi, value, features = self.network_cls.apply(
            train_state.params,
            obs_batch,
            return_features=True
        )
        action = pi.sample(seed=rng_action)[0]
        log_prob = pi.log_prob(action)
        
        # Step environment
        next_obs, next_env_state, reward, done, info = jax.vmap(self.env.step)(
            jax.random.split(rng_step, self.config["NUM_ENVS"]),
            env_state,
            action
        )
        
        # Get future observations
        future_obs = self.obs_processor.process_future_sequence(
            next_obs,
            self.config["FUTURE_STEPS"]
        )
        
        # Create transition
        transition = Transition(
            done=done,
            action=action,
            value=value,
            reward=reward,
            log_prob=log_prob,
            obs=obs_batch,
            features=features,
            future_obs=future_obs
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
            # Get policy outputs
            pi, value, features, future_preds = self.network_cls.apply(
                params,
                traj_batch.obs,
                return_features=True,
                return_future=True
            )
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
            
            # CPC loss
            cpc_loss = self._compute_cpc_loss(features, traj_batch.future_obs, future_preds)
            
            total_loss = (
                loss_actor
                + self.config["VF_COEF"] * value_loss
                - self.config["ENT_COEF"] * entropy
                + self.config["CPC_COEF"] * cpc_loss
            )
            return total_loss, (value_loss, loss_actor, entropy, cpc_loss)
            
        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        total_loss, grads = grad_fn(
            train_state.params,
            traj_batch,
            advantages,
            targets
        )
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, total_loss
        
    def _compute_cpc_loss(
        self,
        features: jnp.ndarray,
        future_obs: jnp.ndarray,
        future_preds: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute contrastive predictive coding loss."""
        # Normalize features and predictions
        features = features / (jnp.linalg.norm(features, axis=-1, keepdims=True) + 1e-8)
        future_preds = future_preds / (jnp.linalg.norm(future_preds, axis=-1, keepdims=True) + 1e-8)
        
        # Compute similarity matrix
        sim_matrix = jnp.matmul(features, future_preds.transpose(0, 2, 1))
        
        # Create positive and negative masks
        pos_mask = jnp.eye(sim_matrix.shape[-1], dtype=jnp.float32)
        neg_mask = 1 - pos_mask
        
        # Compute InfoNCE loss
        exp_sim = jnp.exp(sim_matrix / self.config["TEMPERATURE"])
        pos_sim = jnp.sum(exp_sim * pos_mask, axis=-1)
        neg_sim = jnp.sum(exp_sim * neg_mask, axis=-1)
        nce_loss = -jnp.mean(jnp.log(pos_sim / (pos_sim + neg_sim)))
        
        return nce_loss
        
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
        last_obs_batch = self.obs_processor.process_batch(last_obs)
        _, last_val, _ = self.network_cls.apply(
            train_state.params,
            last_obs_batch,
            return_features=True
        )
        
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
        metric["shaped_reward"] = metric["shaped_reward"] * self.rew_shaping_schedule(current_timestep)
        
        # Update runner state
        runner_state = TrainingState(
            train_state=train_state,
            env_state=env_state,
            last_obs=last_obs,
            update_step=update_step + 1,
            rng=update_state[-1]
        )
        
        return runner_state, metric 