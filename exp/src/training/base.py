"""Base training module for JAX-based RL implementations."""

from typing import Dict, NamedTuple, Optional, Tuple, Type, Union
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
import numpy as np

from ..models.actor_critic import ActorCritic
from ..models.backbones.cnn import CNN
from ..models.backbones.rnn import RNN
from ..models.backbones.ff import FeedForward
from ..utils.data import process_observations, create_initial_obs

class TrainingState(NamedTuple):
    """Container for training state."""
    train_state: TrainState
    env_state: Dict
    last_obs: Dict
    update_step: int
    rng: jnp.ndarray

class BaseTrainer:
    """Base class for training implementations."""
    
    def __init__(
        self,
        config: Dict,
        env_name: str,
        env_kwargs: Dict,
        network_cls: Type[ActorCritic],
        backbone_cls: Type[Union[CNN, RNN, FeedForward]],
        backbone_config: Dict,
        num_envs: int,
        num_steps: int,
        num_updates: int,
        num_minibatches: int,
        total_timesteps: int,
        lr: float,
        gamma: float,
        gae_lambda: float,
        clip_eps: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        anneal_lr: bool = True,
        rew_shaping_horizon: Optional[int] = None
    ):
        """Initialize trainer with configuration."""
        self.config = config
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.network_cls = network_cls
        self.backbone_cls = backbone_cls
        self.backbone_config = backbone_config
        
        # Training parameters
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.num_updates = num_updates
        self.num_minibatches = num_minibatches
        self.total_timesteps = total_timesteps
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.anneal_lr = anneal_lr
        self.rew_shaping_horizon = rew_shaping_horizon
        
        # Initialize environment
        self.env = self._init_env()
        
        # Calculate derived parameters
        self.num_actors = self.env.num_agents * num_envs
        self.minibatch_size = self.num_actors * num_steps // num_minibatches
        
        # Setup learning rate schedule
        self.lr_schedule = self._setup_lr_schedule()
        
        # Setup reward shaping schedule if needed
        if rew_shaping_horizon is not None:
            self.rew_shaping_schedule = optax.linear_schedule(
                init_value=1.,
                end_value=0.,
                transition_steps=rew_shaping_horizon
            )
        else:
            self.rew_shaping_schedule = None

    def _init_env(self):
        """Initialize environment with logging wrapper."""
        env = jaxmarl.make(self.env_name, **self.env_kwargs)
        return jaxmarl.wrappers.baselines.LogWrapper(env, replace_info=False)

    def _setup_lr_schedule(self):
        """Setup learning rate annealing schedule."""
        if self.anneal_lr:
            return lambda count: self.lr * (
                1.0 - (count // (self.num_minibatches * self.config["UPDATE_EPOCHS"])) / self.num_updates
            )
        return self.lr

    def _init_network(self, rng: jnp.ndarray) -> Tuple[ActorCritic, Dict]:
        """Initialize network with proper observation shape."""
        network = self.network_cls(
            action_dim=self.env.action_space().n,
            backbone_cls=self.backbone_cls,
            backbone_config=self.backbone_config,
            activation=self.config["ACTIVATION"]
        )
        
        init_x = create_initial_obs(self.env.observation_space().shape, self.config)
        network_params = network.init(rng, init_x)
        
        return network, network_params

    def _setup_optimizer(self) -> optax.GradientTransformation:
        """Setup optimizer with optional learning rate annealing."""
        if self.anneal_lr:
            return optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.adam(learning_rate=self.lr_schedule, eps=1e-5)
            )
        return optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(self.lr, eps=1e-5)
        )

    def _init_training_state(self, rng: jnp.ndarray) -> TrainingState:
        """Initialize training state including network, optimizer, and environment."""
        # Initialize network and optimizer
        network, network_params = self._init_network(rng)
        optimizer = self._setup_optimizer()
        
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=optimizer
        )
        
        # Initialize environment states
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, self.num_envs)
        obsv, env_state = jax.vmap(self.env.reset, in_axes=(0,))(reset_rng)
        
        return TrainingState(
            train_state=train_state,
            env_state=env_state,
            last_obs=obsv,
            update_step=0,
            rng=rng
        )

    def _process_observations(self, obs: Dict) -> jnp.ndarray:
        """Process observations based on architecture type."""
        return process_observations(
            obs,
            self.env.agents,
            self.num_actors,
            self.env.observation_space().shape,
            self.config
        )

    def _calculate_gae(
        self,
        traj_batch: NamedTuple,
        last_val: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Calculate Generalized Advantage Estimation."""
        def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            done, value, reward = transition.done, transition.value, transition.reward
            delta = reward + self.gamma * next_value * (1 - done) - value
            gae = delta + self.gamma * self.gae_lambda * (1 - done) * gae
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True,
            unroll=16
        )
        return advantages, advantages + traj_batch.value

    def _compute_loss(
        self,
        params: Dict,
        traj_batch: NamedTuple,
        gae: jnp.ndarray,
        targets: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Dict]:
        """Compute PPO loss components."""
        pi, value = self.network.apply(params, traj_batch.obs)
        log_prob = pi.log_prob(traj_batch.action)

        # Value loss
        value_pred_clipped = traj_batch.value + (
            value - traj_batch.value
        ).clip(-self.clip_eps, self.clip_eps)
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

        # Actor loss
        ratio = jnp.exp(log_prob - traj_batch.log_prob)
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss_actor1 = ratio * gae
        loss_actor2 = jnp.clip(
            ratio,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps,
        ) * gae
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
        entropy = pi.entropy().mean()

        total_loss = (
            loss_actor
            + self.vf_coef * value_loss
            - self.ent_coef * entropy
        )
        
        return total_loss, {
            'value_loss': value_loss,
            'policy_loss': loss_actor,
            'entropy_loss': entropy,
            'total_loss': total_loss
        }

    def _update_epoch(
        self,
        update_state: Tuple[TrainState, NamedTuple, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        unused: None
    ) -> Tuple[Tuple[TrainState, NamedTuple, jnp.ndarray, jnp.ndarray, jnp.ndarray], Dict]:
        """Update network for one epoch."""
        def _update_minibatch(train_state, batch_info):
            traj_batch, advantages, targets = batch_info
            
            grad_fn = jax.value_and_grad(self._compute_loss, has_aux=True)
            total_loss, grads = grad_fn(
                train_state.params, traj_batch, advantages, targets
            )
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, total_loss

        train_state, traj_batch, advantages, targets, rng = update_state
        rng, _rng = jax.random.split(rng)
        
        # Reshape and shuffle data
        batch_size = self.minibatch_size * self.num_minibatches
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
                x, [self.num_minibatches, -1] + list(x.shape[1:])
            ),
            shuffled_batch,
        )
        
        train_state, total_loss = jax.lax.scan(
            _update_minibatch, train_state, minibatches
        )
        
        update_state = (train_state, traj_batch, advantages, targets, rng)
        return update_state, total_loss

    def train(self, rng: jnp.ndarray) -> Dict:
        """Main training loop."""
        # Initialize training state
        runner_state = self._init_training_state(rng)
        
        # Training loop
        def _update_step(runner_state, unused):
            # Collect trajectories
            runner_state, (traj_batch, info) = self._collect_trajectories(runner_state)
            
            # Calculate advantages
            advantages, targets = self._calculate_gae(traj_batch, self._get_last_value(runner_state))
            
            # Update network
            update_state = (runner_state.train_state, traj_batch, advantages, targets, runner_state.rng)
            update_state, loss_info = jax.lax.scan(
                self._update_epoch, update_state, None, self.config["UPDATE_EPOCHS"]
            )
            
            # Update runner state
            train_state = update_state[0]
            runner_state = TrainingState(
                train_state=train_state,
                env_state=runner_state.env_state,
                last_obs=runner_state.last_obs,
                update_step=runner_state.update_step + 1,
                rng=update_state[-1]
            )
            
            # Log metrics
            metric = self._process_metrics(info, runner_state.update_step)
            
            return runner_state, metric
        
        # Run training
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, self.num_updates
        )
        
        return {
            "runner_state": runner_state,
            "metrics": metrics
        }

    def _collect_trajectories(self, runner_state: TrainingState) -> Tuple[TrainingState, Tuple[NamedTuple, Dict]]:
        """Collect trajectories from environment."""
        raise NotImplementedError("Subclasses must implement _collect_trajectories")

    def _get_last_value(self, runner_state: TrainingState) -> jnp.ndarray:
        """Get value estimate for last observation."""
        raise NotImplementedError("Subclasses must implement _get_last_value")

    def _process_metrics(self, info: Dict, update_step: int) -> Dict:
        """Process and log training metrics."""
        raise NotImplementedError("Subclasses must implement _process_metrics") 