# Transformer encoder for PPO

class TransformerFeatureExtractor(nn.Module):
    """Minimal Transformer Encoder for PPO with sequence processing."""
    num_layers: int = 2  # Number of Transformer blocks
    model_dim: int = 64  # Hidden dimension
    num_heads: int = 4   # Attention heads
    seq_len: int = 5     # Number of timesteps to consider

    def setup(self):
        """Define Transformer Encoder layers."""
        self.positional_encoding = self.create_positional_encoding(self.seq_len, self.model_dim)

        self.encoder_blocks = [
            nn.SelfAttention(
                num_heads=self.num_heads,
                qkv_features=self.model_dim,
                kernel_init=orthogonal(np.sqrt(2))
            ) for _ in range(self.num_layers)
        ]

        self.final_dense = nn.Dense(features=self.model_dim, kernel_init=orthogonal(np.sqrt(2)))

    def create_positional_encoding(self, seq_len, model_dim):
        """Create sinusoidal positional encoding for sequence processing."""
        positions = jnp.arange(seq_len)[:, None]  # Shape (seq_len, 1)
        div_term = jnp.exp(jnp.arange(0, model_dim, 2) * -(np.log(10000.0) / model_dim))
        pos_enc = jnp.zeros((seq_len, model_dim))

        pos_enc = pos_enc.at[:, 0::2].set(jnp.sin(positions * div_term))
        pos_enc = pos_enc.at[:, 1::2].set(jnp.cos(positions * div_term))
        return pos_enc

    def __call__(self, x):
        """Apply Transformer Encoder to input."""
        # Ensure x has shape (B, T, F) or (T, F)
        x = x + self.positional_encoding  # Add position information

        for block in self.encoder_blocks:
            x = block(x)  # Apply self-attention layers

        x = self.final_dense(x)  # Final feature projection
        return x

class ActorCritic(nn.Module):
    """Actor-Critic model using Transformer feature extractor."""
    action_dim: int
    activation: str = "tanh"
    seq_len: int = 5  # Match transformer sequence length

    def setup(self):
        """Define Transformer-based Actor-Critic architecture."""
        self.act_fn = nn.relu if self.activation == "relu" else nn.tanh

        # Replace CNN with Transformer feature extractor
        self.feature_extractor = TransformerFeatureExtractor(seq_len=self.seq_len)

        # Actor network
        self.actor_hidden = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="actor_hidden"
        )
        self.actor_out = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor_output"
        )

        # Critic network
        self.critic_hidden = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="critic_hidden"
        )
        self.critic_out = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic_output"
        )

    @nn.compact
    def __call__(self, x):
        """Forward pass of the Transformer-based Actor-Critic model."""
        # Extract high-level features using Transformer (now takes sequences)
        features = self.feature_extractor(x)  # Shape (T, F) or (B, T, F)

        # Aggregate features (e.g., take the last timestep)
        last_feature = features[:, -1, :]  # Take last timestep

        # Actor head
        actor_hidden = self.actor_hidden(last_feature)
        actor_hidden = self.act_fn(actor_hidden)
        actor_logits = self.actor_out(actor_hidden)
        pi = distrax.Categorical(logits=actor_logits)

        # Critic head
        critic_hidden = self.critic_hidden(last_feature)
        critic_hidden = self.act_fn(critic_hidden)
        critic_value = self.critic_out(critic_hidden)

        return pi, jnp.squeeze(critic_value, axis=-1)
    
class DelayedObsWrapper(MultiAgentEnv):
    
    def __init__(self, baseEnv, delay):
        self.baseEnv = baseEnv
        self.num_agents = baseEnv.num_agents
        self.agents = baseEnv.agents
        self.window_size = delay

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], StateWindowObs]:
        obs, state = self.baseEnv.reset(key)
        
        dummy_obs, timestep = self.update_obs_dict(obs, -1, dummy=True)
        obs_window = [dummy_obs]
        for _ in range(self.window_size - 1):
            new_dummy_obs, timestep = self.update_obs_dict(obs, timestep, dummy=True)
            obs_window.append(new_dummy_obs)
        new_current_obs, timestep = self.update_obs_dict(obs, timestep)
        obs_window.append(new_current_obs)
        # obs_window = [dummy_obs] * (self.window_size - 1) + [obs]
        state = StateWindowObs(state, obs_window, timestep)
        return jnp.stack(obs_window), state # transformer need full sequence as input

    def update_obs_dict(self, obs_dict, prev_timestep, dummy=False):
        new_obs = {}
        for agent, observ in obs_dict.items():
            if dummy:
                observ = jnp.zeros_like(observ)
            new_dummy_observ, timestep = self.add_timestep_dimension(observ, prev_timestep)
            new_obs[agent] = new_dummy_observ
        return new_obs, timestep
    
    def add_timestep_dimension(self, observation_array, prev_timestep):
        # assert prev_timestep < 255
        timestep_layer = jnp.zeros(observation_array.shape[:-1])
        new_timestep_layer = timestep_layer.at[tuple(0 for _ in range(timestep_layer.ndim))].set(prev_timestep + 1)
        return jnp.concatenate([observation_array, jnp.expand_dims(new_timestep_layer, -1)], axis=-1), prev_timestep + 1
    
        
    @partial(jax.jit, static_argnums=(0,))
    def step_copolicy(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
        co_params,
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Performs step transitions in the environment."""

        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env_copolicy(key, state, actions, co_params)

        obs_re, states_re = self.reset(key_reset)

        # Auto-reset environment based on termination
        states = jax.tree_map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), states_re, states_st
        )
        obs = jax.tree_map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st
        )
        return obs, states, rewards, dones, infos        

    def step_env(
        self, key: chex.PRNGKey, state_window_obs: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Environment-specific step transition."""
        # import pdb; pdb.set_trace()

        obs_st, states_st, rewards, dones, infos = self.baseEnv.step(key, state_window_obs.state, actions)
        obs_st, new_timestep = self.update_obs_dict(obs_st, state_window_obs.most_recent_timestep)
        new_obs_window = state_window_obs.obs_window[1:] + [obs_st]
        new_state_window_obs = StateWindowObs(states_st, new_obs_window, new_timestep)
        curr_obs = new_state_window_obs.obs_window[0]
        return jnp.stack(new_state_window_obs.obs_window), new_state_window_obs, rewards, dones, infos

    
    def step_env_copolicy(
        self, key: chex.PRNGKey, state_window_obs: State, actions: Dict[str, chex.Array], coparams
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Environment-specific step transition."""
        # import pdb; pdb.set_trace()

        obs_st, states_st, rewards, dones, infos = self.baseEnv.step_copolicy(key, state_window_obs.state, actions, coparams)
        obs_st, new_timestep = self.update_obs_dict(obs_st, state_window_obs.most_recent_timestep)
        new_obs_window = state_window_obs.obs_window[1:] + [obs_st]
        new_state_window_obs = StateWindowObs(states_st, new_obs_window, new_timestep)
        curr_obs = new_state_window_obs.obs_window[0]
        # 
        return jnp.stack(new_state_window_obs.obs_window), new_state_window_obs, rewards, dones, infos 

    def observation_space(self, agent: str=''):
        """Observation space for a given agent."""
        assert isinstance(self.baseEnv.observation_space(agent), Box)
        new_low = self.baseEnv.observation_space(agent).low
        new_high = self.baseEnv.observation_space(agent).high
        new_shape = list(self.baseEnv.observation_space(agent).shape)
        new_shape[-1] += 1
        return Box(new_low, new_high, tuple(new_shape))


    def action_space(self, agent: str=''):
        """Action space for a given agent."""
        return self.baseEnv.action_space(agent)

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @property
    def agent_classes(self) -> dict:
        return self.baseEnv.agent_classes()