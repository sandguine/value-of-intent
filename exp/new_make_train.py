def make_train(config):
    # Initialize environment
    dims = config["DIMS"]
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    # Verify dimensions match what we validated in main
    assert np.prod(env.observation_space().shape) == dims["base_obs_dim"], "Observation dimension mismatch"
    assert env.action_space().n == dims["action_dim"], "Action dimension mismatch"

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
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    # Schedule for annealing reward shaping
    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.,
        end_value=0.,
        transition_steps=config["REW_SHAPING_HORIZON"]
    )

    def train(rng, pretrained_params, seq_len):
        network = ActorCritic(
            action_dim=dims["action_dim"],  
            activation=config["ACTIVATION"],
            seq_len=seq_len  # Ensure Transformer knows sequence length
        )

        # Initialize train state
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network.init(jax.random.PRNGKey(0), jnp.zeros((1, seq_len, dims["base_obs_dim"]))),
            tx=optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5)
            )
        )

        # Initialize environment states
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(runner_state, unused, seq_len, pretrained_params):
            train_state, env_state, last_obs, update_step, rng, obs_history_0 = runner_state

            # Ensure `obs_history_0` persists
            if update_step == 0:  
                obs_history_0 = jnp.tile(last_obs['agent_0'][:, None, :], (1, seq_len, 1))

            # Shift observation history before action selection
            obs_history_0 = jnp.roll(obs_history_0, shift=-1, axis=1)
            obs_history_0 = obs_history_0.at[:, -1, :].set(last_obs['agent_0'])

            # Compute policy and value estimates
            pi_0, value_0 = network.apply(train_state.params, obs_history_0)
            action_0 = pi_0.sample(seed=jax.random.split(rng, 1)[0])
            log_prob_0 = pi_0.log_prob(action_0)

            # Compute targets
            _, last_val = network.apply(train_state.params, obs_history_0)
            
            if last_val.ndim == 1:
                last_val = last_val[:, None]  # Ensure it has the correct shape
            
            # Update runner_state
            runner_state = (train_state, env_state, last_obs, update_step, rng, obs_history_0)
            return runner_state, {}

        runner_state = (train_state, env_state, obsv, 0, rng, None)
        runner_state, metrics = jax.lax.scan(
            lambda state, unused: _update_step(state, unused, config["SEQ_LEN"], pretrained_params),
            runner_state,
            None,
            config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train
