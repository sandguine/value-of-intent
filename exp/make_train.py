def make_train(config):
    # Initialize environment
    dims = config["DIMS"]
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    seq_len = config["SEQ_LEN"]

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
            action_dim=dims["action_dim"],  # Use dimension from config
            activation=config["ACTIVATION"],
            seq_len=seq_len
        )

        # Initialize seeds
        rng, _rng = jax.random.split(rng)
        _rng_agent_0, _rng_agent_1 = jax.random.split(_rng)  # Split for two networks

        # Initialize networks with correct dimensions from config
        init_x_agent_0 = jnp.zeros(dims["base_obs_dim"])  # Agent 0 gets base obs
        
        network_params_agent_0 = network.init(_rng_agent_0, init_x_agent_0)
        
        def create_optimizer(config):
            if config["ANNEAL_LR"]:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),  # First transformation
                    optax.adam(learning_rate=linear_schedule, eps=1e-5)  # Second transformation
                )
            else:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(config["LR"], eps=1e-5)
                )
            return tx

        # Create separate optimizer chains for each agent and only keep the one for agent_0 since this is the learning agent
        tx_agent_0 = create_optimizer(config)

        # Create train state for agent_0 only
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params_agent_0,
            tx=tx_agent_0
        )
        
        # Initialize environment states
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        
        # TRAIN LOOP
        def _update_step(runner_state, unused, seq_len, pretrained_params):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused, seq_len, pretrained_params):
                train_state, env_state, last_obs, update_step, rng, obs_history_0 = runner_state
                rng, rng_action_1, rng_action_0, rng_step = jax.random.split(rng, 4)

                num_envs = last_obs['agent_1'].shape[0]  # Should be 16

                agent_1_obs = last_obs['agent_1'].reshape(num_envs, -1)  # Shape: (520,)
                rng_action_1_split = jax.random.split(rng_action_1, num_envs)

                # Vectorized application across all environments
                agent_1_action = jax.vmap(
                    lambda params, obs, rng: network.apply(params, obs)[0].sample(seed=rng),
                    in_axes=(0, 0, 0)
                )(pretrained_params, agent_1_obs, rng_action_1_split)  # agent_1_action: (16,)

                # Initialize history buffer at the start of rollout
                if update_step == 0:  
                    obs_history_0 = jnp.tile(last_obs['agent_0'][:, None, :], (1, seq_len, 1))

                # Shift observation history for agent_0 BEFORE selecting action
                obs_history_0 = jnp.roll(obs_history_0, shift=-1, axis=1)
                obs_history_0 = obs_history_0.at[:, -1, :].set(last_obs['agent_0'])

                # Apply agent_0 policy using trainable parameters
                agent_0_pi, agent_0_value = network.apply(train_state.params, obs_history_0)
                agent_0_action = agent_0_pi.sample(seed=rng_action_0)
                agent_0_log_prob = agent_0_pi.log_prob(agent_0_action)

                # Step the environment
                actions = {"agent_0": agent_0_action, "agent_1": agent_1_action}
                next_obs, next_env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    jax.random.split(rng_step, num_envs),
                    env_state,
                    actions,
                )

                # Create transition
                transition = Transition(
                    done=done["agent_0"],
                    action=agent_0_action,
                    value=agent_0_value,
                    reward=reward["agent_0"],
                    log_prob=agent_0_log_prob,
                    obs=obs_history_0,
                )

                # Pass obs_history_0 forward for next step
                runner_state = (train_state, next_env_state, next_obs, update_step, rng, obs_history_0)
                return runner_state, (transition, info)


            runner_state, (traj_batch, info) = jax.lax.scan(
                lambda state, unused: _env_step(state, unused, seq_len, pretrained_params),  
                runner_state, 
                None, 
                config["NUM_STEPS"]
            )
            
            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, update_step, rng = runner_state
            
            # Ensure sequence length matches Transformer expectation
            last_obs_agent0 = last_obs['agent_0'].reshape(last_obs['agent_0'].shape[0], seq_len, -1)

            # Compute last value for advantage calculation
            _, last_val = network.apply(train_state.params, last_obs_agent0)

            # calculate_gae itself didn't need to be changed because we can use the same advantage function for both agents
            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
        
                    # Calculate delta and GAE per agent
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    #print(f"calculated gae shape: {gae.shape}, value: {gae}")
                    
                    return (gae, value), gae

                # Initialize with agent-specific last value
                init_gae = jnp.zeros_like(last_val)
                init_value = last_val

                # Calculate advantages for an agent
                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (init_gae, init_value),
                    traj_batch,
                    reverse=True,
                    unroll=16
                )
                
                return advantages, advantages + traj_batch.value

            # Calculate advantages 
            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused, seq_len, config):
                def _update_minbatch(train_state, batch_info, seq_len, config):
                    print("\nStarting minibatch update...")
                    # Unpack batch_info which now contains only agent_0 data
                    agent_0_data = batch_info['agent_0']
                    
                    # print("Minibatch shapes:")
                    # print("Agent 0 data:", jax.tree_util.tree_map(lambda x: x.shape, agent_0_data))

                    traj_batch = agent_0_data['traj']
                    advantages = agent_0_data['advantages']
                    targets = agent_0_data['targets']

                    def _loss_fn(params, traj_batch, gae, seq_len, targets, config):
                        """Calculate loss for agent_0."""

                        # Ensure batch size is properly inferred
                        batch_size = traj_batch.obs.shape[0] // seq_len  
                        obs_seq = traj_batch.obs.reshape(batch_size, seq_len, traj_batch.obs.shape[-1])
                        
                        # Forward pass through Transformer
                        pi, value = network.apply(params, obs_seq)
                        log_prob = pi.log_prob(traj_batch.action)

                        # Ensure targets match the sequence format
                        targets_seq = targets.reshape(batch_size, seq_len, -1)

                        # Value loss calculation
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets_seq)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets_seq)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # Normalize GAE correctly along the sequence dimension
                        gae = (gae - gae.mean(axis=1, keepdims=True)) / (gae.std(axis=1, keepdims=True) + 1e-8)

                        # PPO Actor Loss (Clipped Ratio)
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        loss_actor = -jnp.minimum(
                            ratio * gae,
                            jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        ).mean()

                        # Entropy regularization (Exploration bonus)
                        entropy = pi.entropy().mean()

                        # Compute total loss
                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )

                        return total_loss, {
                            'value_loss': value_loss,
                            'actor_loss': loss_actor,
                            'entropy': entropy,
                            'total_loss': total_loss
                        }

                    # Compute gradients for agent 0
                    grad_fn_0 = jax.value_and_grad(lambda p: _loss_fn(p, traj_batch, advantages, seq_len, targets, config), has_aux=True)
                    (loss_0, aux_0), grads_0 = grad_fn_0(train_state.params)

                    # Compute gradient norms correctly
                    grad_norm_0 = optax.global_norm(grads_0)

                    # Update only agent_0
                    train_state = train_state.apply_gradients(grads=grads_0)

                    return train_state, (loss_0, aux_0)
                
                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                # Calculate total batch size and minibatch size
                batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
                minibatch_size = batch_size // config["NUM_MINIBATCHES"]

                # Ensure batch size is evenly divisible
                assert batch_size % config["NUM_MINIBATCHES"] == 0, "Steps * Envs must be divisible by number of minibatches"

                # Reshape function that handles the different observation sizes
                def reshape_agent_data(agent_dict):

                    def reshape_field(x, field_name):
                        if not isinstance(x, (dict, jnp.ndarray)):
                            return x
                        return x.reshape(batch_size, -1) if field_name == 'obs' else x.reshape(batch_size)

                    return {
                        'traj': Transition(**{
                            field: reshape_field(getattr(agent_dict['traj'], field), field)
                            for field in agent_dict['traj']._fields
                        }),
                        'advantages': agent_dict['advantages'].reshape(batch_size),
                        'targets': agent_dict['targets'].reshape(batch_size)
                    }

                # Reshape trajectory data
                agent_data = {
                    "agent_0": {"traj": traj_batch, "advantages": advantages, "targets": targets}
                }
                agent_data = {agent: reshape_agent_data(data) for agent, data in agent_data.items()}

                # Shuffle data
                permutation = jax.random.permutation(_rng, batch_size)
                agent_data = {
                    agent: {
                        'traj': Transition(**{
                            field: jnp.take(getattr(data['traj'], field), permutation, axis=0)
                            for field in data['traj']._fields
                        }),
                        'advantages': jnp.take(data['advantages'], permutation, axis=0),
                        'targets': jnp.take(data['targets'], permutation, axis=0)
                    }
                    for agent, data in agent_data.items()
                }

                # Minibatch function
                def create_minibatches(data):
                    return {
                        'traj': Transition(**{
                            field: getattr(data["traj"], field).reshape((config["NUM_MINIBATCHES"], -1) + getattr(data["traj"], field).shape[1:])
                            for field in data["traj"]._fields  # Use data["traj"]
                        }),
                        'advantages': data["advantages"].reshape((config["NUM_MINIBATCHES"], -1)),
                        'targets': data["targets"].reshape((config["NUM_MINIBATCHES"], -1))
                    }

                # Create minibatches
                minibatches = {agent: create_minibatches(data) for agent, data in agent_data.items()}
                assert advantages.shape[0] % config["NUM_MINIBATCHES"] == 0, "Minibatch size is incorrect!"

                # Perform minibatch updates
                train_state, total_loss = jax.lax.scan(
                    lambda state, data: _update_minbatch(state, data, seq_len, config),  
                    train_state, 
                    minibatches
                )

                return (train_state, traj_batch, advantages, targets, rng), total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                lambda state, _: _update_epoch(state, _, seq_len, config),
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
                wandb.log(
                    metric
                )
            update_step = update_step + 1
            metric = jax.tree.map(lambda x: x.mean(), metric)
            metric["update_step"] = update_step
            metric["env_step"] = update_step*config["NUM_STEPS"]*config["NUM_ENVS"]
            jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, update_step, rng)
            # runner_state = (train_state, next_env_state, next_obs, update_step, rng, obs_history_0)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, 0, _rng)
        runner_state, metrics = jax.lax.scan(
            lambda state, unused: _update_step(state, unused, seq_len, pretrained_params),  
            runner_state,
            None,
            config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train