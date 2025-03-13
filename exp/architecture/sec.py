def infer_sec(Q_values_dict):
    sec_groups = {}
    for policy_id, Q_val in Q_values_dict.items():
        best_actions = tuple(jnp.argmax(Q_val, axis=-1).tolist())
        sec_groups.setdefault(best_actions, []).append(policy_id)
    return sec_groups

# Example Q_values_dict generated from embeddings:
Q_values_dict = {
    policy_id: critic_network.apply(params, embedding)
    for policy_id, embedding in policy_embeddings.items()
}

sec_groups = infer_sec(Q_values_dict)

# Immediately after observing co-player actions during each environment step.
# Update the prior probability of each section based on the likelihood of the co-player actions.

def bayesian_update(prior, likelihood):
    posterior = prior * likelihood
    posterior /= posterior.sum()
    return posterior

'''
# After observing co-player action `a_obs`
prior = policy_belief  # initial uniform or previously updated belief
likelihood = jnp.array([policy.predict_prob(a_obs) for policy in possible_policies])
policy_belief = bayesian_update(prior, likelihood)
'''

'''
def _update_step(runner_state, ...):
    ...
    # Step 1: Compute CPC embeddings and loss
    z_embeddings = network.apply(train_state.params, traj_batch.obs, method=network.feature_extractor)
    cpc_loss = cpc_module(z_embeddings, context, future_embeddings)

    # Step 2: Infer SECs from embeddings
    sec_groups = infer_sec(Q_values_dict)

    # Step 3: Bayesian update based on observed co-player actions
    updated_belief = bayesian_update(prior_belief, observed_action_likelihood)

    # Step 4: PPO loss computation and total loss aggregation
    ppo_loss, aux = compute_ppo_loss(...)
    total_loss = ppo_loss + λ * cpc_loss

    # Continue with optimization step
    ...

'''