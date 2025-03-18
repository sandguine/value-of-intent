import jax
import jax.numpy as jnp
import numpy as np
import pickle
import os

def save_training_results(save_dir, out, config):
    """Save training results and metrics"""
    os.makedirs(save_dir, exist_ok=True)

    def is_pickleable(obj):
        try:
            pickle.dumps(obj)
            return True
        except Exception:
            return False
    
    def process_tree(tree):
        def convert_and_filter(x):
            if isinstance(x, (jax.Array, np.ndarray)):
                x = np.array(x)
            return x if is_pickleable(x) else None
        return jax.tree_util.tree_map(convert_and_filter, tree)
    
    # Convert outputs to numpy and save
    numpy_out = jax.tree_util.tree_map(
        lambda x: np.array(x) if isinstance(x, (jax.Array, np.ndarray)) else x,
        jax.device_get(out)
    )
    
    # Filter and save pickleable objects
    pickle_safe_out = {k: v for k, v in numpy_out.items() if is_pickleable(v)}
    
    # Save complete output
    with open(os.path.join(save_dir, "complete_out.pkl"), 'wb') as f:
        pickle.dump(pickle_safe_out, f)
    np.savez(os.path.join(save_dir, "complete_out.npz"), **pickle_safe_out)
    
    # Save metrics
    if "metrics" in out:
        np.savez(
            os.path.join(save_dir, "metrics.npz"),
            **{k: np.array(v) for k, v in out["metrics"].items()}
        )
    
    # Save config
    with open(os.path.join(save_dir, "config.pkl"), 'wb') as f:
        pickle.dump(config, f)

def save_latent_embeddings(save_dir, latent_storage, action_storage):
    """Save latent embeddings and associated actions."""
    os.makedirs(save_dir, exist_ok=True)

    # Convert JAX arrays to NumPy
    latent_storage = jax.device_get(jnp.concatenate(latent_storage, axis=0))
    action_storage = jax.device_get(jnp.concatenate(action_storage, axis=0))

    # Save as NPZ (compressed format for easy loading)
    np.savez(
        os.path.join(save_dir, "latent_embeddings.npz"),
        latents=latent_storage,
        actions=action_storage
    )

    # Save as Pickle for additional flexibility
    with open(os.path.join(save_dir, "latent_embeddings.pkl"), 'wb') as f:
        pickle.dump({"latents": latent_storage, "actions": action_storage}, f)

    print(f"Latent embeddings saved to {save_dir}")

def load_training_results(load_dir, load_type="complete", config=None):
    """Load training results from specified directory"""
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    if load_type == "params":
        pickle_path = os.path.join(load_dir, "all_seeds_params.pkl")
        if os.path.exists(pickle_path):
            print("Loading params from pickle format...")
            with open(pickle_path, 'rb') as f:
                all_params = pickle.load(f)
            
            num_seeds = len(all_params.keys())
            print("num seeds", num_seeds)

            all_params = flax.core.freeze(all_params)
            num_envs = config["NUM_ENVS"]

            sampled_indices = jax.random.choice(subkey, num_seeds, shape=(num_envs,), replace=False)
            print("sampled_indices", sampled_indices)

            sampled_params_list = [{'params': all_params[f'seed_{i}']['params']} for i in sampled_indices]
            sampled_params = jax.tree_util.tree_map(
                lambda *x: jnp.stack(x, axis=0), *sampled_params_list
            )

            print("Successfully loaded pretrained model.")
            print("Loaded params type:", type(sampled_params))
            print("Shape of sampled_params:", jax.tree_util.tree_map(lambda x: x.shape, sampled_params))

            return sampled_params
                
    elif load_type == "complete":
        pickle_path = os.path.join(load_dir, "complete_out.pkl")
        if os.path.exists(pickle_path):
            print("Loading complete output from pickle format...")
            with open(pickle_path, 'rb') as f:
                out = pickle.load(f)
                return jax.tree_util.tree_map(
                    lambda x: jax.numpy.array(x) if isinstance(x, np.ndarray) else x,
                    out
                )
    
    raise FileNotFoundError(f"No saved {load_type} found in {load_dir}")
