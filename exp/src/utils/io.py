import jax
import jax.numpy as jnp
import numpy as np
import pickle
import os
import flax

from .models import CPCModule
from .models.backbones.cnn import CNN
from .models.backbones.ff import FeedForward
from .models.backbones.rnn import RNN

def save_training_results(save_dir, out, config):
    """Save training results to specified directory"""
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
    
    # Convert outputs to numpy format
    numpy_out = jax.tree_util.tree_map(
        lambda x: np.array(x) if isinstance(x, (jax.Array, np.ndarray)) else x,
        jax.device_get(out)
    )
    
    # Filter and save pickleable objects
    pickle_safe_out = {}
    for key, value in numpy_out.items():
        try:
            pickle.dumps(value)
            pickle_safe_out[key] = value
        except Exception as e:
            print(f"Warning: Skipping unpickleable key '{key}' due to: {str(e)}")

    # Save complete output
    pickle_out_path = os.path.join(save_dir, f"complete_out.pkl")
    with open(pickle_out_path, 'wb') as f:
        pickle.dump(pickle_safe_out, f)

    npz_out_path = os.path.join(save_dir, f"complete_out.npz")
    np.savez(npz_out_path, **pickle_safe_out)

    # Process seed-specific parameters
    all_seeds_params = {}
    for seed_idx in range(config["NUM_SEEDS"]):
        try:
            if "runner_state" not in out or not out["runner_state"]:
                print(f"Warning: No runner_state found for seed {seed_idx}")
                continue
                
            train_state = jax.tree_util.tree_map(
                lambda x: x[seed_idx] if x is not None else None,
                out["runner_state"][0]
            )
            
            processed_state = {}
            
            if hasattr(train_state, 'params'):
                processed_params = process_tree(train_state.params)
                if processed_params is not None:
                    processed_state['params'] = processed_params['params']
            
            if hasattr(train_state, 'step'):
                try:
                    processed_state['step'] = np.array(train_state.step)
                except Exception as e:
                    print(f"Warning: Could not process step for seed {seed_idx}: {str(e)}")
            
            if "metrics" in out:
                processed_metrics = process_tree(
                    jax.tree_util.tree_map(
                        lambda x: x[seed_idx] if isinstance(x, (jax.Array, np.ndarray)) else x,
                        out["metrics"]
                    )
                )
                if processed_metrics:
                    processed_state['metrics'] = processed_metrics
            
            if processed_state:
                all_seeds_params[f"seed_{seed_idx}"] = processed_state
            
        except Exception as e:
            print(f"Warning: Could not process seed {seed_idx} due to: {str(e)}")
            continue
    
    if all_seeds_params:
        pickle_seeds_path = os.path.join(save_dir, f"all_seeds_params.pkl")
        with open(pickle_seeds_path, 'wb') as f:
            pickle.dump(all_seeds_params, f)
        
        npz_seeds_path = os.path.join(save_dir, f"all_seeds_params.npz")
        np.savez(npz_seeds_path, **all_seeds_params)
    else:
        print("Warning: No seed-specific parameters were successfully processed")

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

def load_cpc_encoder(config):
    encoder_type = config['CPC_CONFIG']['encoder_type']
    
    if encoder_type == 'cnn':
        from cnn import CNNEncoder
        def encoder_fn(obs):
            model = CNNEncoder(config['CNN_CONFIG'])
            return model(obs)
        return encoder_fn
    
    elif encoder_type == 'ff':
        from ff import FFEncoder
        def encoder_fn(obs):
            model = FFEncoder(config['FF_CONFIG'])
            return model(obs)
        return encoder_fn

    elif encoder_type == 'rnn':
        from rnn import RNNEncoder
        def encoder_fn(obs):
            model = RNNEncoder(config['RNN_CONFIG'])
            return model(obs)
        return encoder_fn
    
    else:
        raise ValueError(f"Unknown CPC encoder type: {encoder_type}")