import os
import numpy as np
import matplotlib.pyplot as plt
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer


def plot_learning_curves(rewards, config, save_dir):
    """Plot and save learning curves"""
    reward_mean = rewards.mean(0)
    reward_std = rewards.std(0)
    reward_std_err = reward_std / np.sqrt(config["NUM_SEEDS"])

    plt.figure(figsize=(10, 6))
    plt.plot(reward_mean, label="Average Across All Seeds", color='black', linewidth=2)
    plt.fill_between(
        range(len(reward_mean)), 
        reward_mean - reward_std_err,
        reward_mean + reward_std_err,
        alpha=0.2, color='gray', 
        label="Mean ± Std Err"
    )
    
    for seed_idx in range(config["NUM_SEEDS"]):
        plt.plot(rewards[seed_idx], alpha=0.7)
        
    plt.xlabel("Update Step")
    plt.ylabel("Returned Episode Returns")
    plt.title("Per-Seed Performance on Returned Episode Returns")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "learning_curve.png"))
    plt.close()

def create_visualization(train_state, config, filename, save_dir=None, agent_view_size=5):
    """Create and save visualization of agent behavior"""
    base_name = os.path.splitext(os.path.basename(filename))[0]
    clean_filename = f"{base_name}.gif"
    
    state_seq = get_rollout(train_state, config, save_dir)
    viz = OvercookedVisualizer()
    
    if save_dir:
        clean_filename = os.path.join(save_dir, clean_filename)
    viz.animate(state_seq, agent_view_size=agent_view_size, filename=clean_filename)