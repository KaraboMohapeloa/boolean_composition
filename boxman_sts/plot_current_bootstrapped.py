import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import deepdish as dd
import os
from datetime import datetime

def plot_current_training(stats_path, window=20, save_dir='plots'):
    """
    Plot current bootstrapped training progress from intermediate saves.
    Shows SAMPLE EFFICIENCY: reward vs timesteps (not episodes)
    
    Parameters:
    -----------
    stats_path: str
        Path to the HDF5 stats file (intermediate saves)
    window: int
        Rolling window size for smoothing
    save_dir: str
        Directory to save the plot
    """
    if not os.path.exists(stats_path):
        print(f"ERROR: File not found: {stats_path}")
        return
    
    print("="*70)
    print("LOADING CURRENT BOOTSTRAPPED TRAINING DATA")
    print("="*70)
    
    # Load data
    data = dd.io.load(stats_path)
    rewards = data['R']
    total_steps = data['T']
    
    print(f"File: {stats_path}")
    print(f"Total Episodes: {len(rewards)}")
    print(f"Total Timesteps: {total_steps:,}")
    print(f"Episode Reward Range: [{np.min(rewards):.2f}, {np.max(rewards):.2f}]")
    
    # Calculate timesteps per episode (approximate)
    avg_steps_per_episode = total_steps / len(rewards) if len(rewards) > 0 else 0
    print(f"Average Steps per Episode: {avg_steps_per_episode:.1f}")
    
    # Calculate cumulative timesteps for each episode
    # Assuming roughly equal episode lengths
    timesteps = np.linspace(0, total_steps, len(rewards))
    
    # Calculate statistics
    if len(rewards) >= 100:
        avg_last_100 = np.mean(rewards[-100:])
        print(f"Average Last 100 Episodes: {avg_last_100:.2f}")
    
    # Prepare data for plotting
    rewards_smooth = pd.Series(rewards).rolling(window, min_periods=1).mean()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot raw and smoothed rewards vs timesteps
    ax.plot(timesteps, rewards, alpha=0.3, label='Episode Reward (raw)', color='tab:green', linewidth=0.8)
    ax.plot(timesteps, rewards_smooth, label=f'Smoothed (window={window})', linewidth=2.5, color='darkgreen')
    
    # Add horizontal line for average of last 100 episodes
    if len(rewards) >= 100:
        ax.axhline(avg_last_100, color='red', linestyle='--', linewidth=2, 
                   label=f'Avg Last 100 Episodes: {avg_last_100:.2f}')
        # Add text annotation
        ax.text(total_steps * 0.05, avg_last_100, f'{avg_last_100:.2f}', 
                color='red', va='bottom', ha='left', fontsize=11, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', alpha=0.8))
    
    # Add zero line for reference
    ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # Labels (no title)
    ax.set_xlabel('Environment Timesteps (×10⁶)', fontsize=13)
    ax.set_ylabel('Episode Reward', fontsize=13)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis to show timesteps in millions (×10^6)
    if total_steps >= 1e6:
        ax.set_xticklabels([f'{x/1e6:.1f}' for x in ax.get_xticks()])
    
    plt.tight_layout()
    
    # Show plot first
    plt.show()
    
    # Save after viewing
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'bootstrapped_current_progress.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    print("\n" + "="*70)
    print("PLOT SAVED")
    print("="*70)
    print(f"Location: {save_path}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Progress: {total_steps:,} / 2,000,000 timesteps ({total_steps/2000000*100:.1f}%)")
    
    return fig, ax

if __name__ == "__main__":
    # Path to the intermediate training stats
    stats_path = os.path.join('models', 'bootstrapped_blue', 'bdqn_training_stats.h5')
    
    print("\n" + "="*70)
    print("BOOTSTRAPPED DQN - CURRENT TRAINING PROGRESS")
    print("="*70)
    print("This script plots the current training progress from intermediate saves.")
    print("The training is still running - this shows progress so far.")
    print("="*70 + "\n")
    
    # Plot the current progress
    plot_current_training(stats_path, window=20, save_dir='plots')
