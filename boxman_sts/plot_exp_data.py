import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm, rcParams
from matplotlib import rc
import os
import pandas as pd
import seaborn as sns
import deepdish as dd



def plot_side_by_side(vanilla_path, softmax_path, window=20):
    """
    Plot vanilla and softmax reward curves side by side with the same axis calibration and save the combined plot.
    """
    if not os.path.exists(vanilla_path):
        print(f"File not found: {vanilla_path}")
        return
    if not os.path.exists(softmax_path):
        print(f"File not found: {softmax_path}")
        return
    data_stats_vanilla = dd.io.load(vanilla_path)
    data_stats_softmax = dd.io.load(softmax_path)
    rewards_vanilla = data_stats_vanilla[0, 0]['R']
    rewards_softmax = data_stats_softmax[0, 0]['R']
    steps_vanilla = np.arange(1, len(rewards_vanilla) + 1)
    steps_softmax = np.arange(1, len(rewards_softmax) + 1)
    rewards_vanilla_smooth = pd.Series(rewards_vanilla).rolling(window, min_periods=1).mean()
    rewards_softmax_smooth = pd.Series(rewards_softmax).rolling(window, min_periods=1).mean()
    # Determine global y-limits
    min_y = min(np.min(rewards_vanilla), np.min(rewards_softmax))
    max_y = max(np.max(rewards_vanilla), np.max(rewards_softmax))
    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    # Vanilla subplot
    axes[0].plot(steps_vanilla, rewards_vanilla, alpha=0.2, label='Reward (raw)', color='tab:blue')
    axes[0].plot(steps_vanilla, rewards_vanilla_smooth, label=f'Smoothed (w={window})', linewidth=2, color='tab:blue')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Vanilla DQN')
    if len(rewards_vanilla) >= 100:
        avg_last_100_vanilla = np.mean(rewards_vanilla[-100:])
        axes[0].axhline(avg_last_100_vanilla, color='blue', linestyle='--', label=f'Avg last 100: {avg_last_100_vanilla:.2f}')
        axes[0].text(0, avg_last_100_vanilla, f'{avg_last_100_vanilla:.2f}', color='blue', va='bottom', ha='left', fontsize=10, backgroundcolor='white')
    axes[0].set_ylim(min_y, max_y)
    axes[0].legend()
    # Softmax subplot
    axes[1].plot(steps_softmax, rewards_softmax, alpha=0.2, label='Reward (raw)', color='tab:orange')
    axes[1].plot(steps_softmax, rewards_softmax_smooth, label=f'Smoothed (w={window})', linewidth=2, color='tab:orange')
    axes[1].set_xlabel('Episode')
    axes[1].set_title('Softmax DQN')
    if len(rewards_softmax) >= 100:
        avg_last_100_softmax = np.mean(rewards_softmax[-100:])
        axes[1].axhline(avg_last_100_softmax, color='orange', linestyle='--', label=f'Avg last 100: {avg_last_100_softmax:.2f}')
        axes[1].text(0, avg_last_100_softmax, f'{avg_last_100_softmax:.2f}', color='orange', va='bottom', ha='left', fontsize=10, backgroundcolor='white')
    axes[1].set_ylim(min_y, max_y)
    axes[1].legend()
    plt.tight_layout()
    # Show plot interactively
    plt.show()
    # After window is closed, save the plot
    save_dir = os.path.join('plots', 'prime_experiment')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'side_by_side.png')
    fig.savefig(save_path, bbox_inches='tight')
    import datetime
    print(f"Plot saved to: {save_path}")
    print(f"Plot generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    vanilla_path = os.path.join('exps_data', 'vanilla', 'prime_experiment_stats.h5')
    softmax_path = os.path.join('exps_data', 'softmax', 'prime_experiment_stats.h5')
    plot_side_by_side(vanilla_path, softmax_path, window=20)