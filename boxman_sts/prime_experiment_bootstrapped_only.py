import numpy as np
import torch
import warnings

# Suppress CUDA compatibility warning if GPU not available
warnings.filterwarnings('ignore', category=UserWarning, module='torch.cuda')

from bdqn import BootstrappedAgent, BootstrappedDQN
from gym_repoman.envs import CollectEnv
from wrappers import WarpFrame

import deepdish as dd
import os

# Create necessary directories
os.makedirs('exps_data/vanilla', exist_ok=True)
os.makedirs('exps_data/bootstrapped', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('models/bootstrapped_blue', exist_ok=True)

# Print device info
print("="*70)
print("DEVICE INFORMATION")
print("="*70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Training on: GPU")
else:
    print(f"Training on: CPU (this will be MUCH slower)")
    print(f"Recommendation: Install PyTorch with CUDA support")
    print(f"  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
print("="*70 + "\n")

def train_bootstrapped(path, env, n_heads=10):
    """
    Train Bootstrapped DQN with Thompson sampling exploration (original BDQN strategy).
    
    The exploration strategy is Thompson Sampling as per the original Bootstrapped DQN paper
    (Osband et al., 2016). This is adapted for goal-oriented learning with GPI:
    
    EXPLORATION MECHANISM:
    1. Thompson Sampling: Randomly select one head uniformly at random
    2. Goal-Oriented GPI: For selected head, evaluate all goals and take max Q-value
    3. Act greedily with respect to selected head's GPI policy
    
    INITIALIZATION (CRITICAL):
    - All heads start with SAME weights (shared prior)
    - Diversity emerges from bootstrap masking during training
    - NOT from different random initializations
    
    BOOTSTRAP MASKING:
    - Per-step masking: Each transition included in each head with prob=0.8
    - This creates different "views" of the data for each head
    - Leads to diverse Q-function estimates (uncertainty)
    
    This maintains the original BDQN exploration mechanism while incorporating
    Generalized Policy Improvement for multi-goal learning.
    
    Parameters:
    -----------
    path: str
        Path to save models and stats
    env: gym.Env
        Training environment
    n_heads: int
        Number of bootstrap heads (default: 10, as in original paper)
    """
    agent = BootstrappedAgent( 
        env=env, 
        path=path,
        n_heads=n_heads,
        mask_prob=0.8,              # Per-step masking probability (original: 0.5, tuned: 0.8)
        warmup_steps=0,             # No warmup - start Thompson sampling immediately
        exploration_strategy='thompson',  # ORIGINAL BDQN: Thompson sampling with GPI
        max_timesteps=2000000,
        batch_size=128,
        learning_starts=10000,
        learning_rate=1e-4,
        gamma=0.99
    )
    agent.train()
    return agent

start_positions = {'crate_beige': (3, 4),
                   'player': (6, 3),
                   'circle_purple': (7, 7),
                   'circle_beige': (1, 7),
                   'crate_blue': (1, 1),
                   'crate_purple': (8, 1),
                   'circle_blue': (1, 8)}


# --- Experiment: train Bootstrapped DQN for multiple goal conditions ---

# Define your tasks as different goal conditions
Tasks = [
    ('blue', '', lambda x: x.colour == 'blue'),      # Collect blue objects
]

# --- Run experiment for Bootstrapped DQN only ---
num_runs = 1
data_stats_bootstrapped = np.empty((num_runs, len(Tasks)), dtype=object)

print("="*70)
print("BOOTSTRAPPED DQN EXPERIMENT (STANDALONE)")
print("="*70)
print(f"Number of runs: {num_runs}")
print(f"Number of tasks: {len(Tasks)}")
print(f"Tasks: {[(colour + shape) for colour, shape, _ in Tasks]}")
print(f"\nExploration: Thompson Sampling (original BDQN) + GPI (goal-oriented)")
print(f"  - Thompson Sampling: Randomly select one head, act greedily w.r.t. it")
print(f"  - GPI: For selected head, max over all goal Q-values")
print(f"\nConfiguration (matching original BDQN paper):")
print(f"  - Heads: 10 (all start with SAME initialization)")
print(f"  - Mask probability: 0.8 (per-step masking)")
print(f"  - Warmup: DISABLED (Thompson sampling from start)")
print(f"  - Diversity source: Bootstrap masking (NOT random init)")
print(f"\nResults saved separately from vanilla DQN")
print("="*70)
print(f"Results saved separately from vanilla DQN")
print("="*70)

for i in range(num_runs):
    print(f"\n{'='*70}")
    print(f"RUN {i+1}/{num_runs}")
    print(f"{'='*70}\n")
    
    for j, (colour, shape, goal_condition) in enumerate(Tasks):
        task_name = colour + shape
        print(f"\n{'*'*70}")
        print(f"Task {j+1}/{len(Tasks)}: {task_name}")
        print(f"{'*'*70}\n")
        
        base_path_bootstrapped = f'./models/bootstrapped_{task_name}/'
        env = WarpFrame(CollectEnv(start_positions=start_positions, goal_condition=goal_condition))

        # Bootstrapped DQN (Thompson sampling)
        print("--- Training Bootstrapped DQN (Thompson Sampling + GPI) ---")
        try:
            agent_bootstrapped = train_bootstrapped(base_path_bootstrapped, env, n_heads=10)
            data_stats_bootstrapped[i, j] = agent_bootstrapped.training_stats
            torch.save(agent_bootstrapped.q_func.state_dict(), base_path_bootstrapped + 'model_bdqn.pth')
            # Save bootstrapped stats after training
            dd.io.save('exps_data/bootstrapped/prime_experiment_stats.h5', data_stats_bootstrapped)
            
            bootstrapped_final_reward = np.mean(agent_bootstrapped.training_stats['R'][-100:]) if len(agent_bootstrapped.training_stats['R']) > 100 else 0
            print(f"Bootstrapped DQN completed. Final 100-ep reward: {bootstrapped_final_reward:.2f}")
            print(f"Head update counts: {agent_bootstrapped.update_counts}")

        except Exception as e:
            print(f"Bootstrapped DQN training failed: {e}")
            data_stats_bootstrapped[i, j] = {"R": [0], "T": 0, "error": str(e)}
        
        # Task summary
        if 'error' not in data_stats_bootstrapped[i, j]:
            print(f"\n{'='*70}")
            print(f"TASK {task_name} SUMMARY")
            print(f"{'='*70}")
            print(f"Bootstrapped DQN - Final 100-ep avg: {bootstrapped_final_reward:.2f}")
            print(f"Total episodes: {len(agent_bootstrapped.training_stats['R'])}")
            print(f"Total timesteps: {agent_bootstrapped.training_stats.get('T', 'N/A')}")
            print(f"{'='*70}\n")

print("\n" + "="*70)
print("EXPERIMENT COMPLETE")
print("="*70)
print("Results saved to:")
print("  - exps_data/bootstrapped/prime_experiment_stats.h5")
print("\nBootstrapped DQN: Thompson sampling with uncertainty-driven exploration")
print("\nVanilla DQN results (if already run) are preserved in:")
print("  - exps_data/vanilla/prime_experiment_stats.h5")
print("\nTo analyze results comparing both methods, run:")
print("  from analyze_results import plot_learning_curves, analyze_sample_efficiency")
print("  plot_learning_curves()        # Simple learning curve comparison")
print("  analyze_sample_efficiency()   # Detailed 4-panel analysis")
print("="*70)


if __name__ == "__main__":
    import os
    
    # Create necessary directories
    os.makedirs('exps_data/vanilla', exist_ok=True)
    os.makedirs('exps_data/bootstrapped', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
