import numpy as np
import torch
import warnings

# Suppress CUDA compatibility warning if GPU not available
warnings.filterwarnings('ignore', category=UserWarning, module='torch.cuda')

from dqn import Agent as AgentVanilla, DQN as DQNVanilla, FloatTensor as FloatTensorVanilla
from bdqn import BootstrappedAgent, BootstrappedDQN
from gym_repoman.envs import CollectEnv
from wrappers import WarpFrame

import deepdish as dd

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

# Helper functions for each agent type
def train_vanilla(path, env):
    """Train vanilla DQN with epsilon-greedy exploration"""
    agent = AgentVanilla(env, path=path)
    agent.train()
    return agent

def train_bootstrapped(path, env, n_heads=10, exploration_strategy='thompson'):
    """
    Train Bootstrapped DQN with Thompson sampling exploration.
    
    Parameters:
    -----------
    path: str
        Path to save models and stats
    env: gym.Env
        Training environment
    n_heads: int
        Number of bootstrap heads (default: 10)
    exploration_strategy: str
        'thompson': Thompson sampling (recommended for sample efficiency)
        'ucb': Upper confidence bound
        'vote': Ensemble voting
        'mean': Ensemble mean (no exploration bonus)
    """
    agent = BootstrappedAgent(
        env=env, 
        path=path,
        n_heads=n_heads,
        mask_prob=0.8,              # Per-step masking probability
        init_q_range=0.2,           # Optimistic initialization
        warmup_steps=200000,          # Random exploration phase
        exploration_strategy=exploration_strategy,
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


# --- Experiment: train DQN for multiple goal conditions and collect stats ---

# Define your tasks as different goal conditions
# Task 1: Collect blue objects (any shape)
Tasks = [
    ('blue', '', lambda x: x.colour == 'blue'),      # Collect blue objects
]

# --- Run experiment for both vanilla DQN and Bootstrapped DQN ---
num_runs = 1
data_stats_vanilla = np.empty((num_runs, len(Tasks)), dtype=object)
data_stats_bootstrapped = np.empty((num_runs, len(Tasks)), dtype=object)

print("="*70)
print("VANILLA DQN vs BOOTSTRAPPED DQN COMPARISON EXPERIMENT")
print("="*70)
print(f"Number of runs: {num_runs}")
print(f"Number of tasks: {len(Tasks)}")
print(f"Tasks: {[(colour + shape) for colour, shape, _ in Tasks]}")
print(f"Vanilla DQN: Epsilon-greedy exploration")
print(f"Bootstrapped DQN: Thompson sampling with 10 heads")
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
        
        base_path_vanilla = f'./models/vanilla_{task_name}/'
        base_path_bootstrapped = f'./models/bootstrapped_{task_name}/'
        env = WarpFrame(CollectEnv(start_positions=start_positions, goal_condition=goal_condition))

        # Vanilla DQN (epsilon-greedy baseline)
        print("--- Training Vanilla DQN (Epsilon-Greedy) ---")
        try:
            agent_vanilla = train_vanilla(base_path_vanilla, env)
            data_stats_vanilla[i, j] = agent_vanilla.training_stats
            torch.save(agent_vanilla.q_func.state_dict(), base_path_vanilla + 'model.dqn')
            # Save vanilla stats after training
            dd.io.save('exps_data/vanilla/prime_experiment_stats.h5', data_stats_vanilla)
            
            vanilla_final_reward = np.mean(agent_vanilla.training_stats['R'][-100:]) if len(agent_vanilla.training_stats['R']) > 100 else 0
            print(f"Vanilla DQN completed. Final 100-ep reward: {vanilla_final_reward:.2f}")
        except Exception as e:
            print(f"Vanilla DQN training failed: {e}")
            data_stats_vanilla[i, j] = {"R": [0], "T": 0, "error": str(e)}

        # Bootstrapped DQN (Thompson sampling)
        print("\n--- Training Bootstrapped DQN (Thompson Sampling) ---")
        try:
            agent_bootstrapped = train_bootstrapped(base_path_bootstrapped, env, 
                                                    n_heads=10, 
                                                    exploration_strategy='thompson')
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
        
        # Comparison summary
        if ('error' not in data_stats_vanilla[i, j] and 
            'error' not in data_stats_bootstrapped[i, j]):
            print(f"\n{'='*70}")
            print(f"TASK {task_name} SUMMARY")
            print(f"{'='*70}")
            print(f"Vanilla DQN     - Final 100-ep avg: {vanilla_final_reward:.2f}")
            print(f"Bootstrapped DQN - Final 100-ep avg: {bootstrapped_final_reward:.2f}")
            
            improvement = bootstrapped_final_reward - vanilla_final_reward
            improvement_pct = (improvement / abs(vanilla_final_reward)) * 100 if vanilla_final_reward != 0 else 0
            print(f"BDQN improvement: {improvement:+.2f} ({improvement_pct:+.1f}%)")
            print(f"{'='*70}\n")

print("\n" + "="*70)
print("EXPERIMENT COMPLETE")
print("="*70)
print("Results saved to:")
print("  - exps_data/vanilla/prime_experiment_stats.h5")
print("  - exps_data/bootstrapped/prime_experiment_stats.h5")
print("\nVanilla DQN: Epsilon-greedy exploration")
print("Bootstrapped DQN: Thompson sampling with uncertainty-driven exploration")
print("Expected: BDQN shows better sample efficiency (fewer episodes to converge)")
print("\nTo analyze results, run:")
print("  from plots import analyze_sample_efficiency, plot_learning_curves")
print("  analyze_sample_efficiency()  # Detailed 4-panel analysis")
print("  plot_learning_curves()        # Simple learning curve comparison")
print("="*70)


if __name__ == "__main__":
    import os
    
    # Create necessary directories
    os.makedirs('exps_data/vanilla', exist_ok=True)
    os.makedirs('exps_data/bootstrapped', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

