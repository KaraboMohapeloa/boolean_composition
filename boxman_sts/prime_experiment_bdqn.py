import numpy as np
import torch
import os

from bdqn import BootstrappedGoalConditionedAgent, FloatTensor
from dqn import Agent as AgentDQN
from gym_repoman.envs import CollectEnv
from wrappers import WarpFrame

import deepdish as dd

print(torch.cuda.is_available())
print(torch.version.cuda)

# Helper functions for each agent type
def train_dqn(path, env):
    agent = AgentDQN(env, path=path)
    agent.train()
    return agent

def train_bootstrapped(path, env, num_heads=5):
    agent = BootstrappedGoalConditionedAgent(
        env=env, 
        path=path,
        num_heads=num_heads,
        batch_size=32,
        learning_starts=1000
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

# --- Experiment: train DQN and BDQN for multiple goal conditions ---

# Define your tasks as different goal conditions
Tasks = [
    ('purple', '', lambda x: x.colour == 'purple'),
    ('blue', '', lambda x: x.colour == 'blue'),
    ('', 'circle', lambda x: x.shape == 'circle'),
    # Add more (colour, shape, condition) tuples as needed
]

# --- Run experiment for both DQN and Bootstrapped DQN ---
num_runs = 1
num_heads = 5  # Number of bootstrap heads

# Data structures for both agent types
data_stats_dqn = np.empty((num_runs, len(Tasks)), dtype=object)
data_stats_bdqn = np.empty((num_runs, len(Tasks)), dtype=object)

for i in range(num_runs):
    print("run:", i)
    for j, (colour, shape, goal_condition) in enumerate(Tasks):
        print("Task:", j, f"({colour}{shape})")
        name = colour + shape if colour + shape else "default"
        
        # Create paths for both agents
        base_path_dqn = f'./models/dqn_{name}/'
        base_path_bdqn = f'./models/bdqn_{name}/'
        
        # Create environment with specific goal condition
        env = WarpFrame(CollectEnv(start_positions=start_positions, goal_condition=goal_condition))

        # Train DQN
        print("Training DQN...")
        try:
            agent_dqn = train_dqn(base_path_dqn, env)
            data_stats_dqn[i, j] = agent_dqn.training_stats
            torch.save(agent_dqn.q_func.state_dict(), base_path_dqn + 'model.dqn')
            print("DQN training completed successfully")
        except Exception as e:
            print(f"DQN training failed: {e}")
            data_stats_dqn[i, j] = {"R": [0], "T": 0, "error": str(e)}

        # Train Bootstrapped DQN
        print("Training Bootstrapped DQN...")
        try:
            agent_bdqn = train_bootstrapped(base_path_bdqn, env, num_heads=num_heads)
            data_stats_bdqn[i, j] = agent_bdqn.training_stats
            torch.save(agent_bdqn.q_func.state_dict(), base_path_bdqn + 'bootstrapped_model.dqn')
            print("Bootstrapped DQN training completed successfully")
        except Exception as e:
            print(f"Bootstrapped DQN training failed: {e}")
            data_stats_bdqn[i, j] = {"R": [0], "T": 0, "error": str(e)}
        
        print(f"Completed training for task {name}")

# Save all stats
print("Saving experiment statistics...")
os.makedirs('exps_data/dqn/', exist_ok=True)
os.makedirs('exps_data/bdqn/', exist_ok=True)
dd.io.save('exps_data/dqn/comparison_stats.h5', data_stats_dqn)
dd.io.save('exps_data/bdqn/comparison_stats.h5', data_stats_bdqn)

# Analysis functions
def analyze_comparison_results():
    """Analyze and compare results from DQN and Bootstrapped DQN"""
    try:
        dqn_stats = dd.io.load('exps_data/dqn/comparison_stats.h5')
        bdqn_stats = dd.io.load('exps_data/bdqn/comparison_stats.h5')
        
        print("\n" + "="*60)
        print("DQN vs Bootstrapped DQN - COMPARISON RESULTS")
        print("="*60)
        
        for run_idx in range(dqn_stats.shape[0]):
            for task_idx in range(dqn_stats.shape[1]):
                dqn_task_stats = dqn_stats[run_idx, task_idx]
                bdqn_task_stats = bdqn_stats[run_idx, task_idx]
                
                task_name = Tasks[task_idx][0] + Tasks[task_idx][1] if Tasks[task_idx][0] + Tasks[task_idx][1] else "default"
                print(f"\nTask: {task_name} (Run {run_idx})")
                print("-" * 40)
                
                # Check for errors first
                if 'error' in dqn_task_stats:
                    print(f"DQN Error: {dqn_task_stats['error']}")
                if 'error' in bdqn_task_stats:
                    print(f"Bootstrapped DQN Error: {bdqn_task_stats['error']}")
                    continue
                
                # DQN Results
                if dqn_task_stats and 'R' in dqn_task_stats and dqn_task_stats['R']:
                    dqn_rewards = dqn_task_stats['R']
                    dqn_final_100 = np.mean(dqn_rewards[-101:-1]) if len(dqn_rewards) > 100 else np.mean(dqn_rewards)
                    dqn_total_episodes = len(dqn_rewards)
                    dqn_total_steps = dqn_task_stats.get('T', 0)
                    print(f"DQN:")
                    print(f"  Final 100-episode reward: {dqn_final_100:.2f}")
                    print(f"  Total episodes: {dqn_total_episodes}")
                    print(f"  Total steps: {dqn_total_steps}")
                
                # Bootstrapped DQN Results
                if bdqn_task_stats and 'R' in bdqn_task_stats and bdqn_task_stats['R']:
                    bdqn_rewards = bdqn_task_stats['R']
                    bdqn_final_100 = np.mean(bdqn_rewards[-101:-1]) if len(bdqn_rewards) > 100 else np.mean(bdqn_rewards)
                    bdqn_total_episodes = len(bdqn_rewards)
                    bdqn_total_steps = bdqn_task_stats.get('T', 0)
                    
                    print(f"Bootstrapped DQN:")
                    print(f"  Final 100-episode reward: {bdqn_final_100:.2f}")
                    print(f"  Total episodes: {bdqn_total_episodes}")
                    print(f"  Total steps: {bdqn_total_steps}")
                    
                    # Bootstrap-specific metrics
                    if 'head_losses' in bdqn_task_stats:
                        head_losses = bdqn_task_stats['head_losses']
                        if head_losses and len(head_losses) > 0:
                            avg_final_losses = [np.mean(losses[-100:]) if losses and len(losses) > 0 else 0 for losses in head_losses]
                            loss_variance = np.var(avg_final_losses)
                            print(f"  Head loss variance: {loss_variance:.6f}")
                            print(f"  Head losses: {[f'{l:.4f}' for l in avg_final_losses]}")
                
                # Comparison
                if (dqn_task_stats and 'R' in dqn_task_stats and dqn_task_stats['R'] and 
                    bdqn_task_stats and 'R' in bdqn_task_stats and bdqn_task_stats['R']):
                    improvement = bdqn_final_100 - dqn_final_100
                    improvement_pct = (improvement / abs(dqn_final_100)) * 100 if dqn_final_100 != 0 else 0
                    print(f"Comparison:")
                    print(f"  BDQN improvement: {improvement:+.2f} ({improvement_pct:+.1f}%)")
                    
    except FileNotFoundError as e:
        print(f"Results file not found: {e}")
    except Exception as e:
        print(f"Error analyzing results: {e}")

def evaluate_all_models():
    """Evaluate all trained models for comparison"""
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    for colour, shape, goal_condition in Tasks:
        name = colour + shape if colour + shape else "default"
        
        # DQN Evaluation
        dqn_path = f'./models/dqn_{name}/model.dqn'
        if os.path.exists(dqn_path):
            try:
                env = WarpFrame(CollectEnv(start_positions=start_positions, goal_condition=goal_condition))
                agent_dqn = AgentDQN(env, path=f'./models/dqn_{name}/')
                agent_dqn.q_func.load_state_dict(torch.load(dqn_path))
                
                print(f"\nDQN Evaluation - Task {name}:")
                total_reward = 0
                num_eval_episodes = 5
                success_count = 0
                
                for ep in range(num_eval_episodes):
                    obs = env.reset()
                    episode_reward = 0
                    done = False
                    while not done:
                        action = agent_dqn.select_action(obs)
                        obs, reward, done, _ = env.step(int(action[0][0]))
                        episode_reward += reward
                    total_reward += episode_reward
                    if episode_reward > 0:  # Assuming positive reward means success
                        success_count += 1
                
                avg_reward = total_reward / num_eval_episodes
                success_rate = success_count / num_eval_episodes
                print(f"  Average reward: {avg_reward:.2f}")
                print(f"  Success rate: {success_rate:.2f}")
            except Exception as e:
                print(f"  DQN evaluation failed: {e}")
        
        # Bootstrapped DQN Evaluation
        bdqn_path = f'./models/bdqn_{name}/bootstrapped_model.dqn'
        if os.path.exists(bdqn_path):
            try:
                env = WarpFrame(CollectEnv(start_positions=start_positions, goal_condition=goal_condition))
                agent_bdqn = BootstrappedGoalConditionedAgent(
                    env=env,
                    path=f'./models/bdqn_{name}/',
                    num_heads=num_heads
                )
                agent_bdqn.q_func.load_state_dict(torch.load(bdqn_path))
                
                print(f"Bootstrapped DQN Evaluation - Task {name}:")
                avg_reward, success_rate = agent_bdqn.evaluate(num_episodes=5, render=False)
                print(f"  Average reward: {avg_reward:.2f}")
                print(f"  Success rate: {success_rate:.2f}")
            except Exception as e:
                print(f"  Bootstrapped DQN evaluation failed: {e}")

def plot_comparison_results():
    """Create comparison plots (basic version - extend as needed)"""
    try:
        import matplotlib.pyplot as plt
        
        dqn_stats = dd.io.load('exps_data/dqn/comparison_stats.h5')
        bdqn_stats = dd.io.load('exps_data/bdqn/comparison_stats.h5')
        
        # Simple reward comparison plot
        plt.figure(figsize=(12, 8))
        
        for task_idx in range(min(len(Tasks), 4)):  # Limit to 4 subplots
            task_name = Tasks[task_idx][0] + Tasks[task_idx][1] if Tasks[task_idx][0] + Tasks[task_idx][1] else "default"
            
            # DQN rewards
            dqn_task_stats = dqn_stats[0, task_idx]
            # BDQN rewards
            bdqn_task_stats = bdqn_stats[0, task_idx]
            
            plt.subplot(2, 2, task_idx + 1)
            
            if 'R' in dqn_task_stats and dqn_task_stats['R']:
                dqn_rewards = dqn_task_stats['R']
                plt.plot(dqn_rewards, label='DQN', alpha=0.7)
            
            if 'R' in bdqn_task_stats and bdqn_task_stats['R']:
                bdqn_rewards = bdqn_task_stats['R']
                plt.plot(bdqn_rewards, label='Bootstrapped DQN', alpha=0.7)
            
            plt.title(f'Task: {task_name}')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comparison_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Comparison plot saved as 'comparison_results.png'")
        
    except ImportError:
        print("Matplotlib not available for plotting")
    except Exception as e:
        print(f"Could not create plots: {e}")

# Create necessary directories
def setup_directories():
    """Create all necessary directories for the experiment"""
    directories = [
        './models/',
        './exps_data/dqn/',
        './exps_data/bdqn/'
    ]
    
    # Add task-specific directories
    for colour, shape, _ in Tasks:
        name = colour + shape if colour + shape else "default"
        directories.extend([
            f'./models/dqn_{name}/',
            f'./models/bdqn_{name}/'
        ])
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    # Setup directories
    print("Setting up directories...")
    setup_directories()
    
    # Run the main comparison experiment
    print("Starting DQN vs Bootstrapped DQN Comparison Experiment...")
    
    # The main training loop above will execute here
    
    # Analyze results
    print("\nAnalyzing comparison results...")
    analyze_comparison_results()
    
    # Evaluate models
    print("\nEvaluating trained models...")
    evaluate_all_models()
    
    # Create plots
    print("\nCreating comparison plots...")
    plot_comparison_results()
    
    print("\nExperiment completed!")