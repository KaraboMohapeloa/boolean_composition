import numpy as np
import time
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
import deepdish as dd
from GridWorld import GridWorld
from bdqn_library import *

class TaskAlgebraExperiment:
    def __init__(self):
        self.T_states = [(3, 3), (3, 9), (9, 3), (9, 9)]
        self.T_states_list = [[pos, pos] for pos in self.T_states]
        
        # All 16 possible tasks (subsets of 4 goals)
        self.Tasks = [
            [],  # No goals (task 0)
            [(3, 3), (3, 9), (9, 3), (9, 9)],  # All goals (task 1)
            [(3, 3)], [(3, 9)], [(9, 3)], [(9, 9)],  # Single goals (tasks 2-5)
            [(3, 3), (3, 9)], [(3, 9), (9, 3)], [(9, 3), (9, 9)], [(3, 3), (9, 3)],  # Pairs (tasks 6-9)
            [(3, 3), (3, 9), (9, 3)], [(3, 3), (3, 9), (9, 9)],  # Triples (tasks 10-11)
            [(3, 3), (9, 3), (9, 9)], [(3, 9), (9, 3), (9, 9)],  # Triples (tasks 12-13)
            [(3, 3), (9, 9)], [(3, 9), (9, 3)]  # Diagonal pairs (tasks 14-15)
        ]
        
        # Task descriptions for visualization
        self.task_descriptions = [
            "No goals", "All goals", "Top-left", "Top-right", 
            "Bottom-left", "Bottom-right", "Top row", "Anti-diagonal",
            "Bottom row", "Left column", "All except bottom-right",
            "All except bottom-left", "All except top-right", 
            "All except top-left", "Diagonal TL-BR", "Diagonal TR-BL"
        ]
        
        # Load optimal Qs for evaluation
        self.Qs = dd.io.load('exps_data/4Goals_Optimal_Qs.h5')
        self.Qs = [{s: v for (s, v) in Q} for Q in self.Qs]
        self.EQs = dd.io.load('exps_data/4Goals_Optimal_EQs.h5')
        self.EQs = [{s: {s__: v__ for (s__, v__) in v} for (s, v) in EQ} for EQ in self.EQs]

    def run_sample_efficiency_experiment(self, num_runs=5, max_episodes=500):
        """Run the main sample efficiency experiment"""
        print("Starting sample efficiency experiment...")
        
        # Results storage
        results = {
            'samples_Q': np.zeros((num_runs, len(self.Tasks))),
            'samples_EQ': np.zeros((num_runs, len(self.Tasks))),
            'convergence_Q': np.full((num_runs, len(self.Tasks)), -1, dtype=int),
            'convergence_EQ': np.full((num_runs, len(self.Tasks)), -1, dtype=int),
            'final_performance_Q': np.zeros((num_runs, len(self.Tasks))),
            'final_performance_EQ': np.zeros((num_runs, len(self.Tasks))),
            'learning_curves_Q': np.zeros((num_runs, len(self.Tasks), max_episodes)),
            'learning_curves_EQ': np.zeros((num_runs, len(self.Tasks), max_episodes))
        }

        for run in range(num_runs):
            print(f"Run {run+1}/{num_runs}")
            task_order = np.random.permutation(len(self.Tasks))
            
            for task_idx in task_order:
                print(f"  Task {task_idx}: {self.task_descriptions[task_idx]}")
                goals = [[pos, pos] for pos in self.Tasks[task_idx]]
                env = GridWorld(goals=goals, goal_reward=1, step_reward=-0.01, T_states=self.T_states_list)

                # Standard Bootstrapped DQN
                Q_list, stats_Q = Bootstrapped_Q_learning(env, Q_optimal=self.Qs[task_idx], verbose=False)
                results['samples_Q'][run, task_idx] = stats_Q["T"]
                conv_Q = stats_Q.get("ensemble_converged_at_episode")
                results['convergence_Q'][run, task_idx] = conv_Q if conv_Q is not None else -1
                
                # Store learning curve
                R = stats_Q.get("R", [])
                L = min(len(R), max_episodes)
                if L > 0:
                    results['learning_curves_Q'][run, task_idx, :L] = R[:L]

                # Bootstrapped Goal-Oriented DQN (EVF)
                EQ_list, stats_EQ = Bootstrapped_Goal_Oriented_Q_learning(
                    env, T_states=self.T_states_list, Q_optimal=self.EQs[task_idx], verbose=False)
                results['samples_EQ'][run, task_idx] = stats_EQ["T"]
                conv_EQ = stats_EQ.get("ensemble_converged_at_episode")
                results['convergence_EQ'][run, task_idx] = conv_EQ if conv_EQ is not None else -1
                
                # Store learning curve
                R = stats_EQ.get("R", [])
                L = min(len(R), max_episodes)
                if L > 0:
                    results['learning_curves_EQ'][run, task_idx, :L] = R[:L]

        return results

    def analyze_results(self, results):
        """Analyze and plot the results"""
        print("Analyzing results...")
        
        # 1. Average samples per task type
        task_sizes = [len(task) for task in self.Tasks]
        unique_sizes = sorted(set(task_sizes))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Samples vs Task Complexity
        for size in unique_sizes:
            mask = np.array(task_sizes) == size
            if np.any(mask):
                avg_samples_Q = np.mean(results['samples_Q'][:, mask])
                avg_samples_EQ = np.mean(results['samples_EQ'][:, mask])
                std_samples_Q = np.std(results['samples_Q'][:, mask])
                std_samples_EQ = np.std(results['samples_EQ'][:, mask])
                
                axes[0,0].errorbar(size, avg_samples_Q, yerr=std_samples_Q, 
                                  fmt='o-', label='Standard DQN' if size == 0 else "", color='blue', markersize=8)
                axes[0,0].errorbar(size, avg_samples_EQ, yerr=std_samples_EQ, 
                                  fmt='s-', label='EVF DQN' if size == 0 else "", color='red', markersize=8)
        
        axes[0,0].set_xlabel('Number of Goals in Task')
        axes[0,0].set_ylabel('Average Samples to Learn')
        axes[0,0].set_title('Sample Efficiency vs Task Complexity')
        axes[0,0].legend()
        axes[0,0].grid(True)

        # Plot 2: Overall comparison
        tasks_range = range(len(self.Tasks))
        x_pos = np.arange(len(self.Tasks))
        width = 0.35
        
        axes[0,1].bar(x_pos - width/2, np.mean(results['samples_Q'], axis=0), 
                     width, alpha=0.7, label='Standard DQN', color='blue')
        axes[0,1].bar(x_pos + width/2, np.mean(results['samples_EQ'], axis=0), 
                     width, alpha=0.7, label='EVF DQN', color='red')
        axes[0,1].set_xlabel('Task ID')
        axes[0,1].set_ylabel('Average Samples')
        axes[0,1].set_title('Sample Efficiency by Task')
        axes[0,1].set_xticks(x_pos)
        axes[0,1].set_xticklabels([f'T{i}' for i in range(len(self.Tasks))])
        axes[0,1].legend()

        # Plot 3: Cumulative learning cost
        cumulative_Q = np.cumsum(np.mean(results['samples_Q'], axis=0))
        cumulative_EQ = np.cumsum(np.mean(results['samples_EQ'], axis=0))
        
        axes[1,0].plot(cumulative_Q, 'o-', label='Standard DQN', linewidth=2, markersize=6)
        axes[1,0].plot(cumulative_EQ, 's-', label='EVF DQN', linewidth=2, markersize=6)
        axes[1,0].set_xlabel('Number of Tasks Learned')
        axes[1,0].set_ylabel('Cumulative Samples')
        axes[1,0].set_title('Cumulative Learning Cost')
        axes[1,0].legend()
        axes[1,0].grid(True)

        # Plot 4: Learning curves for representative tasks
        representative_tasks = [2, 6, 1]  # Single goal, pair, all goals
        colors = ['blue', 'green', 'red']
        for i, task_idx in enumerate(representative_tasks):
            avg_curve_Q = np.mean(results['learning_curves_Q'][:, task_idx, :], axis=0)
            avg_curve_EQ = np.mean(results['learning_curves_EQ'][:, task_idx, :], axis=0)
            
            # Smooth the curves
            window = 10
            smooth_Q = np.convolve(avg_curve_Q, np.ones(window)/window, mode='valid')
            smooth_EQ = np.convolve(avg_curve_EQ, np.ones(window)/window, mode='valid')
            
            axes[1,1].plot(smooth_Q, color=colors[i], linestyle='-', 
                          label=f'Standard DQN - {self.task_descriptions[task_idx]}', alpha=0.8)
            axes[1,1].plot(smooth_EQ, color=colors[i], linestyle='--', 
                          label=f'EVF DQN - {self.task_descriptions[task_idx]}', alpha=0.8)
        
        axes[1,1].set_xlabel('Episode')
        axes[1,1].set_ylabel('Average Return')
        axes[1,1].set_title('Learning Curves (Smoothed)')
        axes[1,1].legend()
        axes[1,1].grid(True)

        plt.tight_layout()
        plt.savefig('sample_efficiency_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Statistical analysis
        print("\n=== STATISTICAL ANALYSIS ===")
        total_samples_Q = np.sum(results['samples_Q'])
        total_samples_EQ = np.sum(results['samples_EQ'])
        print(f"Total samples Standard DQN: {total_samples_Q:.0f}")
        print(f"Total samples EVF DQN: {total_samples_EQ:.0f}")
        print(f"Ratio (EVF/Standard): {total_samples_EQ/total_samples_Q:.2f}")
        
        # Per-task analysis
        print("\nPer-task analysis:")
        for i, task in enumerate(self.Tasks):
            avg_Q = np.mean(results['samples_Q'][:, i])
            avg_EQ = np.mean(results['samples_EQ'][:, i])
            if avg_Q > 0:  # Avoid division by zero
                print(f"Task {i:2d} ({self.task_descriptions[i]:20}): "
                      f"Standard={avg_Q:6.0f}, EVF={avg_EQ:6.0f}, ratio={avg_EQ/avg_Q:.2f}")

    def visualize_agent_performance(self, task_id, method='both', num_episodes=3):
        """Visualize the agent performing a specific task"""
        print(f"Visualizing task {task_id}: {self.task_descriptions[task_id]}")
        
        goals = [[pos, pos] for pos in self.Tasks[task_id]]
        env = GridWorld(goals=goals, goal_reward=1, step_reward=-0.01, T_states=self.T_states_list)
        
        if method in ['standard', 'both']:
            print("Training Standard Bootstrapped DQN...")
            Q_list, stats_Q = Bootstrapped_Q_learning(env, Q_optimal=self.Qs[task_id], verbose=False)
            self._visualize_single_method(env, Q_list, task_id, 'Standard DQN', num_episodes)
        
        if method in ['evf', 'both']:
            print("Training Bootstrapped Goal-Oriented DQN (EVF)...")
            EQ_list, stats_EQ = Bootstrapped_Goal_Oriented_Q_learning(
                env, T_states=self.T_states_list, Q_optimal=self.EQs[task_id], verbose=False)
            self._visualize_single_method(env, EQ_list, task_id, 'EVF DQN', num_episodes, is_evf=True)

    def _visualize_single_method(self, env, value_functions, task_id, method_name, num_episodes, is_evf=False):
        """Visualize a single method's performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Task {task_id}: {self.task_descriptions[task_id]} - {method_name}', fontsize=16)
        
        # Plot 1: Value function heatmap
        if is_evf:
            # For EVF, show value for the most valuable goal
            avg_value_func = self._get_evf_value_map(value_functions)
        else:
            avg_value_func = self._get_standard_value_map(value_functions)
        
        im1 = axes[0,0].imshow(avg_value_func, cmap='viridis', origin='upper')
        axes[0,0].set_title(f'{method_name} - Value Function')
        plt.colorbar(im1, ax=axes[0,0])
        
        # Mark goals
        for goal in self.Tasks[task_id]:
            axes[0,0].plot(goal[1], goal[0], 'r*', markersize=15, markeredgecolor='white')
        
        # Plot 2: Policy visualization
        policy_map = self._get_policy_map(value_functions, is_evf)
        self._plot_policy(axes[0,1], policy_map, self.Tasks[task_id])
        axes[0,1].set_title(f'{method_name} - Policy')
        
        # Plot 3: Sample trajectories
        trajectories = self._generate_trajectories(env, value_functions, num_episodes, is_evf)
        self._plot_trajectories(axes[1,0], trajectories, self.Tasks[task_id])
        axes[1,0].set_title(f'{method_name} - Sample Trajectories')
        
        # Plot 4: Learning progress (if available)
        # This would require storing learning curves during training
        axes[1,1].text(0.5, 0.5, 'Learning curves\nwould go here', 
                      ha='center', va='center', transform=axes[1,1].transAxes, fontsize=12)
        axes[1,1].set_title('Learning Progress')
        axes[1,1].set_xticks([])
        axes[1,1].set_yticks([])
        
        plt.tight_layout()
        plt.savefig(f'task_{task_id}_{method_name.replace(" ", "_")}_performance.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

    def _get_standard_value_map(self, Q_list):
        """Get average value function for standard DQN"""
        grid_size = 13  # Assuming 13x13 grid
        value_map = np.zeros((grid_size, grid_size))
        
        for Q in Q_list:
            for state, action_values in Q.items():
                if isinstance(state, tuple) and len(state) == 2:
                    i, j = state
                    value_map[i, j] = max(action_values.values())
        
        # Average over ensemble
        return value_map / len(Q_list)

    def _get_evf_value_map(self, EQ_list):
        """Get average value function for EVF DQN (max over goals)"""
        grid_size = 13
        value_map = np.zeros((grid_size, grid_size))
        
        for EQ in EQ_list:
            for state, goal_values in EQ.items():
                if isinstance(state, tuple) and len(state) == 2:
                    i, j = state
                    # Max over goals and actions
                    max_val = -np.inf
                    for goal, action_values in goal_values.items():
                        if isinstance(goal, tuple) and len(goal) == 2:
                            max_val = max(max_val, max(action_values.values()))
                    value_map[i, j] = max_val
        
        return value_map / len(EQ_list)

    def _get_policy_map(self, value_functions, is_evf):
        """Extract policy from value functions"""
        grid_size = 13
        # Actions: 0=up, 1=right, 2=down, 3=left, 4=stay
        action_arrows = ['↑', '→', '↓', '←', '•']
        policy_map = np.full((grid_size, grid_size), '', dtype=object)
        
        for i in range(grid_size):
            for j in range(grid_size):
                action_counts = np.zeros(5)  # Count votes for each action
                
                for vf in value_functions:
                    if is_evf:
                        # For EVF, we need to consider all goals and take max
                        if (i, j) in vf:
                            best_val = -np.inf
                            best_action = 4  # Default to stay
                            for goal, action_vals in vf[(i, j)].items():
                                for action, val in action_vals.items():
                                    if val > best_val:
                                        best_val = val
                                        best_action = action
                            action_counts[best_action] += 1
                    else:
                        # Standard DQN
                        if (i, j) in vf:
                            best_action = max(vf[(i, j)].items(), key=lambda x: x[1])[0]
                            action_counts[best_action] += 1
                
                # Choose action with most votes
                if np.sum(action_counts) > 0:
                    best_action = np.argmax(action_counts)
                    policy_map[i, j] = action_arrows[best_action]
        
        return policy_map

    def _plot_policy(self, ax, policy_map, goals):
        """Plot the policy as arrows"""
        grid_size = policy_map.shape[0]
        ax.imshow(np.zeros((grid_size, grid_size)), cmap='gray', vmin=0, vmax=1)
        
        for i in range(grid_size):
            for j in range(grid_size):
                if policy_map[i, j]:
                    ax.text(j, i, policy_map[i, j], ha='center', va='center', 
                           fontsize=8, fontweight='bold', color='red')
        
        # Mark goals
        for goal in goals:
            ax.plot(goal[1], goal[0], 'g*', markersize=15, markeredgecolor='white')
        
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.grid(True, alpha=0.3)

    def _generate_trajectories(self, env, value_functions, num_episodes, is_evf):
        """Generate sample trajectories using the learned policy"""
        trajectories = []
        
        for ep in range(num_episodes):
            state = env.reset()
            trajectory = [state]
            done = False
            max_steps = 50
            
            while not done and len(trajectory) < max_steps:
                # Use ensemble voting for action selection
                action_votes = np.zeros(env.action_space.n)
                
                for vf in value_functions:
                    if is_evf:
                        # For EVF, we need to handle goal-oriented policy
                        if state in vf:
                            best_val = -np.inf
                            best_action = 0
                            for goal, action_vals in vf[state].items():
                                for action, val in action_vals.items():
                                    if val > best_val:
                                        best_val = val
                                        best_action = action
                            action_votes[best_action] += 1
                    else:
                        # Standard DQN
                        if state in vf:
                            best_action = max(vf[state].items(), key=lambda x: x[1])[0]
                            action_votes[best_action] += 1
                
                if np.sum(action_votes) > 0:
                    action = np.argmax(action_votes)
                else:
                    action = env.action_space.sample()  # Random action if no votes
                
                next_state, reward, done, _ = env.step(action)
                trajectory.append(next_state)
                state = next_state
            
            trajectories.append(trajectory)
        
        return trajectories

    def _plot_trajectories(self, ax, trajectories, goals):
        """Plot the generated trajectories"""
        grid_size = 13
        ax.imshow(np.zeros((grid_size, grid_size)), cmap='gray', vmin=0, vmax=1)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, trajectory in enumerate(trajectories):
            color = colors[i % len(colors)]
            traj_array = np.array(trajectory)
            ax.plot(traj_array[:, 1], traj_array[:, 0], 'o-', color=color, 
                   linewidth=2, markersize=4, label=f'Traj {i+1}')
            
            # Mark start and end
            if len(trajectory) > 0:
                ax.plot(trajectory[0][1], trajectory[0][0], 's', color=color, 
                       markersize=8, markeredgecolor='white')
                if len(trajectory) > 1:
                    ax.plot(trajectory[-1][1], trajectory[-1][0], 'D', color=color, 
                           markersize=8, markeredgecolor='white')
        
        # Mark goals
        for goal in goals:
            ax.plot(goal[1], goal[0], 'y*', markersize=15, markeredgecolor='black')
        
        ax.legend()
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.grid(True, alpha=0.3)

    def demonstrate_zero_shot_composition(self, base_task_ids, composite_task_id):
        """Demonstrate zero-shot composition using EVFs"""
        print("Demonstrating zero-shot composition...")
        print(f"Base tasks: {[self.task_descriptions[i] for i in base_task_ids]}")
        print(f"Target composite task: {self.task_descriptions[composite_task_id]}")
        
        # This would implement the actual composition from the thesis
        # For now, we'll show a conceptual demonstration
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Zero-Shot Composition Demonstration', fontsize=16)
        
        # Plot base tasks
        for i, task_id in enumerate(base_task_ids):
            goals = [[pos, pos] for pos in self.Tasks[task_id]]
            env = GridWorld(goals=goals, goal_reward=1, step_reward=-0.01, T_states=self.T_states_list)
            
            # Learn EVF for base task
            EQ_list, _ = Bootstrapped_Goal_Oriented_Q_learning(
                env, T_states=self.T_states_list, Q_optimal=self.EQs[task_id], verbose=False)
            
            # Visualize base task
            value_map = self._get_evf_value_map(EQ_list)
            axes[0, i].imshow(value_map, cmap='viridis', origin='upper')
            axes[0, i].set_title(f'Base Task {i+1}: {self.task_descriptions[task_id]}')
            
            for goal in self.Tasks[task_id]:
                axes[0, i].plot(goal[1], goal[0], 'r*', markersize=10)
        
        # Plot composite task (would be computed via Boolean operations in real implementation)
        composite_goals = [[pos, pos] for pos in self.Tasks[composite_task_id]]
        env_composite = GridWorld(goals=composite_goals, goal_reward=1, step_reward=-0.01, T_states=self.T_states_list)
        
        # Learn composite task directly for comparison
        EQ_composite, _ = Bootstrapped_Goal_Oriented_Q_learning(
            env_composite, T_states=self.T_states_list, Q_optimal=self.EQs[composite_task_id], verbose=False)
        
        value_map_composite = self._get_evf_value_map(EQ_composite)
        axes[0, 2].imshow(value_map_composite, cmap='viridis', origin='upper')
        axes[0, 2].set_title(f'Direct Learning: {self.task_descriptions[composite_task_id]}')
        
        for goal in self.Tasks[composite_task_id]:
            axes[0, 2].plot(goal[1], goal[0], 'r*', markersize=10)
        
        # Show composition concept
        axes[1, 0].text(0.5, 0.6, f'Base Task 1\n{self.task_descriptions[base_task_ids[0]]}', 
                       ha='center', va='center', fontsize=12, transform=axes[1, 0].transAxes)
        axes[1, 0].text(0.5, 0.4, 'AND/OR/NOT', ha='center', va='center', 
                       fontsize=10, style='italic', transform=axes[1, 0].transAxes)
        
        axes[1, 1].text(0.5, 0.6, f'Base Task 2\n{self.task_descriptions[base_task_ids[1]]}', 
                       ha='center', va='center', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.5, 0.4, '→', ha='center', va='center', 
                       fontsize=16, transform=axes[1, 1].transAxes)
        
        axes[1, 2].text(0.5, 0.6, f'Composite Task\n{self.task_descriptions[composite_task_id]}', 
                       ha='center', va='center', fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.5, 0.4, 'Zero-Shot!', ha='center', va='center', 
                       fontsize=14, fontweight='bold', color='green', transform=axes[1, 2].transAxes)
        
        for ax in axes[1, :]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
        
        plt.tight_layout()
        plt.savefig('zero_shot_composition_demo.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to run experiments and visualizations"""
    experiment = TaskAlgebraExperiment()
    
    # Run sample efficiency experiment
    print("=" * 60)
    print("RUNNING SAMPLE EFFICIENCY EXPERIMENT")
    print("=" * 60)
    results = experiment.run_sample_efficiency_experiment(num_runs=3)  # Reduced for demo
    experiment.analyze_results(results)
    
    # Save results
    dd.io.save('exps_data/bdqn/comprehensive_results.h5', results)
    
    # Visualize specific tasks
    print("\n" + "=" * 60)
    print("VISUALIZING AGENT PERFORMANCE")
    print("=" * 60)
    
    # Visualize different types of tasks
    tasks_to_visualize = [2, 6, 1]  # Single goal, pair, all goals
    
    for task_id in tasks_to_visualize:
        experiment.visualize_agent_performance(task_id, method='both', num_episodes=2)
    
    # Demonstrate zero-shot composition
    print("\n" + "=" * 60)
    print("DEMONSTRATING ZERO-SHOT COMPOSITION")
    print("=" * 60)
    
    # Example: Compose single goals to get a pair task
    experiment.demonstrate_zero_shot_composition(
        base_task_ids=[2, 3],  # Top-left and top-right
        composite_task_id=6     # Top row (both top goals)
    )

if __name__ == "__main__":
    main()