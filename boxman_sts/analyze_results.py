"""
Analysis script for comparing Vanilla DQN vs Bootstrapped DQN results.
Run this after prime_experiment.py completes.
"""

import numpy as np
import deepdish as dd
import matplotlib.pyplot as plt
import os

# Same tasks as in prime_experiment.py
Tasks = [
    ('blue', '', lambda x: x.colour == 'blue'),
]


def analyze_sample_efficiency():
    """
    Analyze sample efficiency: compare number of episodes/timesteps to reach threshold performance
    """
    
    if not os.path.exists('exps_data/vanilla/prime_experiment_stats.h5'):
        print("No vanilla data found. Run prime_experiment.py first.")
        return
    
    if not os.path.exists('exps_data/bootstrapped/prime_experiment_stats.h5'):
        print("No bootstrapped data found. Run prime_experiment.py first.")
        return
    
    print("\n" + "="*70)
    print("SAMPLE EFFICIENCY ANALYSIS: VANILLA DQN vs BOOTSTRAPPED DQN")
    print("="*70)
    
    try:
        vanilla_stats = dd.io.load('exps_data/vanilla/prime_experiment_stats.h5')
        bootstrapped_stats = dd.io.load('exps_data/bootstrapped/prime_experiment_stats.h5')
        
        for run_idx in range(vanilla_stats.shape[0]):
            for task_idx in range(vanilla_stats.shape[1]):
                task_name = Tasks[task_idx][0] + Tasks[task_idx][1]
                
                vanilla_data = vanilla_stats[run_idx, task_idx]
                bootstrapped_data = bootstrapped_stats[run_idx, task_idx]
                
                print(f"\n{'='*70}")
                print(f"TASK: {task_name.upper()} (Run {run_idx + 1})")
                print(f"{'='*70}")
                
                # Skip if there were errors
                if 'error' in vanilla_data or 'error' in bootstrapped_data:
                    if 'error' in vanilla_data:
                        print(f"❌ Vanilla error: {vanilla_data['error']}")
                    if 'error' in bootstrapped_data:
                        print(f"❌ Bootstrapped error: {bootstrapped_data['error']}")
                    continue
                
                # Vanilla DQN analysis
                vanilla_rewards = vanilla_data['R']
                vanilla_episodes = len(vanilla_rewards)
                vanilla_timesteps = vanilla_data.get('T', 0)
                vanilla_final_100 = np.mean(vanilla_rewards[-100:])
                
                # Bootstrapped DQN analysis
                bootstrapped_rewards = bootstrapped_data['R']
                bootstrapped_episodes = len(bootstrapped_rewards)
                bootstrapped_timesteps = bootstrapped_data.get('T', 0)
                bootstrapped_final_100 = np.mean(bootstrapped_rewards[-100:])
                
                print("\n📊 FINAL PERFORMANCE:")
                print(f"  Vanilla DQN:      {vanilla_final_100:7.2f} (avg reward over last 100 episodes)")
                print(f"  Bootstrapped DQN: {bootstrapped_final_100:7.2f} (avg reward over last 100 episodes)")
                improvement = bootstrapped_final_100 - vanilla_final_100
                improvement_pct = (improvement / abs(vanilla_final_100)) * 100 if vanilla_final_100!= 0 else 0
                print(f"  Improvement:      {improvement:+7.2f} ({improvement_pct:+.1f}%)")
                
                print("\n⏱️  TOTAL TRAINING:")
                print(f"  Vanilla DQN:      {vanilla_episodes:6d} episodes, {vanilla_timesteps:8d} timesteps")
                print(f"  Bootstrapped DQN: {bootstrapped_episodes:6d} episodes, {bootstrapped_timesteps:8d} timesteps")
                
                # Smooth rewards for threshold analysis
                window = 100
                vanilla_smooth = np.convolve(vanilla_rewards, np.ones(window)/window, mode='valid')
                bootstrapped_smooth = np.convolve(bootstrapped_rewards, np.ones(window)/window, mode='valid')
                
                print("\n🎯 SAMPLE EFFICIENCY (Episodes to reach performance thresholds):")
                
                for threshold_pct in [50, 75, 90]:
                    # Use the better of the two final performances as reference
                    reference_reward = max(vanilla_final_100, bootstrapped_final_100)
                    threshold = reference_reward * (threshold_pct / 100.0)
                    
                    # Find first episode where smoothed reward exceeds threshold
                    v_ep = next((i for i, r in enumerate(vanilla_smooth) if r >= threshold), len(vanilla_smooth))
                    b_ep = next((i for i, r in enumerate(bootstrapped_smooth) if r >= threshold), len(bootstrapped_smooth))
                    
                    if v_ep < len(vanilla_smooth) and b_ep < len(bootstrapped_smooth):
                        reduction = (v_ep - b_ep) / v_ep * 100
                        print(f"  {threshold_pct}% of final performance ({threshold:.2f} reward):")
                        print(f"    Vanilla: {v_ep:6d} episodes")
                        print(f"    BDQN:    {b_ep:6d} episodes")
                        print(f"    Gain:    {reduction:+6.1f}% fewer episodes {'✅' if reduction > 0 else '❌'}")
                    else:
                        print(f"  {threshold_pct}% threshold: One or both methods did not reach this level")
                
                # Overall timestep efficiency
                if vanilla_timesteps > 0 and bootstrapped_timesteps > 0:
                    timestep_reduction = (vanilla_timesteps - bootstrapped_timesteps) / vanilla_timesteps * 100
                    print(f"\n💡 OVERALL SAMPLE EFFICIENCY:")
                    print(f"  Timestep reduction: {timestep_reduction:+.1f}%")
                    if timestep_reduction > 0:
                        print(f"  ✅ Bootstrapped DQN is more sample efficient!")
                    else:
                        print(f"  ⚠️  Vanilla DQN was more sample efficient in this run")
                
                # Exploration diversity (if available)
                if 'update_counts' in bootstrapped_data:
                    update_counts = bootstrapped_data['update_counts']
                    print(f"\n🔄 BOOTSTRAP HEAD UTILIZATION:")
                    print(f"  Update counts: {update_counts}")
                    print(f"  Mean:  {np.mean(update_counts):.0f}")
                    print(f"  Std:   {np.std(update_counts):.0f}")
                    balance = np.std(update_counts) / np.mean(update_counts)
                    print(f"  Balance: {balance:.3f} (lower is better)")
    
    except FileNotFoundError as e:
        print(f"❌ Data file not found: {e}")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


def plot_learning_curves():
    """
    Plot learning curves comparing vanilla DQN vs Bootstrapped DQN
    """
    
    if not os.path.exists('exps_data/vanilla/prime_experiment_stats.h5'):
        print("No vanilla data found. Run prime_experiment.py first.")
        return
    
    if not os.path.exists('exps_data/bootstrapped/prime_experiment_stats.h5'):
        print("No bootstrapped data found. Run prime_experiment.py first.")
        return
    
    print("\n" + "="*70)
    print("GENERATING LEARNING CURVE COMPARISON PLOTS")
    print("="*70)
    
    try:
        vanilla_stats = dd.io.load('exps_data/vanilla/prime_experiment_stats.h5')
        bootstrapped_stats = dd.io.load('exps_data/bootstrapped/prime_experiment_stats.h5')
        
        # Create figure with subplots for each task
        n_tasks = len(Tasks)
        fig, axes = plt.subplots(1, n_tasks, figsize=(8*n_tasks, 6))
        if n_tasks == 1:
            axes = [axes]
        
        for task_idx in range(n_tasks):
            task_name = Tasks[task_idx][0] + Tasks[task_idx][1]
            
            vanilla_data = vanilla_stats[0, task_idx]
            bootstrapped_data = bootstrapped_stats[0, task_idx]
            
            ax = axes[task_idx]
            
            # Plot vanilla DQN
            if 'R' in vanilla_data and vanilla_data['R']:
                vanilla_rewards = vanilla_data['R']
                # Smooth with running average
                window = 100
                vanilla_smooth = np.convolve(vanilla_rewards, 
                                           np.ones(window)/window, 
                                           mode='valid')
                ax.plot(vanilla_smooth, label='Vanilla DQN (ε-greedy)', 
                       color='#2E86AB', alpha=0.8, linewidth=2.5)
            
            # Plot bootstrapped DQN
            if 'R' in bootstrapped_data and bootstrapped_data['R']:
                bootstrapped_rewards = bootstrapped_data['R']
                bootstrapped_smooth = np.convolve(bootstrapped_rewards, 
                                                 np.ones(window)/window, 
                                                 mode='valid')
                ax.plot(bootstrapped_smooth, label='Bootstrapped DQN (Thompson)', 
                       color='#A23B72', alpha=0.8, linewidth=2.5)
            
            ax.set_xlabel('Episode', fontsize=14, fontweight='bold')
            ax.set_ylabel('Reward (100-ep moving avg)', fontsize=14, fontweight='bold')
            ax.set_title(f'Learning Curve: {task_name.upper()}', 
                        fontsize=16, fontweight='bold')
            ax.legend(fontsize=12, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/vanilla_vs_bootstrapped_comparison.png', 
                   dpi=300, bbox_inches='tight')
        print("\n✅ Learning curves saved to: plots/vanilla_vs_bootstrapped_comparison.png")
        plt.show()
        
    except ImportError:
        print("❌ Matplotlib not available. Install with: pip install matplotlib")
    except Exception as e:
        print(f"❌ Error creating plots: {e}")
        import traceback
        traceback.print_exc()


def generate_comparison_table():
    """
    Generate a markdown table comparing the two methods
    """
    
    if not os.path.exists('exps_data/vanilla/prime_experiment_stats.h5'):
        print("No vanilla data found.")
        return
    
    if not os.path.exists('exps_data/bootstrapped/prime_experiment_stats.h5'):
        print("No bootstrapped data found.")
        return
    
    try:
        vanilla_stats = dd.io.load('exps_data/vanilla/prime_experiment_stats.h5')
        bootstrapped_stats = dd.io.load('exps_data/bootstrapped/prime_experiment_stats.h5')
        
        print("\n" + "="*70)
        print("COMPARISON TABLE")
        print("="*70)
        print("\n| Metric | Vanilla DQN | Bootstrapped DQN | Improvement |")
        print("|--------|-------------|------------------|-------------|")
        
        for task_idx in range(len(Tasks)):
            task_name = Tasks[task_idx][0] + Tasks[task_idx][1]
            
            vanilla_data = vanilla_stats[0, task_idx]
            bootstrapped_data = bootstrapped_stats[0, task_idx]
            
            if 'R' in vanilla_data and 'R' in bootstrapped_data:
                v_final = np.mean(vanilla_data['R'][-100:])
                b_final = np.mean(bootstrapped_data['R'][-100:])
                v_episodes = len(vanilla_data['R'])
                b_episodes = len(bootstrapped_data['R'])
                v_timesteps = vanilla_data.get('T', 0)
                b_timesteps = bootstrapped_data.get('T', 0)
                
                reward_improvement = ((b_final - v_final) / abs(v_final)) * 100 if v_final != 0 else 0
                episode_reduction = ((v_episodes - b_episodes) / v_episodes) * 100
                timestep_reduction = ((v_timesteps - b_timesteps) / v_timesteps) * 100
                
                print(f"| **Task: {task_name}** | | | |")
                print(f"| Final Reward | {v_final:.2f} | {b_final:.2f} | {reward_improvement:+.1f}% |")
                print(f"| Total Episodes | {v_episodes} | {b_episodes} | {episode_reduction:+.1f}% |")
                print(f"| Total Timesteps | {v_timesteps:,} | {b_timesteps:,} | {timestep_reduction:+.1f}% |")
        
        print("\n")
        
    except Exception as e:
        print(f"Error generating table: {e}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("VANILLA DQN vs BOOTSTRAPPED DQN - RESULTS ANALYSIS")
    print("="*70)
    
    # Run all analyses
    analyze_sample_efficiency()
    plot_learning_curves()
    generate_comparison_table()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
