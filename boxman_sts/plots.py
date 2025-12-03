"""
Plotting and analysis functions for DQN experiments
"""
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm, rcParams
from matplotlib import rc
import os
import pandas as pd
import seaborn as sns
import deepdish as dd


def analyze_sample_efficiency():
    """
    Analyze and plot sample efficiency: compare number of episodes/timesteps 
    to reach threshold performance between vanilla and bootstrapped DQN
    """
    if not os.path.exists('exps_data/vanilla/prime_experiment_stats.h5'):
        print("Vanilla DQN data not found. Run prime_experiment.py or the vanilla portion first.")
        return
    
    if not os.path.exists('exps_data/bootstrapped/prime_experiment_stats.h5'):
        print("Bootstrapped DQN data not found. Run prime_experiment_bootstrapped_only.py first.")
        print("Note: The bootstrapped experiment must complete successfully for stats to be saved.")
        return
    
    print("\n" + "="*70)
    print("SAMPLE EFFICIENCY ANALYSIS")
    print("="*70)
    
    try:
        vanilla_stats = dd.io.load('exps_data/vanilla/prime_experiment_stats.h5')
        bootstrapped_stats = dd.io.load('exps_data/bootstrapped/prime_experiment_stats.h5')
        
        # Import Tasks from prime_experiment
        from prime_experiment import Tasks
        
        # Prepare data for plotting
        n_tasks = vanilla_stats.shape[1]
        n_runs = vanilla_stats.shape[0]
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 10))
        
        for run_idx in range(n_runs):
            for task_idx in range(n_tasks):
                task_name = Tasks[task_idx][0] + Tasks[task_idx][1]
                
                vanilla_data = vanilla_stats[run_idx, task_idx]
                bootstrapped_data = bootstrapped_stats[run_idx, task_idx]
                
                print(f"\nTask: {task_name} (Run {run_idx})")
                print("-" * 40)
                
                # Skip if there were errors
                if 'error' in vanilla_data or 'error' in bootstrapped_data:
                    if 'error' in vanilla_data:
                        print(f"Vanilla error: {vanilla_data['error']}")
                    if 'error' in bootstrapped_data:
                        print(f"Bootstrapped error: {bootstrapped_data['error']}")
                    continue
                
                # Vanilla DQN analysis
                vanilla_metrics = {}
                if 'R' in vanilla_data and vanilla_data['R']:
                    vanilla_rewards = vanilla_data['R']
                    vanilla_episodes = len(vanilla_rewards)
                    vanilla_timesteps = vanilla_data.get('T', 0)
                    vanilla_final_100 = np.mean(vanilla_rewards[-100:])
                    
                    # Find episodes to reach 50%, 75%, 90% of final performance
                    threshold_50 = vanilla_final_100 * 0.5
                    threshold_75 = vanilla_final_100 * 0.75
                    threshold_90 = vanilla_final_100 * 0.9
                    
                    # Running average to smooth noise
                    window = 100
                    vanilla_smooth = np.convolve(vanilla_rewards, 
                                                np.ones(window)/window, 
                                                mode='valid')
                    
                    ep_50 = next((i for i, r in enumerate(vanilla_smooth) if r >= threshold_50), vanilla_episodes)
                    ep_75 = next((i for i, r in enumerate(vanilla_smooth) if r >= threshold_75), vanilla_episodes)
                    ep_90 = next((i for i, r in enumerate(vanilla_smooth) if r >= threshold_90), vanilla_episodes)
                    
                    vanilla_metrics = {
                        'episodes': vanilla_episodes,
                        'timesteps': vanilla_timesteps,
                        'final_reward': vanilla_final_100,
                        'ep_50': ep_50,
                        'ep_75': ep_75,
                        'ep_90': ep_90,
                        'smooth': vanilla_smooth,
                        'rewards': vanilla_rewards
                    }
                    
                    print(f"Vanilla DQN:")
                    print(f"  Total episodes: {vanilla_episodes}")
                    print(f"  Total timesteps: {vanilla_timesteps}")
                    print(f"  Final reward: {vanilla_final_100:.2f}")
                    print(f"  Episodes to 50% performance: {ep_50}")
                    print(f"  Episodes to 75% performance: {ep_75}")
                    print(f"  Episodes to 90% performance: {ep_90}")
                
                # Bootstrapped DQN analysis
                bootstrapped_metrics = {}
                if 'R' in bootstrapped_data and bootstrapped_data['R']:
                    bootstrapped_rewards = bootstrapped_data['R']
                    bootstrapped_episodes = len(bootstrapped_rewards)
                    bootstrapped_timesteps = bootstrapped_data.get('T', 0)
                    bootstrapped_final_100 = np.mean(bootstrapped_rewards[-100:])
                    
                    # Find episodes to reach 50%, 75%, 90% of final performance
                    threshold_50 = bootstrapped_final_100 * 0.5
                    threshold_75 = bootstrapped_final_100 * 0.75
                    threshold_90 = bootstrapped_final_100 * 0.9
                    
                    # Running average
                    window = 100
                    bootstrapped_smooth = np.convolve(bootstrapped_rewards, 
                                                     np.ones(window)/window, 
                                                     mode='valid')
                    
                    ep_50_b = next((i for i, r in enumerate(bootstrapped_smooth) if r >= threshold_50), bootstrapped_episodes)
                    ep_75_b = next((i for i, r in enumerate(bootstrapped_smooth) if r >= threshold_75), bootstrapped_episodes)
                    ep_90_b = next((i for i, r in enumerate(bootstrapped_smooth) if r >= threshold_90), bootstrapped_episodes)
                    
                    bootstrapped_metrics = {
                        'episodes': bootstrapped_episodes,
                        'timesteps': bootstrapped_timesteps,
                        'final_reward': bootstrapped_final_100,
                        'ep_50': ep_50_b,
                        'ep_75': ep_75_b,
                        'ep_90': ep_90_b,
                        'smooth': bootstrapped_smooth,
                        'rewards': bootstrapped_rewards
                    }
                    
                    print(f"\nBootstrapped DQN:")
                    print(f"  Total episodes: {bootstrapped_episodes}")
                    print(f"  Total timesteps: {bootstrapped_timesteps}")
                    print(f"  Final reward: {bootstrapped_final_100:.2f}")
                    print(f"  Episodes to 50% performance: {ep_50_b}")
                    print(f"  Episodes to 75% performance: {ep_75_b}")
                    print(f"  Episodes to 90% performance: {ep_90_b}")
                    
                    # Sample efficiency comparison
                    if vanilla_metrics:
                        print(f"\nSample Efficiency Gain:")
                        if vanilla_metrics['ep_50'] > 0:
                            gain_50 = (vanilla_metrics['ep_50'] - ep_50_b) / vanilla_metrics['ep_50'] * 100
                            print(f"  50% threshold: {gain_50:.1f}% fewer episodes")
                        if vanilla_metrics['ep_75'] > 0:
                            gain_75 = (vanilla_metrics['ep_75'] - ep_75_b) / vanilla_metrics['ep_75'] * 100
                            print(f"  75% threshold: {gain_75:.1f}% fewer episodes")
                        if vanilla_metrics['ep_90'] > 0:
                            gain_90 = (vanilla_metrics['ep_90'] - ep_90_b) / vanilla_metrics['ep_90'] * 100
                            print(f"  90% threshold: {gain_90:.1f}% fewer episodes")
                        
                        timestep_gain = (vanilla_metrics['timesteps'] - bootstrapped_timesteps) / vanilla_metrics['timesteps'] * 100
                        print(f"  Total timesteps: {timestep_gain:.1f}% reduction")
                
                # Create plots if we have both sets of metrics
                if vanilla_metrics and bootstrapped_metrics:
                    # Plot 1: Learning curves with threshold lines
                    ax1 = plt.subplot(2, 2, 1)
                    ax1.plot(vanilla_metrics['smooth'], label='Vanilla DQN', 
                            alpha=0.8, linewidth=2, color='steelblue')
                    ax1.plot(bootstrapped_metrics['smooth'], label='Bootstrapped DQN', 
                            alpha=0.8, linewidth=2, color='darkorange')
                    
                    # Add threshold lines
                    max_ep = max(len(vanilla_metrics['smooth']), len(bootstrapped_metrics['smooth']))
                    ax1.axhline(vanilla_metrics['final_reward'] * 0.5, 
                               color='gray', linestyle='--', alpha=0.5, label='50% threshold')
                    ax1.axhline(vanilla_metrics['final_reward'] * 0.75, 
                               color='gray', linestyle='-.', alpha=0.5, label='75% threshold')
                    ax1.axhline(vanilla_metrics['final_reward'] * 0.9, 
                               color='gray', linestyle=':', alpha=0.5, label='90% threshold')
                    
                    ax1.set_xlabel('Episode', fontsize=11)
                    ax1.set_ylabel('Reward (100-ep moving avg)', fontsize=11)
                    ax1.set_title(f'Learning Curves: {task_name}', fontsize=12, fontweight='bold')
                    ax1.legend(fontsize=9)
                    ax1.grid(True, alpha=0.3)
                    
                    # Plot 2: Sample efficiency comparison (bar chart)
                    ax2 = plt.subplot(2, 2, 2)
                    thresholds = ['50%', '75%', '90%']
                    vanilla_eps = [vanilla_metrics['ep_50'], vanilla_metrics['ep_75'], vanilla_metrics['ep_90']]
                    bootstrapped_eps = [bootstrapped_metrics['ep_50'], bootstrapped_metrics['ep_75'], bootstrapped_metrics['ep_90']]
                    
                    x = np.arange(len(thresholds))
                    width = 0.35
                    
                    bars1 = ax2.bar(x - width/2, vanilla_eps, width, label='Vanilla DQN', 
                                   color='steelblue', alpha=0.8)
                    bars2 = ax2.bar(x + width/2, bootstrapped_eps, width, label='Bootstrapped DQN', 
                                   color='darkorange', alpha=0.8)
                    
                    ax2.set_xlabel('Performance Threshold', fontsize=11)
                    ax2.set_ylabel('Episodes to Reach', fontsize=11)
                    ax2.set_title(f'Sample Efficiency: {task_name}', fontsize=12, fontweight='bold')
                    ax2.set_xticks(x)
                    ax2.set_xticklabels(thresholds)
                    ax2.legend(fontsize=9)
                    ax2.grid(True, alpha=0.3, axis='y')
                    
                    # Add value labels on bars
                    for bars in [bars1, bars2]:
                        for bar in bars:
                            height = bar.get_height()
                            ax2.annotate(f'{int(height)}',
                                       xy=(bar.get_x() + bar.get_width() / 2, height),
                                       xytext=(0, 3),
                                       textcoords="offset points",
                                       ha='center', va='bottom', fontsize=8)
                    
                    # Plot 3: Cumulative timesteps comparison
                    ax3 = plt.subplot(2, 2, 3)
                    vanilla_cumulative = np.cumsum([len(ep) if hasattr(ep, '__len__') else 1 
                                                   for ep in vanilla_metrics['rewards']])
                    bootstrapped_cumulative = np.cumsum([len(ep) if hasattr(ep, '__len__') else 1 
                                                        for ep in bootstrapped_metrics['rewards']])
                    
                    ax3.plot(vanilla_cumulative, label='Vanilla DQN', 
                            alpha=0.8, linewidth=2, color='steelblue')
                    ax3.plot(bootstrapped_cumulative, label='Bootstrapped DQN', 
                            alpha=0.8, linewidth=2, color='darkorange')
                    
                    ax3.set_xlabel('Episode', fontsize=11)
                    ax3.set_ylabel('Cumulative Timesteps', fontsize=11)
                    ax3.set_title(f'Timesteps Usage: {task_name}', fontsize=12, fontweight='bold')
                    ax3.legend(fontsize=9)
                    ax3.grid(True, alpha=0.3)
                    
                    # Plot 4: Efficiency gain percentages
                    ax4 = plt.subplot(2, 2, 4)
                    metrics_names = ['50% Threshold', '75% Threshold', '90% Threshold', 'Total Timesteps']
                    gains = []
                    
                    if vanilla_metrics['ep_50'] > 0:
                        gains.append((vanilla_metrics['ep_50'] - bootstrapped_metrics['ep_50']) / 
                                   vanilla_metrics['ep_50'] * 100)
                    else:
                        gains.append(0)
                    
                    if vanilla_metrics['ep_75'] > 0:
                        gains.append((vanilla_metrics['ep_75'] - bootstrapped_metrics['ep_75']) / 
                                   vanilla_metrics['ep_75'] * 100)
                    else:
                        gains.append(0)
                    
                    if vanilla_metrics['ep_90'] > 0:
                        gains.append((vanilla_metrics['ep_90'] - bootstrapped_metrics['ep_90']) / 
                                   vanilla_metrics['ep_90'] * 100)
                    else:
                        gains.append(0)
                    
                    if vanilla_metrics['timesteps'] > 0:
                        gains.append((vanilla_metrics['timesteps'] - bootstrapped_metrics['timesteps']) / 
                                   vanilla_metrics['timesteps'] * 100)
                    else:
                        gains.append(0)
                    
                    colors = ['green' if g > 0 else 'red' for g in gains]
                    bars = ax4.barh(metrics_names, gains, color=colors, alpha=0.7)
                    
                    ax4.set_xlabel('% Improvement (Fewer Episodes/Timesteps)', fontsize=11)
                    ax4.set_title(f'BDQN Efficiency Gains: {task_name}', fontsize=12, fontweight='bold')
                    ax4.axvline(0, color='black', linewidth=0.8)
                    ax4.grid(True, alpha=0.3, axis='x')
                    
                    # Add value labels
                    for i, (bar, gain) in enumerate(zip(bars, gains)):
                        ax4.text(gain + (2 if gain > 0 else -2), i, f'{gain:.1f}%',
                               ha='left' if gain > 0 else 'right', va='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save the plot
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/sample_efficiency_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n{'='*70}")
        print("Sample efficiency plots saved to: plots/sample_efficiency_analysis.png")
        print(f"{'='*70}")
        plt.show()
    
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


def plot_learning_curves():
    """
    Plot learning curves comparing vanilla DQN vs Bootstrapped DQN
    """
    if not os.path.exists('exps_data/vanilla/prime_experiment_stats.h5'):
        print("Vanilla DQN data not found. Run prime_experiment.py or the vanilla portion first.")
        return
    
    if not os.path.exists('exps_data/bootstrapped/prime_experiment_stats.h5'):
        print("Bootstrapped DQN data not found. Run prime_experiment_bootstrapped_only.py first.")
        print("Note: The bootstrapped experiment must complete successfully for stats to be saved.")
        return
    
    try:
        vanilla_stats = dd.io.load('exps_data/vanilla/prime_experiment_stats.h5')
        bootstrapped_stats = dd.io.load('exps_data/bootstrapped/prime_experiment_stats.h5')
        
        # Import Tasks from prime_experiment
        from prime_experiment import Tasks
        
        # Create figure with subplots for each task
        n_tasks = len(Tasks)
        fig, axes = plt.subplots(1, n_tasks, figsize=(6*n_tasks, 5))
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
                       alpha=0.8, linewidth=2, color='steelblue')
            
            # Plot bootstrapped DQN
            if 'R' in bootstrapped_data and bootstrapped_data['R']:
                bootstrapped_rewards = bootstrapped_data['R']
                bootstrapped_smooth = np.convolve(bootstrapped_rewards, 
                                                 np.ones(window)/window, 
                                                 mode='valid')
                ax.plot(bootstrapped_smooth, label='Bootstrapped DQN (Thompson)', 
                       alpha=0.8, linewidth=2, color='darkorange')
            
            ax.set_xlabel('Episode', fontsize=12)
            ax.set_ylabel('Reward (100-ep moving avg)', fontsize=12)
            ax.set_title(f'Task: {task_name}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/vanilla_vs_bootstrapped_comparison.png', 
                   dpi=300, bbox_inches='tight')
        print("\nLearning curves saved to: plots/vanilla_vs_bootstrapped_comparison.png")
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Install with: pip install matplotlib")
    except Exception as e:
        print(f"Error creating plots: {e}")
        import traceback
        traceback.print_exc()


def plot1():
    tasks = ['Blue','Square','OR', 'AND', 'XOR']
        
    s = 20
    rc_ = {'figure.figsize':(10,5),'axes.labelsize': 30, 'font.size': 30, 
          'legend.fontsize': 20, 'axes.titlesize': 30}
    sns.set(rc=rc_, style="darkgrid",font_scale = 1.8)
    rc('text', usetex=False)
        
    data0 = dd.io.load('data/exp_returns_0.h5')[:1000,:]
    data1 = dd.io.load('data/exp_returns_1.h5')[:1000,:]
    types = ["Optimal",
              "Composed",
            ]
    
    data = pd.DataFrame(
    [[data0[i,t] for t in range(len(tasks))]+[types[0]] for i in range(len(data1))] +
    [[data1[i,t] for t in range(len(tasks))]+[types[1]] for i in range(len(data1))],
      columns=tasks+[""])
    data = pd.melt(data, "", var_name="Tasks", value_name="Average Returns")
    
    fig, ax = plt.subplots()
    ax = sns.boxplot(x="Tasks", y="Average Returns", hue="", data=data, linewidth=3, showfliers = False)
    plt.show()
    fig.savefig("plots/returns.pdf", bbox_inches='tight')

#####################################################################################

def plot2():
    tasks = ['blue','square','or', 'and', 'xor']
        
    s = 20
    rc_ = {'figure.figsize':(4,5),'axes.labelsize': 30, 'font.size': 30, 
          'legend.fontsize': 20, 'axes.titlesize': 30}
    sns.set(rc=rc_, style="darkgrid",font_scale = 1.8)
    rc('text', usetex=False)
        
    data0 = dd.io.load('data/exp_returns_0.h5')[:1000,:]
    data1 = dd.io.load('data/exp_returns_1.h5')[:1000,:]
    types = ["Optimal",
              "Composed",
            ]
    
    for task in range(len(tasks)):
        data = data0[:,:2]
        data[:,0] = data0[:,task]
        data[:,1] = data1[:,task]
        data = pd.DataFrame(
        [[data[i,t] for t in range(len(data[i]))] for i in range(len(data))],
          columns=types)
        # data = pd.melt(data, "", var_name="Tasks", value_name="Average Returns")
        
        fig, ax = plt.subplots()
        ax = sns.boxplot(data=data, linewidth=3, showfliers = False)
        plt.xlabel('Tasks')
        plt.ylabel('Average Returns')
        # plt.show()
        fig.savefig("plots/returns_{0}.pdf".format(tasks[task]), bbox_inches='tight')

# analyze_sample_efficiency()
# plot_learning_curves()
plot1()
plot2()
